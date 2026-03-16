#!/bin/bash
# =============================================================================
# nanochat FULL TRAINING JOB — d26 (GPT-2 grade), ~16-20h on 1xH100
#
# CHECKPOINT STRATEGY:
#   1. On first run:   starts training from scratch, saves every 1000 steps
#   2. On requeue:     auto-detects latest LOCAL checkpoint and resumes
#   3. If no local:    downloads the latest checkpoint from HuggingFace Hub
#   4. On preemption:  SIGUSR1 handler triggers a clean checkpoint save
#                      before Slurm kills the job, then requeues automatically
#   5. Checkpoints are continuously synced to HuggingFace Hub in the background
#
# USAGE:
#   First run:   sbatch train_nanochat.sbatch
#   Manual resume (if needed): sbatch --export=ALL,FORCE_RESUME=1 train_nanochat.sbatch
# =============================================================================
#SBATCH --job-name=nanochat_train
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/nanochat_train_%j.out
#SBATCH --error=logs/nanochat_train_%j.err
# Send SIGUSR1 120 seconds before the time limit so we can checkpoint cleanly
#SBATCH --signal=B:USR1@120

# ── Paths — edit these ───────────────────────────────────────────────────────
NANOCHAT_DIR="$HOME/nanochat-exp1"
SIF_IMAGE="$HOME/containers/nanochat.sif"
CHECKPOINT_DIR="$HOME/checkpoints/nanochat_d26"
RUN_NAME="nanochat_d26"

# ── HuggingFace Hub — edit these ─────────────────────────────────────────────
HF_REPO_ID="pandevim/nanochat-d26-checkpoints"   # ← change this
HF_SYNC_INTERVAL=300   # seconds between background syncs (5 min)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
echo "[$(date)] ============================================"
echo "[$(date)] nanochat training job ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Depth:   26 (GPT-2 grade)"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "HF Repo: ${HF_REPO_ID}"
echo "[$(date)] ============================================"

mkdir -p logs "${CHECKPOINT_DIR}"

# ── Fast local SSD ────────────────────────────────────────────────────────────
LOCAL_DIR="/local/${SLURM_JOB_ID}"
mkdir -p \
    "${LOCAL_DIR}/hf_cache" \
    "${LOCAL_DIR}/apptainer_cache" \
    "${LOCAL_DIR}/apptainer_tmp"

export APPTAINER_CACHEDIR="${LOCAL_DIR}/apptainer_cache"
export APPTAINER_TMPDIR="${LOCAL_DIR}/apptainer_tmp"
export APPTAINER_BIND="\
${LOCAL_DIR}/hf_cache:/root/.cache/huggingface,\
${NANOCHAT_DIR}:/workspace,\
${CHECKPOINT_DIR}:/checkpoints"

module load apptainer

# Load secrets (must contain HF_TOKEN and optionally WANDB_API_KEY)
source ~/.secrets/nanochat.key

# ── Helper: run Python inside the container ───────────────────────────────────
run_in_container() {
    apptainer exec --nv \
        --pwd /workspace \
        --env HF_TOKEN="${HF_TOKEN}" \
        "${SIF_IMAGE}" \
        python -c "$1"
}

# ── Helper: upload checkpoints to HuggingFace Hub ────────────────────────────
sync_checkpoints_to_hf() {
    echo "[$(date)] Syncing checkpoints to HuggingFace Hub (${HF_REPO_ID})..."
    run_in_container "
from huggingface_hub import HfApi, create_repo
import glob, os

api = HfApi()

# Create repo if it doesn't exist (private by default)
try:
    create_repo('${HF_REPO_ID}', repo_type='model', private=True, exist_ok=True)
except Exception as e:
    print(f'Repo creation note: {e}')

# Upload all checkpoint files
ckpt_files = glob.glob('/checkpoints/step_*.pt')
for f in ckpt_files:
    basename = os.path.basename(f)
    print(f'  Uploading {basename}...')
    try:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=basename,
            repo_id='${HF_REPO_ID}',
            repo_type='model',
        )
        print(f'  ✓ {basename} uploaded')
    except Exception as e:
        print(f'  ✗ {basename} failed: {e}')

print('Sync complete.')
" 2>&1 || echo "[$(date)] WARNING: HF sync failed (non-fatal)"
}

# ── Background sync loop ─────────────────────────────────────────────────────
# Periodically uploads new checkpoints to HF while training runs
background_hf_sync() {
    while true; do
        sleep "${HF_SYNC_INTERVAL}"
        sync_checkpoints_to_hf
    done
}

# ── Checkpoint resume detection ───────────────────────────────────────────────
# Step 1: Check LOCAL checkpoints first
# Step 2: If none found, try downloading the latest from HuggingFace Hub
RESUME_FLAG=""
LATEST_CKPT=$(run_in_container "
import glob, re
ckpts = glob.glob('/checkpoints/step_*.pt')
if not ckpts:
    print('')
else:
    steps = [int(re.search(r'step_(\d+)', f).group(1)) for f in ckpts if re.search(r'step_(\d+)', f)]
    if steps:
        latest = max(steps)
        print(f'/checkpoints/step_{latest:06d}.pt')
    else:
        print('')
" 2>/dev/null || echo "")

if [[ -n "${LATEST_CKPT}" ]]; then
    echo "[$(date)] Found LOCAL checkpoint: ${LATEST_CKPT} — resuming training"
    RESUME_FLAG="--resume=${LATEST_CKPT}"
else
    echo "[$(date)] No local checkpoint found. Checking HuggingFace Hub..."
    LATEST_CKPT=$(run_in_container "
from huggingface_hub import HfApi, hf_hub_download
import re

api = HfApi()
try:
    files = api.list_repo_files('${HF_REPO_ID}', repo_type='model')
    ckpt_files = [f for f in files if re.match(r'step_\d+\.pt', f)]
    if not ckpt_files:
        print('')
    else:
        steps = [(int(re.search(r'step_(\d+)', f).group(1)), f) for f in ckpt_files]
        latest_step, latest_file = max(steps, key=lambda x: x[0])
        # Download to the local checkpoint dir
        local_path = hf_hub_download(
            repo_id='${HF_REPO_ID}',
            filename=latest_file,
            repo_type='model',
            local_dir='/checkpoints',
        )
        print(f'/checkpoints/{latest_file}')
except Exception as e:
    print('', end='')  # silently fall through to train from scratch
" 2>/dev/null || echo "")

    if [[ -n "${LATEST_CKPT}" ]]; then
        echo "[$(date)] Downloaded checkpoint from HF: ${LATEST_CKPT} — resuming training"
        RESUME_FLAG="--resume=${LATEST_CKPT}"
    else
        echo "[$(date)] No checkpoint found anywhere — starting from scratch"
    fi
fi

# ── Graceful preemption handler ───────────────────────────────────────────────
# When Slurm sends SIGUSR1 (120s before kill), we write a sentinel file
# that tells the training process to checkpoint and exit cleanly.
# After the job exits, Slurm requeues it automatically.
PREEMPT_FLAG_FILE="${CHECKPOINT_DIR}/.preempt_requested"

handle_preemption() {
    echo "[$(date)] Received SIGUSR1 — time limit approaching. Requesting checkpoint..."
    touch "${PREEMPT_FLAG_FILE}"
    # Give the training process up to 60s to finish its checkpoint
    sleep 60
    # Final sync to HF before requeue
    echo "[$(date)] Final checkpoint sync to HuggingFace before requeue..."
    sync_checkpoints_to_hf
    echo "[$(date)] Requeuing job ${SLURM_JOB_ID}..."
    scontrol requeue "${SLURM_JOB_ID}"
}
trap handle_preemption USR1

# ── Main training command ─────────────────────────────────────────────────────
# Notes on single-GPU settings:
#   --device_batch_size 32    : adjust down to 16/8 if you get OOM
#   --save-every 1000         : checkpoint every 1000 steps (~30 min)
#   NANOCHAT_DTYPE=bfloat16   : H100 native precision

apptainer exec --nv \
    --pwd /workspace \
    --env NANOCHAT_DTYPE=bfloat16 \
    --env WANDB_RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    "${SIF_IMAGE}" \
    python -m scripts.base_train -- \
        --depth=26 \
        --run="${RUN_NAME}" \
        --model-tag="d26" \
        --save-dir=/checkpoints \
        --save-every=1000 \
        --device_batch_size=32 \
        ${RESUME_FLAG} &

TRAIN_PID=$!
echo "[$(date)] Training started with PID ${TRAIN_PID}"

# Start background HF sync loop
background_hf_sync &
HF_SYNC_PID=$!
echo "[$(date)] Background HF sync started (every ${HF_SYNC_INTERVAL}s, PID ${HF_SYNC_PID})"

# Wait for training while keeping signal handler alive
wait ${TRAIN_PID}
TRAIN_EXIT=$?

# Kill background sync loop
kill ${HF_SYNC_PID} 2>/dev/null || true
wait ${HF_SYNC_PID} 2>/dev/null || true

# ── Post-job: final sync + cleanup ───────────────────────────────────────────
rm -f "${PREEMPT_FLAG_FILE}"

echo "[$(date)] Final checkpoint sync to HuggingFace..."
sync_checkpoints_to_hf

if [[ ${TRAIN_EXIT} -eq 0 ]]; then
    echo "[$(date)] Training completed successfully!"
    echo "Checkpoints saved locally: ${CHECKPOINT_DIR}"
    echo "Checkpoints saved on HF:   https://huggingface.co/${HF_REPO_ID}"
    echo "To chat with your model, run:"
    echo "  apptainer exec --nv ${SIF_IMAGE} python -m scripts.chat_web"
else
    echo "[$(date)] Training exited with code ${TRAIN_EXIT}"
fi

exit ${TRAIN_EXIT}
