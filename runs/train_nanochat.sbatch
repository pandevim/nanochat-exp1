#!/bin/bash
# =============================================================================
# nanochat FULL TRAINING JOB — d26 (GPT-2 grade), ~16-20h on 1xH100
#
# CHECKPOINT STRATEGY:
#   1. On first run:   starts training from scratch, saves every 1000 steps
#   2. On requeue:     auto-detects latest checkpoint and resumes
#   3. On preemption:  SIGUSR1 handler triggers a clean checkpoint save
#                      before Slurm kills the job, then requeues automatically
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
NANOCHAT_DIR="$HOME/nanochat"
SIF_IMAGE="$HOME/containers/nanochat.sif"
CHECKPOINT_DIR="$HOME/checkpoints/nanochat_d26"
RUN_NAME="nanochat_d26"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
echo "[$(date)] ============================================"
echo "[$(date)] nanochat training job ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Depth:   26 (GPT-2 grade)"
echo "Checkpoints: ${CHECKPOINT_DIR}"
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

# Uncomment to load WandB / other secrets:
# source ~/.secrets/nanochat.key

# ── Checkpoint resume detection ───────────────────────────────────────────────
# Finds the latest step checkpoint in the save directory
RESUME_FLAG=""
LATEST_CKPT=$(apptainer exec --nv \
    --pwd /workspace \
    "${SIF_IMAGE}" \
    python -c "
import os, glob, re
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
    echo "[$(date)] Found checkpoint: ${LATEST_CKPT} — resuming training"
    RESUME_FLAG="--resume=${LATEST_CKPT}"
else
    echo "[$(date)] No checkpoint found — starting from scratch"
fi

# ── Graceful preemption handler ───────────────────────────────────────────────
# When Slurm sends SIGUSR1 (120s before kill), we write a sentinel file
# that tells the training process to checkpoint and exit cleanly.
# After the job exits, Slurm requeues it automatically.
PREEMPT_FLAG_FILE="${CHECKPOINT_DIR}/.preempt_requested"

handle_preemption() {
    echo "[$(date)] Received SIGUSR1 — time limit approaching. Requesting checkpoint..."
    touch "${PREEMPT_FLAG_FILE}"
    # Give the training process up to 100s to finish its checkpoint
    sleep 100
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

# Wait for training while keeping signal handler alive
wait ${TRAIN_PID}
TRAIN_EXIT=$?

# ── Post-job cleanup ──────────────────────────────────────────────────────────
rm -f "${PREEMPT_FLAG_FILE}"

if [[ ${TRAIN_EXIT} -eq 0 ]]; then
    echo "[$(date)] Training completed successfully!"
    echo "To chat with your model, run:"
    echo "  apptainer exec --nv ${SIF_IMAGE} python -m scripts.chat_web"
else
    echo "[$(date)] Training exited with code ${TRAIN_EXIT}"
fi

exit ${TRAIN_EXIT}
