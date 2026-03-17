#!/bin/bash
# =============================================================================
# nanochat TEST JOB — d12 model, ~5-10 minutes
# Validates the full stack: container, GPU, dataset download, training loop
# =============================================================================
#SBATCH --job-name=nanochat_test
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/logs/nanochat_test_%j.out
#SBATCH --error=/home/%u/logs/nanochat_test_%j.err

# ── Paths — edit these to match your environment ─────────────────────────────
NANOCHAT_DIR="$HOME/nanochat-exp1"          # Where you cloned the repo
SIF_IMAGE="$HOME/containers/nanochat.sif"
CHECKPOINT_DIR="$HOME/checkpoints/nanochat_test"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
echo "[$(date)] Starting nanochat TEST job ${SLURM_JOB_ID}"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Create output dirs
mkdir -p logs "$CHECKPOINT_DIR"

# ── Fast local SSD setup (prevents home-dir quota issues) ────────────────────
LOCAL_DIR="/local/${SLURM_JOB_ID}"
mkdir -p \
    "${LOCAL_DIR}/hf_cache" \
    "${LOCAL_DIR}/apptainer_cache" \
    "${LOCAL_DIR}/apptainer_tmp"

export APPTAINER_CACHEDIR="${LOCAL_DIR}/apptainer_cache"
export APPTAINER_TMPDIR="${LOCAL_DIR}/apptainer_tmp"

# ── Apptainer bind mounts ─────────────────────────────────────────────────────
# Bind: HuggingFace cache → fast SSD, nanochat source → /workspace
export APPTAINER_BIND="\
${LOCAL_DIR}/hf_cache:/root/.cache/huggingface,\
$HOME/.cache/nanochat:/root/.cache/nanochat,\
${NANOCHAT_DIR}:/workspace,\
${CHECKPOINT_DIR}:/checkpoints"

export NANOCHAT_BASE_DIR=/root/.cache/nanochat

module load apptainer

# ── Load secrets (WandB key etc.) ────────────────────────────────────────────
# Uncomment if you use dotenvx like your reference script:
source ~/.secrets/nanochat.key
# OR export directly:
# export WANDB_API_KEY="your-key-here"

echo "[$(date)] Container and environment ready. Starting training..."

# ── Run nanochat d12 test ─────────────────────────────────────────────────────
# d12 = GPT-1 scale, ~5 min on 1xH100 — enough to confirm everything works
~/.local/bin/dotenvx run -- apptainer exec --nv \
    --pwd /workspace \
    "${SIF_IMAGE}" \
    python -m scripts.base_train \
    --depth=12 \
    --run="nanochat_test_${SLURM_JOB_ID}" \
    --model-tag="test_d12" \
    --save-every=200 \
    --core-metric-every=-1 \
    --sample-every=-1

echo "[$(date)] Test job finished. Check logs/nanochat_test_${SLURM_JOB_ID}.out"
echo "If successful, submit the full training job: sbatch train_nanochat.sbatch"
