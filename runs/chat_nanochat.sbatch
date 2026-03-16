#!/bin/bash
# =============================================================================
# nanochat INFERENCE JOB — serves the web UI after training
# Access via: http://<node-ip>:8000
# =============================================================================
#SBATCH --job-name=nanochat_chat
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nanochat_chat_%j.out

NANOCHAT_DIR="$HOME/nanochat"
SIF_IMAGE="$HOME/containers/nanochat.sif"
CHECKPOINT_DIR="$HOME/checkpoints/nanochat_d26"

LOCAL_DIR="/local/${SLURM_JOB_ID}"
mkdir -p "${LOCAL_DIR}/hf_cache" "${LOCAL_DIR}/apptainer_cache"

export APPTAINER_CACHEDIR="${LOCAL_DIR}/apptainer_cache"
export APPTAINER_BIND="\
${LOCAL_DIR}/hf_cache:/root/.cache/huggingface,\
${NANOCHAT_DIR}:/workspace,\
${CHECKPOINT_DIR}:/checkpoints"

module load apptainer

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Chat UI will be available at: http://${NODE_IP}:8000"
echo "(Allow 30-60s for the model to load)"

apptainer exec --nv \
    --pwd /workspace \
    "${SIF_IMAGE}" \
    python -m scripts.chat_web \
        --checkpoint=/checkpoints \
        --host=0.0.0.0 \
        --port=8000
