#!/bin/bash
# =============================================================================
# Build the nanochat Apptainer image (Slurm version)
# Runs on a compute node with fast local SSD for better I/O performance.
# Submit with:  sbatch build_image.sbatch
# =============================================================================
#SBATCH --job-name=nanochat_build
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nanochat_build_%j.out
#SBATCH --error=logs/nanochat_build_%j.err

set -euo pipefail

MODULE_NAME="apptainer"
DEF_FILE="$HOME/nanochat-exp1/runs/nanochat.def"
SIF_OUT="$HOME/containers/nanochat.sif"

# Use fast local SSD for the build (much faster than home/NFS)
BUILD_TMPDIR="/local/${SLURM_JOB_ID}/nanochat_build"
mkdir -p "$BUILD_TMPDIR" "$HOME/containers" logs

echo "[$(date)] Starting Apptainer image build (job ${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Def file: ${DEF_FILE}"
echo "Output:   ${SIF_OUT}"
echo "Temp dir: ${BUILD_TMPDIR}"

echo "[$(date)] Loading apptainer module..."
module load "$MODULE_NAME"

echo "[$(date)] Building image — this may take 10-20 minutes..."

APPTAINER_TMPDIR="$BUILD_TMPDIR" \
    apptainer build --fakeroot "${SIF_OUT}" "${DEF_FILE}"

echo "[$(date)] Done! Image saved to: ${SIF_OUT}"
echo "Size: $(du -sh "${SIF_OUT}" | cut -f1)"

# Cleanup temp files
rm -rf "$BUILD_TMPDIR"
echo "[$(date)] Build complete. Run 'sbatch test_nanochat.sbatch' to validate."
