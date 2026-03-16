#!/bin/bash
# =============================================================================
# Build the nanochat Apptainer image (Slurm version)
# Two-phase build: pull base image first (resumable), then build on top.
# Submit with:  sbatch build_image.sbatch
# =============================================================================
#SBATCH --job-name=nanochat_build
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/nanochat_build_%j.out
#SBATCH --error=/home/%u/logs/nanochat_build_%j.err

set -euo pipefail

MODULE_NAME="apptainer"
BASE_IMAGE="docker://nvcr.io/nvidia/pytorch:25.01-py3"
BASE_SIF="$HOME/containers/pytorch_25.01-py3.sif"
DEF_FILE="$HOME/nanochat-exp1/runs/nanochat.def"
SIF_OUT="$HOME/containers/nanochat.sif"

# Use fast local SSD for the build (much faster than home/NFS)
BUILD_TMPDIR="${TMPDIR}/nanochat_build_$$"
mkdir -p "$BUILD_TMPDIR" "$HOME/containers" "$HOME/logs"

echo "[$(date)] Starting Apptainer image build (job ${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Def file: ${DEF_FILE}"
echo "Output:   ${SIF_OUT}"
echo "Temp dir: ${BUILD_TMPDIR}"

echo "[$(date)] Loading apptainer module..."
module load "$MODULE_NAME"

# ── Phase 1: Pull base image (skipped if already cached) ──────────────
if [ -f "$BASE_SIF" ]; then
    echo "[$(date)] Base image already exists at ${BASE_SIF}, skipping pull."
else
    echo "[$(date)] Pulling base image: ${BASE_IMAGE}"
    echo "         This downloads ~15-20 GB and may take a while on first run."
    echo "         Layers are cached, so retries resume where they left off."
    APPTAINER_TMPDIR="$BUILD_TMPDIR" \
        apptainer pull --force "$BASE_SIF" "$BASE_IMAGE"
    echo "[$(date)] Base image pulled: $(du -sh "$BASE_SIF" | cut -f1)"
fi

# ── Phase 2: Build nanochat image from local base ─────────────────────
# Generate a modified def file that uses the local base image
BUILD_DEF="${BUILD_TMPDIR}/nanochat_local.def"
sed 's|^Bootstrap:.*|Bootstrap: localimage|; s|^From:.*|From: '"${BASE_SIF}"'|' \
    "$DEF_FILE" > "$BUILD_DEF"

echo "[$(date)] Building nanochat image from local base..."
APPTAINER_TMPDIR="$BUILD_TMPDIR" \
    apptainer build --fakeroot "${SIF_OUT}" "${BUILD_DEF}"

echo "[$(date)] Done! Image saved to: ${SIF_OUT}"
echo "Size: $(du -sh "${SIF_OUT}" | cut -f1)"

# Cleanup temp files
rm -rf "$BUILD_TMPDIR"
echo "[$(date)] Build complete. Run 'sbatch test_nanochat.sbatch' to validate."
