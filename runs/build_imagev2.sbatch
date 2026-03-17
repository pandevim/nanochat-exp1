#!/bin/bash
# =============================================================================
# Build the nanochat Apptainer image (Slurm version)
# No %post installs — packages live in ~/nanochat-env on host, bind-mounted at runtime.
# Submit with:  sbatch build_image.sbatch
# =============================================================================
#SBATCH --job-name=nanochat_build
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/logs/nanochat_build_%j.out
#SBATCH --error=/home/%u/logs/nanochat_build_%j.err

set -euo pipefail

BASE_IMAGE="docker://nvcr.io/nvidia/pytorch:25.01-py3"
BASE_SIF="$HOME/containers/pytorch_25.01-py3.sif"
DEF_FILE="$HOME/nanochat-exp1/runs/nanochatv2.def"
SIF_OUT="$HOME/containers/nanochat.sif"

# Use fast local SSD for the build (avoids NFS stall)
BUILD_TMPDIR="/local/${SLURM_JOB_ID}/nanochat_build"
mkdir -p "$BUILD_TMPDIR" "$HOME/containers" "$HOME/logs"

echo "[$(date)] Starting Apptainer image build (job ${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Def file: ${DEF_FILE}"
echo "Output:   ${SIF_OUT}"
echo "Temp dir: ${BUILD_TMPDIR}"

module load apptainer

# ── Phase 1: Pull base image (skipped if already cached) ──────────────
if [ -f "$BASE_SIF" ]; then
    echo "[$(date)] Base image already exists at ${BASE_SIF}, skipping pull."
else
    echo "[$(date)] Pulling base image: ${BASE_IMAGE}"
    APPTAINER_TMPDIR="$BUILD_TMPDIR" \
    apptainer pull --force "$BASE_SIF" "$BASE_IMAGE"
    echo "[$(date)] Base image pulled: $(du -sh "$BASE_SIF" | cut -f1)"
fi

# ── Copy base SIF to local SSD (avoids NFS stall during build) ────────
LOCAL_BASE_SIF="${BUILD_TMPDIR}/pytorch_25.01-py3.sif"
echo "[$(date)] Copying base image to local SSD..."
cp "$BASE_SIF" "$LOCAL_BASE_SIF"
echo "[$(date)] Copy done: $(du -sh "$LOCAL_BASE_SIF" | cut -f1)"

# ── Phase 2: Build nanochat image from local base ─────────────────────
BUILD_DEF="${BUILD_TMPDIR}/nanochat_local.def"
sed 's|^Bootstrap:.*|Bootstrap: localimage|; s|^From:.*|From: '"${LOCAL_BASE_SIF}"'|' \
    "$DEF_FILE" > "$BUILD_DEF"

echo "[$(date)] Building nanochat image (no --fakeroot needed, no %post installs)..."
APPTAINER_TMPDIR="$BUILD_TMPDIR" \
apptainer build --force "${SIF_OUT}" "${BUILD_DEF}"

echo "[$(date)] Done! Image saved to: ${SIF_OUT}"
echo "Size: $(du -sh "${SIF_OUT}" | cut -f1)"

rm -rf "$BUILD_TMPDIR"
echo "[$(date)] Build complete."