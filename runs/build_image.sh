#!/bin/bash
# =============================================================================
# Build the nanochat Apptainer image
# Run this ONCE on a login node (or build node) before submitting jobs.
# =============================================================================
set -euo pipefail

MODULE_NAME="apptainer"   # adjust if your cluster uses a different name
DEF_FILE="nanochat.def"
SIF_OUT="$HOME/containers/nanochat.sif"

# Scratch space for the build (uses fast local storage if available)
BUILD_TMPDIR="${TMPDIR}/nanochat_build_$$"
mkdir -p "$BUILD_TMPDIR" "$HOME/containers"

echo "Loading apptainer module..."
module load "$MODULE_NAME"

echo "Building ${SIF_OUT} from ${DEF_FILE}..."
echo "This will pull ~10GB from NGC and may take 10-20 minutes."

# --fakeroot lets non-root users run %post scripts
# APPTAINER_TMPDIR must point to a disk with enough space (>15 GB)
APPTAINER_TMPDIR="$BUILD_TMPDIR" \
    apptainer build --fakeroot "${SIF_OUT}" "${DEF_FILE}"

echo "Done! Image saved to: ${SIF_OUT}"
echo "Size: $(du -sh "${SIF_OUT}" | cut -f1)"

# Cleanup temp files
rm -rf "$BUILD_TMPDIR"
echo "Run 'sbatch test_nanochat.sbatch' to validate the image."
