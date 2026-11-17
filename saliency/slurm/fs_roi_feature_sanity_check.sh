#!/usr/bin/env bash
#SBATCH --job-name=fs_roi_feature_sanity_check
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x.out
#
# Runs src/fs_roi_feature_sanity_check.py for all 3 targets (age, sex, bmi)
# -- the independent per-ROI sanity check: for each of fs_roi's ~110
# FreeSurfer regions, fit a single-feature RidgeCV (age/bmi) or
# LogisticRegressionCV (sex) model FROM SCRATCH, ignoring every other
# region, and report its held-out R^2/RMSE (or ROC-AUC/accuracy) via nested
# cross-validation -- see that script's module docstring for the full
# rationale and its ALIGNMENT NOTE on why it re-derives labels from the
# demographics file rather than trusting the cached _y_{target}.npy
# directly (that cache is currently stale for age/sex, not bmi).
#
# CPU-only, single-shot (not an array job): each target is ~110 cheap
# single-feature model fits (~1 minute total per target on 1 core), so all
# 3 targets run sequentially in one job rather than 3 separate submissions.
#
# Output per target, in saliency_outputs/fs_roi_feature_sanity_check/:
#   fs_roi_per_feature_{target}.csv
#   fs_roi_per_feature_{target}.png

set -uo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/fs_roi_feature_sanity_check.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

for TARGET in age sex bmi; do
    echo "=== fs_roi_feature_sanity_check.py --target ${TARGET} ==="
    python3 src/fs_roi_feature_sanity_check.py --target "$TARGET"
done
