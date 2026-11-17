#!/usr/bin/env bash
#SBATCH --job-name=train_all_models
#SBATCH --time=15:0:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --output=slurm/logs/%x_%a.out
#SBATCH --array=0-2
#
# Fits a linear head -- RidgeCV regression for --target age/bmi, a binary
# LogisticRegressionCV classifier for --target sex (see train_linear.py's
# --target) -- for all 7 models (optim_uncnn, brainiac, simclr, swinbrain,
# anatcl_global, anatcl_local, fs_roi) on top of each model's frozen
# backbone features (fs_roi has no backbone -- see below), over the full
# NKI cohort. Each invocation trains on the whole cohort at once
# (train_linear.py --demographics), it never processes subjects one at a
# time -- so an array index here selects a TARGET, not a subject.
#
# TARGET selects which head is trained, one of three ways (highest priority
# first):
#   1. Array job: submit with --array=0-2 (e.g. `sbatch --array=0-2
#      slurm/train_all_models.sh`) -- TASK_TARGETS below maps
#      SLURM_ARRAY_TASK_ID 0/1/2 to age/sex/bmi, one target per array task,
#      all 7 models trained for that one target. --array=1-2 (sex+bmi only)
#      or --array=0 (age only) work too -- any subset of {0,1,2}.
#   2. Plain (non-array) job with TARGET set: `TARGET=sex sbatch
#      slurm/train_all_models.sh` -- trains that one target only.
#   3. Plain job, TARGET unset: defaults to age -- reproduces the original
#      age-only behavior and filenames exactly.
#
# Writes {cache_name}_{brain_age,sex,bmi}_head.pt + feature caches into
# artifacts/ (train_linear.py's DEFAULT_OUTPUT_DIR). Copy the small head
# checkpoints you want to keep into checkpoints/heads/ afterward -- artifacts/
# itself is gitignored (feature caches are large and regenerable).
#
# fs_roi's prerequisite (artifacts/nki_freesurfer_roi.csv, built by
# src/build_freesurfer_roi_tables.py from src/fetch_freesurfer_tsvs.py's
# datalad-get output) must already exist before this runs -- both of those
# are login-node-only steps (network access to the git-annex remote), not
# part of this job. See fetch_freesurfer_tsvs.py's own module docstring.

set -uo pipefail

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/train_all_models.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# Array-index -> target mapping (see header comment). Only consulted when
# SLURM_ARRAY_TASK_ID is actually set (i.e. this was submitted with
# --array=...) -- a plain, non-array submission falls through to the
# TARGET env var / age default instead.
TASK_TARGETS=(age sex bmi)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no entry in TASK_TARGETS (${TASK_TARGETS[*]}) -- use --array=0-$((${#TASK_TARGETS[@]} - 1)) or a subset of it." >&2
        exit 1
    fi
    TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
else
    TARGET="${TARGET:-age}"
fi
echo "TARGET=$TARGET"

echo "unCNN"
python3 src/train_linear.py --model optim_uncnn --target "$TARGET" --no-cache

echo "BrainIAC"
python3 src/train_linear.py --model brainiac --target "$TARGET" --no-cache

echo "SIMCLR"
python3 src/train_linear.py --model simclr --target "$TARGET" --no-cache

echo "SWINBRAIN"
python3 src/train_linear.py --model swinbrain --target "$TARGET" --no-cache

echo "ANATCL GLOBAL"
export ANATCL_SUFFIX=global_
python3 src/train_linear.py --model anatcl --anatcl-folds 0 --anatcl-level global --target "$TARGET" --no-cache

echo "ANATCL LOCAL"
export ANATCL_SUFFIX=local_
python3 src/train_linear.py --model anatcl --anatcl-folds 0 --anatcl-level local --target "$TARGET" --no-cache

echo "FS ROI (FreeSurfer aparc.DKTatlas + subcortical volumes)"
python3 src/train_linear.py --model fs_roi --target "$TARGET" --no-cache
