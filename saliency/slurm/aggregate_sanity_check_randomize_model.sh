#!/usr/bin/env bash
#SBATCH --job-name=aggregate_sanity_check_randomize_model
#SBATCH --time=1:0:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Averages slurm/compute_sanity_checks_randomize_model.sh's per-subject
# randomized-model ("arch_") maps across the full cohort into the same
# arch_*.nii.gz files every downstream consumer already reads
# (saliency_affine/<target>/(<dataset>/)arch_*.nii.gz --
# visualize_paper_nilearn.py --sanity-check arch,
# compute_saliency_correlations.py). Run this ONCE all of that script's
# array tasks for the same target (and DATASET, see below) have finished --
# e.g. with SLURM job dependencies:
#   RM_JOB=$(sbatch --array=0-2873 --parsable slurm/compute_sanity_checks_randomize_model.sh)
#   sbatch --dependency=afterok:$RM_JOB slurm/aggregate_sanity_check_randomize_model.sh
# or just submit this manually once you've confirmed (e.g. via `sacct`) that
# every array task for a target completed.
#
# --array=0-2 = age/sex/bmi, same TASK_TARGETS convention as every other
# per-target script here (slurm/visualize_aggregate.sh in particular, whose
# own optim_uncnn-channel-averaging invocation this mirrors below).
#
# Uses src/aggregate_saliency.py --sanity-check arch (see that script's
# module docstring for the full reasoning) over arch_linear_persubject/ --
# one subfolder per subject, written by
# slurm/compute_sanity_checks_randomize_model.sh's --randomize-model-mode
# cascade --randomize-max-layers 1 run (resets ONLY the linear head, keeps
# the backbone) -- for every model in MODELS below, INCLUDING fs_roi (its
# arch_ map is painted onto the DKT-ROI grid, but that grid is identical
# across subjects, so averaging it across subjects is exactly as valid as
# averaging its baseline already is in .../aggregate/).
#
# This is the ONLY arch_ variant any current script can (re)produce --
# there is no selectable "all" (full-model, backbone included) mode here
# anymore: an older full-model variant exists on disk for NKI from a
# since-removed code path in slurm/compute_sanity_checks.sh's git history,
# but no current script can regenerate it for any dataset, so it was
# dropped from this script rather than kept as a dead, misleading option.
# Output filenames are tagged "arch_{model}_step01_head_...".
#
# DATASET env var (OPTIONAL, same convention as every other DATASET-aware
# script here): unset/empty (default) reads/writes
# saliency_affine/<target>/{arch_linear_persubject,.}/ (this repo's own
# checkpoints, NKI only). Set to DATASET=nki|ppmi|hbn instead reads/writes
# saliency_affine_pretrained/<target>/<dataset>/{arch_linear_persubject,.}/
# (the externally pre-trained heads) -- requires
# slurm/compute_sanity_checks_randomize_model.sh to have already been run
# with the SAME DATASET for this target. fs_roi dropped from MODELS only
# for DATASET=ppmi (no compatible checkpoint there).

set -uo pipefail
shopt -s nullglob

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/aggregate_sanity_check_randomize_model.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/aggregate_sanity_check_randomize_model.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"

# DATASET (optional, see header comment above): empty/unset resolves to
# this script's original saliency_affine/<target>/ (NKI, own checkpoints)
# behavior exactly; nki/ppmi/hbn switches to the pretrained-heads tree.
DATASET="${DATASET:-}"
case "$DATASET" in
    "") SALIENCY_DIR="${REPO_DIR}/saliency_affine/${TARGET}" ;;
    nki|ppmi|hbn) SALIENCY_DIR="${REPO_DIR}/saliency_affine_pretrained/${TARGET}/${DATASET}" ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected unset, nki, ppmi, or hbn)" >&2
        exit 1
        ;;
esac

# The only arch_ variant any current script can (re)produce -- see header
# comment above. VARIANT_TAG is the exact "arch_{model}_<VARIANT_TAG>_..."
# filename segment calculate_gradcam.py baked into that run's outputs --
# passed straight through to src/aggregate_saliency.py's
# --randomize-model-variant (see its own docstring for the full
# "step{NN:02d}_{module}" naming contract).
PERSUBJECT_DIR="${SALIENCY_DIR}/arch_linear_persubject"
OUTPUT_DIR="${SALIENCY_DIR}"
VARIANT_TAG="step01_head"

if [ ! -d "$PERSUBJECT_DIR" ]; then
    echo "ERROR: $PERSUBJECT_DIR does not exist -- run slurm/compute_sanity_checks_randomize_model.sh for this target (and DATASET=$DATASET, if set) first." >&2
    exit 1
fi

# unCNN's 3 channels aggregated separately (raw/rank/sobel are heterogeneous
# units -- never pooled directly, same reasoning as everywhere else in this
# repo); optim_uncnn_avg is built AFTER, once per kind below, from these
# already-cross-subject-averaged per-channel files -- not per-subject, so
# the averaging-across-channels and averaging-across-subjects steps each
# run exactly once, in either order (both are plain means, so composing
# them either way gives the same result -- this order is just cheaper: one
# channel-average pass instead of one per subject). fs_roi dropped only for
# DATASET=ppmi -- no compatible checkpoint there (see
# slurm/compute_saliency_pretrained_heads.sh's own SCOPE comment).
MODELS=(optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac swinbrain simclr anatcl_global anatcl_local)
if [ "$DATASET" != "ppmi" ]; then
    MODELS+=(fs_roi)
fi

python3 src/aggregate_saliency.py --sanity-check arch --target "$TARGET" \
    --saliency-dir "$PERSUBJECT_DIR" --output-dir "$OUTPUT_DIR" \
    --kinds signed abs --models "${MODELS[@]}" \
    --randomize-model-variant "$VARIANT_TAG"

echo "=== Averaging optim_uncnn raw/rank/sobel arch_ maps into optim_uncnn_avg ==="
python3 src/average_uncnn_channels.py --aggregate-dir "$OUTPUT_DIR" --stats x \
    --kinds signed abs --tag-prefixes ig_reference_ \
    --input-template "arch_optim_uncnn_${VARIANT_TAG}_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
    --output-template "arch_optim_uncnn_${VARIANT_TAG}_avg_{tag_prefix}{kind}_saliency.nii.gz"

echo "=== Done: aggregated randomize-model sanity check for $TARGET ==="
