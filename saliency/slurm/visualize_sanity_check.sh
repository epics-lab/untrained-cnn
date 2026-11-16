#!/usr/bin/env bash
#SBATCH --job-name=visualize_sanity_check
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --output=slurm/logs/%x.out
#
# src/visualize_sanity_check.py: one figure with EVERY model (rows) x EVERY
# target's (age/sex/bmi) 3 Adebayo et al. sanity checks (arch_/data_/input_,
# i.e. random-weights/shuffled-labels/shuffled-voxels) as columns, ONE
# global color scale for the whole grid -- see that script's own module
# docstring for why only a single anatomical view per cell (not the usual
# ortho+montage block) keeps this legible. Unlike slurm/visualize_aggregate.sh
# and slurm/visualize_paper_nilearn.sh, this does NOT need a --array=0-2
# per-target split: one run already covers all 3 targets in a single figure.
# Loops over the 3 views instead (cheap -- CPU-only plotting, no GPU).
#
# Prerequisites (same as slurm/visualize_paper_nilearn.sh's own sanity-check
# figures):
#   - slurm/compute_sanity_checks.sh (data_/input_) and slurm/
#     aggregate_sanity_check_randomize_model.sh (arch_, after slurm/
#     compute_sanity_checks_randomize_model.sh's per-subject array) already
#     run for every target in SANITY_TARGETS below, so SALIENCY_AFFINE_ROOT/
#     <target>/(<DATASET>/) has arch_*/data_*/input_* ig_reference_*_
#     saliency.nii.gz for every model in SANITY_MODELS. arch_ specifically
#     only ever exists as the linear-head-only "step01_head" variant (see
#     SANITY_ARCH_VARIANT below) -- the older full-model "all" variant that
#     exists on disk for NKI predates a since-removed code path and cannot
#     be regenerated for any dataset.
#   - if SANITY_MODELS includes optim_uncnn_avg (the default), src/average_
#     uncnn_channels.py must already have averaged optim_uncnn's raw/rank/
#     sobel arch_/data_/input_ maps into one optim_uncnn_avg map each (done
#     below, same as slurm/visualize_paper_nilearn.sh's own pre-step) -- a
#     model/target/check combination missing its file is otherwise just
#     skipped (WARN + blank cell), never a hard failure.
#
# DATASET env var (OPTIONAL, same convention as slurm/visualize_aggregate.sh):
# unset/empty (default) reads/writes saliency_affine/ (this repo's own
# checkpoints, NKI only). Set to DATASET=nki|ppmi|hbn instead reads/writes
# saliency_affine_pretrained/<target>/<dataset>/ (the external externally
# pre-trained heads). SANITY_MODELS' default drops fs_roi for DATASET=ppmi
# (no compatible checkpoint there, same as every other script).
#
# matplotlib>=3.8 requirement: nilearn.plotting refuses to import against
# the matplotlib pinned in env.sh's own brainiac_env -- see
# slurm/visualize_paper_nilearn.sh's identical note. NILEARN_VENV is
# re-activated on top of env.sh for the same reason here.

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_sanity_check.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

DATASET="${DATASET:-}"
case "$DATASET" in
    "") ;;
    nki|ppmi|hbn) ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected unset, nki, ppmi, or hbn)" >&2
        exit 1
        ;;
esac
if [ -n "$DATASET" ]; then
    SALIENCY_AFFINE_ROOT="${REPO_DIR}/saliency_affine_pretrained"
else
    SALIENCY_AFFINE_ROOT="${REPO_DIR}/saliency_affine"
fi
OUTPUT_DIR="${SALIENCY_AFFINE_ROOT}"
mkdir -p "$OUTPUT_DIR"

DATASET_ARGS=()
if [ -n "$DATASET" ]; then
    DATASET_ARGS=(--dataset "$DATASET")
fi

# Only arch variant any current script can (re)produce -- see header
# comment above.
SANITY_ARCH_VARIANT="${SANITY_ARCH_VARIANT:-step01_head}"

# See the matplotlib>=3.8 note above -- override with e.g.
# `NILEARN_VENV=/path/to/other/venv sbatch slurm/visualize_sanity_check.sh`
# (required: there is no default venv).
NILEARN_VENV="${NILEARN_VENV:?set NILEARN_VENV to a venv with matplotlib>=3.8 and nilearn}"
source "${NILEARN_VENV}/bin/activate"
python3 -c "import matplotlib; assert tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 8)" || {
    echo "FATAL: $NILEARN_VENV's matplotlib is < 3.8 -- nilearn.plotting cannot import. Set NILEARN_VENV to a venv with a current matplotlib+nilearn." >&2
    exit 1
}

SANITY_TARGETS=(${SANITY_TARGETS:-age sex bmi})
#anatcl_global brainiac optim_uncnn_avg simclr swinbrain
# fs_roi has no compatible checkpoint for DATASET=ppmi -- default drops it
# there (same convention as every other DATASET-aware script).
if [ -z "${SANITY_MODELS+x}" ]; then
    if [ "$DATASET" = "ppmi" ]; then
        SANITY_MODELS=(anatcl_global anatcl_local brainiac optim_uncnn_avg simclr swinbrain)
    else
        SANITY_MODELS=(fs_roi)
    fi
else
    SANITY_MODELS=(${SANITY_MODELS})
fi
SANITY_CHECKS=(${SANITY_CHECKS:-arch data input})
SANITY_KIND="${SANITY_KIND:-abs}"
SANITY_THRESHOLD_PERCENTILE="${SANITY_THRESHOLD_PERCENTILE:-0.0}"
SANITY_LABEL_SCALE="${SANITY_LABEL_SCALE:-1.0}"
SANITY_VIEWS=(${SANITY_VIEWS:-axial coronal sagittal})
# 1 (default): rescale each map onto the shared [-1,1]/[0,1] range before pooling into the
# figure's one global vmax/threshold (recommended -- see visualize_sanity_check.py's own
# compare_sanity_checks_all_targets docstring for why). 0: pass --no-norm, using each map's raw
# values instead -- exploratory/debugging use only, not a recommended way to compare models.
SANITY_NORM="${SANITY_NORM:-0}"
SANITY_NORM_FLAG=()
if [ "$SANITY_NORM" = "0" ]; then
    SANITY_NORM_FLAG=(--no-norm)
fi

# Build optim_uncnn_avg's arch_/data_/input_ maps by averaging the 3 real
# channels (raw/rank/sobel), same as slurm/visualize_paper_nilearn.sh's own
# pre-step -- skipped if optim_uncnn_avg isn't requested.
if printf '%s\n' "${SANITY_MODELS[@]}" | grep -qx "optim_uncnn_avg"; then
    for TARGET in "${SANITY_TARGETS[@]}"; do
        echo "=== Averaging optim_uncnn raw/rank/sobel sanity-check maps into optim_uncnn_avg (${TARGET}) ==="
        if [ -n "$DATASET" ]; then
            SALIENCY_DIR="${SALIENCY_AFFINE_ROOT}/${TARGET}/${DATASET}"
        else
            SALIENCY_DIR="${SALIENCY_AFFINE_ROOT}/${TARGET}"
        fi
        python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
            --kinds "$SANITY_KIND" --tag-prefixes ig_reference_ \
            --input-template "arch_optim_uncnn_${SANITY_ARCH_VARIANT}_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
            --output-template "arch_optim_uncnn_${SANITY_ARCH_VARIANT}_avg_{tag_prefix}{kind}_saliency.nii.gz"
        python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
            --kinds "$SANITY_KIND" --tag-prefixes ig_reference_ \
            --input-template "data_optim_uncnn_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
            --output-template "data_optim_uncnn_avg_{tag_prefix}{kind}_saliency.nii.gz"
        python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
            --kinds "$SANITY_KIND" --tag-prefixes ig_reference_ \
            --input-template "input_optim_uncnn_permute_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
            --output-template "input_optim_uncnn_permute_avg_{tag_prefix}{kind}_saliency.nii.gz"
    done
fi

for VIEW in "${SANITY_VIEWS[@]}"; do
    NORM_SUFFIX=""
    if [ "$SANITY_NORM" = "0" ]; then
        NORM_SUFFIX="_raw"
    fi
    DATASET_SUFFIX=""
    if [ -n "$DATASET" ]; then
        DATASET_SUFFIX="_${DATASET}"
    fi
    OUT="${OUTPUT_DIR}/sanity_check_all_models_all_targets_${VIEW}_${SANITY_KIND}${DATASET_SUFFIX}${NORM_SUFFIX}.png"
    if python3 src/visualize_sanity_check.py \
        --saliency-affine-root "$SALIENCY_AFFINE_ROOT" "${DATASET_ARGS[@]}" \
        --no-title \
        --targets "${SANITY_TARGETS[@]}" --models "${SANITY_MODELS[@]}" --checks "${SANITY_CHECKS[@]}" \
        --arch-variant "$SANITY_ARCH_VARIANT" \
        --view "$VIEW" --kind "$SANITY_KIND" --threshold-percentile "$SANITY_THRESHOLD_PERCENTILE" \
        --label-scale "$SANITY_LABEL_SCALE" "${SANITY_NORM_FLAG[@]}" --output "$OUT"; then
        echo "  wrote $OUT"
    else
        echo "  WARN: visualize_sanity_check.py failed for view $VIEW"
    fi
done

echo "=== Done: visualize_sanity_check ==="
