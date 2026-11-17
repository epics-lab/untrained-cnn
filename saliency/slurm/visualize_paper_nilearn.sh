#!/usr/bin/env bash
#SBATCH --job-name=viz_paper_nilearn
#SBATCH --time=0:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Paper figure(s): src/visualize_paper_nilearn.py's cross-model ig_popmean
# comparison rendered with nilearn's own plot_stat_map (black background,
# cold_hot/hot colormap, ortho view + axial montage strip per model) -- the
# cleaner, nilearn-native sibling of slurm/visualize_paper.sh (which drives
# src/visualize_paper.py's hand-rolled matplotlib overlay instead). See that
# Python script's module docstring for what it draws and why.
#
# Prerequisite: slurm/visualize_aggregate.sh (or src/aggregate_saliency.py
# directly) must have already been run, so that saliency_affine/aggregate/
# already has {model}_ig_reference_aggregate_{stat}_{signed,abs}_saliency.
# nii.gz for every model in PAPER_NL_MODELS -- this script only visualizes,
# it never computes aggregates itself. PAPER_NL_MODELS' default
# (optim_uncnn_avg, not the 3 separate raw/rank/sobel channels) needs
# visualize_aggregate.sh's average_uncnn_channels.py step to have run too.
#
# ALSO produces a second, region-averaged comparison figure
# (paper_ig_popmean_nilearn_${STAT}_${PAPER_NL_KIND}_roi.png): fs_roi's
# saliency is one value per FreeSurfer ROI (painted onto a DKT segmentation
# by calculate_gradcam_tabular.py), not per-voxel like every other model's
# -- so the usual figure above compares fs_roi's coarse map against the
# other models' fine-grained ones, not apples-to-apples. src/regionalize_
# saliency.py resamples every OTHER model's aggregate map onto that same
# DKT segmentation and replaces each region's voxels with their mean,
# so all 7 models land at the identical ROI granularity fs_roi is native
# to; fs_roi's own file needs no such processing and is copied through
# unchanged (see that script's own module docstring for why, and the
# axis-order pitfall its code works around). This is a deliberate exception
# to "this script only visualizes, it never computes aggregates itself"
# above -- the regionalized files are a byproduct specific to this
# figure, not a general-purpose aggregate any other script depends on.
#
# ALSO produces 3 sanity-check comparison figures
# (paper_sanity_check_nilearn_{arch,data,input}_${PAPER_NL_KIND}.png),
# same nilearn styling/one row per model, but showing slurm/
# compute_sanity_checks.sh's single-reference-subject arch_/data_/input_
# randomization maps directly from saliency_affine/<target>/ instead of the
# across-subject aggregate -- see src/visualize_paper_nilearn.py's
# --sanity-check. Requires slurm/compute_sanity_checks.sh to have already
# been run for this target. Unlike the two figures above, these have no
# --stat dimension (the sanity-check maps are single reference-subject
# maps, not an across-subject statistic), so they're produced once per
# invocation, not once per PAPER_NL_STATS entry. If "optim_uncnn_avg" is in
# PAPER_NL_MODELS, its row is built by averaging optim_uncnn's raw/rank/
# sobel sanity-check maps on the fly via src/average_uncnn_channels.py
# (mirroring what visualize_aggregate.sh already does for the aggregate
# figures) -- no separate prerequisite step needed for that model.
#
# matplotlib>=3.8 requirement: nilearn.plotting refuses to import against
# the matplotlib 3.7.0 pinned in brainiac_env (env.sh's own venv, shared
# with BrainIAC and every other slurm/*.sh script here) -- verified directly
# (ImportError at import time, not a runtime warning). Rather than upgrading
# that shared venv (risking a BrainIAC-side break), this script re-activates
# NILEARN_VENV (a separate venv with a current matplotlib+nilearn, default
# below) on top of it after sourcing env.sh, so every other exported
# variable (TEMPLATEFLOW_MNI_DIR in particular) still comes from the same
# single source of truth as every other script in this directory.

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_paper_nilearn.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list), same
# convention as slurm/visualize_aggregate.sh and slurm/compute_correlations.sh
# -- selects which saliency_affine/<target>/aggregate{,_roi} subtree to read
# from and write into. Submit a single target's task directly with e.g.
# `sbatch --array=0 slurm/visualize_paper_nilearn.sh` (age only), `--array=1`
# (sex only), `--array=2` (bmi only). Requires visualize_aggregate.sh to have
# already been run for that same target (see the prerequisite note above).
TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/visualize_paper_nilearn.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"

# DATASET env var (OPTIONAL, same convention as slurm/visualize_aggregate.sh):
# unset/empty (default) preserves this script's original behavior exactly --
# reads/writes saliency_affine/<target>/{aggregate,aggregate_roi}/ (this
# repo's own checkpoints, NKI only). Set to DATASET=nki|ppmi|hbn instead
# reads/writes saliency_affine_pretrained/<target>/<dataset>/{aggregate,
# aggregate_roi}/ (the externally pre-trained heads) -- requires
# slurm/visualize_aggregate.sh to have already been run with the SAME
# DATASET for this target (see that script's own DATASET env var). The
# sanity-check companion figures below (arch/data/input) will still be
# empty/skipped for the pretrained tree even with DATASET set --
# slurm/compute_sanity_checks.sh (data_/input_) has never been made
# DATASET-aware, and compute_sanity_checks_randomize_model.sh's DATASET-set
# branch writes a different filename shape (arch_{model}_step01_head_...,
# the head-only cascade variant) than this script's sanity_prefix() default
# (arch_{model}_all_...) -- a separate gap, not fixed here.
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
    SALIENCY_DIR="${REPO_DIR}/saliency_affine_pretrained/${TARGET}/${DATASET}"
else
    SALIENCY_DIR="${REPO_DIR}/saliency_affine/${TARGET}"
fi
AGGREGATE_DIR="${SALIENCY_DIR}/aggregate"
ROI_AGGREGATE_DIR="${SALIENCY_DIR}/aggregate_roi"

# See the matplotlib>=3.8 note above -- override with e.g.
# `NILEARN_VENV=/path/to/other/venv sbatch slurm/visualize_paper_nilearn.sh`
# (required: there is no default venv).
NILEARN_VENV="${NILEARN_VENV:?set NILEARN_VENV to a venv with matplotlib>=3.8 and nilearn}"
source "${NILEARN_VENV}/bin/activate"
python3 -c "import matplotlib; assert tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 8)" || {
    echo "FATAL: $NILEARN_VENV's matplotlib is < 3.8 -- nilearn.plotting cannot import. Set NILEARN_VENV to a venv with a current matplotlib+nilearn." >&2
    exit 1
}

# visualize_paper_nilearn.py's own default -- override with e.g.
# `PAPER_NL_MODELS="optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac" sbatch slurm/visualize_paper_nilearn.sh`
# to see optim_uncnn's 3 channels separately instead of the averaged row.
# fs_roi dropped from the default only for DATASET=ppmi -- no compatible
# checkpoint exists there (see slurm/visualize_aggregate.sh's own MODELS
# comment for why).
if [ -z "${PAPER_NL_MODELS+x}" ]; then
    if [ "$DATASET" = "ppmi" ]; then
        PAPER_NL_MODELS=(optim_uncnn_avg brainiac swinbrain simclr anatcl_global)
    else
        PAPER_NL_MODELS=(fs_roi optim_uncnn_avg brainiac swinbrain simclr anatcl_global)
    fi
else
    PAPER_NL_MODELS=(${PAPER_NL_MODELS})
fi
PAPER_NL_STATS=(${PAPER_NL_STATS:-mean})
PAPER_NL_KIND="${PAPER_NL_KIND:-abs}"                          # signed or abs
PAPER_NL_N_MONTAGE="${PAPER_NL_N_MONTAGE:-4}"                     # 0 disables the axial montage strip
PAPER_NL_THRESHOLD_PERCENTILE="${PAPER_NL_THRESHOLD_PERCENTILE:--0.1}"
PAPER_NL_CUT_COORDS=(${PAPER_NL_CUT_COORDS:--17 1 15})            # shared (x y z) mm ortho cut across every row; set empty to let each row pick its own peak

# Region-averaged companion figure's inputs -- see the header comment above.
# PAPER_NL_ROI_SEG_TEMPLATE defaults to the SAME segmentation
# slurm/compute_saliency.sh's fs_roi block actually paints its own saliency
# onto (--seg-template there), not calculate_gradcam_tabular.py's own
# internal (unused-in-production) DEFAULT_SEG_TEMPLATE -- keep the two in
# sync if that ever changes.
PAPER_NL_ROI_SEG_TEMPLATE="${PAPER_NL_ROI_SEG_TEMPLATE:-${REPO_DIR}/templates/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz}"
PAPER_NL_ROI_LUT="${PAPER_NL_ROI_LUT:-${FREESURFER_HOME:-}/FreeSurferColorLUT.txt}"

# Title is optional (visualize_paper_nilearn.py's own --title, default off)
# -- on by default here to keep this script's existing output unchanged;
# set PAPER_NL_SHOW_TITLE=0 to drop the suptitle entirely (e.g. for a paper
# figure that gets its caption from the manuscript, not the image itself).
PAPER_NL_SHOW_TITLE="${PAPER_NL_SHOW_TITLE:-0}"

# Shared colorbar (bottom of the figure) and per-row model-name label size --
# both new visualize_paper_nilearn.py flags (--no-colorbar/--label-scale).
# On/1x by default here, same "keep existing output unchanged unless
# overridden" convention as PAPER_NL_SHOW_TITLE above. Set
# PAPER_NL_SHOW_COLORBAR=0 to drop the colorbar (e.g. for a paper figure
# whose color scale is given in the caption), and/or PAPER_NL_LABEL_SCALE to
# something >1 to make the row labels bigger.
PAPER_NL_SHOW_COLORBAR="${PAPER_NL_SHOW_COLORBAR:-0}"
PAPER_NL_LABEL_SCALE="${PAPER_NL_LABEL_SCALE:-2}"

# Row label for the untrained-CNN control -- visualize_paper_nilearn.py's
# --uncnn-label collapses "optim_uncnn_avg"/"_raw"/"_rank"/"_sobel" (whichever
# channel PAPER_NL_MODELS requested) down to plain "unCNN". Off by default
# here, same "keep existing output unchanged unless overridden" convention as
# the flags above; set PAPER_NL_UNCNN_LABEL=1 to turn it on.
PAPER_NL_UNCNN_LABEL="${PAPER_NL_UNCNN_LABEL:-1}"

# --sanity-check arch only (see src/visualize_paper_nilearn.py's own
# sanity_prefix() docstring): the only arch variant any current script can
# (re)produce -- slurm/compute_sanity_checks_randomize_model.sh's
# linear-head-only cascade run. Harmless no-op for --sanity-check data/input.
PAPER_NL_ARCH_VARIANT="${PAPER_NL_ARCH_VARIANT:-step01_head}"

COMMON_FLAGS=(
    --models "${PAPER_NL_MODELS[@]}"
    --kind "$PAPER_NL_KIND"
    --n-montage "$PAPER_NL_N_MONTAGE"
    --threshold-percentile "$PAPER_NL_THRESHOLD_PERCENTILE"
    --label-scale "$PAPER_NL_LABEL_SCALE"
)
if [ "$PAPER_NL_SHOW_COLORBAR" = "0" ]; then
    COMMON_FLAGS+=(--no-colorbar)
fi
if [ "$PAPER_NL_UNCNN_LABEL" = "1" ]; then
    COMMON_FLAGS+=(--uncnn-label)
fi
if [ "${#PAPER_NL_CUT_COORDS[@]}" -eq 3 ]; then
    COMMON_FLAGS+=(--cut-coords "${PAPER_NL_CUT_COORDS[@]}")
fi

# Build the region-averaged companion inputs once per invocation (not once
# per STAT below -- it does its own internal stats loop), scoped to exactly
# what this run's PAPER_NL_STATS/PAPER_NL_KIND need, not a speculative full
# precompute. PAPER_NL_MODELS (fs_roi included) is passed through
# unfiltered -- regionalize_saliency.py itself recognizes --fs-roi-prefix
# and copies that one model's file through unchanged rather than
# resampling+averaging it (see that script's module docstring).
mkdir -p "$ROI_AGGREGATE_DIR"
if ! python3 src/regionalize_saliency.py \
    --saliency-dir "$AGGREGATE_DIR" --output-dir "$ROI_AGGREGATE_DIR" \
    --seg-template "$PAPER_NL_ROI_SEG_TEMPLATE" --lut "$PAPER_NL_ROI_LUT" \
    --models "${PAPER_NL_MODELS[@]}" --stats "${PAPER_NL_STATS[@]}" --kinds "$PAPER_NL_KIND"; then
    echo "  WARN: regionalize_saliency.py failed -- the region-averaged figure below will be skipped/incomplete"
fi

for STAT in "${PAPER_NL_STATS[@]}"; do
    TITLE_FLAGS=()
    if [ "$PAPER_NL_SHOW_TITLE" != "0" ]; then
        TITLE_FLAGS=(--title "ig_popmean -- across-subject aggregate (${STAT})")
    fi
    OUT="$AGGREGATE_DIR/final_paper_ig_popmean_nilearn_${STAT}_${PAPER_NL_KIND}.pdf"
    if python3 src/visualize_paper_nilearn.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
        --saliency-dir "$AGGREGATE_DIR" \
        "${TITLE_FLAGS[@]}" \
        --output "$OUT"; then
        echo "  wrote $OUT"
    else
        echo "  WARN: visualize_paper_nilearn.py failed for stat $STAT"
    fi

    # Region-averaged companion figure -- reuses COMMON_FLAGS (same
    # --models/--kind/--n-montage/--cut-coords/--threshold-percentile) with
    # only --saliency-dir/--output/--title overridden. Zero code changes to
    # visualize_paper_nilearn.py itself: the regionalized files above
    # already share its usual naming convention and already sit on its own
    # default --template's exact grid (verified: aparc.DKTatlas+aseg_
    # subject_MNI152Lin.nii.gz's affine matches tpl-MNI152Lin_res-01_T1w.
    # nii.gz's), so its own internal resample step is a no-op for these.
    ROI_TITLE_FLAGS=()
    if [ "$PAPER_NL_SHOW_TITLE" != "0" ]; then
        ROI_TITLE_FLAGS=(--title "ig_popmean -- region-averaged (DKT atlas) aggregate (${STAT})")
    fi
    OUT_ROI="$ROI_AGGREGATE_DIR/paper_ig_popmean_nilearn_${STAT}_${PAPER_NL_KIND}_roi.png"
    if python3 src/visualize_paper_nilearn.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
        --saliency-dir "$ROI_AGGREGATE_DIR" \
        "${ROI_TITLE_FLAGS[@]}" \
        --output "$OUT_ROI"; then
        echo "  wrote $OUT_ROI"
    else
        echo "  WARN: visualize_paper_nilearn.py failed for ROI-averaged stat $STAT"
    fi
done

# Sanity-check comparison figures (arch/data/input randomization, single
# reference subject -- see slurm/compute_sanity_checks.sh): one row per
# model like the aggregate figures above, but a completely separate figure
# per test type since arch_/data_/input_ answer different questions (see
# that script's own module docstring) and have no --stat dimension to loop
# over, so this runs once per invocation rather than once per
# PAPER_NL_STATS entry.
if printf '%s\n' "${PAPER_NL_MODELS[@]}" | grep -qx "optim_uncnn_avg"; then
    echo "=== Averaging optim_uncnn raw/rank/sobel sanity-check maps into optim_uncnn_avg ==="
    python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
        --kinds "$PAPER_NL_KIND" --tag-prefixes ig_reference_ \
        --input-template "arch_optim_uncnn_${PAPER_NL_ARCH_VARIANT}_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
        --output-template "arch_optim_uncnn_${PAPER_NL_ARCH_VARIANT}_avg_{tag_prefix}{kind}_saliency.nii.gz"
    python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
        --kinds "$PAPER_NL_KIND" --tag-prefixes ig_reference_ \
        --input-template "data_optim_uncnn_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
        --output-template "data_optim_uncnn_avg_{tag_prefix}{kind}_saliency.nii.gz"
    python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
        --kinds "$PAPER_NL_KIND" --tag-prefixes ig_reference_ \
        --input-template "input_optim_uncnn_permute_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
        --output-template "input_optim_uncnn_permute_avg_{tag_prefix}{kind}_saliency.nii.gz"
fi

for SANITY_TEST in arch data input; do
    SANITY_TITLE_FLAGS=()
    if [ "$PAPER_NL_SHOW_TITLE" != "0" ]; then
        SANITY_TITLE_FLAGS=(--title "ig_reference -- ${SANITY_TEST} sanity check")
    fi
    OUT_SANITY="$AGGREGATE_DIR/paper_sanity_check_nilearn_${SANITY_TEST}_${PAPER_NL_KIND}.png"
    if python3 src/visualize_paper_nilearn.py "${COMMON_FLAGS[@]}" --sanity-check "$SANITY_TEST" \
        --arch-variant "$PAPER_NL_ARCH_VARIANT" \
        --saliency-dir "$SALIENCY_DIR" \
        "${SANITY_TITLE_FLAGS[@]}" \
        --output "$OUT_SANITY"; then
        echo "  wrote $OUT_SANITY"
    else
        echo "  WARN: visualize_paper_nilearn.py failed for sanity-check test $SANITY_TEST"
    fi
done

echo "=== Done: nilearn paper figures ==="
