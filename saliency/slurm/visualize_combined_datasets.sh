#!/usr/bin/env bash
#SBATCH --job-name=viz_combined
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x.out
#
# Same figures as slurm/visualize_paper_nilearn.sh (+ the cross-target grid of
# slurm/visualize_sanity_check.sh when sanity checks are on), but each model's
# map is ONE map pooled across NKI *and* HBN instead of one dataset's own.
#
# Steps, per target in TARGETS (age sex bmi):
#   1. src/combine_datasets_saliency.py pools every NKI + HBN subject's
#      {model}_ig_reference_{kind}_saliency.nii.gz from
#      saliency_affine_pretrained/<target>/{nki,hbn}/<SUBJECT>/ into
#      saliency_affine_pretrained/<target>/nki_hbn/aggregate/
#      {model}_ig_reference_aggregate_{mean,std}_{signed,abs}_saliency.nii.gz
#      -- the same filenames slurm/visualize_aggregate.sh writes per dataset, so
#      everything below is unchanged. Subject-weighted pool (each subject counts
#      once; see that script's docstring), NOT a mean of the two dataset means.
#      Replaces slurm/visualize_aggregate.sh's per-dataset aggregate step, so
#      that script does NOT need to have been run first (only the per-subject
#      maps from compute_saliency_pretrained_heads.sh are required).
#   2. src/average_uncnn_channels.py -> optim_uncnn_avg (raw/rank/sobel mean).
#   3. src/regionalize_saliency.py + src/visualize_paper_nilearn.py: the
#      across-subject figure (final_paper_ig_popmean_nilearn_{STAT}_{KIND}.pdf)
#      and the region-averaged DKT figure (..._roi.png), as in
#      visualize_paper_nilearn.sh.
#
# COMPUTE_SANITY_CHECKS=1 (flag, off by default; "1"/"true"/"yes" turn it on)
# additionally, per target:
#   4. combines the Adebayo et al. sanity-check maps across the same datasets
#      into saliency_affine_pretrained/<target>/nki_hbn/:
#        arch  : per-subject arch_linear_persubject/ maps of both datasets pooled
#                (what aggregate_sanity_check_randomize_model.sh does for one
#                dataset);
#        data / input : the single reference-subject map of each dataset,
#                averaged. A (model, check) missing in ANY dataset is skipped
#                with a message, not written from the other dataset alone.
#      then optim_uncnn_avg for those, then the 3 per-test figures
#      (paper_sanity_check_nilearn_{arch,data,input}_{KIND}.png), and, after all
#      targets, the cross-target grid (sanity_check_all_models_all_targets_
#      {VIEW}_{KIND}_nki_hbn[_raw].png, in saliency_affine_pretrained/) like
#      visualize_sanity_check.sh.
#   Prerequisites: the per-dataset arch_linear_persubject/ runs
#   (compute_sanity_checks_randomize_model.sh with DATASET=nki / hbn) and the
#   per-dataset data_/input_ reference maps. As of writing the NKI pretrained
#   tree has none of these top-level data_/input_ maps and an empty
#   arch_linear_persubject/, so those cells are skipped until they exist.
#
# Env vars (all optional):
#   EXCLUDE_SUBJECTS (default: the 3 duplicated-T1w HBN subjects)
#   TARGETS="age sex bmi"     DATASETS="nki hbn" (output dir is the names joined
#   with "_", e.g. nki_hbn)   COMPUTE_SANITY_CHECKS=0   NILEARN_VENV=...
#   PAPER_NL_MODELS / PAPER_NL_STATS / PAPER_NL_KIND / PAPER_NL_N_MONTAGE /
#   PAPER_NL_THRESHOLD_PERCENTILE / PAPER_NL_CUT_COORDS / PAPER_NL_ROI_* /
#   PAPER_NL_SHOW_TITLE / PAPER_NL_SHOW_COLORBAR / PAPER_NL_LABEL_SCALE /
#   PAPER_NL_UNCNN_LABEL / PAPER_NL_ARCH_VARIANT   -- as in visualize_paper_nilearn.sh
#   SANITY_MODELS / SANITY_VIEWS / SANITY_KIND / SANITY_NORM / SANITY_THRESHOLD_PERCENTILE /
#   SANITY_LABEL_SCALE -- as in visualize_sanity_check.sh
#
# matplotlib>=3.8: nilearn.plotting needs it, brainiac_env pins 3.7 -- NILEARN_VENV is
# re-activated on top of env.sh exactly as in visualize_paper_nilearn.sh.
#
# Submit from the repo root:  sbatch slurm/visualize_combined_datasets.sh
#                             COMPUTE_SANITY_CHECKS=1 sbatch slurm/visualize_combined_datasets.sh

set -uo pipefail
shopt -s nullglob

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_combined_datasets.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

TARGETS=(${TARGETS:-age sex bmi})
DATASETS=(${DATASETS:-nki hbn})
COMBINED_NAME="$(IFS=_; echo "${DATASETS[*]}")"
PRETRAINED_ROOT="${REPO_DIR}/saliency_affine_pretrained"

case "${COMPUTE_SANITY_CHECKS:-0}" in
    1|true|TRUE|True|yes|YES) COMPUTE_SANITY_CHECKS=1 ;;
    0|false|FALSE|False|no|NO|"") COMPUTE_SANITY_CHECKS=0 ;;
    *) echo "ERROR: COMPUTE_SANITY_CHECKS=${COMPUTE_SANITY_CHECKS} (expected 1/true/yes or 0/false/no)" >&2; exit 1 ;;
esac
echo "TARGETS=${TARGETS[*]} DATASETS=${DATASETS[*]} -> ${COMBINED_NAME}  COMPUTE_SANITY_CHECKS=${COMPUTE_SANITY_CHECKS}"

NILEARN_VENV="${NILEARN_VENV:?set NILEARN_VENV to a venv with matplotlib>=3.8 and nilearn}"
source "${NILEARN_VENV}/bin/activate"
python3 -c "import matplotlib; assert tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 8)" || {
    echo "FATAL: $NILEARN_VENV's matplotlib is < 3.8 -- nilearn.plotting cannot import. Set NILEARN_VENV to a venv with a current matplotlib+nilearn." >&2
    exit 1
}

N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

# Same 3 duplicated-T1w HBN subjects compute_saliency_pretrained_heads.sh /
# compute_reference_baselines.sh exclude. NDARYC287UFV's baseline folder still sits in
# saliency_affine_pretrained/<target>/hbn/ from before that exclusion, so it must be
# dropped here explicitly. Override with e.g. EXCLUDE_SUBJECTS="" to pool everything.
EXCLUDE_SUBJECTS=(${EXCLUDE_SUBJECTS-NDAREF662XZT NDARVN370YL8 NDARYC287UFV})

# ---- figure settings: identical defaults to visualize_paper_nilearn.sh ----
if [ -z "${PAPER_NL_MODELS+x}" ]; then
    PAPER_NL_MODELS=(fs_roi optim_uncnn_avg brainiac swinbrain simclr anatcl_global)
else
    PAPER_NL_MODELS=(${PAPER_NL_MODELS})
fi
PAPER_NL_STATS=(${PAPER_NL_STATS:-mean})
PAPER_NL_KIND="${PAPER_NL_KIND:-abs}"
PAPER_NL_N_MONTAGE="${PAPER_NL_N_MONTAGE:-4}"
PAPER_NL_THRESHOLD_PERCENTILE="${PAPER_NL_THRESHOLD_PERCENTILE:--0.1}"
PAPER_NL_CUT_COORDS=(${PAPER_NL_CUT_COORDS:--17 1 15})
PAPER_NL_ROI_SEG_TEMPLATE="${PAPER_NL_ROI_SEG_TEMPLATE:-${REPO_DIR}/templates/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz}"
PAPER_NL_ROI_LUT="${PAPER_NL_ROI_LUT:-${FREESURFER_HOME:-}/FreeSurferColorLUT.txt}"
PAPER_NL_SHOW_TITLE="${PAPER_NL_SHOW_TITLE:-0}"
PAPER_NL_SHOW_COLORBAR="${PAPER_NL_SHOW_COLORBAR:-0}"
PAPER_NL_LABEL_SCALE="${PAPER_NL_LABEL_SCALE:-2}"
PAPER_NL_UNCNN_LABEL="${PAPER_NL_UNCNN_LABEL:-1}"
PAPER_NL_ARCH_VARIANT="${PAPER_NL_ARCH_VARIANT:-step01_head}"

COMMON_FLAGS=(
    --models "${PAPER_NL_MODELS[@]}"
    --kind "$PAPER_NL_KIND"
    --n-montage "$PAPER_NL_N_MONTAGE"
    --threshold-percentile "$PAPER_NL_THRESHOLD_PERCENTILE"
    --label-scale "$PAPER_NL_LABEL_SCALE"
)
[ "$PAPER_NL_SHOW_COLORBAR" = "0" ] && COMMON_FLAGS+=(--no-colorbar)
[ "$PAPER_NL_UNCNN_LABEL" = "1" ] && COMMON_FLAGS+=(--uncnn-label)
[ "${#PAPER_NL_CUT_COORDS[@]}" -eq 3 ] && COMMON_FLAGS+=(--cut-coords "${PAPER_NL_CUT_COORDS[@]}")

# Real (per-subject) models pooled in step 1; optim_uncnn_avg is derived from the
# 3 channels afterwards, never pooled directly (no per-subject source files).
POOL_MODELS=(optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac swinbrain simclr anatcl_global anatcl_local fs_roi)
# fs_roi has no data_/input_ sanity-check maps at all (see calculate_gradcam_tabular.py); arch only.
SANITY_POOL_MODELS=(optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac swinbrain simclr anatcl_global anatcl_local)

for TARGET in "${TARGETS[@]}"; do
    echo "################ TARGET=$TARGET ################"
    DATASET_DIRS=()
    for D in "${DATASETS[@]}"; do
        DATASET_DIRS+=("${PRETRAINED_ROOT}/${TARGET}/${D}")
    done
    SALIENCY_DIR="${PRETRAINED_ROOT}/${TARGET}/${COMBINED_NAME}"
    AGGREGATE_DIR="${SALIENCY_DIR}/aggregate"
    ROI_AGGREGATE_DIR="${SALIENCY_DIR}/aggregate_roi"
    mkdir -p "$AGGREGATE_DIR" "$ROI_AGGREGATE_DIR"

    echo "=== [$TARGET] pooling ${DATASETS[*]} baseline maps into one map per model ==="
    if ! python3 src/combine_datasets_saliency.py --dataset-dirs "${DATASET_DIRS[@]}" \
        --output-dir "$AGGREGATE_DIR" --models "${POOL_MODELS[@]}" --kinds signed abs --stats mean std \
        --n-jobs "$N_JOBS" --exclude-subjects "${EXCLUDE_SUBJECTS[@]}"; then
        echo "  WARN: combine_datasets_saliency.py wrote nothing for $TARGET -- skipping this target"
        continue
    fi

    python3 src/average_uncnn_channels.py --aggregate-dir "$AGGREGATE_DIR" --stats mean std --kinds signed abs \
        --tag-prefixes ig_reference_

    if ! python3 src/regionalize_saliency.py \
        --saliency-dir "$AGGREGATE_DIR" --output-dir "$ROI_AGGREGATE_DIR" \
        --seg-template "$PAPER_NL_ROI_SEG_TEMPLATE" --lut "$PAPER_NL_ROI_LUT" \
        --models "${PAPER_NL_MODELS[@]}" --stats "${PAPER_NL_STATS[@]}" --kinds "$PAPER_NL_KIND"; then
        echo "  WARN: regionalize_saliency.py failed -- the region-averaged figure below will be skipped/incomplete"
    fi

    for STAT in "${PAPER_NL_STATS[@]}"; do
        TITLE_FLAGS=(); ROI_TITLE_FLAGS=()
        if [ "$PAPER_NL_SHOW_TITLE" != "0" ]; then
            TITLE_FLAGS=(--title "ig_popmean -- ${COMBINED_NAME} across-subject aggregate (${STAT})")
            ROI_TITLE_FLAGS=(--title "ig_popmean -- ${COMBINED_NAME} region-averaged (DKT atlas) aggregate (${STAT})")
        fi
        OUT="$AGGREGATE_DIR/final_paper_ig_popmean_nilearn_${STAT}_${PAPER_NL_KIND}.pdf"
        if python3 src/visualize_paper_nilearn.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
            --saliency-dir "$AGGREGATE_DIR" "${TITLE_FLAGS[@]}" --output "$OUT"; then
            echo "  wrote $OUT"
        else
            echo "  WARN: visualize_paper_nilearn.py failed for stat $STAT"
        fi
        OUT_ROI="$ROI_AGGREGATE_DIR/paper_ig_popmean_nilearn_${STAT}_${PAPER_NL_KIND}_roi.png"
        if python3 src/visualize_paper_nilearn.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
            --saliency-dir "$ROI_AGGREGATE_DIR" "${ROI_TITLE_FLAGS[@]}" --output "$OUT_ROI"; then
            echo "  wrote $OUT_ROI"
        else
            echo "  WARN: visualize_paper_nilearn.py failed for ROI-averaged stat $STAT"
        fi
    done

    [ "$COMPUTE_SANITY_CHECKS" = "1" ] || continue

    echo "=== [$TARGET] combining sanity-check maps across ${DATASETS[*]} ==="
    python3 src/combine_datasets_saliency.py --sanity-check arch --arch-variant "$PAPER_NL_ARCH_VARIANT" \
        --dataset-dirs "${DATASET_DIRS[@]}" --output-dir "$SALIENCY_DIR" \
        --models "${POOL_MODELS[@]}" --kinds signed abs --n-jobs "$N_JOBS" --exclude-subjects "${EXCLUDE_SUBJECTS[@]}"
    for SANITY_TEST in data input; do
        python3 src/combine_datasets_saliency.py --sanity-check "$SANITY_TEST" \
            --dataset-dirs "${DATASET_DIRS[@]}" --output-dir "$SALIENCY_DIR" \
            --models "${SANITY_POOL_MODELS[@]}" --kinds signed abs
    done

    # optim_uncnn_avg for arch_/data_/input_, same templates as visualize_paper_nilearn.sh.
    python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
        --kinds signed abs --tag-prefixes ig_reference_ \
        --input-template "arch_optim_uncnn_${PAPER_NL_ARCH_VARIANT}_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
        --output-template "arch_optim_uncnn_${PAPER_NL_ARCH_VARIANT}_avg_{tag_prefix}{kind}_saliency.nii.gz"
    python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
        --kinds signed abs --tag-prefixes ig_reference_ \
        --input-template "data_optim_uncnn_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
        --output-template "data_optim_uncnn_avg_{tag_prefix}{kind}_saliency.nii.gz"
    python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
        --kinds signed abs --tag-prefixes ig_reference_ \
        --input-template "input_optim_uncnn_permute_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
        --output-template "input_optim_uncnn_permute_avg_{tag_prefix}{kind}_saliency.nii.gz"

    for SANITY_TEST in arch data input; do
        SANITY_TITLE_FLAGS=()
        [ "$PAPER_NL_SHOW_TITLE" != "0" ] && SANITY_TITLE_FLAGS=(--title "ig_reference -- ${COMBINED_NAME} ${SANITY_TEST} sanity check")
        OUT_SANITY="$AGGREGATE_DIR/paper_sanity_check_nilearn_${SANITY_TEST}_${PAPER_NL_KIND}.png"
        if python3 src/visualize_paper_nilearn.py "${COMMON_FLAGS[@]}" --sanity-check "$SANITY_TEST" \
            --arch-variant "$PAPER_NL_ARCH_VARIANT" --saliency-dir "$SALIENCY_DIR" \
            "${SANITY_TITLE_FLAGS[@]}" --output "$OUT_SANITY"; then
            echo "  wrote $OUT_SANITY"
        else
            echo "  WARN: visualize_paper_nilearn.py failed for sanity-check test $SANITY_TEST"
        fi
    done
done

# Cross-target sanity-check grid (all models x all targets' arch/data/input in one
# figure) -- once, after every target's combined maps exist.
if [ "$COMPUTE_SANITY_CHECKS" = "1" ]; then
    if [ -z "${SANITY_MODELS+x}" ]; then
        SANITY_MODELS=(optim_uncnn_avg brainiac swinbrain simclr anatcl_global anatcl_local)
    else
        SANITY_MODELS=(${SANITY_MODELS})
    fi
    SANITY_VIEWS=(${SANITY_VIEWS:-axial coronal sagittal})
    SANITY_KIND="${SANITY_KIND:-abs}"
    SANITY_NORM="${SANITY_NORM:-0}"
    SANITY_NORM_FLAG=(); NORM_SUFFIX=""
    if [ "$SANITY_NORM" = "0" ]; then SANITY_NORM_FLAG=(--no-norm); NORM_SUFFIX="_raw"; fi
    for VIEW in "${SANITY_VIEWS[@]}"; do
        OUT="${PRETRAINED_ROOT}/sanity_check_all_models_all_targets_${VIEW}_${SANITY_KIND}_${COMBINED_NAME}${NORM_SUFFIX}.png"
        if python3 src/visualize_sanity_check.py \
            --saliency-affine-root "$PRETRAINED_ROOT" --dataset "$COMBINED_NAME" --no-title \
            --targets "${TARGETS[@]}" --models "${SANITY_MODELS[@]}" --checks arch data input \
            --arch-variant "$PAPER_NL_ARCH_VARIANT" \
            --view "$VIEW" --kind "$SANITY_KIND" \
            --threshold-percentile "${SANITY_THRESHOLD_PERCENTILE:-0.0}" \
            --label-scale "${SANITY_LABEL_SCALE:-1.0}" "${SANITY_NORM_FLAG[@]}" --output "$OUT"; then
            echo "  wrote $OUT"
        else
            echo "  WARN: visualize_sanity_check.py failed for view $VIEW"
        fi
    done
fi

echo "=== Done: combined ${COMBINED_NAME} figures ==="
