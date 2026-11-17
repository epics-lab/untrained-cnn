#!/usr/bin/env bash
#SBATCH --job-name=all_sanity_checks
#SBATCH --time=48:0:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Runs TWO of the Adebayo et al. (2018) sanity-check battery's THREE checks
# -- --randomize-input, --randomize-data -- for all 6 image-based models PLUS
# fs_roi (via calculate_gradcam_tabular.py, same flags -- see
# run_model_tabular below and that script's own module docstring for what
# had to be added there), against ONE reference subject, writing straight
# into saliency_outputs/ (not a per-subject subfolder). This is deliberately
# separate from
# compute_saliency.sh, which only ever runs --randomize-input as a
# byproduct of its per-subject baseline sweep across the full cohort: the
# other check here is far more expensive (--randomize-data refits a ridge
# head and needs the full cached feature matrix) and, per Adebayo et al.'s
# own methodology, is a diagnostic check on the *method*, not per-subject
# saliency outputs -- there's no reason to repeat it for all 15 subjects.
#
# --randomize-model (the THIRD check) does NOT run here -- it moved to
# slurm/compute_sanity_checks_randomize_model.sh, which runs it as a
# genuinely per-(target, subject) SLURM ARRAY job across the FULL cohort
# (not one reference subject) with a FIXED --randomize-seed, then
# slurm/aggregate_sanity_check_randomize_model.sh averages the results --
# see that script's own module docstring for why (population-level
# comparison against the equally-population-averaged trained aggregate,
# rather than a single noisy reference-subject sample). Run those two
# scripts, in that order, for the arch_*.nii.gz files this script used to
# produce directly.
#
# Output lands directly in saliency_outputs/<target>/ (data_*, input_*
# prefixes, no subject-ID prefix -- calculate_gradcam.py/
# calculate_gradcam_tabular.py insert the <target> path component
# themselves, see those scripts' own docstrings) so it drops straight into
# visualize_subjects.sh's existing globs (see that script's DATA_PREFIX/
# INPUT_PREFIX construction) without any changes there -- and,
# for every non-default attribution method, with calculate_gradcam.py's own
# gxi_/ig_zero_/ig_reference_ tag inserted, which visualize_saliency.py's
# display_label() now surfaces as its own correctly-labeled row (see
# split_attribution_tag() there) instead of silently collapsing into
# "vanillagrad_...".
#
# unCNN's single run above writes 3 separate channel maps (raw/rank/sobel)
# per data_/input_ check (arch_'s own channel-averaging happens in
# slurm/aggregate_sanity_check_randomize_model.sh instead, see above);
# right after it here, src/average_uncnn_channels.py is also run (with
# --input-template/--output-template) to average those 3 into one extra
# data_optim_uncnn_avg_/input_optim_uncnn_permute_avg_ map per attribution
# method -- written ALONGSIDE the 3 channel maps, matching
# visualize_aggregate.sh's optim_uncnn_avg row for the across-subject
# aggregate figures. See that script's CAVEAT before reading too much into
# the avg map specifically (raw/rank/sobel are heterogeneous units).
#
# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list) instead
# of 3 sequential `TARGET=... sbatch` submissions -- selects which head/
# checkpoint suffix and cached label file are used, same as
# compute_saliency.sh's own TASK_TARGETS convention. Submit a single
# target's task directly with e.g. `sbatch --array=0 slurm/compute_sanity_checks.sh`
# (age only), `--array=1` (sex only), `--array=2` (bmi only).
#
# Prerequisites:
#   - checkpoints/heads/*_{brain_age,sex,bmi}_head_affine.pt (slurm/train_all_models.sh,
#     train_linear.py appends the _affine suffix itself)
#   - FEATURES_DIR below (default: artifacts/, train_linear.py's own
#     DEFAULT_OUTPUT_DIR) must contain {cache_name}_X_features_raw.npy and
#     _y_{age,sex,bmi}.npy for every model -- also written by train_all_models.sh.
#     --randomize-data fails loudly (FileNotFoundError) per model if these
#     are missing; --randomize-input doesn't need them.
#   - reference_baselines/nki/*_reference.npy (slurm/compute_reference_baselines.sh)
#     for the ig_reference spec below specifically. If missing, only that
#     one spec fails per model (calculate_gradcam.py raises
#     FileNotFoundError); every other spec/model still runs (set -uo
#     pipefail, no -e -- this script doesn't abort on one failed invocation).
#
# DATASET env var (OPTIONAL): unset/empty (default) preserves this script's
# original behavior exactly -- this repo's own checkpoints/heads/, NKI
# only, output under saliency_affine/. Set to DATASET=nki|ppmi|hbn instead
# switches to the externally pre-trained heads (same
# $HEADS_DIR/
# checkpoints slurm/compute_saliency_pretrained_heads.sh uses, discovered
# via src/list_pretrained_heads.py), writing under
# saliency_affine_pretrained/<target>/<dataset>/ instead -- a single
# reference subject (first row of that dataset's own subject list), same
# as the DATASET-unset case. --randomize-data needs a cached feature
# matrix (FEATURES_DIR/{cache_name}_X_features_raw.npy/_y_{target}.npy)
# this repo's own train_linear.py builds -- no such cache exists for
# the external checkpoints (they were fit externally), so the 6 image models
# under DATASET nki/ppmi/hbn only ever run --randomize-input, gracefully
# skipping --randomize-data via run_model's own existing cache-miss WARNING
# (same code path as a genuinely missing NKI cache, not a new one). fs_roi
# is the exception: it reuses the external {DATASET}_FS_ROI_{target}.pt
# checkpoint (nki/hbn only, same as compute_saliency_pretrained_heads.sh --
# no compatible checkpoint for ppmi) against THIS repo's own fs_roi feature
# cache (artifacts/ for nki, artifacts/hbn_fs_roi/ for hbn -- built once via
# train_linear.py, see compute_saliency_pretrained_heads.sh's own SCOPE
# comment), so --randomize-data DOES work for fs_roi wherever that cache's
# own --target matches (age only for the hbn cache as of this writing).
#
# COST NOTES (see docs/attribution_methods.md / calculate_gradcam.py's
# --randomize-*/--attribution-method help text for the full picture):
#   - Runs ALL 4 attribution specs (gradient, gradient_x_input, ig_zero,
#     ig_reference -- matching compute_saliency.sh's ATTRIBUTION_SPECS) x
#     the 2 sanity checks left here x 6 models. integrated_gradients costs
#     ~IG_STEPS forward+backward passes per saliency computation it's asked
#     for (baseline + --randomize-input + every --randomize-data seed)
#     instead of 1 -- IG_STEPS below is deliberately far below
#     calculate_gradcam.py's own --ig-steps default (50) to keep this
#     tractable on CPU (no GPU requested here), same tradeoff
#     compute_saliency.sh makes. Even so, the two IG specs dominate this
#     script's runtime by roughly IG_STEPS/1 over the gradient/gxi specs --
#     --time above is a rough starting guess, not a validated number; raise
#     it or split ATTRIBUTION_SPECS across separate job submissions
#     (comment out entries below) if it doesn't fit your queue budget.
#   - --randomize-data-seeds averages the Spearman correlation over several
#     permutation draws (see fit_ridge_head_inmemory's docstring for why a
#     single draw is noisier here than in Adebayo et al.'s deep-net case);
#     each extra seed is one more RidgeCV refit + forward/backward pass (or
#     IG_STEPS of them, for the IG specs) per spec.

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_sanity_checks.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# train_linear.py's own default cache location (DEFAULT_OUTPUT_DIR); override
# with `export FEATURES_DIR=...` in slurm/env.sh if train_all_models.sh was
# pointed at a different --output-dir on this machine. Only meaningful for
# DATASET unset/nki/hbn's fs_roi (see DATASET header comment) and, for
# DATASET unset, --randomize-data's own cache check for every model.
FEATURES_DIR="${FEATURES_DIR:-${REPO_DIR}/artifacts}"

# ---------------------------------------------------------------------------
# DATASET (optional, see header comment above): empty/unset resolves to
# this script's original NKI/own-checkpoints behavior exactly; nki/ppmi/hbn
# switches to the externally pre-trained heads, same per-dataset
# resolution as slurm/compute_saliency_pretrained_heads.sh.
# ---------------------------------------------------------------------------
DATASET="${DATASET:-}"
case "$DATASET" in
    "")
        SALIENCY_ROOT="saliency_affine"
        SUBJECTS_FILE="$NKI_DEMOGRAPHICS"
        TURBOPREP_ROOT="$NKI_AFFINE_TURBOPREP_ROOT"
        CAT12_ROOTS=("$NKI_BIDS_ROOT")
        HEADS_DIR="${REPO_DIR}/checkpoints/heads"
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="$FEATURES_DIR"
        ;;
    nki)
        SALIENCY_ROOT="saliency_affine_pretrained"
        SUBJECTS_FILE="$NKI_DEMOGRAPHICS"
        TURBOPREP_ROOT="$NKI_AFFINE_TURBOPREP_ROOT"
        CAT12_ROOTS=("$NKI_BIDS_ROOT")
        HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="$FEATURES_DIR"
        ;;
    ppmi)
        SALIENCY_ROOT="saliency_affine_pretrained"
        SUBJECTS_FILE="$PPMI_SUBJECTS_LIST"
        TURBOPREP_ROOT="$PPMI_TURBOPREP_ROOT"
        CAT12_ROOTS=("$PPMI_CAT12_BIDS_ROOT")
        HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"
        FS_ROI_SUPPORTED=false
        ;;
    hbn)
        SALIENCY_ROOT="saliency_affine_pretrained"
        SUBJECTS_FILE="$HBN_SUBJECTS_LIST"
        TURBOPREP_ROOT="$HBN_TURBOPREP_ROOT"
        CAT12_ROOTS=("$HBN_CAT12_BIDS_ROOT_1" "$HBN_CAT12_BIDS_ROOT_2")
        HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="${REPO_DIR}/artifacts/hbn_fs_roi"
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected unset, nki, ppmi, or hbn)" >&2
        exit 1
        ;;
esac

TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/compute_sanity_checks.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
case "$TARGET" in
    age) CKPT_SUFFIX=brain_age_head_affine ;;
    sex) CKPT_SUFFIX=sex_head_affine ;;
    bmi) CKPT_SUFFIX=bmi_head_affine ;;
    *) echo "ERROR: unknown TARGET=$TARGET (expected age/sex/bmi)" >&2; exit 1 ;;
esac
echo "DATASET=${DATASET:-<unset>} TARGET=$TARGET (checkpoint suffix: ${CKPT_SUFFIX})"

if [ -n "$DATASET" ]; then
    SALIENCY_DIR="${REPO_DIR}/${SALIENCY_ROOT}/${DATASET}"
    mkdir -p "${REPO_DIR}/${SALIENCY_ROOT}/${TARGET}/${DATASET}"
else
    SALIENCY_DIR="${REPO_DIR}/${SALIENCY_ROOT}"
    mkdir -p "${SALIENCY_DIR}/${TARGET}"
fi

RANDOMIZE_DATA_SEEDS=1

IG_STEPS=50

# "tag:extra flags" pairs, matching compute_saliency.sh's ATTRIBUTION_SPECS
# exactly -- flags are intentionally word-split on use (no individual flag
# value here contains a space). Comment out entries to split this across
# multiple job submissions (e.g. gradient+gxi in one job, the two IG specs
# in a separate, longer---time job) if the combined runtime doesn't fit
# your queue.
# DATASET set passes --reference-baselines-dir explicitly -- same fix as
# compute_saliency_pretrained_heads.sh/compute_sanity_checks_randomize_
# model.sh: calculate_gradcam.py's own default is reference_baselines/nki
# regardless of --model, which would otherwise be silently wrong for
# ppmi/hbn.
IG_FLAGS="--attribution-method integrated_gradients --ig-baseline reference --ig-steps ${IG_STEPS}"
if [ -n "$DATASET" ]; then
    IG_FLAGS="${IG_FLAGS} --reference-baselines-dir reference_baselines/${DATASET}"
fi
ATTRIBUTION_SPECS=(
    # "gradient:--attribution-method gradient"
    # "gradient_x_input:--attribution-method gradient_x_input"
    # "ig_zero:--attribution-method integrated_gradients --ig-baseline zero --ig-steps ${IG_STEPS}"
    "ig_reference:${IG_FLAGS}"
)

# Reference subject: first row of this dataset's own subject list (first
# row of $NKI_DEMOGRAPHICS when DATASET is unset, matching
# compute_saliency.sh's --array=0 subject) -- override with
# `export REFERENCE_SUBJECT=...` before submitting to pick a different one.
mapfile -t subjects < <(cut -f1 "$SUBJECTS_FILE")
REFERENCE_SUBJECT="${REFERENCE_SUBJECT:-${subjects[0]}}"
echo "Reference subject: $REFERENCE_SUBJECT"

TURBOPREP_NII="${TURBOPREP_ROOT}/${REFERENCE_SUBJECT}/normalized.nii.gz"

# run_model LABEL MODEL CHECKPOINT SUBJECT_NII CACHE_NAME [extra calculate_gradcam.py args...]
# CACHE_NAME must match the FeatureSpec.cache_name train_linear.py used for
# this model/level (see make_feature_spec in train_linear.py) -- only used
# here to check --randomize-data's prerequisite files exist once per model
# (not once per attribution spec -- the cache doesn't depend on attribution
# method) before spending specs on a model that will just fail that one flag.
run_model() {
    local LABEL="$1" MODEL="$2" CKPT="$3" NII="$4" CACHE_NAME="$5"
    shift 5
    local EXTRA_ARGS=("$@")

    local RANDOMIZE_DATA_FLAG="--randomize-data"
    if [ ! -e "${FEATURES_DIR}/${CACHE_NAME}_X_features_raw.npy" ] || [ ! -e "${FEATURES_DIR}/${CACHE_NAME}_y_${TARGET}.npy" ]; then
        echo "  WARNING: missing cached features at ${FEATURES_DIR}/${CACHE_NAME}_{X_features_raw,y_${TARGET}}.npy" \
             "-- skipping --randomize-data for ${LABEL} (run slurm/train_all_models.sh first, or set FEATURES_DIR)."
        RANDOMIZE_DATA_FLAG=""
    fi

    local SPEC TAG FLAGS
    for SPEC in "${ATTRIBUTION_SPECS[@]}"; do
        TAG="${SPEC%%:*}"
        FLAGS="${SPEC#*:}"
        echo "=== ${LABEL} (${TAG}) ==="
        # shellcheck disable=SC2086  # FLAGS/RANDOMIZE_DATA_FLAG are intentionally word-split, see above.
        time python3 src/calculate_gradcam.py --model "$MODEL" \
            --checkpoint-path "$CKPT" --target "$TARGET" \
            --subject-nii "$NII" --output-dir "$SALIENCY_DIR" \
            --features-dir "$FEATURES_DIR" \
            --randomize-input \
            --randomize-data-seeds "$RANDOMIZE_DATA_SEEDS" \
            "${EXTRA_ARGS[@]}" \
            $FLAGS \
            $RANDOMIZE_DATA_FLAG
    done
}



# run_model_tabular LABEL CHECKPOINT CACHE_NAME [FEATURES_DIR_OVERRIDE]
# fs_roi's analogue of run_model -- calculate_gradcam_tabular.py takes
# --subject-id directly (see compute_saliency.sh's identical
# run_model_tabular), not --subject-nii, so there's no NII argument here.
# Its data_/input_ sanity-check maps land straight in --output-dir
# (no subject-id join, matching every other model's shared reference-
# subject battery here) -- only its OWN baseline saliency gets a
# subject-id subfolder, same as compute_saliency.sh's per-subject sweep --
# see calculate_gradcam_tabular.py's subject_output_dir vs. args.output_dir.
#
# FEATURES_DIR_OVERRIDE (optional, 4th arg): defaults to the global
# $FEATURES_DIR (this repo's own NKI cache) when omitted -- pass
# $FS_ROI_FEATURES_DIR explicitly for DATASET=hbn, whose fs_roi feature
# cache lives in a separate directory (artifacts/hbn_fs_roi/, see DATASET
# header comment) so it was never built with the FEATURES_DIR global.
#
# KNOWN BLOCKER as of this writing: fs_roi's cached
# artifacts/fs_roi_dktatlas_grayvol_y_age.npy has 957 rows vs.
# _X_features_raw.npy/_subjects.npy's 956 -- an off-by-one row-count
# mismatch in the cache itself (not a bug in --randomize-data), which
# fit_ridge_head_inmemory (sklearn) rejects outright. --randomize-data will
# fail (loudly, but non-fatally -- no `-e` above) for fs_roi until that
# cache is regenerated with train_linear.py --model fs_roi so X/y/subjects
# are all the same length again. --randomize-input is unaffected (it never
# touches y).
run_model_tabular() {
    local LABEL="$1" CKPT="$2" CACHE_NAME="$3"
    local LOCAL_FEATURES_DIR="${4:-$FEATURES_DIR}"

    local RANDOMIZE_DATA_FLAG="--randomize-data"
    if [ ! -e "${LOCAL_FEATURES_DIR}/${CACHE_NAME}_X_features_raw.npy" ] || [ ! -e "${LOCAL_FEATURES_DIR}/${CACHE_NAME}_y_${TARGET}.npy" ]; then
        echo "  WARNING: missing cached features at ${LOCAL_FEATURES_DIR}/${CACHE_NAME}_{X_features_raw,y_${TARGET}}.npy" \
             "-- skipping --randomize-data for ${LABEL} (run slurm/train_all_models.sh first, or set FEATURES_DIR)."
        RANDOMIZE_DATA_FLAG=""
    fi

    local SPEC TAG FLAGS
    for SPEC in "${ATTRIBUTION_SPECS[@]}"; do
        TAG="${SPEC%%:*}"
        FLAGS="${SPEC#*:}"
        echo "=== ${LABEL} (${TAG}) ==="
        # shellcheck disable=SC2086  # FLAGS/RANDOMIZE_DATA_FLAG are intentionally word-split, see above.
        time python3 src/calculate_gradcam_tabular.py \
            --checkpoint-path "$CKPT" --target "$TARGET" \
            --subject-id "$REFERENCE_SUBJECT" --output-dir "$SALIENCY_DIR" \
            --features-dir "$LOCAL_FEATURES_DIR" --cache-name "$CACHE_NAME" \
            --randomize-input --randomize-data-seeds "$RANDOMIZE_DATA_SEEDS" \
            $FLAGS \
            $RANDOMIZE_DATA_FLAG
    done
}

if [ -z "$DATASET" ]; then
    # DATASET unset: this script's original behavior, unchanged -- this
    # repo's own checkpoints/heads/, NKI only.
    run_model "unCNN" optim_uncnn "${HEADS_DIR}/optim_uncnn_turboprep_covpool_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII" \
        "optim_uncnn_turboprep_covpool"

    # The "unCNN" run above already wrote 3 separate channel maps (raw/rank/
    # sobel, see calculate_gradcam.py's UNCNN_CHANNELS) per data_/input_ sanity
    # check (arch_'s own channel-averaging happens in
    # slurm/aggregate_sanity_check_randomize_model.sh instead) -- average them
    # into one extra "avg" map per check, the same way visualize_aggregate.sh's
    # optim_uncnn_avg row averages the across-subject aggregate maps
    # (src/average_uncnn_channels.py, generalized via
    # --input-template/--output-template so the one load/verify/mean/save
    # routine works for both the aggregate-dir files there and these
    # reference-subject sanity-check files here). Written ALONGSIDE, never
    # instead of, the 3 channel files -- see that script's CAVEAT on what
    # averaging raw/rank/sobel together actually means before reading too much
    # into the avg row specifically. --stats x is a required-but-unused dummy
    # (these filenames have no stat component) so average_uncnn_channels.py's
    # stat loop still runs exactly once.
    # echo "=== unCNN: averaging raw/rank/sobel sanity-check maps into optim_uncnn_avg ==="
    # python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
    #     --input-template "data_optim_uncnn_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
    #     --output-template "data_optim_uncnn_avg_{tag_prefix}{kind}_saliency.nii.gz"
    # python3 src/average_uncnn_channels.py --aggregate-dir "$SALIENCY_DIR" --stats x \
    #     --input-template "input_optim_uncnn_permute_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \
    #     --output-template "input_optim_uncnn_permute_avg_{tag_prefix}{kind}_saliency.nii.gz"

    run_model "BrainIAC" brainiac "${HEADS_DIR}/brainiac_turboprep_96_cls_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII" \
        "brainiac_turboprep_96_cls"
    run_model "SIMCLR" simclr "${HEADS_DIR}/simclr_turboprep_transpose_crop_150x192x192_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII" \
        "simclr_turboprep_transpose_crop_150x192x192"
    run_model "SWINBRAIN" swinbrain "${HEADS_DIR}/swinbrain_turboprep_128x128x64_encoder10_avgpool_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII" \
        "swinbrain_turboprep_128x128x64_encoder10_avgpool"

    # anatcl needs the subject's CAT12 mwp1 file; its filename has a variable
    # ses-/acq- suffix per subject (see compute_saliency.sh's identical block),
    # so glob for it rather than assuming one fixed filename.
    mapfile -t ANATCL_NII_MATCHES < <(compgen -G "${NKI_BIDS_ROOT}/sub-${REFERENCE_SUBJECT}/ses-*/anat/mri/mwp1sub-${REFERENCE_SUBJECT}*.nii")
    if [ "${#ANATCL_NII_MATCHES[@]}" -ne 1 ]; then
      echo "ERROR: expected exactly 1 anatcl CAT12 mwp1 file for subject ${REFERENCE_SUBJECT}, found ${#ANATCL_NII_MATCHES[@]}: ${ANATCL_NII_MATCHES[*]}" >&2
      exit 1
    fi
    ANATCL_NII="${ANATCL_NII_MATCHES[0]}"

    export ANATCL_SUFFIX=global_
    run_model "ANATCL GLOBAL" anatcl "${HEADS_DIR}/anatcl_global_cat12vbm_fold0_${CKPT_SUFFIX}.pt" "$ANATCL_NII" \
        "anatcl_global_cat12vbm_fold0" --anatcl-level global --anatcl-folds 0

    export ANATCL_SUFFIX=local_
    run_model "ANATCL LOCAL" anatcl "${HEADS_DIR}/anatcl_local_cat12vbm_fold0_${CKPT_SUFFIX}.pt" "$ANATCL_NII" \
        "anatcl_local_cat12vbm_fold0" --anatcl-level local --anatcl-folds 0

    run_model_tabular "FS ROI" "${HEADS_DIR}/fs_roi_dktatlas_grayvol_${CKPT_SUFFIX}.pt" "fs_roi_dktatlas_grayvol"
else
    # DATASET set: the externally pre-trained heads, discovered via
    # src/list_pretrained_heads.py -- same checkpoint-discovery/ANATCL_SUFFIX
    # logic as slurm/compute_saliency_pretrained_heads.sh. CACHE_NAME="" for
    # the 6 image models -- no cached feature matrix exists for the external
    # checkpoints (fit externally), so run_model's own existing cache-miss
    # check gracefully disables --randomize-data for all of them (see
    # DATASET header comment) -- not a new code path, the same WARNING a
    # genuinely missing NKI cache already triggers.
    ANATCL_NII=""
    resolve_anatcl_nii() {
        if [ -n "$ANATCL_NII" ]; then
            return
        fi
        local ROOT MATCHES=()
        for ROOT in "${CAT12_ROOTS[@]}"; do
            mapfile -t -O "${#MATCHES[@]}" MATCHES < <(compgen -G "${ROOT}/sub-${REFERENCE_SUBJECT}/ses-*/anat/mri/mwp1*.nii")
        done
        if [ "${#MATCHES[@]}" -ne 1 ]; then
            echo "ERROR: expected exactly 1 anatcl CAT12 mwp1 file for DATASET=$DATASET SUBJECT=${REFERENCE_SUBJECT} across ${CAT12_ROOTS[*]}, found ${#MATCHES[@]}: ${MATCHES[*]}" >&2
            exit 1
        fi
        ANATCL_NII="${MATCHES[0]}"
    }

    mapfile -t HEAD_ROWS < <(python3 src/list_pretrained_heads.py \
        --heads-dir "$HEADS_DIR" --dataset "$DATASET" --target "$TARGET")

    if [ "${#HEAD_ROWS[@]}" -eq 0 ]; then
        echo "WARN: no usable pretrained heads found for DATASET=$DATASET TARGET=$TARGET -- nothing to do."
    fi

    for ROW in "${HEAD_ROWS[@]}"; do
        IFS=$'\t' read -r LABEL MODEL_ARGS CKPT_PATH <<< "$ROW"
        read -ra MODEL_PARTS <<< "$MODEL_ARGS"
        MODEL="${MODEL_PARTS[0]}"
        EXTRA_ARGS=("${MODEL_PARTS[@]:1}")

        if [ "$MODEL" = "anatcl" ]; then
            resolve_anatcl_nii
            ANATCL_LEVEL=""
            for i in "${!EXTRA_ARGS[@]}"; do
                if [ "${EXTRA_ARGS[$i]}" = "--anatcl-level" ]; then
                    ANATCL_LEVEL="${EXTRA_ARGS[$((i + 1))]}"
                fi
            done
            if [ -z "$ANATCL_LEVEL" ]; then
                echo "ERROR: anatcl row ($LABEL, model_args=$MODEL_ARGS) has no --anatcl-level -- refusing to run it, it would silently overwrite another anatcl level's output." >&2
                exit 1
            fi
            export ANATCL_SUFFIX="${ANATCL_LEVEL}_"
            run_model "$LABEL" "$MODEL" "$CKPT_PATH" "$ANATCL_NII" "" "${EXTRA_ARGS[@]}"
        else
            export ANATCL_SUFFIX=""
            run_model "$LABEL" "$MODEL" "$CKPT_PATH" "$TURBOPREP_NII" "" "${EXTRA_ARGS[@]}"
        fi
    done

    if [ "$FS_ROI_SUPPORTED" = true ]; then
        run_model_tabular "FS ROI" "${HEADS_DIR}/${DATASET^^}_FS_ROI_${TARGET}.pt" "fs_roi_dktatlas_grayvol" "$FS_ROI_FEATURES_DIR"
    else
        echo "WARN: fs_roi skipped for DATASET=$DATASET -- no compatible feature schema (see DATASET header comment)."
    fi
fi

echo "=== Done: sanity checks for DATASET=${DATASET:-<unset>} reference subject $REFERENCE_SUBJECT ==="
