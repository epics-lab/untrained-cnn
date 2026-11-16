#!/usr/bin/env bash
#SBATCH --job-name=sanity_check_randomize_model_linear_hbn
#SBATCH --time=2:30:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --array=0-2873
#SBATCH --output=slurm/logs/%x_%a.out
#
# --randomize-model half of the Adebayo et al. (2018) sanity-check battery,
# split out of slurm/compute_sanity_checks.sh (which still owns
# --randomize-input/--randomize-data, unchanged, against a single reference
# subject -- see that script's own docstring for why those two stay cheap
# and single-subject) into its own per-(target, subject) SLURM ARRAY job,
# same TARGET_IDX/SUBJECT_IDX index-flattening convention as
# slurm/compute_saliency.sh -- so this runs across the FULL cohort, not one
# reference subject, and slurm/aggregate_sanity_check_randomize_model.sh
# (run once ALL these array tasks finish) averages the per-subject
# randomized-model maps into the same arch_*.nii.gz files every downstream
# consumer (visualize_paper_nilearn.py --sanity-check arch,
# compute_saliency_correlations.py) already reads.
#
# DATASET env var (OPTIONAL, unlike slurm/compute_saliency_pretrained_heads.sh's
# required one): unset/empty (default) preserves every line of this script's
# original behavior -- NKI, this repo's own checkpoints/heads/, output under
# saliency_affine/. Set to DATASET=nki|ppmi|hbn instead switches to
# the externally pre-trained heads (same
# $HEADS_DIR/ checkpoints,
# discovered via src/list_pretrained_heads.py exactly like
# slurm/compute_saliency_pretrained_heads.sh does), writing under
# saliency_affine_pretrained/<target>/<dataset>/arch_linear_persubject/<subject>/
# instead -- so this same head-only randomization battery can now be run
# against either checkpoint source. fs_roi is never randomized in the
# DATASET-set branch: list_pretrained_heads.py already excludes the 3
# FreeSurfer-surface models (no matching checkpoint for fs_roi's DKT-volume
# feature set exists among the external heads either), so run_model_tabular
# stays exclusive to the DATASET-unset (NKI) branch below. --array bounds
# differ per DATASET exactly like compute_saliency_pretrained_heads.sh's own
# (same SUBJECTS_FILE/TASK_TARGETS resolution): nki 0-2873, ppmi 0-1787,
# hbn 0-2993 (vs. this script's own NKI-only 0-2873 default when DATASET is
# unset -- same number only because NKI's own N_SUBJECTS/TASK_TARGETS happen
# to match).
#
# NOTE: aggregating this head-only (step01_head cascade) variant across the
# cohort still needs its own separate pass, regardless of DATASET --
# slurm/aggregate_sanity_check_randomize_model.sh, run once this script's
# array tasks for a given (TARGET, DATASET) have all finished.
#
# HEAD-ONLY VARIANT (and the ONLY one this script, or any current script,
# produces/can reproduce): --randomize-model-mode cascade with
# --randomize-max-layers 1, which calculate_gradcam.py's
# cascading_randomize_pipeline() resets output-side first -- i.e. only the
# linear head, leaving the backbone (the pretrained foundation-model
# weights, or optim_uncnn's already-random-and-frozen untrained CNN)
# exactly as loaded from checkpoint. This isolates "does the saliency map
# depend on the trained LINEAR HEAD" from "does it depend on the backbone
# at all". calculate_gradcam.py/calculate_gradcam_tabular.py name these
# outputs "arch_{model}_step01_head_..." (cascade's per-step filename),
# written under a separate arch_linear_persubject/ subtree here (see
# OUT_DIR below). A full-model (backbone included, "arch_{model}_all_...")
# variant also exists on disk for NKI, from an older, now-removed code
# path in slurm/compute_sanity_checks.sh's git history -- it predates this
# script's own split-out and is NOT reproducible by any current script,
# for any dataset, so it isn't offered as an option here.
#
# WHY per-subject + average, not one reference subject: --randomize-seed
# below is a FIXED constant (same for every array task), so every subject's
# model gets reset to the IDENTICAL random weights -- only the real input
# varies across tasks. Averaging the resulting maps across subjects then
# means "this one randomly-initialized model's average explanation across
# many real inputs," directly analogous to how the TRAINED aggregate
# (slurm/visualize_aggregate.sh) is already an across-subject average, not a
# single subject's map -- putting the arch_ comparison on the same
# population-level footing as what it's compared against, rather than
# comparing a population average to a single noisy sample.
#
# Prerequisites: identical to slurm/compute_saliency.sh (checkpoints/heads/
# *_{brain_age,sex,bmi}_head_affine.pt from slurm/train_all_models.sh;
# reference_baselines/nki/*_reference.npy from slurm/compute_reference_baselines.sh
# for the ig_reference spec -- if missing, only that spec fails per model,
# see calculate_gradcam.py's own FileNotFoundError).
#
# Output: saliency_affine/<target>/arch_linear_persubject/<SUBJECT>/arch_*.nii.gz
# (DATASET unset) or
# saliency_affine_pretrained/<target>/<dataset>/arch_linear_persubject/<SUBJECT>/arch_*.nii.gz
# (DATASET set, see DATASET env var above) -- a TEMPORARY per-subject
# location, not a final aggregate destination (see NOTE above --
# slurm/aggregate_sanity_check_randomize_model.sh consumes this subtree,
# run separately once every array task here finishes). The baseline
# (trained-model) map compute_and_save_saliency always writes alongside it
# (calculate_gradcam.py has no flag to skip that step) is deleted
# immediately after each call -- it's a byproduct duplicate of what
# slurm/compute_saliency.sh's (or, DATASET set,
# slurm/compute_saliency_pretrained_heads.sh's) own per-subject sweep
# already stores; keeping it here too would ~double this script's disk
# footprint for nothing.
#
# COST: this is calculate_gradcam.py's --randomize-model path (one full
# extra forward+backward saliency computation on top of the always-computed
# baseline) run ONCE PER SUBJECT instead of once total -- roughly N_SUBJECTS
# times slurm/compute_sanity_checks.sh's own --randomize-model cost. IG_STEPS
# below defaults LOWER than compute_sanity_checks.sh's own IG_STEPS=50
# specifically because of that multiplier -- raise it (and --time above) if
# your queue/budget allows; the population-mean reasoning above still holds
# at fewer steps, just with a coarser per-subject IG approximation feeding
# into an average that's already smoothing over many subjects.
#
# --array=0-2873 above matches compute_saliency.sh's own default (DATASET
# unset: all 3 targets x N_SUBJECTS=958 rows in $NKI_DEMOGRAPHICS as of this
# writing -- RECOMPUTE if that row count ever changes, e.g. `cut -f1
# "$NKI_DEMOGRAPHICS" | wc -l` after sourcing slurm/env.sh, then
# --array=0-$((3*N_SUBJECTS-1))). DATASET set instead uses that dataset's
# own TASK_TARGETS/N_SUBJECTS, same bounds as
# compute_saliency_pretrained_heads.sh's own header table (N_SUBJECTS
# already EXCLUDE_SUBJECTS-filtered -- HBN's raw subject-list row count is
# 998, but N_SUBJECTS is 997 since NDARYC287UFV is excluded, see that case
# statement):
#   DATASET=nki:  N_SUBJECTS=958, TASK_TARGETS=(age sex bmi) -> --array=0-2873
#   DATASET=ppmi: N_SUBJECTS=894, TASK_TARGETS=(age sex)     -> --array=0-1787
#   DATASET=hbn:  N_SUBJECTS=997, TASK_TARGETS=(age sex bmi) -> --array=0-2990
# Same TARGET_IDX/SUBJECT_IDX layout as compute_saliency.sh: one contiguous
# N_SUBJECTS-sized block per target, in TASK_TARGETS order -- submit a
# single target's block directly (e.g. DATASET unset, age: 0..N_SUBJECTS-1,
# sex: N_SUBJECTS..2*N_SUBJECTS-1, bmi: 2*N_SUBJECTS..3*N_SUBJECTS-1). See
# compute_saliency.sh's own header comment for the off-by-one/stale-
# N_SUBJECTS caveats, which apply here identically.

set -uo pipefail
shopt -s nullglob

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_sanity_checks_randomize_model.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

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
        TASK_TARGETS=(age sex bmi)
        TURBOPREP_ROOT="$NKI_AFFINE_TURBOPREP_ROOT"
        HEADS_DIR="${REPO_DIR}/checkpoints/heads"
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="${REPO_DIR}/artifacts"
        EXCLUDE_SUBJECTS=()
        ;;
    nki)
        SALIENCY_ROOT="saliency_affine_pretrained"
        SUBJECTS_FILE="$NKI_DEMOGRAPHICS"
        TASK_TARGETS=(age sex bmi)
        TURBOPREP_ROOT="$NKI_AFFINE_TURBOPREP_ROOT"
        CAT12_ROOTS=("$NKI_BIDS_ROOT")
        HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="${REPO_DIR}/artifacts"
        EXCLUDE_SUBJECTS=()
        ;;
    ppmi)
        SALIENCY_ROOT="saliency_affine_pretrained"
        SUBJECTS_FILE="$PPMI_SUBJECTS_LIST"
        TASK_TARGETS=(age sex)  # no bmi heads for PPMI
        TURBOPREP_ROOT="$PPMI_TURBOPREP_ROOT"
        CAT12_ROOTS=("$PPMI_CAT12_BIDS_ROOT")
        HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"
        FS_ROI_SUPPORTED=false
        # Same exclusion slurm/compute_reference_baselines.sh's own PPMI
        # block uses -- currently a no-op (3510 isn't in
        # $PPMI_SUBJECTS_LIST, verified directly), kept here for parity/
        # documentation in case that ever changes.
        EXCLUDE_SUBJECTS=(3510)
        ;;
    hbn)
        SALIENCY_ROOT="saliency_affine_pretrained"
        SUBJECTS_FILE="$HBN_SUBJECTS_LIST"
        TASK_TARGETS=(age sex bmi)
        TURBOPREP_ROOT="$HBN_TURBOPREP_ROOT"
        CAT12_ROOTS=("$HBN_CAT12_BIDS_ROOT_1" "$HBN_CAT12_BIDS_ROOT_2")  # two roots, search both
        HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="${REPO_DIR}/artifacts/hbn_fs_roi"
        # Same 3 duplicated-T1w subjects slurm/compute_reference_baselines.sh's
        # own HBN block excludes. NDAREF662XZT/NDARVN370YL8 are already
        # absent from $HBN_SUBJECTS_LIST (verified directly); NDARYC287UFV
        # is NOT -- without this, it would get a full per-subject arch_ run
        # here scored against a reference_baselines/hbn population mean
        # that excluded it.
        EXCLUDE_SUBJECTS=(NDAREF662XZT NDARVN370YL8 NDARYC287UFV)
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected unset, nki, ppmi, or hbn)" >&2
        exit 1
        ;;
esac

# Same for every array task -- see header comment (WHY per-subject +
# average) for why this MUST be fixed, not derived from $SUBJECT.
RANDOMIZE_SEED="${RANDOMIZE_SEED:-0}"

IG_STEPS="${IG_STEPS:-10}"

mapfile -t subjects < <(cut -f1 "$SUBJECTS_FILE")
if [ "${#EXCLUDE_SUBJECTS[@]}" -gt 0 ]; then
    FILTERED_SUBJECTS=()
    for SUBJ in "${subjects[@]}"; do
        EXCLUDED=false
        for EX in "${EXCLUDE_SUBJECTS[@]}"; do
            if [ "$SUBJ" = "$EX" ]; then
                EXCLUDED=true
                break
            fi
        done
        [ "$EXCLUDED" = false ] && FILTERED_SUBJECTS+=("$SUBJ")
    done
    if [ "${#FILTERED_SUBJECTS[@]}" -ne "${#subjects[@]}" ]; then
        echo "Excluded $(( ${#subjects[@]} - ${#FILTERED_SUBJECTS[@]} )) subject(s) from DATASET=${DATASET:-<unset>}'s EXCLUDE_SUBJECTS (${EXCLUDE_SUBJECTS[*]})"
    fi
    subjects=("${FILTERED_SUBJECTS[@]}")
fi
N_SUBJECTS=${#subjects[@]}

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-\$((\${#TASK_TARGETS[@]}*N_SUBJECTS-1)) (see header comment for DATASET=$DATASET's own bounds), sbatch slurm/compute_sanity_checks_randomize_model.sh}"
TARGET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SUBJECTS ))
SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID % N_SUBJECTS ))
if [ "$TARGET_IDX" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> target index $TARGET_IDX, out of range for TASK_TARGETS (${TASK_TARGETS[*]}) with N_SUBJECTS=$N_SUBJECTS." >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$TARGET_IDX]}"
SUBJECT="${subjects[$SUBJECT_IDX]}"

case "$TARGET" in
    age) CKPT_SUFFIX=brain_age_head_affine ;;
    sex) CKPT_SUFFIX=sex_head_affine ;;
    bmi) CKPT_SUFFIX=bmi_head_affine ;;
    *) echo "ERROR: unknown TARGET=$TARGET (expected age/sex/bmi)" >&2; exit 1 ;;
esac
echo "DATASET=${DATASET:-<unset>} TARGET=$TARGET (checkpoint suffix: ${CKPT_SUFFIX}), SUBJECT=$SUBJECT, RANDOMIZE_SEED=$RANDOMIZE_SEED"

# Uniform FINAL per-subject location every model's arch_ file ends up in,
# regardless of which of the two path-insertion conventions below produced
# it (own subtree, arch_linear_persubject, see HEAD-ONLY VARIANT header
# comment). DATASET set inserts an extra <dataset> path component (see
# DATASET env var header comment) -- SALIENCY_ROOT alone (not SALIENCY_ROOT/<target>)
# is what calculate_gradcam.py's own insertion logic below needs to see,
# since it inserts <target> right after whichever component starts with
# "saliency" regardless of what follows it.
if [ -n "$DATASET" ]; then
    OUT_DIR="${REPO_DIR}/${SALIENCY_ROOT}/${TARGET}/${DATASET}/arch_linear_persubject/${SUBJECT}"
    IMAGE_OUTPUT_DIR="${SALIENCY_ROOT}/${DATASET}/arch_linear_persubject/${SUBJECT}"
    TABULAR_BASE_DIR="${SALIENCY_ROOT}/${DATASET}/arch_linear_persubject/${SUBJECT}"
else
    OUT_DIR="${REPO_DIR}/${SALIENCY_ROOT}/${TARGET}/arch_linear_persubject/${SUBJECT}"
    # calculate_gradcam.py's own --output-dir handling INSERTS $TARGET as a new
    # path component right after whichever part of the path starts with
    # "saliency" (see its own comment at the insertion site) -- so passing
    # "$SALIENCY_ROOT/arch_linear_persubject/$SUBJECT" (deliberately WITHOUT
    # $TARGET; baking it in here would duplicate it, landing at
    # .../$TARGET/$TARGET/...) makes IT produce exactly $OUT_DIR above.
    IMAGE_OUTPUT_DIR="${SALIENCY_ROOT}/arch_linear_persubject/${SUBJECT}"

    # calculate_gradcam_tabular.py (fs_roi) instead just APPENDS $TARGET at the
    # very end of whatever --output-dir it's given (no "insert after saliency
    # root" logic) -- so the SAME base path here lands somewhere DIFFERENT:
    # $SALIENCY_ROOT/arch_linear_persubject/$SUBJECT/$TARGET/ (target nested
    # inside subject, not the other way around), with its arch_ file written
    # directly there and its (unwanted) baseline one level deeper under
    # .../$SUBJECT/. run_model_tabular below moves the arch_ file into $OUT_DIR
    # afterward so fs_roi's temp files still end up in the same uniform layout
    # every other model's do. fs_roi/run_model_tabular only ever runs in this
    # (DATASET unset) branch -- see DATASET env var header comment.
    TABULAR_BASE_DIR="${SALIENCY_ROOT}/arch_linear_persubject/${SUBJECT}"
fi
mkdir -p "$OUT_DIR"

TURBOPREP_NII="${TURBOPREP_ROOT}/${SUBJECT}/normalized.nii.gz"

# Single ig_reference spec, matching compute_sanity_checks.sh's own default
# ATTRIBUTION_SPECS entry. DATASET set passes --reference-baselines-dir
# explicitly -- calculate_gradcam.py's own default is reference_baselines/nki
# regardless of --model, which would otherwise be silently wrong for
# ppmi/hbn (same fix as compute_saliency_pretrained_heads.sh's own
# ATTRIBUTION_SPECS; reference_baselines/nki is already correct as the
# default when DATASET is unset).
FLAGS="--attribution-method integrated_gradients --ig-baseline reference --ig-steps ${IG_STEPS}"
if [ -n "$DATASET" ]; then
    FLAGS="${FLAGS} --reference-baselines-dir reference_baselines/${DATASET}"
fi

# cleanup_baseline PREFIX -- deletes the always-written trained-model
# baseline file(s) this task doesn't need (see header comment). PREFIX is
# the exact non-arch_ filename stem calculate_and_save_saliency uses
# (matches compute_saliency.sh's own prefix construction), globbed since
# optim_uncnn writes 3 channel files, not 1.
cleanup_baseline() {
    local PREFIX="$1"
    rm -f "${OUT_DIR}/${PREFIX}"*_saliency.nii.gz
}

# run_model LABEL MODEL CHECKPOINT SUBJECT_NII [extra calculate_gradcam.py args...]
run_model() {
    local LABEL="$1" MODEL="$2" CKPT="$3" NII="$4"
    shift 4
    local EXTRA_ARGS=("$@")
    echo "=== ${LABEL} ==="
    # shellcheck disable=SC2086  # FLAGS is intentionally word-split, see above.
    time python3 src/calculate_gradcam.py --model "$MODEL" \
        --checkpoint-path "$CKPT" --target "$TARGET" \
        --subject-nii "$NII" --output-dir "$IMAGE_OUTPUT_DIR" \
        --randomize-model --randomize-model-mode cascade --randomize-max-layers 1 \
        --randomize-seed "$RANDOMIZE_SEED" \
        "${EXTRA_ARGS[@]}" \
        $FLAGS
    cleanup_baseline "${MODEL}_${ANATCL_SUFFIX:-}"
}

# run_model_tabular LABEL CHECKPOINT [FEATURES_DIR_OVERRIDE]
# FEATURES_DIR_OVERRIDE (optional 3rd arg): omitted -- calculate_gradcam_
# tabular.py's own defaults apply (this repo's NKI cache, DATASET unset).
# Pass $FS_ROI_FEATURES_DIR explicitly for DATASET=nki/hbn -- HBN's fs_roi
# feature cache lives in a separate directory (artifacts/hbn_fs_roi/, see
# compute_saliency_pretrained_heads.sh's own SCOPE comment for why).
run_model_tabular() {
    local LABEL="$1" CKPT="$2"
    local FEATURES_DIR_ARGS=()
    if [ -n "${3:-}" ]; then
        FEATURES_DIR_ARGS=(--features-dir "$3" --cache-name fs_roi_dktatlas_grayvol)
    fi
    echo "=== ${LABEL} ==="
    # shellcheck disable=SC2086  # FLAGS is intentionally word-split, see above.
    time python3 src/calculate_gradcam_tabular.py \
        --checkpoint-path "$CKPT" --target "$TARGET" \
        --subject-id "$SUBJECT" --output-dir "$TABULAR_BASE_DIR" \
        "${FEATURES_DIR_ARGS[@]}" \
        --randomize-model --randomize-model-mode cascade --randomize-max-layers 1 \
        --randomize-seed "$RANDOMIZE_SEED" \
        $FLAGS
    # See TABULAR_BASE_DIR comment above -- its actual output landed at
    # .../$TARGET/ (arch_ file directly there, baseline one level deeper
    # under .../$TARGET/$SUBJECT/). Move only the arch_ file into $OUT_DIR,
    # then drop the whole intermediate tree (arch_ file already moved out,
    # baseline discarded, same as cleanup_baseline does for every other
    # model). Cascade mode (--randomize-max-layers 1) names its single step
    # "arch_fs_roi_step01_<module>_..." not "arch_fs_roi_all_...", so glob on
    # the step-prefixed shape instead.
    local TABULAR_TARGET_DIR="${TABULAR_BASE_DIR}/${TARGET}"
    mv "${TABULAR_TARGET_DIR}/arch_fs_roi_step"* "$OUT_DIR"/ 2>/dev/null || true
    # rm the whole TABULAR_BASE_DIR (not just TABULAR_TARGET_DIR): nothing
    # else ever writes directly into TABULAR_BASE_DIR itself (only its
    # $TARGET subfolder, just removed above) -- calculate_gradcam_tabular.py
    # implicitly creates TABULAR_BASE_DIR as a side effect of mkdir(parents=True)
    # on .../$TARGET/, so leaving it behind after rm -rf-ing only the $TARGET
    # child stranded an empty .../$SUBJECT/ directory here every single run.
    rm -rf "$TABULAR_BASE_DIR"
}

if [ -z "$DATASET" ]; then
    # DATASET unset: this script's original behavior, unchanged -- this
    # repo's own checkpoints/heads/, NKI only.
    run_model "unCNN" optim_uncnn "${HEADS_DIR}/optim_uncnn_turboprep_covpool_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"
    run_model "BrainIAC" brainiac "${HEADS_DIR}/brainiac_turboprep_96_cls_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"
    run_model "SIMCLR" simclr "${HEADS_DIR}/simclr_turboprep_transpose_crop_150x192x192_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"
    run_model "SWINBRAIN" swinbrain "${HEADS_DIR}/swinbrain_turboprep_128x128x64_encoder10_avgpool_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"

    # anatcl needs the subject's CAT12 mwp1 file; variable ses-/acq- suffix per
    # subject, same glob-and-fail-loudly convention as compute_saliency.sh/
    # compute_sanity_checks.sh's identical blocks.
    mapfile -t ANATCL_NII_MATCHES < <(compgen -G "${NKI_BIDS_ROOT}/sub-${SUBJECT}/ses-*/anat/mri/mwp1sub-${SUBJECT}*.nii")
    if [ "${#ANATCL_NII_MATCHES[@]}" -ne 1 ]; then
      echo "ERROR: expected exactly 1 anatcl CAT12 mwp1 file for subject ${SUBJECT}, found ${#ANATCL_NII_MATCHES[@]}: ${ANATCL_NII_MATCHES[*]}" >&2
      exit 1
    fi
    ANATCL_NII="${ANATCL_NII_MATCHES[0]}"

    export ANATCL_SUFFIX=global_
    run_model "ANATCL GLOBAL" anatcl "${HEADS_DIR}/anatcl_global_cat12vbm_fold0_${CKPT_SUFFIX}.pt" "$ANATCL_NII" \
        --anatcl-level global --anatcl-folds 0

    export ANATCL_SUFFIX=local_
    run_model "ANATCL LOCAL" anatcl "${HEADS_DIR}/anatcl_local_cat12vbm_fold0_${CKPT_SUFFIX}.pt" "$ANATCL_NII" \
        --anatcl-level local --anatcl-folds 0

    run_model_tabular "FS ROI" "${HEADS_DIR}/fs_roi_dktatlas_grayvol_${CKPT_SUFFIX}.pt"
else
    # DATASET set: the externally pre-trained heads, discovered via
    # src/list_pretrained_heads.py -- same checkpoint-discovery/ANATCL_SUFFIX
    # logic as slurm/compute_saliency_pretrained_heads.sh (see that script's
    # own comments for the overwrite hazard this guards against). fs_roi is
    # NOT part of this discovery loop (list_pretrained_heads.py excludes the
    # FreeSurfer-surface models, and no checkpoint matches fs_roi's
    # DKT-volume feature set among the external heads either) -- it's called
    # separately below via run_model_tabular, same external
    # {DATASET}_FS_ROI_{target}.pt checkpoint
    # slurm/compute_sanity_checks.sh/compute_saliency_pretrained_heads.sh
    # already use, for DATASET=nki/hbn only (see FS_ROI_SUPPORTED above).
    ANATCL_NII=""
    resolve_anatcl_nii() {
        if [ -n "$ANATCL_NII" ]; then
            return
        fi
        local ROOT MATCHES=()
        for ROOT in "${CAT12_ROOTS[@]}"; do
            mapfile -t -O "${#MATCHES[@]}" MATCHES < <(compgen -G "${ROOT}/sub-${SUBJECT}/ses-*/anat/mri/mwp1*.nii")
        done
        if [ "${#MATCHES[@]}" -ne 1 ]; then
            echo "ERROR: expected exactly 1 anatcl CAT12 mwp1 file for DATASET=$DATASET SUBJECT=${SUBJECT} across ${CAT12_ROOTS[*]}, found ${#MATCHES[@]}: ${MATCHES[*]}" >&2
            exit 1
        fi
        ANATCL_NII="${MATCHES[0]}"
    }

    mapfile -t HEAD_ROWS < <(python3 src/list_pretrained_heads.py \
        --heads-dir "$HEADS_DIR" --dataset "$DATASET" --target "$TARGET")

    if [ "${#HEAD_ROWS[@]}" -eq 0 ]; then
        echo "WARN: no usable pretrained heads found for DATASET=$DATASET TARGET=$TARGET -- nothing to do."
        exit 0
    fi

    for ROW in "${HEAD_ROWS[@]}"; do
        IFS=$'\t' read -r LABEL MODEL_ARGS CKPT_PATH <<< "$ROW"
        read -ra MODEL_PARTS <<< "$MODEL_ARGS"
        MODEL="${MODEL_PARTS[0]}"
        EXTRA_ARGS=("${MODEL_PARTS[@]:1}")

        if [ "$MODEL" = "anatcl" ]; then
            resolve_anatcl_nii
            # Same ANATCL_SUFFIX overwrite hazard as
            # compute_saliency_pretrained_heads.sh -- AnatCL_Global/
            # AnatCL_Local both map to --model anatcl and would otherwise
            # collide on the same "arch_anatcl_step01_..." filename.
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
            run_model "$LABEL" "$MODEL" "$CKPT_PATH" "$ANATCL_NII" "${EXTRA_ARGS[@]}"
        else
            export ANATCL_SUFFIX=""
            run_model "$LABEL" "$MODEL" "$CKPT_PATH" "$TURBOPREP_NII" "${EXTRA_ARGS[@]}"
        fi
    done

    if [ "$FS_ROI_SUPPORTED" = true ]; then
        run_model_tabular "FS ROI" "${HEADS_DIR}/${DATASET^^}_FS_ROI_${TARGET}.pt" "$FS_ROI_FEATURES_DIR"
    else
        echo "WARN: fs_roi skipped for DATASET=$DATASET -- no compatible feature schema (see SCOPE/FS_ROI_SUPPORTED above)."
    fi
fi

echo "=== Done: head-only randomize-model sanity check for DATASET=${DATASET:-<unset>} $TARGET/$SUBJECT (linear layer randomized, backbone kept) -- outputs under arch_linear_persubject/; no aggregate script consumes this subtree yet ==="
