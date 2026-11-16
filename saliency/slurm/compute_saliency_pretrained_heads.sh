#!/usr/bin/env bash
#SBATCH --job-name=compute_saliency_pretrained_nki
#SBATCH --time=4:30:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --array=0-2873
#SBATCH --output=slurm/logs/%x_%a.out
#
# slurm/compute_saliency.sh's sibling for the INDEPENDENTLY pre-trained
# linear heads ($HEADS_DIR/,
# one heads_affine_linear_sklearn_summary_{NKI,PPMI,HBN}.csv manifest per
# dataset, inside that same directory) -- 58 checkpoints spanning 3 datasets
# (NKI/PPMI/HBN) x up to 9 model families x up to 3 tasks (age/sex/bmi --
# PPMI has none for bmi), one `{dataset}_{model}_{task}.pt` per combination
# that exists. A separate script, not a --dataset flag bolted onto
# compute_saliency.sh itself, because almost everything about checkpoint
# DISCOVERY differs (a CSV-driven manifest + per-checkpoint train_linear_model
# field instead of a fixed per-target filename suffix) even though the actual
# saliency computation (src/calculate_gradcam.py) is unchanged and reused as-is
# -- see src/list_pretrained_heads.py's own module docstring for the full
# reasoning on checkpoint discovery.
#
# NOTE on *_sklearn specifically (vs. this repo's own compute_saliency.sh /
# src/train_linear.py, and vs. the external older, non-"_sklearn"
# heads_affine_linear/ checkpoint set): Untrained_CNN's features here are fed
# RAW into StandardScaler+Ridge/LogisticRegression -- the covariance-pooled
# channels do NOT get the signed-square-root normalization
# (BrainAgePipeline._signed_sqrt / train_ridge_head's cov_mask branch in
# src/train_linear.py) that this repo's own optim_uncnn head and the external
# older heads_affine_linear/ checkpoints both apply. Every checkpoint under
# heads_affine_linear_sklearn/ has cov_mask all-False (verified via
# torch.load), so BrainAgePipeline.forward skips that nonlinearity entirely
# for these heads. See docs/known_limitations.md's "Untrained_CNN features
# are raw in the externally pre-trained *_sklearn heads" section for what this changes about
# the resulting saliency maps (different fitted feature space + no
# signed-sqrt derivative-singularity artifact near zero).
#
# SCOPE: the 6 image-based model families (Untrained_CNN/AnatCL_Global/
# AnatCL_Local/BrainIAC/SwinBrain/3D-Neuro-SimCLR) -- discovered via
# list_pretrained_heads.py below -- PLUS fs_roi (hardcoded below, not part
# of that discovery: calculate_gradcam_tabular.py has a different CLI than
# the image models' calculate_gradcam.py, so list_pretrained_heads.py's
# generic "<model> <extra args>" row format doesn't fit it, and the external
# {NKI,HBN}_FS_ROI_{target}.pt checkpoints aren't listed in the summary CSV
# yet anyway -- see fs_roi SCOPE below). The remaining 16 of the external
# checkpoints (FreeSurfer, FS_Schaefer, FS_aparc) use cortical surface
# features (thickness/surface-area) this repo has no feature-extraction or
# atlas-painting logic for yet (different from fs_roi's DKT-volume pathway)
# -- deferred, see list_pretrained_heads.py's FREESURFER_SURFACE_MODELS.
#
# fs_roi SCOPE: DATASET=nki/hbn only, NOT ppmi. Uses the external own
# {HEADS_DIR}/{DATASET}_FS_ROI_{target}.pt (110 DKT-atlas GrayVol features:
# 62 cortical + 48 subcortical from brainmeasures.tsv) -- verified directly
# (torch.load): train_linear_model="fs_roi", feature_dim=110, cov_mask
# all-False, and its own feature_names field matches this repo's
# artifacts/{nki,hbn}_freesurfer_roi.csv column order exactly for both
# datasets. No external checkpoint exists for PPMI's fs_roi feature set
# (PPMI's own roi table, artifacts/ppmi_freesurfer_roi.csv, is a completely
# different, incompatible schema: cth_*/sa_* cortical thickness/
# surface-area, no GrayVol, no subcortical data at all) -- fs_roi is
# skipped (WARN, not a hard failure) for DATASET=ppmi for this reason.
# HBN's own 110-column roi table (hence the feature-order match above) only
# exists because of a fix to src/build_freesurfer_roi_tables.py's
# load_hbn_freesurfer_df earlier this session (it used to be cortical-only,
# 62 columns, missing brainmeasures.tsv's subcortical volumes entirely).
#
# fs_roi ALSO needs its own per-dataset feature CACHE (--features-dir,
# separate from --checkpoint-path) for calculate_gradcam_tabular.py's
# per-subject row lookup (load_cached_row reads {features_dir}/
# fs_roi_dktatlas_grayvol_{X_features_raw,subjects,feature_names}.*, NOT
# --roi-csv directly, and NOT the checkpoint's own feature_names/subjects
# fields either, despite the external checkpoint carrying those) --
# src/train_linear.py's fs_roi branch hardcodes its cache filename
# regardless of --roi-csv, so NKI's own cache already lives at the default
# (artifacts/); HBN's was built once into a SEPARATE directory
# (artifacts/hbn_fs_roi/, via `python3 src/train_linear.py --model fs_roi
# --roi-csv artifacts/hbn_freesurfer_roi.csv --demographics
# "$HBN_DEMOGRAPHICS" --target age --output-dir artifacts/hbn_fs_roi
# --turboprep-root "$HBN_TURBOPREP_ROOT"`, one-time prep, NOT part of this
# script's own per-subject array loop) specifically so it could never
# overwrite NKI's existing cache at the shared default location. The
# checkpoint that run actually trained+saved there (fit on HBN's own
# labels, via this repo's own RidgeCV, not the external sklearn pipeline) is
# NOT used below -- only its X_features_raw/subjects/feature_names side
# effect matters; the actual head weights come from the external checkpoint.
#
# DATASET env var (REQUIRED, not part of the array index -- submit once per
# dataset): DATASET=nki | DATASET=ppmi | DATASET=hbn. Each dataset has its own
# N_SUBJECTS and TASK_TARGETS (PPMI has no bmi heads), so --array bounds
# differ per dataset -- as of this writing (N_SUBJECTS already EXCLUDE_
# SUBJECTS-filtered below, see that case statement -- HBN's raw subject-list
# row count is 998, but N_SUBJECTS is 997 since NDARYC287UFV is excluded):
#   DATASET=nki:  N_SUBJECTS=958, TASK_TARGETS=(age sex bmi) -> --array=0-2873
#   DATASET=ppmi: N_SUBJECTS=894, TASK_TARGETS=(age sex)     -> --array=0-1787
#   DATASET=hbn:  N_SUBJECTS=997, TASK_TARGETS=(age sex bmi) -> --array=0-2990
# RECOMPUTE if a dataset's subject-list row count changes -- see
# compute_saliency.sh's own header comment for the general index-layout
# reasoning (target_idx * N_SUBJECTS + subject_idx, one contiguous block per
# target in TASK_TARGETS order), which applies identically here once
# N_SUBJECTS/TASK_TARGETS are resolved for the chosen DATASET below.
#
# Output: saliency_affine_pretrained/<target>/<dataset>/<subject>/ -- a
# SEPARATE top-level root from compute_saliency.sh's own saliency_affine/, so
# these externally-trained-head runs can never collide with or be picked up
# by this repo's own saliency_affine/<target>/... aggregate/compare scripts
# (calculate_gradcam.py inserts <target> right after whichever output-dir path
# component starts with "saliency" -- passing
# "saliency_affine_pretrained/<dataset>/<subject>/" here, deliberately without
# <target> baked in, is what makes it land at exactly that layout -- same
# insertion-ordering trick compute_saliency.sh's own output-dir already uses).
#
# Prerequisites: reference_baselines/<dataset>/*_reference.npy (from
# slurm/compute_reference_baselines.sh) for the ig_reference spec -- same
# graceful-degradation convention as compute_saliency.sh: if missing, only
# that spec fails per model (set -uo pipefail, no -e), everything else still
# runs. ATTRIBUTION_SPECS below passes --reference-baselines-dir
# reference_baselines/${DATASET} explicitly for this reason --
# calculate_gradcam.py's own default is reference_baselines/nki regardless of
# --model, so DATASET=ppmi/hbn runs would otherwise silently score against
# NKI's population-mean baseline instead of their own (a real bug fixed here;
# any ig_reference output already computed for DATASET=ppmi/hbn before this
# fix was computed against the wrong baseline).

set -uo pipefail
shopt -s nullglob

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_saliency_pretrained_heads.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

IG_STEPS=50

HEADS_DIR="${HEADS_DIR:?set HEADS_DIR to the directory of externally pre-trained linear heads}"

# ---------------------------------------------------------------------------
# Resolve DATASET -> turboprep root / subject list / CAT12 roots (for AnatCL's
# mwp1 lookup) / this dataset's own TASK_TARGETS. Every *_hdbet_affine root
# below already uses BARE subject IDs (no "sub-" prefix) matching each
# dataset's own subject-list file exactly -- verified directly against all
# three on disk. CAT12 BIDS roots, by contrast, DO use "sub-${SUBJECT}"
# directory names (see ANATCL_NII lookup below).
# ---------------------------------------------------------------------------
DATASET="${DATASET:?DATASET unset -- submit with DATASET=nki|ppmi|hbn sbatch slurm/compute_saliency_pretrained_heads.sh (see header comment for the array bounds of each dataset)}"
# FS_ROI_FEATURES_DIR: this dataset's own fs_roi feature cache (see SCOPE
# comment above -- NOT --roi-csv, which load_cached_row never reads).
# FS_ROI_SUPPORTED=false for ppmi -- no compatible feature schema exists.
case "$DATASET" in
    nki)
        TURBOPREP_ROOT="$NKI_AFFINE_TURBOPREP_ROOT"
        SUBJECTS_FILE="$NKI_DEMOGRAPHICS"
        CAT12_ROOTS=("$NKI_BIDS_ROOT")
        TASK_TARGETS=(age sex bmi)
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="${REPO_DIR}/artifacts"
        EXCLUDE_SUBJECTS=()
        ;;
    ppmi)
        TURBOPREP_ROOT="$PPMI_TURBOPREP_ROOT"
        SUBJECTS_FILE="$PPMI_SUBJECTS_LIST"
        CAT12_ROOTS=("$PPMI_CAT12_BIDS_ROOT")
        TASK_TARGETS=(age sex)  # no bmi heads for PPMI
        FS_ROI_SUPPORTED=false
        # Same exclusion slurm/compute_reference_baselines.sh's own PPMI
        # block uses -- currently a no-op (3510 isn't in
        # $PPMI_SUBJECTS_LIST, verified directly), kept here for parity/
        # documentation in case that ever changes.
        EXCLUDE_SUBJECTS=(3510)
        ;;
    hbn)
        TURBOPREP_ROOT="$HBN_TURBOPREP_ROOT"
        SUBJECTS_FILE="$HBN_SUBJECTS_LIST"
        CAT12_ROOTS=("$HBN_CAT12_BIDS_ROOT_1" "$HBN_CAT12_BIDS_ROOT_2")  # two roots, search both
        TASK_TARGETS=(age sex bmi)
        FS_ROI_SUPPORTED=true
        FS_ROI_FEATURES_DIR="${REPO_DIR}/artifacts/hbn_fs_roi"
        # Same 3 duplicated-T1w subjects slurm/compute_reference_baselines.sh's
        # own HBN block excludes (see neurostars.org/t/36444 comment there).
        # NDAREF662XZT/NDARVN370YL8 are already absent from
        # $HBN_SUBJECTS_LIST (verified directly); NDARYC287UFV is NOT --
        # without this, it would get a full per-subject run here scored
        # against a reference_baselines/hbn population mean that excluded
        # it, and would land in any later aggregate statistic alongside the
        # subjects that mean was actually computed from.
        EXCLUDE_SUBJECTS=(NDAREF662XZT NDARVN370YL8 NDARYC287UFV)
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected nki|ppmi|hbn)" >&2
        exit 1
        ;;
esac

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
        echo "Excluded $(( ${#subjects[@]} - ${#FILTERED_SUBJECTS[@]} )) subject(s) from DATASET=$DATASET's EXCLUDE_SUBJECTS (${EXCLUDE_SUBJECTS[*]})"
    fi
    subjects=("${FILTERED_SUBJECTS[@]}")
fi
N_SUBJECTS=${#subjects[@]}

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-\$((${#TASK_TARGETS[@]}*N_SUBJECTS-1)) for DATASET=$DATASET (see header comment), sbatch slurm/compute_saliency_pretrained_heads.sh}"
TARGET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SUBJECTS ))
SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID % N_SUBJECTS ))
if [ "$TARGET_IDX" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> target index $TARGET_IDX, out of range for DATASET=$DATASET's TASK_TARGETS (${TASK_TARGETS[*]}) with N_SUBJECTS=$N_SUBJECTS." >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$TARGET_IDX]}"
SUBJECT="${subjects[$SUBJECT_IDX]}"
echo "DATASET=$DATASET TARGET=$TARGET SUBJECT=$SUBJECT"

OUT_DIR="saliency_affine_pretrained/${TARGET}/${DATASET}/${SUBJECT}"
mkdir -p "$OUT_DIR"

TURBOPREP_NII="${TURBOPREP_ROOT}/${SUBJECT}/normalized.nii.gz"

# AnatCL's mwp1 CAT12 file, resolved lazily (only looked up if a discovered
# checkpoint row below actually needs it) and cached in ANATCL_NII once
# found. Scoped to sub-${SUBJECT}'s own directory tree and matched by
# LOCATION, not filename content -- NKI's/HBN's mwp1 filenames embed
# "sub-${SUBJECT}" (mwp1sub-${SUBJECT}..._T1w.nii) but PPMI's do NOT
# (mwp1PPMI_<id>_MPRAGE_GRAPPA.nii, no "sub-" anywhere in the filename) --
# verified directly on disk for all three CAT12 roots. Searching within
# "$ROOT/sub-${SUBJECT}" works uniformly across all three regardless of that
# filename difference. Fails loudly on zero or multiple matches, same as
# compute_saliency.sh's own anatcl lookup.
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

# "tag:extra flags" pairs -- flags are intentionally word-split on use (no
# individual flag value here contains a space). Same convention as
# compute_saliency.sh.
ATTRIBUTION_SPECS=(
    "ig_reference:--attribution-method integrated_gradients --ig-baseline reference --ig-steps ${IG_STEPS} --reference-baselines-dir reference_baselines/${DATASET}"
)

# run_model LABEL MODEL CHECKPOINT NII [extra calculate_gradcam.py args...]
run_model() {
    local LABEL="$1" MODEL="$2" CKPT="$3" NII="$4"
    shift 4
    local EXTRA_ARGS=("$@")
    local SPEC TAG FLAGS
    for SPEC in "${ATTRIBUTION_SPECS[@]}"; do
        TAG="${SPEC%%:*}"
        FLAGS="${SPEC#*:}"
        echo "${LABEL} (${TAG})"
        # shellcheck disable=SC2086  # FLAGS is intentionally word-split, see above.
        time python3 src/calculate_gradcam.py --model "$MODEL" \
            --checkpoint-path "$CKPT" --target "$TARGET" \
            --subject-nii "$NII" --output-dir "saliency_affine_pretrained/${DATASET}/${SUBJECT}/" \
            "${EXTRA_ARGS[@]}" \
            $FLAGS
    done
}

# run_model_tabular LABEL CHECKPOINT
# fs_roi's analogue of run_model -- calculate_gradcam_tabular.py takes
# --subject-id directly (a row lookup into FS_ROI_FEATURES_DIR's cached
# feature matrix, see SCOPE comment above), not --subject-nii/a file-glob,
# so there's no NII argument here at all. Only called when
# FS_ROI_SUPPORTED=true (see DATASET case statement above).
#
# Output-dir handling differs from run_model/calculate_gradcam.py:
# calculate_gradcam_tabular.py unconditionally APPENDS /{target}/{subject_id}
# to whatever --output-dir it's given (args.output_dir = args.output_dir /
# target_spec.name, then / args.subject_id) -- there is no "insert <target>
# after saliency" convention to rely on here, so --output-dir can't be
# pointed directly at $OUT_DIR without duplicating $TARGET/$SUBJECT. Write
# to a throwaway temp dir instead and move its real output into $OUT_DIR
# afterward -- same pattern slurm/compute_sanity_checks_randomize_model.sh's
# own TABULAR_BASE_DIR handling already uses for this exact reason.
run_model_tabular() {
    local LABEL="$1" CKPT="$2"
    local SPEC TAG FLAGS
    local TMP_DIR
    TMP_DIR="$(mktemp -d)"
    for SPEC in "${ATTRIBUTION_SPECS[@]}"; do
        TAG="${SPEC%%:*}"
        FLAGS="${SPEC#*:}"
        echo "${LABEL} (${TAG})"
        # shellcheck disable=SC2086  # FLAGS is intentionally word-split, see above.
        time python3 src/calculate_gradcam_tabular.py \
            --checkpoint-path "$CKPT" --target "$TARGET" \
            --subject-id "$SUBJECT" --output-dir "$TMP_DIR" \
            --features-dir "$FS_ROI_FEATURES_DIR" \
            --seg-template "${REPO_DIR}/templates/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz" \
            $FLAGS
    done
    mv "${TMP_DIR}/${TARGET}/${SUBJECT}/"* "$OUT_DIR"/ 2>/dev/null || true
    rm -rf "$TMP_DIR"
}

# Discover this (DATASET, TARGET)'s usable checkpoints -- see
# src/list_pretrained_heads.py's own module docstring. Each row:
# "<label>\t<model-and-extra-args>\t<checkpoint-path>".
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
        # calculate_gradcam.py bakes ANATCL_SUFFIX into EVERY model's output filename prefix
        # unconditionally (prefix = f"{model}_{anatcl_suffix}", read fresh from the environment on
        # every invocation -- not just for anatcl) -- without this, AnatCL_Global and AnatCL_Local
        # (both --model anatcl, just different --anatcl-level) collide on the identical filename
        # "anatcl_ig_reference_*", and whichever row runs second silently overwrites the first's
        # output. Derived from this row's own "--anatcl-level global|local" (present in every
        # anatcl train_linear_model string -- see list_pretrained_heads.py), not hardcoded, so it
        # stays correct if a row's level ever doesn't match what the loop order would suggest.
        ANATCL_LEVEL=""
        for i in "${!EXTRA_ARGS[@]}"; do
            if [ "${EXTRA_ARGS[$i]}" = "--anatcl-level" ]; then
                ANATCL_LEVEL="${EXTRA_ARGS[$((i + 1))]}"
            fi
        done
        if [ -z "$ANATCL_LEVEL" ]; then
            echo "ERROR: anatcl row ($LABEL, model_args=$MODEL_ARGS) has no --anatcl-level -- refusing to run it, it would silently overwrite another anatcl level's output (see ANATCL_SUFFIX comment above)." >&2
            exit 1
        fi
        export ANATCL_SUFFIX="${ANATCL_LEVEL}_"
        run_model "$LABEL" "$MODEL" "$CKPT_PATH" "$ANATCL_NII" "${EXTRA_ARGS[@]}"
    else
        # Explicitly cleared (not just "never set") -- ANATCL_SUFFIX is exported above and would
        # otherwise leak into every subsequent non-anatcl row's own filename prefix too, since
        # calculate_gradcam.py reads it unconditionally regardless of --model.
        export ANATCL_SUFFIX=""
        run_model "$LABEL" "$MODEL" "$CKPT_PATH" "$TURBOPREP_NII" "${EXTRA_ARGS[@]}"
    fi
done

# fs_roi -- not part of the external-checkpoint discovery loop above (see
# SCOPE header comment: calculate_gradcam_tabular.py has a different CLI
# than the image models, so list_pretrained_heads.py's generic row format
# doesn't fit it). Uses the external {DATASET}_FS_ROI_{target}.pt
# checkpoint directly -- NOT list_pretrained_heads.py/the summary CSV,
# which doesn't have an FS_ROI row yet for either dataset (the .pt files on
# disk are newer than the CSV -- verified directly, Oct 26 vs Oct 22 -- so
# this is a manifest lag, not a missing checkpoint).
if [ "$FS_ROI_SUPPORTED" = true ]; then
    run_model_tabular "FS ROI" "${HEADS_DIR}/${DATASET^^}_FS_ROI_${TARGET}.pt"
else
    echo "WARN: fs_roi skipped for DATASET=$DATASET -- no compatible feature schema (see SCOPE header comment)."
fi

echo "=== Done: compute_saliency_pretrained_heads for $DATASET/$TARGET/$SUBJECT ==="
