#!/usr/bin/env bash
#SBATCH --job-name=compute_saliency_affine
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --array=0-2873
#SBATCH --output=slurm/logs/%x_%a.out
#
# Per-(target, subject) array job: like slurm/train_all_models.sh, the array
# index now encodes TWO dimensions flattened into one, so age/sex/bmi run as
# genuinely parallel SLURM array tasks instead of three sequential
# `TARGET=... sbatch` submissions. For the subject/target pair at index
# $SLURM_ARRAY_TASK_ID, computes saliency + the Adebayo et al. sanity check
# (--randomize-input) for all 6 image-based models, across all 4 attribution
# methods (gradient, gradient_x_input, integrated_gradients with a zero
# baseline, integrated_gradients with the population-mean reference
# baseline), writing into saliency_outputs/<target>/<SUBJECT_ID>/ (both
# calculate_gradcam.py and calculate_gradcam_tabular.py insert the <target>
# path component themselves, so age/sex/bmi runs never share an output
# directory and overwrite each other -- see those scripts' own docstrings).
# That's 6 models x
# 4 attribution specs = 24 calculate_gradcam.py invocations per (target,
# subject), plus fs_roi (calculate_gradcam_tabular.py, baseline saliency
# only -- no --randomize-input sanity check ported yet, see that script's
# own docstring) x the same 4 attribution specs = 4 more.
#
# Index layout: TASK_TARGETS=(age sex bmi) below, N_SUBJECTS = number of
# rows in $NKI_DEMOGRAPHICS, computed FRESH at runtime from that file (`cut
# -f1 "$NKI_DEMOGRAPHICS" | wc -l` below) -- NOT a fixed constant, so these
# bounds shift whenever the cohort's row count changes. Index = target_idx *
# N_SUBJECTS + subject_idx, i.e. one contiguous block of N_SUBJECTS indices
# per target, in TASK_TARGETS order -- NOT interleaved -- so a single
# target's block is always a plain contiguous --array range. As of this
# writing $NKI_DEMOGRAPHICS has N_SUBJECTS=958 rows, giving:
#   --array=0-2873  all three targets, all subjects, in parallel (default)
#   --array=0-957   age only
#   --array=958-1915   sex only
#   --array=1916-2873  bmi only
#   --array=0-1915  age + sex only (skip bmi)
# RECOMPUTE THESE BOUNDS if $NKI_DEMOGRAPHICS's row count ever changes --
# check with `cut -f1 "$NKI_DEMOGRAPHICS" | wc -l` first, then use
# target_idx * N_SUBJECTS + [0, N_SUBJECTS-1] for target_idx in 0 (age), 1
# (sex), 2 (bmi). Submitting a stale range from an old N_SUBJECTS (e.g. this
# script's own earlier N_SUBJECTS=15 example) doesn't error -- it silently
# lands entirely inside the wrong target's block (or, worse, entirely inside
# "age" for every target you meant to hit), so if a submitted job only ever
# shows TARGET=age in its log, check whether N_SUBJECTS grew since your
# --array range was chosen. The original version of this script fixed an
# off-by-one here (--array=1-15 indexing a 0-based bash `subjects` array
# skipped subjects[0] entirely) -- that fix is preserved (subject_idx below
# is still 0-based).
#
# COST WARNING: integrated_gradients costs ~IG_STEPS times more than
# gradient/gradient_x_input -- one forward+backward pass per interpolation
# step, x2 since --randomize-input computes a second attribution map the
# same way. IG_STEPS below is deliberately far below calculate_gradcam.py's
# own --ig-steps default (50) to keep this array job tractable on CPU (no
# GPU is requested here). Raise it (and --time above) if your queue/budget
# allows, or lower it further if this still doesn't fit -- exact runtime is
# very hardware/subject-dependent, so treat --time=6:0:00 as a starting
# guess, not a validated number.
#
# Requires checkpoints/heads/*_{brain_age,sex,bmi}_head_affine.pt (from
# slurm/train_all_models.sh, train_linear.py appends the _affine suffix
# itself) to already exist, and -- for the ig_reference
# spec specifically -- reference_baselines/nki/*_reference.npy (from
# slurm/compute_reference_baselines.sh). If the latter is missing, only that
# one spec fails per model (calculate_gradcam.py raises FileNotFoundError);
# every other spec/model still runs -- this script doesn't abort on a single
# failed invocation (set -uo pipefail, no -e).

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_saliency.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

IG_STEPS=50

mapfile -t subjects < <(cut -f1 "$NKI_DEMOGRAPHICS")
N_SUBJECTS=${#subjects[@]}

# Array-index -> (target, subject) mapping (see header comment above).
TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=..., e.g. sbatch --array=0-$((${#TASK_TARGETS[@]} * N_SUBJECTS - 1)) slurm/compute_saliency.sh}"
TARGET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SUBJECTS ))
SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID % N_SUBJECTS ))
if [ "$TARGET_IDX" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> target index $TARGET_IDX, out of range for TASK_TARGETS (${TASK_TARGETS[*]}) with N_SUBJECTS=$N_SUBJECTS. Use --array=0-$((${#TASK_TARGETS[@]} * N_SUBJECTS - 1)) or a single target's contiguous block (see header comment)." >&2
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
echo "TARGET=$TARGET (checkpoint suffix: ${CKPT_SUFFIX}), SUBJECT=$SUBJECT"

mkdir -p "saliency_affine/${TARGET}/${SUBJECT}"

HEADS_DIR="${REPO_DIR}/checkpoints/heads"
TURBOPREP_NII="${NKI_AFFINE_TURBOPREP_ROOT}/${SUBJECT}/normalized.nii.gz"

# "tag:extra flags" pairs -- flags are intentionally word-split on use (no
# individual flag value here contains a space).
ATTRIBUTION_SPECS=(
    # "gradient:--attribution-method gradient"
    # "gradient_x_input:--attribution-method gradient_x_input"
    # "ig_zero:--attribution-method integrated_gradients --ig-baseline zero --ig-steps ${IG_STEPS}"
    "ig_reference:--attribution-method integrated_gradients --ig-baseline reference --ig-steps ${IG_STEPS}"
)

# run_model LABEL MODEL CHECKPOINT SUBJECT_NII [extra calculate_gradcam.py args...]
# Runs calculate_gradcam.py once per ATTRIBUTION_SPECS entry, on top of
# whatever model-specific args (e.g. --anatcl-level) are passed through.
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
            --subject-nii "$NII" --output-dir "saliency_affine/${SUBJECT}/" \
            "${EXTRA_ARGS[@]}" \
            $FLAGS
    done
}

# run_model_tabular LABEL CHECKPOINT
# fs_roi's analogue of run_model -- calculate_gradcam_tabular.py takes
# --subject-id directly (a row lookup into the already-cached feature
# matrix), not --subject-nii/a file-existence glob, so there's no NII/path
# argument here at all -- simpler than every image-based model's call.
run_model_tabular() {
    local LABEL="$1" CKPT="$2"
    local SPEC TAG FLAGS
    for SPEC in "${ATTRIBUTION_SPECS[@]}"; do
        TAG="${SPEC%%:*}"
        FLAGS="${SPEC#*:}"
        echo "${LABEL} (${TAG})"
        # shellcheck disable=SC2086  # FLAGS is intentionally word-split, see above.
        time python3 src/calculate_gradcam_tabular.py \
            --checkpoint-path "$CKPT" --target "$TARGET" \
            --subject-id "$SUBJECT" --output-dir "saliency_affine/" \
            --seg-template "${REPO_DIR}/templates/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz" \
            $FLAGS
    done
}

# run_model "unCNN" optim_uncnn "${HEADS_DIR}/optim_uncnn_turboprep_covpool_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"
# run_model "BrainIAC" brainiac "${HEADS_DIR}/brainiac_turboprep_96_cls_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"
# run_model "SIMCLR" simclr "${HEADS_DIR}/simclr_turboprep_transpose_crop_150x192x192_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"
# run_model "SWINBRAIN" swinbrain "${HEADS_DIR}/swinbrain_turboprep_128x128x64_encoder10_avgpool_${CKPT_SUFFIX}.pt" "$TURBOPREP_NII"

# anatcl needs the subject's CAT12 mwp1 file; its filename has a variable
# ses-/acq- suffix per subject (not always ses-BAS1, acq- segment differs or
# is absent), so glob for it rather than assuming one fixed filename, and
# fail loudly on zero or multiple matches rather than silently using the
# wrong file (both anatcl levels read the same CAT12 volume).
mapfile -t ANATCL_NII_MATCHES < <(compgen -G "${NKI_BIDS_ROOT}/sub-${SUBJECT}/ses-*/anat/mri/mwp1sub-${SUBJECT}*.nii")
if [ "${#ANATCL_NII_MATCHES[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly 1 anatcl CAT12 mwp1 file for subject ${SUBJECT}, found ${#ANATCL_NII_MATCHES[@]}: ${ANATCL_NII_MATCHES[*]}" >&2
  exit 1
fi
ANATCL_NII="${ANATCL_NII_MATCHES[0]}"

# export ANATCL_SUFFIX=global_
# run_model "ANATCL GLOBAL" anatcl "${HEADS_DIR}/anatcl_global_cat12vbm_fold0_${CKPT_SUFFIX}.pt" "$ANATCL_NII" \
#     --anatcl-level global --anatcl-folds 0

# export ANATCL_SUFFIX=local_
# run_model "ANATCL LOCAL" anatcl "${HEADS_DIR}/anatcl_local_cat12vbm_fold0_${CKPT_SUFFIX}.pt" "$ANATCL_NII" \
#     --anatcl-level local --anatcl-folds 0

run_model_tabular "FS ROI" "${HEADS_DIR}/fs_roi_dktatlas_grayvol_${CKPT_SUFFIX}.pt"
