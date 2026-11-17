#!/usr/bin/env bash
#SBATCH --job-name=viz_subjects
#SBATCH --time=0:50:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x_%a.out
#SBATCH --array=0-44
#
# Per-(target, subject) array job, like slurm/compute_saliency.sh: the array
# index encodes TWO dimensions flattened into one (TASK_TARGETS=(age sex bmi)
# below, index = target_idx * N_SUBJECTS + subject_idx), so age/sex/bmi run
# as genuinely parallel SLURM array tasks instead of 3 sequential
# `TARGET=... sbatch` submissions. N_SUBJECTS below must match the number of
# per-subject folders under saliency_outputs/<target>/ for EVERY target --
# recompute --array's bound (${#TASK_TARGETS[@]} * N_SUBJECTS - 1) if that
# count changes, e.g. `ls -d saliency_outputs/age/A*/ | wc -l`. Submit a
# single target's contiguous block directly, e.g. `--array=0-14` (age only),
# `--array=15-29` (sex only), `--array=30-44` (bmi only), with N_SUBJECTS=15.
#
# For each of the 15 per-subject folders under saliency_outputs/<target>/, generate:
#   1. One PNG per model (6 total: optim_uncnn, brainiac, swinbrain, simclr,
#      anatcl_global, anatcl_local) comparing that subject's own baseline
#      ("reference") saliency map against the sanity-check maps
#      (arch_{MODEL}_all_*, data_{MODEL}_*, input_{MODEL}_permute_*) that
#      live one directory up in saliency_outputs/ itself.
#   2. One more PNG comparing all 6 models' subject-level baseline maps to
#      each other.
# 7 PNGs per subject, written into that subject's own folder.
#
# The reference-row baseline and the sanity-check rows live in two different
# directories (subject folder vs. its parent), and visualize_saliency.py only
# takes one --saliency-dir, so each per-model figure stages symlinks to both
# sources into a throwaway temp directory before calling it.
#
# Also produces one all-subjects-per-model figure per model per target (see
# the block at the end, guarded to run only on each target's first subject
# task): every subject's baseline map for that model alongside its arch_/
# data_/input_ sanity-check battery, in
# saliency_outputs/<target>/{model}_all_subjects_sanity_check_abs.png.

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_subjects.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

TASK_TARGETS=(age sex bmi)
N_SUBJECTS=15
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-$((${#TASK_TARGETS[@]} * N_SUBJECTS - 1)), e.g. sbatch --array=0-$((${#TASK_TARGETS[@]} * N_SUBJECTS - 1)) slurm/visualize_subjects.sh}"
TARGET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SUBJECTS ))
SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID % N_SUBJECTS ))
if [ "$TARGET_IDX" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> target index $TARGET_IDX, out of range for TASK_TARGETS (${TASK_TARGETS[*]}) with N_SUBJECTS=$N_SUBJECTS." >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$TARGET_IDX]}"
echo "TARGET=$TARGET"
SALIENCY_DIR="${REPO_DIR}/saliency_outputs/${TARGET}"

# One kind per figure, matching the "6 PNGs (not 12)" spec. Add "abs" here
# (or rerun with KINDS=(abs)) if you also want the abs-saliency versions.
KINDS=(abs)
# fs_roi has no arch_/data_/input_ sanity-check files (calculate_gradcam_
# tabular.py doesn't implement --randomize-model/--randomize-data/
# --randomize-input yet) -- it still flows through the per-model loop below
# same as every other model, so it still gets a "sanity check" figure with
# its own baseline row, just with no sanity-check rows alongside it (same
# graceful-degradation pattern visualize_aggregate.sh's MODELS/SANITY_MODELS
# already relies on for fs_roi, and optim_uncnn_avg relies on there too).
# The all_models_${KIND}.png cross-model comparison below has no such
# dependency and benefits from fs_roi unconditionally.
MODELS=(optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac swinbrain simclr anatcl_global anatcl_local fs_roi)

SUBJECT_DIRS=("$SALIENCY_DIR"/A*/)
if [ "$SUBJECT_IDX" -ge "${#SUBJECT_DIRS[@]}" ]; then
    echo "ERROR: SUBJECT_IDX=$SUBJECT_IDX out of range for $SALIENCY_DIR (found ${#SUBJECT_DIRS[@]} subject folders) -- check N_SUBJECTS matches" >&2
    exit 1
fi
SUBJECT_DIR="${SUBJECT_DIRS[$SUBJECT_IDX]%/}"
SUBJECT_ID="$(basename "$SUBJECT_DIR")"
echo "=== Subject: $SUBJECT_ID ($SUBJECT_DIR) ==="

for MODEL in "${MODELS[@]}"; do
    # anatcl's arch_/data_/input_ sanity-check outputs use "anatcl" with a
    # trailing _global_/_local_ suffix (see calculate_gradcam.py's
    # ANATCL_SUFFIX handling), not "anatcl_global"/"anatcl_local" as an
    # infix -- build the right glob prefixes per model accordingly.
    # JSON_STEM: calculate_gradcam.py always names the sanity-checks JSON
    # after the real model it ran (args_saliency.model), never a pseudo-model
    # name -- optim_uncnn's 3 channels (raw/rank/sobel) are one run, one
    # shared "optim_uncnn_sanity_checks.json", not "optim_uncnn_raw_...json"
    # etc (see bin_saliency.py's sanity_json_stem, same mapping). anatcl
    # DOES get a real per-level JSON ("anatcl_global_..."/"anatcl_local_...",
    # via calculate_gradcam.py's ANATCL_SUFFIX), so $MODEL is already correct
    # there and in every other (single-real-model) case.
    case "$MODEL" in
        anatcl_*)
            LEVEL="${MODEL#anatcl_}"
            ARCH_PREFIX="arch_anatcl_all_${LEVEL}_"
            DATA_PREFIX="data_anatcl_${LEVEL}_"
            INPUT_PREFIX="input_anatcl_permute_${LEVEL}_"
            JSON_STEM="$MODEL"
            ;;
        optim_uncnn_*)
            CHANNEL="${MODEL#optim_uncnn_}"
            ARCH_PREFIX="arch_optim_uncnn_all_${CHANNEL}_"
            DATA_PREFIX="data_optim_uncnn_${CHANNEL}_"
            INPUT_PREFIX="input_optim_uncnn_permute_${CHANNEL}_"
            JSON_STEM="optim_uncnn"
            ;;
        *)
            ARCH_PREFIX="arch_${MODEL}_all_"
            DATA_PREFIX="data_${MODEL}_"
            INPUT_PREFIX="input_${MODEL}_permute_"
            JSON_STEM="$MODEL"
            ;;
    esac

    for KIND in "${KINDS[@]}"; do
        STAGE_DIR="$(mktemp -d)"

        # Reference row(s): this subject's own baseline map(s) + its
        # sanity-checks JSON (for the true/predicted-age suptitle). Globs
        # (not an exact filename match) so every attribution-method variant
        # calculate_gradcam.py wrote for this model (gxi_/ig_zero_/
        # ig_reference_, in addition to plain gradient) gets its own row --
        # see attribution_method_tag/split_attribution_tag for how that tag
        # sits right before "_${KIND}_saliency.nii.gz".
        for f in "$SUBJECT_DIR/${MODEL}_"*"${KIND}_saliency.nii.gz"; do
            [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/"
        done

        if ! compgen -G "$STAGE_DIR/${MODEL}_*${KIND}_saliency.nii.gz" > /dev/null; then
            echo "  SKIP ${MODEL} (${KIND}): no baseline map for subject ${SUBJECT_ID}"
            rm -rf "$STAGE_DIR"
            continue
        fi

        # Uses JSON_STEM (not $MODEL -- see above) and globs so every
        # attribution-method variant's JSON gets staged too, not just
        # gradient's -- metadata_for_file()'s prefix-startswith matching
        # (visualize_saliency.py) still resolves every row correctly from
        # whichever JSON(s) actually exist, but only if at least one is here.
        for f in "$SUBJECT_DIR/${JSON_STEM}"*"_sanity_checks.json"; do
            [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/"
        done

        # Sanity-check rows: shared reference-subject battery, one dir up.
        for f in "$SALIENCY_DIR/${ARCH_PREFIX}"*"${KIND}_saliency.nii.gz" \
                 "$SALIENCY_DIR/${DATA_PREFIX}"*"${KIND}_saliency.nii.gz" \
                 "$SALIENCY_DIR/${INPUT_PREFIX}"*"${KIND}_saliency.nii.gz"; do
            [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/"
        done

        # --models controls both filtering AND row order (visualize_saliency.py
        # appends matches in the order its entries are given, so this list's
        # order IS the figure's row order): grouped by attribution method
        # first (vanillagrad's baseline+3 checks together, then gxi's, then
        # ig_zero's, then ig_reference's), not by check-type first. Each
        # entry is a full, unambiguous filename (not just a prefix stub) --
        # e.g. "${MODEL}_${KIND}_saliency.nii.gz" for the untagged gradient
        # baseline, "${MODEL}_gxi_${KIND}_saliency.nii.gz" for its gxi
        # counterpart -- so a plain prefix match can't let one method's rows
        # swallow another's (a bare "$MODEL" would match "${MODEL}_gxi_..."
        # too). See attribution_method_tag/split_attribution_tag for where
        # this tag sits in the filename.
        METHOD_TAGS=("" "gxi_" "ig_zero_" "ig_reference_")
        MODEL_FILTERS=()
        for TAG in "${METHOD_TAGS[@]}"; do
            MODEL_FILTERS+=(
                "${MODEL}_${TAG}${KIND}_saliency.nii.gz"
                "${ARCH_PREFIX}${TAG}${KIND}_saliency.nii.gz"
                "${DATA_PREFIX}${TAG}${KIND}_saliency.nii.gz"
                "${INPUT_PREFIX}${TAG}${KIND}_saliency.nii.gz"
            )
        done

        OUT="$SUBJECT_DIR/${MODEL}_sanity_check_${KIND}.png"
        if python3 src/visualize_saliency.py \
            --saliency-dir "$STAGE_DIR" --kind "$KIND" --norm \
            --models "${MODEL_FILTERS[@]}" \
            --title "${MODEL} sanity check -- subject ${SUBJECT_ID}" \
            --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
            echo "  wrote $OUT"
        else
            echo "  WARN: visualize_saliency.py failed for ${MODEL} (${KIND}), subject ${SUBJECT_ID}"
        fi
        rm -rf "$STAGE_DIR"
    done
done

# Cross-model comparison: all 6 models' subject-level baselines already live
# together in the subject's own folder -- no staging needed.
for KIND in "${KINDS[@]}"; do
    OUT="$SUBJECT_DIR/all_models_${KIND}.png"
    if python3 src/visualize_saliency.py \
        --saliency-dir "$SUBJECT_DIR" --kind "$KIND" --norm \
        --models "${MODELS[@]}" \
        --title "Model comparison -- subject ${SUBJECT_ID}" \
        --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
        echo "  wrote $OUT"
    else
        echo "  WARN: cross-model comparison failed for kind $KIND, subject ${SUBJECT_ID}"
    fi
done

# All-subjects-per-model figure: for each model, one PNG with every
# subject's baseline map alongside that model's sanity-check battery
# (arch_/data_/input_), so all subjects can be scanned against the random
# sanity checks in one figure instead of one-per-subject. Baseline maps for
# a given model share the same filename across subject folders, so each is
# symlinked into the stage dir under a subject-ID-prefixed name to avoid
# collisions; no --models filter is needed since subject IDs ("A00...")
# sort before the arch_/data_/input_ prefixes, giving subject rows first.
# Only each target's first subject task (SUBJECT_IDX == 0) does this for
# that target (it needs every subject dir for TARGET, not just its own, and
# doesn't depend on the other array tasks -- the baseline maps it reads come
# from calculate_gradcam.py, not from this script) -- runs once per target,
# not just once globally, since each target has its own SALIENCY_DIR.
if [ "$SUBJECT_IDX" -eq 0 ]; then
    echo "=== All-subjects-per-model figures ==="
    for MODEL in "${MODELS[@]}"; do
        case "$MODEL" in
            anatcl_*)
                LEVEL="${MODEL#anatcl_}"
                ARCH_PREFIX="arch_anatcl_all_${LEVEL}_"
                DATA_PREFIX="data_anatcl_${LEVEL}_"
                INPUT_PREFIX="input_anatcl_permute_${LEVEL}_"
                ;;
            optim_uncnn_*)
                CHANNEL="${MODEL#optim_uncnn_}"
                ARCH_PREFIX="arch_optim_uncnn_all_${CHANNEL}_"
                DATA_PREFIX="data_optim_uncnn_${CHANNEL}_"
                INPUT_PREFIX="input_optim_uncnn_permute_${CHANNEL}_"
                ;;
            *)
                ARCH_PREFIX="arch_${MODEL}_all_"
                DATA_PREFIX="data_${MODEL}_"
                INPUT_PREFIX="input_${MODEL}_permute_"
                ;;
        esac

        for KIND in "${KINDS[@]}"; do
            STAGE_DIR="$(mktemp -d)"

            # Globs (not an exact filename match), same reasoning as the
            # per-subject reference row above -- every attribution-method
            # variant gets its own row. Destination keeps the source's own
            # basename (tag included) behind the subject-ID prefix, rather
            # than a hardcoded "${MODEL}_${KIND}" name, so different
            # attribution-method variants for the same subject don't
            # collide/overwrite each other under one symlink target.
            for SUBJ_DIR in "${SUBJECT_DIRS[@]}"; do
                SUBJ_DIR="${SUBJ_DIR%/}"
                SUBJ_ID="$(basename "$SUBJ_DIR")"
                for f in "$SUBJ_DIR/${MODEL}_"*"${KIND}_saliency.nii.gz"; do
                    [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/${SUBJ_ID}_$(basename "$f")"
                done
            done

            if ! compgen -G "$STAGE_DIR/*_${MODEL}_*${KIND}_saliency.nii.gz" > /dev/null; then
                echo "  SKIP ${MODEL} (${KIND}): no baseline maps found across subjects"
                rm -rf "$STAGE_DIR"
                continue
            fi

            for f in "$SALIENCY_DIR/${ARCH_PREFIX}"*"${KIND}_saliency.nii.gz" \
                     "$SALIENCY_DIR/${DATA_PREFIX}"*"${KIND}_saliency.nii.gz" \
                     "$SALIENCY_DIR/${INPUT_PREFIX}"*"${KIND}_saliency.nii.gz"; do
                [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/"
            done

            OUT="$SALIENCY_DIR/${MODEL}_all_subjects_sanity_check_${KIND}.png"
            if python3 src/visualize_saliency.py \
                --saliency-dir "$STAGE_DIR" --kind "$KIND" --norm \
                --title "${MODEL} sanity check -- all subjects" \
                --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
                echo "  wrote $OUT"
            else
                echo "  WARN: visualize_saliency.py failed for ${MODEL} (${KIND}), all-subjects figure"
            fi
            rm -rf "$STAGE_DIR"
        done
    done
fi

echo "=== Done: $SUBJECT_ID ==="
