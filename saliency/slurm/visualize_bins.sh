#!/usr/bin/env bash
#SBATCH --job-name=viz_bins
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Two group-level views of the per-subject saliency maps, each built by
# bin_saliency.py (mean + std per (model, bin) group) and then plotted with
# visualize_saliency.py:
#
#   1. Age bins: subjects grouped by true age (0-20, 20-40, 40-60, 60-80,
#      80-100 by default -- see bin_saliency.py --age-edges). Since age
#      doesn't depend on which model is looking, the same subjects land in
#      the same bin for every model.
#   2. Error bins: subjects grouped by |predicted_age - true_age| *for that
#      model* (within 2y, 2-5y, 5-10y, over 10y by default -- see
#      bin_saliency.py --error-edges). A subject's bin can differ model to
#      model since predictions differ.
#
# For each binning, one figure per model is written (6 models x 2 stats = 12
# figures per binning), each with one row per (attribution method, bin) for
# that model -- all 4 of calculate_gradcam.py's attribution methods
# (gradient aka vanillagrad, gradient_x_input aka gxi, integrated_gradients
# with a zero baseline aka ig_zero, with the population-mean baseline aka
# ig_reference/ig_popmean -- see bin_saliency.py's --attribution-tags),
# grouped by method first (vanillagrad's bins together, then gxi's, etc.),
# bin second -- same grouping convention visualize_subjects.sh uses for its
# sanity-check figures.
# Output: saliency_outputs/age_bins/{model}_age_bins_{mean,std}_{kind}.png and
#         saliency_outputs/error_bins/{model}_error_bins_{mean,std}_{kind}.png
#
# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list) instead
# of 3 sequential `TARGET=... sbatch` submissions -- selects which
# sanity_checks.json target-key/checkpoint convention is read and which
# --bin-by modes are valid for it. Submit a single target's task directly
# with e.g. `sbatch --array=0 slurm/visualize_bins.sh` (age only), `--array=1`
# (sex only), `--array=2` (bmi only).
#   age (default): --bin-by age (value binning) + error, as always.
#   bmi: --bin-by value (edges from VALUE_EDGES below, since BMI's scale
#        differs from age's) + error -- no "age" alias, output lands in
#        value_bins/ and error_bins/, target=bmi is passed through.
#   sex: --bin-by class only (bin = the true class label itself, no edges) --
#        "error" (a continuous-residual concept) has no meaning for a binary
#        target, so it's skipped entirely for TARGET=sex.

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_bins.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/visualize_bins.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"
# calculate_gradcam.py/calculate_gradcam_tabular.py now write into
# saliency_outputs/<target>/ (see those scripts' own docstrings), so this
# must point at the same per-target subtree, not the flat root.
SALIENCY_DIR="${REPO_DIR}/saliency_outputs/${TARGET}"

KINDS=(signed)
# fs_roi needs calculate_gradcam_tabular.py to have written a
# fs_roi_..._sanity_checks.json per subject (true/predicted value) for
# bin_saliency.py's load_meta to bin it at all -- true for any run since
# that script started writing this JSON unconditionally (see its own
# docstring); older per-subject fs_roi baselines computed before that change
# won't have one and are silently skipped by bin_saliency.py (true_value is
# None -> that subject just doesn't appear in any bin for fs_roi, not a
# crash -- see bin_saliency.py's load_meta).
MODELS=(optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac swinbrain simclr anatcl_global anatcl_local fs_roi)
AGE_EDGES=(0 20 40 60 80 100)
VALUE_EDGES=(${VALUE_EDGES:-15 20 25 30 35 40})   # --target bmi only -- override via `VALUE_EDGES="..." TARGET=bmi sbatch ...`
ERROR_EDGES=(2 5 10)

# calculate_gradcam.py's 4 attribution methods (bin_saliency.py's own
# --attribution-tags default, listed explicitly here since the loop below
# needs the same names/order to build filename prefixes and figure row
# grouping). "" is gradient's own filename fragment (calculate_gradcam.py
# never inserts a tag for it); the other 3 match attribution_method_tag()'s
# tags exactly, trailing underscore included since they sit directly before
# "agebin"/"errorbin" in the filename (see bin_saliency.py's model_tag_prefix).
ATTRIBUTION_TAGS=(gradient gxi ig_zero ig_reference)
FILENAME_TAG_PREFIXES=("" "gxi_" "ig_zero_" "ig_reference_")

# Generic "N edges -> N-1 range labels" (0-20, 20-40, ...) -- used for both
# --bin-by age (AGE_EDGES) and --bin-by value (VALUE_EDGES, --target bmi).
# Takes the edges array by name (bash nameref).
range_bin_labels() {
    local -n edges_ref=$1
    local prev=""
    for e in "${edges_ref[@]}"; do
        [ -n "$prev" ] && echo "${prev}-${e}"
        prev="$e"
    done
}

error_bin_labels() {
    local prev=""
    for e in "${ERROR_EDGES[@]}"; do
        if [ -z "$prev" ]; then
            echo "within${e}"
        else
            echo "${prev}to${e}"
        fi
        prev="$e"
    done
    echo "over${prev}"
}

# class_bin_labels: fixed, matching targets.py's TargetSpec.decode_display
# for sex exactly ("Female" if >=0.5 else "Male") -- no edges, unlike the
# other three modes.
class_bin_labels() {
    echo "Female"
    echo "Male"
}

BIN_BY_MODES=()
case "$TARGET" in
    age)
        python3 src/bin_saliency.py --saliency-dir "$SALIENCY_DIR" --bin-by age \
            --age-edges "${AGE_EDGES[@]}" --kinds "${KINDS[@]}" --models "${MODELS[@]}" \
            --attribution-tags "${ATTRIBUTION_TAGS[@]}"
        python3 src/bin_saliency.py --saliency-dir "$SALIENCY_DIR" --bin-by error \
            --error-edges "${ERROR_EDGES[@]}" --kinds "${KINDS[@]}" --models "${MODELS[@]}" \
            --attribution-tags "${ATTRIBUTION_TAGS[@]}"
        BIN_BY_MODES=(age error)
        ;;
    bmi)
        python3 src/bin_saliency.py --saliency-dir "$SALIENCY_DIR" --bin-by value --target bmi \
            --age-edges "${VALUE_EDGES[@]}" --kinds "${KINDS[@]}" --models "${MODELS[@]}" \
            --attribution-tags "${ATTRIBUTION_TAGS[@]}"
        python3 src/bin_saliency.py --saliency-dir "$SALIENCY_DIR" --bin-by error --target bmi \
            --error-edges "${ERROR_EDGES[@]}" --kinds "${KINDS[@]}" --models "${MODELS[@]}" \
            --attribution-tags "${ATTRIBUTION_TAGS[@]}"
        BIN_BY_MODES=(value error)
        ;;
    sex)
        # No "error" mode for a binary target -- see header comment.
        python3 src/bin_saliency.py --saliency-dir "$SALIENCY_DIR" --bin-by class --target sex \
            --kinds "${KINDS[@]}" --models "${MODELS[@]}" \
            --attribution-tags "${ATTRIBUTION_TAGS[@]}"
        BIN_BY_MODES=(class)
        ;;
    *)
        echo "ERROR: unknown TARGET=$TARGET (expected age/sex/bmi)" >&2
        exit 1
        ;;
esac

for BIN_BY in "${BIN_BY_MODES[@]}"; do
    BIN_DIR="$SALIENCY_DIR/${BIN_BY}_bins"
    case "$BIN_BY" in
        age) mapfile -t BIN_LABELS < <(range_bin_labels AGE_EDGES) ;;
        value) mapfile -t BIN_LABELS < <(range_bin_labels VALUE_EDGES) ;;
        error) mapfile -t BIN_LABELS < <(error_bin_labels) ;;
        class) mapfile -t BIN_LABELS < <(class_bin_labels) ;;
    esac

    for MODEL in "${MODELS[@]}"; do
        # Row order: grouped by attribution method first (each method's bins
        # together, in ATTRIBUTION_TAGS order), bin second. Each prefix is
        # built with the tag's filename fragment already embedded (e.g.
        # "${MODEL}_gxi_${BIN_BY}bin_..."), matching bin_saliency.py's
        # model_tag_prefix construction exactly -- no ambiguity risk from a
        # bare "$MODEL" prefix swallowing every method's rows, since
        # "${MODEL}_agebin_" (gradient) and "${MODEL}_gxi_agebin_" (gxi)
        # diverge at the very next character.
        for STAT in mean std; do
            STAT_PREFIXES=()
            for i in "${!ATTRIBUTION_TAGS[@]}"; do
                TAG_PREFIX="${FILENAME_TAG_PREFIXES[$i]}"
                for LABEL in "${BIN_LABELS[@]}"; do
                    STAT_PREFIXES+=("${MODEL}_${TAG_PREFIX}${BIN_BY}bin_${LABEL}_${STAT}")
                done
            done

            for KIND in "${KINDS[@]}"; do
                OUT="$BIN_DIR/${MODEL}_${BIN_BY}_bins_${STAT}_${KIND}.png"
                if python3 src/visualize_saliency.py \
                    --saliency-dir "$BIN_DIR" --kind "$KIND" --norm \
                    --models "${STAT_PREFIXES[@]}" \
                    --title "${MODEL} ${BIN_BY} bins (${STAT}) -- all attribution methods" \
                    --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
                    echo "  wrote $OUT"
                else
                    echo "  WARN: visualize_saliency.py failed for ${MODEL} ${BIN_BY} bins (${STAT}, ${KIND})"
                fi
            done
        done
    done
done

echo "=== Done: bins ==="
