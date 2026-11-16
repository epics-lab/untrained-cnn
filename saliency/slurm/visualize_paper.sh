#!/usr/bin/env bash
#SBATCH --job-name=viz_paper
#SBATCH --time=0:02:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Paper figure(s): src/visualize_paper.py's cross-model ig_popmean
# (calculate_gradcam.py's "ig_reference" attribution method -- integrated
# gradients against the population-mean baseline) comparison, aggregated
# across subjects, one row per model -- see that script's module docstring
# for the full picture. Produces TWO figures per stat in PAPER_STATS:
#   - paper_ig_popmean_{stat}_permap.png -- each model's own row
#     independently rescaled onto the shared unit-free range (visualize_
#     saliency.py's usual --norm behavior).
#   - paper_ig_popmean_{stat}_refnorm_{PAPER_NORM_REFERENCE_MODEL}.png --
#     every row instead rescaled by the ONE scale computed from
#     PAPER_NORM_REFERENCE_MODEL's own map, so relative magnitude
#     differences between models stay visible (see visualize_paper.py's
#     --norm-reference-model docs).
#
# Prerequisite: slurm/visualize_aggregate.sh (or src/aggregate_saliency.py
# directly) must have already been run, so that saliency_outputs/aggregate/
# already has {model}_ig_reference_aggregate_{stat}_{signed,abs}_saliency.
# nii.gz for every model in PAPER_MODELS -- this script only visualizes,
# it never computes aggregates itself. In particular PAPER_MODELS' default
# (optim_uncnn_avg, not the 3 separate raw/rank/sobel channels) needs
# visualize_aggregate.sh's average_uncnn_channels.py step to have run too
# (unconditional there as of this writing -- see that script's comments).

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_paper.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list), same
# convention as slurm/visualize_aggregate.sh and slurm/compute_correlations.sh
# -- selects which saliency_outputs/<target>/aggregate subtree to read from
# and write into. Submit a single target's task directly with e.g.
# `sbatch --array=0 slurm/visualize_paper.sh` (age only), `--array=1` (sex
# only), `--array=2` (bmi only). Requires visualize_aggregate.sh to have
# already been run for that same target (see the prerequisite note above).
TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/visualize_paper.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"
AGGREGATE_DIR="${REPO_DIR}/saliency_outputs/${TARGET}/aggregate"

# visualize_paper.py's own default -- override with e.g.
# `PAPER_MODELS="optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac" sbatch slurm/visualize_paper.sh`
# to see optim_uncnn's 3 channels separately instead of the averaged row.
PAPER_MODELS=(${PAPER_MODELS:-optim_uncnn_avg brainiac swinbrain simclr anatcl_global anatcl_local})
PAPER_STATS=(${PAPER_STATS:-mean})
PAPER_KIND="${PAPER_KIND:-abs}"                                   # signed or abs -- ignored if PAPER_COMBINE_KINDS=1
PAPER_COMBINE_KINDS="${PAPER_COMBINE_KINDS:-0}"                   # 1 = signed+abs side by side (--combine-kinds)
PAPER_NORM_REFERENCE_MODEL="${PAPER_NORM_REFERENCE_MODEL:-optim_uncnn_avg}"
PAPER_NORM_METHOD="${PAPER_NORM_METHOD:-percentile}"
PAPER_MONTAGE="${PAPER_MONTAGE:-1}"                               # 1 = add the axial-slice montage strip
PAPER_MONTAGE_AXIAL_Z_MM=(${PAPER_MONTAGE_AXIAL_Z_MM:--18 0 30})

COMMON_FLAGS=(--models "${PAPER_MODELS[@]}" --no-brain-mask --mask-thresh-frac -0.1)
if [ "$PAPER_COMBINE_KINDS" = "1" ]; then
    COMMON_FLAGS+=(--combine-kinds)
else
    COMMON_FLAGS+=(--kind "$PAPER_KIND")
fi
if [ "$PAPER_MONTAGE" = "1" ]; then
    COMMON_FLAGS+=(--montage --montage-axial-z-mm "${PAPER_MONTAGE_AXIAL_Z_MM[@]}")
fi

for STAT in "${PAPER_STATS[@]}"; do
    # OUT_PERMAP="$AGGREGATE_DIR/paper_ig_popmean_${STAT}_permap.png"
    # if python3 src/visualize_paper.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
    #     --norm --norm-method "$PAPER_NORM_METHOD" \
    #     --title "ig_popmean -- across-subject aggregate (${STAT}), per-map normalization" \
    #     --output "$OUT_PERMAP"; then
    #     echo "  wrote $OUT_PERMAP"
    # else
    #     echo "  WARN: visualize_paper.py (per-map) failed for stat $STAT"
    # fi

    OUT_REFNORM="$AGGREGATE_DIR/paper_ig_popmean_${STAT}_refnorm_${PAPER_NORM_REFERENCE_MODEL}.png"
    if python3 src/visualize_paper.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
        --saliency-dir "$AGGREGATE_DIR" \
        --norm --norm-method "$PAPER_NORM_METHOD"  \
        --norm-reference-model "$PAPER_NORM_REFERENCE_MODEL" \
        --title "ig_popmean -- across-subject aggregate (${STAT}), reference-normalized on ${PAPER_NORM_REFERENCE_MODEL}" \
        --output "$OUT_REFNORM"; then
        echo "  wrote $OUT_REFNORM"
    else
        echo "  WARN: visualize_paper.py (reference-normalized) failed for stat $STAT"
    fi

    # OUT_NONORM="$AGGREGATE_DIR/paper_ig_popmean_${STAT}_nonorm.png"
    # if python3 src/visualize_paper.py "${COMMON_FLAGS[@]}" --stat "$STAT" \
    #     --title "ig_popmean -- across-subject aggregate (${STAT}), no normalization" \
    #     --output "$OUT_NONORM"; then
    #     echo "  wrote $OUT_NONORM"
    # else
    #     echo "  WARN: visualize_paper.py (no normalization) failed for stat $STAT"
    # fi
done

echo "=== Done: paper figures ==="
