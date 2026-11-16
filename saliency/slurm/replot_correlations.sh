#!/usr/bin/env bash
#SBATCH --job-name=saliency_correlations_replot
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Restyles compute_correlations_gpu.sh's already-computed pairwise heatmap
# grids (saliency_outputs/<target>/correlations/pairwise_model_subject_*)
# WITHOUT recomputing or touching any CSV -- runs
# compute_saliency_correlations.py --skip-compute, which (per that flag's
# help) "just replot[s] the CSVs already in --output-dir". No GPU, no
# --gres, minimal --mem/--time: this only re-reads CSVs and re-draws
# matplotlib figures, none of the resampling/pairwise-matrix cost that
# drives compute_correlations_gpu.sh's/compute_correlations.sh's resource
# requests.
#
# TARGET/--saliency-root/LEVEL/KIND/METRICS/--attribution-tags/
# --exclude-models/--brain-mask below are copied VERBATIM from
# compute_correlations_gpu.sh (this script's counterpart with --skip-compute
# added instead of a from-scratch run) -- they must match exactly, since
# --output-dir/CSV filenames are derived from these same flags (mask_suffix,
# pairwise_stem, --kind, --metrics, --attribution-tags) and --skip-compute
# looks for CSVs already on disk under that exact path, not the whole
# --output-dir tree. If you change compute_correlations_gpu.sh's flags,
# mirror the change here too or this script will hit the "--skip-compute but
# no CSVs exist yet" SystemExit.
#
# --no-title / --tick-label-scale / --annotation-scale are the new
# (--skip-compute-only in practice) style flags this script exists to
# exercise -- see draw_pairwise_heatmap/plot_pairwise_heatmap_grid in
# compute_saliency_correlations.py. All three apply only to the pairwise
# heatmap grid PNG (pairwise_model_subject_*.png), not the sanity-check/
# cross-model figures. --annotation-scale also widens the gap between the
# stacked z:*/p:* significance lines (in points, not data units, so it
# stays legible regardless of block size) -- raise it together with
# --tick-label-scale, not instead of it, or the z:/p: text stays readable
# but the now much-bigger tick labels dominate the figure instead. Override
# at submission, e.g.
# `sbatch --export=ALL,NO_TITLE=,TICK_SCALE=1,ANNOTATION_SCALE=1 slurm/replot_correlations.sh`
# to fall back to the original (titled, base-size) style.
NO_TITLE="${NO_TITLE:---no-title}"
TICK_SCALE="${TICK_SCALE:-10}"
ANNOTATION_SCALE="${ANNOTATION_SCALE:-3}"

set -uo pipefail

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/replot_correlations.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/replot_correlations.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"

# Matches compute_correlations_gpu.sh's METRICS/LEVEL exactly (see header
# comment above) -- saliency_outputs is that script's (and this one's)
# --saliency-root default, so it's left unset here too.
METRICS=(spearman pearson mutual_information ssd)
LEVEL="${LEVEL:-subject}"

for KIND in abs; do
    python3 -u src/compute_saliency_correlations.py --target "$TARGET" \
    --kind "$KIND" --level "$LEVEL" --attribution-tags ig_reference \
    --exclude-models anatcl_local --metrics "${METRICS[@]}" \
    --brain-mask --skip-compute $NO_TITLE --tick-label-scale "$TICK_SCALE" --annotation-scale "$ANNOTATION_SCALE"
done
