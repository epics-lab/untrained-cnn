#!/usr/bin/env bash
#SBATCH --job-name=saliency_correlations_gpu
#SBATCH --time=5:0:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out

# GPU counterpart of compute_correlations.sh -- runs
# compute_saliency_correlations_gpu.py instead of compute_saliency_correlations.py.
# Identical outputs (same CSV/PNG/HTML filenames, same --level/--metrics/
# --resume semantics) -- see that script's module docstring for what each
# comparison/file means. --gres=gpu:1 is this repo's first GPU-requesting
# SLURM job; if your cluster (Narval/Béluga/Cedar/Graham,
# etc.) uses a different GPU resource flag (e.g. --gpus-per-node=1), adjust
# the line above accordingly -- check `sbatch --help` / that cluster's docs
# if this is rejected at submission.
#
# WHAT GPU ACTUALLY SPEEDS UP HERE (read before assuming this is faster):
# compute_saliency_correlations_gpu.py only replaces comparison #1's
# (pairwise) matrix computation with a GPU implementation -- resampling
# (still the dominant cost per the base script's own COST NOTE) and the
# per-subject sanity-check comparison (#2) are untouched, CPU/IO-bound
# exactly as before. That means:
#   - LEVEL=aggregate (default, same as compute_correlations.sh): comparison
#     #1 is already tiny (one N_models x N_models matrix, single digits) on
#     CPU -- expect little to no speedup from this script over the CPU one
#     for a normal run at this level.
#   - LEVEL=subject: comparison #1 becomes an (model, subject) x
#     (model, subject) matrix, thousands of labels -- THIS is where GPU
#     acceleration actually pays off, especially combined with
#     mutual_information/ssd/ssim (unvectorized, O(n_labels^2) on CPU -- see
#     compute_saliency_correlations.py's COST NOTE; mutual_information/ssd
#     become blocked matmuls on GPU here, ssim a from-scratch windowed-
#     convolution reimplementation -- see compute_saliency_correlations_gpu.py's
#     module docstring for both, and for why ssim -- local/windowed, not
#     pointwise -- is the most expensive of the three per pair even on GPU).
# If your actual bottleneck is resampling, this script won't help -- run
# compute_saliency.sh's resampling more in parallel (--n-jobs) instead.
#
# MEMORY: BLOCK_SIZE below is deliberately much smaller than
# compute_correlations.sh's CPU default (200) -- GPU block matrices must fit
# in GPU memory (typically far smaller than the --mem system RAM budget
# above), not just system RAM. At the unmasked template grid (~7M
# voxels/vector) and float64, one row/col block is
# BLOCK_SIZE * 7e6 * 8 bytes -- BLOCK_SIZE=200 alone is already ~11GB, easily
# more than one GPU's worth once both blocks + matmul workspace are
# resident. 32 is a conservative starting point for a 16-24GB GPU; raise it
# if `nvidia-smi` (or the job's own OutOfMemoryError message, which names
# the offending block size) shows headroom, lower it if the job dies with a
# GPU OOM. --brain-mask cuts n_voxels (and so this memory cost) substantially
# if you don't need the unmasked grid.
BLOCK_SIZE="${BLOCK_SIZE:-32}"

set -uo pipefail

# Fail FAST if this job didn't actually get a working GPU -- --gres=gpu:1
# being granted at submission doesn't guarantee CUDA actually works on
# whatever node the scheduler assigned (a flaky node, a driver that didn't
# come up, etc. -- on a shared/free GPU pool this is a per-job roll of the
# dice, not something --gres alone can prevent). Without this check,
# compute_saliency_correlations_gpu.py silently falls back to a CPU torch
# implementation and just runs (correctly, but catastrophically slowly) --
# see slurm/logs/saliency_correlations_gpu_0.out for a real example: one
# ssim row-block alone took ~10.5 HOURS on CPU before this job's --time
# budget ran out, with nothing in the log until then to say the GPU never
# attached. Checked here, before sourcing env.sh/doing any real work, so a
# bad node is caught in seconds, not after burning the whole --time budget.
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed on this node -- no working GPU/driver despite --gres=gpu:1. " \
         "Not proceeding (compute_saliency_correlations_gpu.py would silently fall back to its slow CPU path " \
         "instead of failing loudly). Resubmit -- this is typically node-specific, not a request problem " \
         "(an earlier run on this same account/allocation did get a real GPU)." >&2
    exit 1
fi

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_correlations_gpu.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list) instead
# of 3 sequential `TARGET=... sbatch` submissions -- selects which
# saliency_outputs/<target>/ subtree to analyze. Submit a single target's
# task directly with e.g. `sbatch --array=0 slurm/compute_correlations_gpu.sh`
# (age only), `--array=1` (sex only), `--array=2` (bmi only).
TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/compute_correlations_gpu.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"

# Declared explicitly (matching this repo's ATTRIBUTION_SPECS/STATS-array
# convention elsewhere) rather than relying on the script's own --metrics
# default (which is the same all-5 list) -- comment out an entry here to
# skip it. mutual_information/ssd/ssim are the metrics that benefit most
# from this GPU script (see the header comment above) -- included here
# (unlike compute_correlations.sh's CPU default of just spearman/pearson)
# since they're now cheap enough on GPU to be worth computing by default.
# ssim in particular is the most expensive of the three per pair on CPU
# (windowed filtering, not one histogram/sum -- see
# compute_saliency_correlations.py's own "ssim" docstring entry) and the
# only one this GPU script accelerates via a from-scratch reimplementation
# rather than an algebraic matmul identity -- see
# compute_saliency_correlations_gpu.py's module docstring for what that
# means for exact reproducibility vs. the CPU script's skimage-based output.
# METRICS=(spearman pearson mutual_information ssd ssim)
METRICS=(ssim)

# LEVEL selects the granularity of the pairwise/cross-model comparisons (see
# compute_saliency_correlations.py's --level). "subject" is set as the
# default HERE (unlike compute_correlations.sh, which defaults to
# "aggregate") because that's the level where this GPU script's speedup
# actually shows up -- see the header comment above. Override at submission
# with e.g. `sbatch --export=ALL,LEVEL=aggregate slurm/compute_correlations_gpu.sh`.
# Output filenames encode the level, so an aggregate run and a subject run
# never overwrite each other's CSVs/figures.
LEVEL="${LEVEL:-subject}"

# Off by default: at LEVEL=subject (thousands of (model, subject) labels,
# this script's own default -- see header comment), the cross-model heatmap/
# boxplot figures (compute_saliency_correlations.py's comparison #3) can be
# noticeably slower to render than the pairwise/sanity-check ones, and
# aren't always needed for a given run. Set CROSS_MODEL_FIGURES=1 (e.g.
# `sbatch --export=ALL,CROSS_MODEL_FIGURES=1 slurm/compute_correlations_gpu.sh`)
# to generate them -- the underlying --skip-cross-model-figures flag only
# gates the two PNGs; the cross_model_*.csv itself is still written either
# way (cheap, derived from the pairwise CSV already on disk -- see that
# flag's own help text), so it's always available to plot later via
# --skip-compute --no CROSS_MODEL_FIGURES if you change your mind without
# rerunning this whole job.
CROSS_MODEL_FIGURES="${CROSS_MODEL_FIGURES:-0}"
CROSS_MODEL_FLAG="--skip-cross-model-figures"
if [ "$CROSS_MODEL_FIGURES" = "1" ]; then
    CROSS_MODEL_FLAG=""
fi

for KIND in abs signed; do
    # -u: unbuffered stdout -- without it, Python fully block-buffers stdout
    # once it's redirected to this job's log file, so progress/timing prints
    # are lost if the job is killed before a normal exit flushes them.
    # shellcheck disable=SC2086  # CROSS_MODEL_FLAG is intentionally word-split (empty when unset).
    python3 -u src/compute_saliency_correlations_gpu.py --target "$TARGET" \
    --kind "$KIND" --level "$LEVEL" --attribution-tags ig_reference \
    --saliency-root saliency_affine \
    --exclude-models anatcl_local --metrics "${METRICS[@]}" --n-jobs 5 \
    --block-size "$BLOCK_SIZE" --timing --brain-mask --resume \
    $CROSS_MODEL_FLAG
done
