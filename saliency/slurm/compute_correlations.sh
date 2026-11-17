#!/usr/bin/env bash
#SBATCH --job-name=saliency_correlations_affine
#SBATCH --time=2:0:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Affine-registration counterpart of the original compute_correlations.sh --
# reads from saliency_affine/<target>/ (slurm/compute_saliency.sh's
# NKI_AFFINE_TURBOPREP_ROOT/checkpoints/heads/*_affine.pt pipeline) instead of
# saliency_outputs/<target>/ (the original rigid-registration pipeline), via
# compute_saliency_correlations.py's --saliency-root flag. Every model
# tag/filename convention, --level/--metrics semantics, and output layout
# are otherwise identical -- only the top-level tree (and so, by
# construction, which registration each saliency map was produced from)
# differs; see that script's --saliency-root help. Output lands under
# saliency_affine/<target>/correlations/, so it never overwrites a
# saliency_outputs run's CSVs/figures even if both are run from the same
# --output-dir default.
#
# Runs compute_saliency_correlations.py, which writes into
# saliency_affine/<target>/correlations/, separately per attribution method
# (--attribution-tags, default all 4: gradient/vanillagrad, gxi, ig_zero,
# ig_reference) AND per agreement/dissimilarity metric (--metrics, default
# all 4: spearman, pearson, mutual_information, ssd -- see that script's
# module docstring for what each one measures), across THREE separate
# comparisons that test different things -- never mixed into one another's
# CSVs/figures. The pairwise and cross-model comparisons' granularity is
# controlled by LEVEL below (see that script's --level): at the "aggregate"
# default, each model's baseline maps are averaged across every subject
# FIRST, so "pairwise" and "cross-model" both collapse to one score per
# model pair; at "subject" (the original behavior), every individual
# (model, subject) baseline map is its own row/column instead.
#   - pairwise_model_{LEVEL}_{METRIC}_{KIND}_{TAG}.csv ("pairwise_model_subject_..."
#     at LEVEL=subject, keeping its pre-existing name): at LEVEL=aggregate, a
#     small N_models x N_models score matrix over each model's mean map; at
#     LEVEL=subject, the full NxN matrix across every (model, subject)
#     baseline map, for that method.
#   - sanity_check_{METRIC}_{KIND}_{TAG}.csv: each subject's baseline vs. the
#     shared arch_/data_/input_ sanity-check maps for that model and method
#     (robustness/sanity, not cross-model agreement) -- always per-subject,
#     unaffected by LEVEL.
#   - cross_model_{LEVEL}_{METRIC}_{KIND}_{TAG}.csv ("cross_model_..." at
#     LEVEL=subject, keeping its pre-existing name): at LEVEL=aggregate,
#     every pair of DIFFERENT models' mean maps, one score per pair; at
#     LEVEL=subject, every pair of DIFFERENT models' baseline maps for each
#     subject (e.g. optim_uncnn_rank vs. simclr, for that one subject) --
#     "do different architectures agree on what's salient". Derived directly
#     from the pairwise CSV above (see extract_cross_model_pairs(_aggregate)),
#     so it costs no extra resampling on top of an already-running pairwise
#     comparison.
# Plus 5 combined figures per (KIND, METRIC) (2x2-grid PNGs per attribution
# method for each CSV family, plus a plotly-interactive HTML version of the
# sanity-check figure with per-point subject hover):
#   pairwise_model_{LEVEL}_{METRIC}_{KIND}.png
#   sanity_check_{METRIC}_{KIND}.png
#   sanity_check_{METRIC}_{KIND}.html
#   cross_model_{LEVEL}_heatmap_{METRIC}_{KIND}.png   (model x model)
#   cross_model_{LEVEL}_boxplot_{METRIC}_{KIND}.png   (per-model-pair spread; one point per subject at LEVEL=subject, one point at LEVEL=aggregate)
# i.e. 5 files (4 PNG + 1 HTML) per (kind, metric) combination -- 2 kinds x
# 4 metrics x 5 files = 40 figures total, at the defaults below.
#
# COST: this resamples every baseline/sanity-check map separately per
# attribution method (4x the work at the default --attribution-tags) --
# --time above is a starting guess, raise it if this doesn't fit. Requesting
# several --metrics in one invocation (the default: all 4) does NOT multiply
# the resampling cost the same way -- resampling depends on kind/tags/models,
# never on which metric scores the resampled vectors, so
# compute_saliency_correlations.py shares one resampled vector_cache across
# every requested metric (see its module docstring COST NOTE) -- only the
# (comparatively cheap) scoring step repeats per metric, and even that is
# non-trivially more expensive for mutual_information/ssd than for
# spearman/pearson's vectorized pairwise computation (again, see that
# docstring). --n-jobs below parallelizes resampling (the dominant cost --
# see compute_saliency_correlations.py's --timing) across --cpus-per-task
# above; keep the two in sync if you change one. Add --timing here for
# per-file resampling time if you need to debug a slow run further.
#
# --resume below means resubmitting this exact script after a timeout (or
# any other kill) picks up where it left off: any (metric, tag, pairwise/
# sanity-check) CSV already written and parseable is skipped instead of
# recomputed, and resampling itself is narrowed to only the files the
# remaining work still needs -- see compute_saliency_correlations.py's
# --resume/plan_remaining_work for exactly what counts as "already done".
# Safe to leave on for a from-scratch run too: with an empty output-dir,
# nothing is skipped and behavior is identical to not passing it.
#
# MEMORY: --mem was previously bumped to 700G after this job OOM'd -- that
# was masking, not fixing, the actual cause: compute_saliency_correlations.py
# used to hold every unique resampled map (a few hundred subjects x ~6
# models x baseline+sanity files, ~28MB each at the unmasked template grid)
# in one in-RAM dict for the whole run, AND built the full (model, subject)
# x (model, subject) pairwise matrix as one np.vstack/np.corrcoef call over
# every one of those maps at once (with ~5-6k labels once every subject is
# included, that single matrix alone was on the order of 100+GB). Both are
# now bounded: resampled vectors are cached to disk (one .npy per file,
# under <output-dir>/.correlations_cache/) and read back via memmap instead
# of held in RAM together, and the pairwise matrix is computed in
# --block-size (default 200) label-sized blocks instead of all at once (see
# compute_pairwise_matrix in that script). 64G is a generous starting
# guess for the new bounded footprint -- check `seff <jobid>` after a run
# and lower this (or pass a smaller `--block-size` below) if it still looks
# oversized, or raise --block-size for a modest speedup if 64G leaves
# headroom.

set -uo pipefail

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_correlations.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list) instead
# of 3 sequential `TARGET=... sbatch` submissions -- selects which
# saliency_affine/<target>/ subtree to analyze. Submit a single target's
# task directly with e.g. `sbatch --array=0 slurm/compute_correlations.sh`
# (age only), `--array=1` (sex only), `--array=2` (bmi only).
TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/compute_correlations.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"

# Declared explicitly (matching this repo's ATTRIBUTION_SPECS/STATS-array
# convention elsewhere) rather than relying on compute_saliency_correlations.py's
# own --metrics default (which is the same all-4 list) -- comment out an
# entry here to skip it, e.g. to drop the pairwise-comparison-heavy
# mutual_information/ssd metrics (see the COST NOTE above) from a quick run.
# METRICS=(spearman pearson mutual_information ssd)
METRICS=(spearman pearson)

# LEVEL selects the granularity of the pairwise/cross-model comparisons (see
# compute_saliency_correlations.py's --level): "aggregate" (default here and
# in that script) compares each model's MEAN map across every subject --
# an N_models x N_models matrix, cheap -- while "subject" reproduces the
# original, much larger (model, subject) x (model, subject) matrix (see
# compute_pairwise_matrix's memory/cost notes there). Override at submission
# with e.g. `sbatch --export=ALL,LEVEL=subject slurm/compute_correlations.sh`
# for the old per-subject behavior; output filenames encode the level, so an
# aggregate run and a subject run never overwrite each other's CSVs/figures.
LEVEL="${LEVEL:-aggregate}"

for KIND in signed abs; do
    # -u: unbuffered stdout -- without it, Python fully block-buffers stdout
    # once it's redirected to this job's log file (rather than a terminal),
    # so every print() (resampling progress, per-file --timing, per-tag/
    # metric status) sits in memory and is lost if the job is killed (e.g.
    # SLURM's SIGTERM/SIGKILL at --time) before a normal exit flushes it --
    # exactly what made the CANCELLED DUE TO TIME LIMIT logs come back empty.
    python3 -u src/compute_saliency_correlations.py --target "$TARGET" \
    --saliency-root saliency_affine \
    --kind "$KIND" --level "$LEVEL" --attribution-tags ig_reference \
    --exclude-models anatcl_local --metrics "${METRICS[@]}" --n-jobs 5 \
    --timing --resume
done
