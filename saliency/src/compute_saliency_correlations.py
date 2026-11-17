"""
Cross-map agreement statistics across saliency maps, three ways:

  1. Pairwise, across every (model, subject) baseline map found under
     saliency_outputs/<SUBJECT>/ -- a full NxN matrix, so you can see which
     models/subjects produce similar maps to which others.
  2. Each subject's baseline vs. the shared sanity-check maps for that same
     model (arch_{MODEL}_all_*, data_{MODEL}_*, input_{MODEL}_permute_*,
     living one directory up in saliency_outputs/ itself -- see
     visualize_subjects.sh for the same layout) -- extends the single-subject
     correlations calculate_gradcam.py already writes into
     {model}_sanity_checks.json to every subject's own baseline, against the
     one shared reference-subject sanity-check battery.
  3. Cross-model, SAME subject: for each subject, every pair of DIFFERENT
     models' baseline maps (e.g. optim_uncnn_rank vs. simclr, for that one
     subject) -- "do different architectures agree on what's salient for
     the same brain", a genuinely different question from #1 (which also
     mixes in different-subject pairs, diluting this signal inside a much
     larger matrix) and #2 (same model, real vs. randomized -- a sanity
     check, not a cross-model agreement measure). Derived directly from
     comparison #1's already-computed pairwise matrix (see
     extract_cross_model_pairs) -- no extra resampling, just a filter +
     regroup of scores #1 already computed, so this is nearly free on top
     of an existing run. Saved to its own CSV/figure files, never mixed
     into the #1 or #2 outputs (see "Also generates figures" below) --
     they test different things and answering "did the cross-model
     comparison change" shouldn't require diffing a sanity-check figure.

--metrics selects which agreement statistic is used for both comparisons
(default: spearman):
  - spearman: Spearman rank correlation (Pearson-on-ranks: rankdata each map
    once, then a single np.corrcoef over the stacked rank matrix, rather
    than calling scipy.stats.spearmanr per pair -- mathematically identical,
    but avoids O(n_maps^2) redundant rank computations). Range [-1, 1].
  - pearson: Pearson correlation coefficient on the raw (unranked) values.
    Range [-1, 1]. Unlike Spearman, sensitive to the linearity (not just
    monotonicity) of the relationship between two maps.
  - mutual_information: normalized mutual information (sklearn's
    normalized_mutual_info_score) on each map discretized into --mi-bins
    quantile bins of its own distribution -- the standard histogram-based MI
    approach used in image registration (e.g. ANTs' Mattes MI). Captures
    non-monotonic relationships correlation can miss entirely. Range [0, 1],
    1.0 on the diagonal (a map always has perfect mutual information with
    itself). Not vectorizable across all pairs at once the way correlation
    is (see COST NOTE) -- computed pairwise.
  - ssd: sum of squared differences between two maps' raw voxel values,
    Range [0, inf), 0 = identical, with NO upper bound -- this is a
    dissimilarity (not similarity) measure, the only one of the four where
    LOWER means more alike; every plot/label below says so explicitly where
    it matters (see METRIC_HAS_ZERO_BASELINE). Like raw gradient magnitude
    (see visualize_saliency.py's module docstring), SSD is sensitive to each
    model's own scale, so cross-model SSD comparisons conflate "different
    magnitude" with "different pattern" -- treat absolute SSD values as
    meaningful mainly within-model (e.g. one subject's baseline vs. that
    same model's sanity-check maps), not as a magnitude-free score the way
    the other three metrics are.
  - ssim: structural similarity (Wang et al., 2004), the one metric here
    that is LOCAL/windowed rather than a single global statistic over every
    voxel independently -- local mean/variance/covariance over a sliding
    window (skimage's structural_similarity, 7-voxel default window),
    combined into a per-voxel SSIM map, then averaged (over in-brain voxels
    only when --brain-mask is set, same as every other metric here -- see
    score_pair). Every other metric here is purely POINTWISE (one voxel's
    value in map A vs. the SAME voxel's value in map B, with no notion of
    its neighbors), so SSIM answers a genuinely different question --
    "do these maps have the same local SPATIAL PATTERN" -- rather than
    "are individual voxels numerically similar." Range [-1, 1], 1.0 on the
    diagonal. This is also one of the metrics Adebayo et al. (2018,
    "Sanity Checks for Saliency Maps") themselves use, unlike the other
    four here. Requires reconstructing each map's true (D, H, W) volume
    from its flattened/masked vector (see prepare_vector's "ssim" branch,
    and _SSIM_TEMPLATE_SHAPE/_SSIM_BRAIN_MASK) -- the only metric here
    that needs that step, since it's the only one where voxel adjacency
    matters. Like mutual_information/ssd, not vectorizable across all
    pairs at once (see COST NOTE) -- computed pairwise, and meaningfully
    more expensive per pair than either (multiple windowed
    convolutions/filters per pair, not one histogram or one sum).

Both comparisons are computed separately per attribution method
(--attribution-tags, default all 4: gradient aka vanillagrad,
gradient_x_input aka gxi, integrated_gradients with a zero baseline aka
ig_zero, with the population-mean baseline aka ig_reference -- see
calculate_gradcam.py's attribution_method_tag) -- these are genuinely
different saliency maps, not variations on one map, so pooling them into a
single score would average over that difference rather than reveal it. Each
attribution method gets its own CSV and its own subplot panel in the
combined figures.

COST NOTE: this multiplies resampling/scoring work by the number of
--attribution-tags (4x at the default) -- each tag's baseline/sanity-check
maps are genuinely different files needing their own resample, not something
the vector_cache below can share across tags. Additionally, spearman/pearson
use a vectorized pairwise computation (one np.corrcoef call over every map
at once); mutual_information/ssd/ssim have no equivalent single-call trick
and are computed with an explicit O(n_maps^2) pairwise loop instead -- expect
the pairwise (not sanity-check) comparison to be noticeably slower for those
three metrics once n_maps is in the dozens, and ssim in particular is the
slowest of the three per pair (windowed filtering, not one histogram/sum --
see --metrics' own "ssim" entry above). compute_saliency_correlations_gpu.py
moves ssim (like mutual_information/ssd) onto GPU for real speedup at
--level subject; see that file's own module docstring.

Reuses visualize_saliency.py's template/resampling machinery so every
comparison happens on the identical grid used for the figures. --brain-mask
is OFF by default (every voxel in the resampled volume is used, background
included) -- pass --brain-mask (optionally with a path; bare --brain-mask
uses the template's own desc-brain_mask.nii.gz) to restrict to brain-masked
voxels instead, which is worth doing deliberately: background/skull-stripped
padding is trivially tied (exactly 0 across every map being compared), which
inflates agreement (correlation and MI alike) unless masked out. Whenever a
mask is applied, every output filename gets a "_masked" suffix (e.g.
sanity_check_spearman_signed_masked.png) so masked and unmasked runs of the
same --kind/metric never overwrite each other.

Also generates figures from those same CSVs (skip computation and just
replot with --skip-compute, e.g. after tweaking style):
  - pairwise_model_subject_{metric}_{kind}.png: one heatmap subplot per
    attribution method (2x2 grid), rows/cols grouped by model so same-model
    blocks are contiguous.
  - sanity_check_{metric}_{kind}.png: one per-model boxplot subplot per
    attribution method (2x2 grid; one box per test -- arch/data/input --
    overlaid with each subject's own point) of baseline-vs-sanity-check
    agreement.
  - sanity_check_{metric}_{kind}.html: plotly interactive version of the
    above -- same data, but hovering a point shows exactly which subject it
    is (matplotlib's static PNG can't do this; the module docstring for
    plot_sanity_check_interactive has the fallback-format reasoning).
  - cross_model_heatmap_{metric}_{kind}.png: one model x model heatmap
    subplot per attribution method (2x2 grid) of each model PAIR's MEAN
    score across subjects -- "at a glance, which models produce similar
    maps to which others", same visual idiom as the pairwise heatmap above
    but collapsed to one cell per model pair instead of one per
    (model, subject) pair.
  - cross_model_boxplot_{metric}_{kind}.png: one per-model-pair boxplot
    subplot per attribution method (2x2 grid; one box per model pair,
    overlaid with each subject's own point) -- the full across-subject
    spread the heatmap's mean alone collapses away (two model pairs can
    share the same mean while one is tight across subjects and the other
    swings from strongly agreeing to strongly disagreeing subject to
    subject).

--level controls the granularity of comparisons #1 and #3 above (#2, the
sanity-check comparison, is always per-subject regardless):
  - aggregate (default): each model's baseline maps are averaged across
    every subject FIRST (one mean map per model), then compared -- an
    N_models x N_models matrix (single digits), computed directly with no
    need for compute_pairwise_matrix's block/parallel machinery. Cross-model
    (#3) is then just that matrix's off-diagonal, since there's no subject
    dimension left to derive "same subject" pairs from. Cheap, and the
    default because most uses of this script want "do these models agree on
    average", not a full subject-resolved matrix.
  - subject: the original per-(model, subject) comparison -- see
    compute_pairwise_matrix's docstring for its block-parallel, disk-backed
    approach to keeping that much larger matrix's memory bounded.
See compute_aggregate_baseline_vectors/compute_aggregate_pairwise_matrix/
extract_cross_model_pairs_aggregate for the aggregate-level implementation,
mirroring resolve_baseline_vector/compute_pairwise_matrix/
extract_cross_model_pairs on the subject-level side.

Usage:
    python compute_saliency_correlations.py --kind signed
    python compute_saliency_correlations.py --kind signed --metrics mutual_information
    python compute_saliency_correlations.py --kind signed --skip-compute
    python compute_saliency_correlations.py --kind abs --attribution-tags gradient gxi
    python compute_saliency_correlations.py --kind signed --level subject
"""

import argparse
import hashlib
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import threadpoolctl
from plotly.subplots import make_subplots
from scipy.stats import rankdata, ttest_1samp
from skimage.metrics import structural_similarity
from sklearn.metrics import normalized_mutual_info_score

from visualize_saliency import (
    DEFAULT_BRAIN_MASK,
    DEFAULT_SALIENCY_DIR,
    DEFAULT_TEMPLATE,
    _ATTRIBUTION_DISPLAY,
    load_brain_mask,
    load_canonical_template,
    resample_saliency,
)

# Agreement/dissimilarity metrics available via --metrics (see module
# docstring for what each one means and its range).
METRICS = ("spearman", "pearson", "mutual_information", "ssd", "ssim")

METRIC_DISPLAY = {
    "spearman": "Spearman correlation",
    "pearson": "Pearson correlation",
    "mutual_information": "Normalized mutual information",
    "ssd": "Sum of squared differences",
    "ssim": "Structural similarity (SSIM)",
}

# Whether 0 is the metric's meaningful "chance"/independent baseline that a
# sanity-check score should move TOWARD once the model/labels are
# randomized (True: spearman/pearson/MI -- an uninformative map is
# uncorrelated/independent, i.e. near 0), or not (False: ssd -- an
# uninformative map is typically MORE different from the baseline, i.e. a
# LARGE positive SSD, not 0; 0 for SSD means "identical", which is the
# opposite of what randomization is expected to produce). Controls whether
# draw_sanity_check_panel/plot_sanity_check_interactive draw a 0 reference
# line.
METRIC_HAS_ZERO_BASELINE = {"spearman": True, "pearson": True, "mutual_information": True, "ssd": False, "ssim": True}

# Fixed (vmin, vmax) for the pairwise heatmap color scale, or None to derive
# it from the data (SSD has no universal upper bound the way a correlation
# coefficient or normalized MI does -- see draw_pairwise_heatmap). SSIM is
# bounded [-1, 1] by construction (same range as Pearson/Spearman), so it
# gets the same fixed range.
METRIC_FIXED_RANGE = {
    "spearman": (-1.0, 1.0), "pearson": (-1.0, 1.0), "mutual_information": (0.0, 1.0), "ssd": None,
    "ssim": (-1.0, 1.0),
}

# Diverging (centered at 0) for signed correlation-like metrics (SSIM
# included -- same [-1, 1] range/interpretation as Pearson/Spearman);
# sequential (low->high, no meaningful negative/center) for MI/SSD.
METRIC_CMAP = {
    "spearman": "coolwarm", "pearson": "coolwarm", "mutual_information": "viridis", "ssd": "viridis",
    "ssim": "coolwarm",
}

DEFAULT_MI_BINS = 32

# The 3 optim_uncnn channels (raw/rank/sobel -- see UnCNNDataset in
# train_linear.py) always co-exist per subject (one forward pass fans out
# into all 3), but are heterogeneous units (raw intensity, rank-filtered
# value, Sobel gradient magnitude -- see average_uncnn_channels.py's
# CAVEAT). UNCNN_AVG_MODEL is a synthetic model name, never backed by an
# on-disk baseline/sanity file: resolve_baseline_vector/resolve_sanity_vector
# below build it in-memory as a per-voxel mean of the 3 channels' already-
# resampled vectors, so model-to-model comparisons (pairwise + cross-model,
# both derived from the same labels) see one optim_uncnn_avg row instead of
# 3 separate channel rows.
UNCNN_CHANNELS = ("optim_uncnn_raw", "optim_uncnn_rank", "optim_uncnn_sobel")
UNCNN_AVG_MODEL = "optim_uncnn_avg"

# fs_roi has no arch_/data_/input_ sanity-check maps (calculate_gradcam_
# tabular.py doesn't implement --randomize-model/--randomize-data/
# --randomize-input yet -- see slurm/compute_sanity_checks.sh) -- it still
# gets a pairwise/cross-model row like every other model (its baseline maps
# exist and resample onto the same template via resample_saliency), it just
# has no sanity-check rows of its own: sanity_filenames(fs_roi, ...) resolves
# to files that don't exist, so its sanity_check_*.csv columns come back
# blank (see the "no {test} sanity-check map found" warning in
# compute_correlations), and permutation_null_significance's null pool for
# any (fs_roi, X) pair falls back to X's own sanity-check scores alone.
# Matches the same graceful-degradation precedent visualize_aggregate.sh
# already established for fs_roi (and for optim_uncnn_avg, which has the
# same no-sanity-files gap).
MODELS = [
    UNCNN_AVG_MODEL,
    "brainiac",
    "swinbrain",
    "simclr",
    "anatcl_global",
    "anatcl_local",
    "fs_roi",
]

# --models/--exclude-models also accept the individual optim_uncnn channels
# (not in the default MODELS list) for debugging a single channel in
# isolation -- see MODELS' docstring-comment above.
MODEL_CHOICES = MODELS + list(UNCNN_CHANNELS)

# calculate_gradcam.py's attribution_method_tag() tags, plus "gradient" for
# its own untagged default -- matches bin_saliency.py's ATTRIBUTION_TAGS/
# visualize_saliency.py's split_attribution_tag exactly. Filename fragment is
# "" for gradient (calculate_gradcam.py never inserts a tag for it) and
# "{tag}_" for the other three, inserted directly before the kind
# ("abs"/"signed") in every saliency filename this script reads.
ATTRIBUTION_TAGS = ("gradient", "gxi", "ig_zero", "ig_reference")
ATTRIBUTION_FILENAME_PREFIX = {"gradient": "", "gxi": "gxi_", "ig_zero": "ig_zero_", "ig_reference": "ig_reference_"}


def _center_unit(vec):
    """
    Mean-centered, L2-normalized copy of `vec` -- the dot product of two
    such vectors equals their Pearson correlation coefficient (the same
    center/scale identity np.corrcoef uses internally: corr(x, y) =
    (x-xbar).(y-ybar) / (||x-xbar|| * ||y-ybar||)). Putting every prepared
    spearman/pearson vector into this form once is what lets
    compute_pairwise_matrix replace one big np.corrcoef call (or a per-pair
    scipy call) with a plain matrix multiply over blocks of vectors, without
    ever holding every vector in RAM at once -- see that function's
    docstring and the module docstring's memory note.

    NaN (matching np.corrcoef's own behavior) for a constant vector, whose
    correlation with anything is undefined.
    """
    centered = vec.astype(np.float64) - vec.mean()
    norm = np.linalg.norm(centered)
    if norm == 0:
        return np.full_like(centered, np.nan, dtype=np.float32)
    return (centered / norm).astype(np.float32)


# SSIM's context (the template grid shape, and the brain mask it was
# resampled onto, if any) -- set ONCE per run via set_ssim_context(), then
# read by prepare_vector's/score_pair's "ssim" branches below, rather than
# threading two more parameters through every prepare_vector/score_pair call
# site (compute_correlations's several closures, compute_aggregate_pairwise_
# matrix, _pairwise_block_worker's per-pair loop, the GPU script's own
# reuse of prepare_vector). Mirrors _worker_template_img/_worker_brain_mask's
# existing module-global pattern above (used for the SAME reason in the
# resampling worker pool) rather than introducing a new idiom. A
# ProcessPoolExecutor worker (see _init_pairwise_worker) sets its own copy
# via the same setter, passed through initargs -- never relies on fork
# inheriting the main process's globals, for the same portability reason
# _init_worker's own initargs exist.
_SSIM_TEMPLATE_SHAPE = None
_SSIM_BRAIN_MASK = None


def set_ssim_context(template_shape, brain_mask):
    """Sets the module-global template shape/brain mask prepare_vector/score_pair's "ssim" branches read. See their docstrings."""
    global _SSIM_TEMPLATE_SHAPE, _SSIM_BRAIN_MASK
    _SSIM_TEMPLATE_SHAPE = template_shape
    _SSIM_BRAIN_MASK = brain_mask


def prepare_vector(vec, metric, mi_bins=DEFAULT_MI_BINS):
    """
    Per-file transform applied once before pairwise scoring (cached by the
    caller -- see compute_correlations's prepared_cache), specific to
    `metric`:
      - spearman: rankdata (Pearson-on-ranks == Spearman; see module
        docstring for why this beats calling scipy.stats.spearmanr per pair),
        then mean-centered/L2-normalized (see _center_unit).
      - pearson: mean-centered/L2-normalized raw values (see _center_unit) --
        Pearson is scale/shift invariant, so this loses no information but
        turns "correlation between two vectors" into "dot product between
        two vectors", which is what makes compute_pairwise_matrix's blocked
        matmul possible.
      - ssd: identity (raw values) -- SSD is defined directly on raw values.
      - mutual_information: quantile-digitized into `mi_bins` bins of this
        vector's OWN distribution (each file gets its own bin edges, the
        standard histogram-MI convention -- e.g. ANTs' Mattes MI) --
        normalized_mutual_info_score needs discrete labels, not continuous
        values.
      - ssim: reconstructs the true (D, H, W) volume `vec` was flattened
        from (see apply_mask/_SSIM_TEMPLATE_SHAPE/_SSIM_BRAIN_MASK) --
        unlike every other metric here, SSIM is a WINDOWED/local statistic
        (see score_pair), so it needs real spatial adjacency between
        voxels, which a flat vector doesn't preserve (a boolean-masked
        selection in particular scrambles it entirely). If a brain mask
        was used, this scatters `vec`'s values back into the masked
        positions of a zero-filled volume -- exact, not an approximation:
        apply_mask's masked branch (data[brain_mask]) only ever DROPS the
        background voxels this reconstructs as 0, it never modifies the
        in-brain ones, and background is already ~0 in the source resampled
        map (skull-stripped) anyway. Unmasked, this is just vec.reshape(...)
        (ravel() is exactly invertible).
    """
    if metric == "spearman":
        return _center_unit(rankdata(vec))
    if metric == "pearson":
        return _center_unit(vec)
    if metric == "ssim":
        if _SSIM_BRAIN_MASK is not None:
            vol = np.zeros(_SSIM_TEMPLATE_SHAPE, dtype=np.float32)
            vol[_SSIM_BRAIN_MASK] = vec
            return vol
        return vec.reshape(_SSIM_TEMPLATE_SHAPE).astype(np.float32)
    if metric == "mutual_information":
        edges = np.quantile(vec, np.linspace(0.0, 1.0, mi_bins + 1))
        # np.digitize's rightmost bin is right-open ([edges[-2], edges[-1]));
        # nudge the true minimum down slightly so it doesn't fall below the
        # first interior edge and get digitized as bin -1's neighbor.
        edges[0] -= 1e-6
        # np.digitize defaults to int64; bin indices only ever range
        # 0..mi_bins-1 (mi_bins is expected well under 256), so uint8 is
        # exact and cuts this vector's memmap/disk footprint 8x -- material
        # at ~7M voxels/vector times thousands of labels (see compute_
        # pairwise_matrix's block-parallel scoring, which is what made this
        # 8x actually matter: N concurrent workers each touching full-size
        # blocks at once).
        return np.digitize(vec, edges[1:-1]).astype(np.uint8)
    return vec  # ssd


def score_pair(prepared_a, prepared_b, metric):
    """
    Agreement/dissimilarity between two ALREADY-prepared vectors (see
    prepare_vector) for `metric`. spearman/pearson are both already
    mean-centered/L2-normalized (see _center_unit), so their "correlation"
    is just a dot product -- compute_pairwise_matrix's blocked path uses the
    same identity directly (a block matrix multiply) rather than calling
    this function once per pair; this per-pair form is used directly for
    mutual_information/ssd (no equivalent single-call vectorization exists
    for either, see module docstring COST NOTE) and for every metric in the
    smaller sanity-check comparison.
    """
    if metric in ("spearman", "pearson"):
        # Accumulate in float64 even though the prepared vectors are stored
        # as float32 on disk (see _center_unit) -- matches np.corrcoef's own
        # internal promotion (numpy's cov/corrcoef always computes in at
        # least float64), which a plain float32 dot product over ~7M-length
        # vectors would drift from measurably (verified against the
        # pre-refactor np.corrcoef output: float32 accumulation alone showed
        # up to ~7e-4 error on real data, float64 accumulation matches to
        # ~1e-6).
        return float(np.dot(prepared_a.astype(np.float64), prepared_b.astype(np.float64)))
    if metric == "mutual_information":
        return float(normalized_mutual_info_score(prepared_a, prepared_b))
    if metric == "ssd":
        return float(np.sum((prepared_a - prepared_b) ** 2))
    if metric == "ssim":
        # prepared_a/prepared_b are the reconstructed (D, H, W) volumes (see
        # prepare_vector's "ssim" branch), not flat vectors -- SSIM's local
        # windows need real spatial adjacency. data_range is computed PER
        # PAIR from the two volumes' own combined min/max (skimage requires
        # an explicit data_range for float input; there's no fixed [0, 255]-
        # style range here the way there is for 8-bit images) -- an exact
        # (self-vs-self) 0 range means two identical, typically all-zero
        # (fully masked-out) volumes, which SSIM=1 trivially/correctly
        # represents rather than dividing by zero.
        lo = min(float(prepared_a.min()), float(prepared_b.min()))
        hi = max(float(prepared_a.max()), float(prepared_b.max()))
        data_range = hi - lo
        if data_range == 0:
            return 1.0
        if _SSIM_BRAIN_MASK is not None:
            # Restrict the averaged score to in-brain voxels (see
            # _SSIM_BRAIN_MASK/prepare_vector) -- the reconstructed volumes'
            # zero-filled background would otherwise trivially/identically
            # agree between ANY two maps and dominate the windowed average,
            # the exact same background-inflation problem --brain-mask
            # already guards against for every other metric here (see
            # apply_mask's docstring) -- full=True gets the whole per-voxel
            # SSIM map instead of just its own (unmasked) global mean so
            # that restriction is possible.
            _, ssim_map = structural_similarity(prepared_a, prepared_b, data_range=data_range, full=True)
            return float(ssim_map[_SSIM_BRAIN_MASK].mean())
        return float(structural_similarity(prepared_a, prepared_b, data_range=data_range))
    raise ValueError(f"Unknown metric: {metric!r}")


def sanity_filenames(model, tag, kind):
    """
    Exact arch/data/input sanity-check filenames for (model, attribution
    tag, kind) -- same prefix construction as visualize_subjects.sh
    (including anatcl's and optim_uncnn's channel special cases), with the
    attribution tag's filename fragment inserted directly before the kind,
    matching calculate_gradcam.py's compute_and_save_saliency exactly. No
    wildcard/glob needed: calculate_gradcam.py's naming is fully
    deterministic once (model, tag, kind, test) are known.
    """
    tag_prefix = ATTRIBUTION_FILENAME_PREFIX[tag]
    if model.startswith("anatcl_"):
        level = model[len("anatcl_") :]
        stems = {
            "arch": f"arch_anatcl_all_{level}_",
            "data": f"data_anatcl_{level}_",
            "input": f"input_anatcl_permute_{level}_",
        }
    elif model.startswith("optim_uncnn_"):
        channel = model[len("optim_uncnn_") :]
        stems = {
            "arch": f"arch_optim_uncnn_all_{channel}_",
            "data": f"data_optim_uncnn_{channel}_",
            "input": f"input_optim_uncnn_permute_{channel}_",
        }
    else:
        stems = {
            "arch": f"arch_{model}_all_",
            "data": f"data_{model}_",
            "input": f"input_{model}_permute_",
        }
    return {test: f"{stem}{tag_prefix}{kind}_saliency.nii.gz" for test, stem in stems.items()}


def baseline_filename(model, tag, kind):
    return f"{model}_{ATTRIBUTION_FILENAME_PREFIX[tag]}{kind}_saliency.nii.gz"


def baseline_prepare_key(model, subj_dir, tag, kind):
    """
    The same cache key load_prepared_baseline's closure computes for a
    (model, subj_dir) baseline -- factored out to a top-level function (pure,
    no closure state) so compute_pairwise_matrix's parallel block workers can
    compute it too, without needing to pickle/capture the closure itself.
    See load_prepared_baseline's docstring for why UNCNN_AVG_MODEL needs a
    synthetic (not path-based) key.
    """
    if model != UNCNN_AVG_MODEL:
        return str((subj_dir / baseline_filename(model, tag, kind)).resolve())
    return f"AVG::{subj_dir}::{tag}::{kind}"


def already_on_template_grid(img, template_img, atol=1e-3):
    """
    Whether `img` already sits on `template_img`'s exact grid -- i.e.
    resample_to_img is a no-op identity resample for it, not a real
    interpolation (see check_resolutions.py, which surfaced that this
    stopped holding for every file here once DEFAULT_TEMPLATE was pointed
    at MNI152Lin instead of turboprep's own MNI152NLin2009cAsym target --
    that's very likely why this script slowed down: every single resample
    became a real trilinear interpolation over a 181x217x181 grid instead
    of a cheap/no-op copy).
    """
    return tuple(img.shape) == tuple(template_img.shape) and np.allclose(img.affine, template_img.affine, atol=atol)


# Populated once per worker process by _init_worker (not per task -- reused
# across every file that worker resamples), so parallel resampling
# (--n-jobs) doesn't reload the template/brain mask from disk per file.
_worker_template_img = None
_worker_brain_mask = None


def _init_worker(template_path, brain_mask_path):
    # Pin this worker process's own BLAS/OpenMP thread pool to 1 -- without
    # this, each of --n-jobs worker PROCESSES independently multi-threads
    # its own BLAS calls (resample_saliency's nilearn/scipy interpolation
    # does use BLAS internally), so the ambient thread-pool default (often
    # the whole node's core count, not this job's --cpus-per-task
    # allocation -- see slurm/env.sh's OMP_NUM_THREADS) gets multiplied by
    # --n-jobs, oversubscribing the allocation instead of --n-jobs
    # processes cleanly dividing it. threadpoolctl (not just OMP_NUM_THREADS)
    # because the BLAS backend here may not be OpenMP-threaded (e.g. plain
    # OpenBLAS uses its own thread pool) -- this pins whichever backend is
    # actually loaded. Persists for the worker's whole lifetime (called once
    # by the pool's initializer, not per task).
    threadpoolctl.threadpool_limits(1)
    global _worker_template_img, _worker_brain_mask
    _worker_template_img = load_canonical_template(template_path)
    _worker_brain_mask = load_brain_mask(brain_mask_path, _worker_template_img) if brain_mask_path else None


def apply_mask(data, brain_mask):
    """data[brain_mask] (brain-masked voxels only) if a mask is given, else data.ravel() (every voxel, unmasked)."""
    return data[brain_mask].astype(np.float32) if brain_mask is not None else data.ravel().astype(np.float32)


def resolve_baseline_vector(model, tag, kind, subj_dir, vector_cache):
    """
    Raw (unprepared -- see prepare_vector) baseline vector for (model, tag,
    kind, subj_dir). For a real model, a direct vector_cache lookup by
    resolved path (None if the file doesn't exist/wasn't resampled). For
    UNCNN_AVG_MODEL, the per-voxel mean of its 3 channels' baseline vectors
    for this same subject (None if any channel is missing for this subject
    -- averaging 2 of 3 would silently change what "optim_uncnn_avg" means
    subject to subject, so this is skipped rather than partially computed).
    """
    if model != UNCNN_AVG_MODEL:
        f = subj_dir / baseline_filename(model, tag, kind)
        return vector_cache.get(str(f.resolve())) if f.exists() else None
    channel_vecs = []
    for channel in UNCNN_CHANNELS:
        f = subj_dir / baseline_filename(channel, tag, kind)
        vec = vector_cache.get(str(f.resolve())) if f.exists() else None
        if vec is None:
            return None
        channel_vecs.append(vec)
    return np.mean(np.stack(channel_vecs, axis=0), axis=0)


def compute_aggregate_baseline_vectors(models, tag, kind, subject_dirs, vector_cache):
    """
    {model: mean baseline vector across every subject in `subject_dirs` that
    has one} -- the aggregate-level counterpart of resolve_baseline_vector's
    per-subject vector (see --level in the module docstring). Raw
    (unprepared -- see prepare_vector), and averaged BEFORE any per-metric
    prep (rankdata/digitizing/centering), so "aggregate" means one mean
    ANATOMICAL map per model, not an average of per-subject scores.

    A model with zero available subjects (e.g. --subjects narrowed to
    subjects that model has no baseline for) is simply omitted from the
    result rather than raising, so a partial --models/--subjects run doesn't
    kill the whole aggregate comparison over one missing model.
    """
    aggregates = {}
    for model in models:
        vecs = [
            v for v in (resolve_baseline_vector(model, tag, kind, subj_dir, vector_cache) for subj_dir in subject_dirs)
            if v is not None
        ]
        if vecs:
            aggregates[model] = np.mean(np.stack(vecs, axis=0), axis=0)
    return aggregates


def resolve_sanity_vector(model, tag, kind, test, saliency_dir, vector_cache):
    """
    Raw (unprepared) sanity-check vector for (model, tag, kind, test --
    "arch"/"data"/"input"). Mirrors resolve_baseline_vector: a direct lookup
    for a real model, or the per-voxel mean of the 3 optim_uncnn channels'
    sanity-check maps for that same test when model == UNCNN_AVG_MODEL (None
    if any channel's map is missing).
    """
    if model != UNCNN_AVG_MODEL:
        f = saliency_dir / sanity_filenames(model, tag, kind)[test]
        return vector_cache.get(str(f.resolve())) if f.exists() else None
    channel_vecs = []
    for channel in UNCNN_CHANNELS:
        f = saliency_dir / sanity_filenames(channel, tag, kind)[test]
        vec = vector_cache.get(str(f.resolve())) if f.exists() else None
        if vec is None:
            return None
        channel_vecs.append(vec)
    return np.mean(np.stack(channel_vecs, axis=0), axis=0)


def _cache_filename(cache_dir, key):
    """Stable on-disk .npy filename for a cache `key` (a resolved path string
    or a synthetic UNCNN_AVG_MODEL key) -- hashed since raw paths/keys can be
    longer than common filesystem filename limits and contain characters
    that don't belong in one."""
    return cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.npy"


class DiskVectorCache:
    """
    dict-like {key: vector} interface (.get/__getitem__/__contains__, same
    as resolve_baseline_vector/resolve_sanity_vector already expect) backed
    by .npy files under `cache_dir` instead of a plain dict holding every
    unique file's resampled vector in RAM at once.

    This is the fix for this script's dominant memory cost: with N subjects
    x M models x (baseline + up to 3 sanity-check maps), the old in-RAM
    vector_cache held all of those simultaneously for the whole run (~28MB
    each at the unmasked 181x217x181 template grid -- easily hundreds of GB
    once N is in the hundreds). Each .get()/__getitem__ call here instead
    memory-maps its one file, so a caller that only touches a bounded subset
    at a time (e.g. compute_pairwise_matrix's blocks) only ever pages in a
    bounded amount of real memory, not the full cache.
    """

    def __init__(self, cache_dir, index):
        self._cache_dir = cache_dir
        self._index = index  # {key: cache_path}

    def get(self, key, default=None):
        cache_path = self._index.get(key)
        return np.load(cache_path, mmap_mode="r") if cache_path is not None else default

    def __getitem__(self, key):
        return np.load(self._index[key], mmap_mode="r")

    def __contains__(self, key):
        return key in self._index


def _resample_worker(task):
    """Runs in a worker process (see _init_worker): resamples one file and
    writes its vector straight to the on-disk cache itself (rather than
    pickling a ~28MB array back through the process-pool pipe to the parent,
    which would briefly double memory for that file), returning only the
    small bits the parent needs to build its {key: cache_path} index."""
    path_str, key, cache_dir_str = task
    t0 = time.time()
    path = Path(path_str)
    is_identity = already_on_template_grid(nib.load(path), _worker_template_img)
    data = resample_saliency(path, _worker_template_img)
    vec = apply_mask(data, _worker_brain_mask)
    cache_path = _cache_filename(Path(cache_dir_str), key)
    np.save(cache_path, vec)
    return key, str(cache_path), time.time() - t0, is_identity


def models_to_resample(models):
    """
    `models` with UNCNN_AVG_MODEL expanded into its 3 backing channels (which
    are the actual files on disk -- see UNCNN_AVG_MODEL's docstring-comment)
    and any real channel already in `models` deduplicated against that
    expansion, so a file needed by both an explicit channel selection and
    the synthetic average is only resampled once.
    """
    out = []
    for model in models:
        candidates = UNCNN_CHANNELS if model == UNCNN_AVG_MODEL else (model,)
        for candidate in candidates:
            if candidate not in out:
                out.append(candidate)
    return out


def collect_needed_paths(args, subject_dirs, tags=None):
    """
    Every unique file this run will need to resample (baseline maps across
    `tags` (default: all --attribution-tags) x --models/subjects, plus
    sanity-check maps across the same tags/models), deduplicated by resolved
    path -- computed once up front so the resample step below
    (parallelizable, and the dominant cost -- see --n-jobs/--timing) never
    resamples the same file twice, instead of the old lazy-cache-on-first-use
    approach interleaving resampling with correlation math and hiding the
    true unique-file count.

    `tags` lets --resume narrow this to only the tags that still have
    remaining work across any requested metric (resampling depends on
    kind/tag/model, never on metric, so a tag with nothing left to compute
    for ANY metric needs no resampling at all -- see plan_remaining_work).

    UNCNN_AVG_MODEL is never itself a file on disk (see its docstring-
    comment) -- collect its 3 backing channels' files instead, via
    models_to_resample.
    """
    tags = args.attribution_tags if tags is None else tags
    models = models_to_resample(args.models)
    paths = {}
    for tag in tags:
        for subj_dir in subject_dirs:
            for model in models:
                f = subj_dir / baseline_filename(model, tag, args.kind)
                if f.exists():
                    paths[str(f.resolve())] = f
        for model in models:
            for fname in sanity_filenames(model, tag, args.kind).values():
                f = args.saliency_dir / fname
                if f.exists():
                    paths[str(f.resolve())] = f
    return paths


def resample_all(paths_by_key, args, template_img, brain_mask, cache_dir):
    """
    Resamples every path in `paths_by_key` onto `template_img`'s grid
    (sequentially, or across --n-jobs worker processes), returning a
    DiskVectorCache over `cache_dir` -- brain-masked if `brain_mask` is
    given, every voxel (unmasked) if it's None (see apply_mask). Each
    vector is written straight to its own .npy file under `cache_dir`
    instead of being accumulated in an in-RAM dict (see DiskVectorCache's
    docstring for why: that dict is what drove this script's memory use into
    the hundreds of GB). `cache_dir` is namespaced by kind/mask by the
    caller, and persists across runs, so re-running against the same
    --output-dir (e.g. under --resume, or just to add more --metrics) skips
    re-resampling any file whose cache entry already exists on disk.

    Prints per-file timing (elapsed + whether it needed a real interpolation
    or was already on the template's grid) and a final summary -- this is
    the actual "why is it slow" instrumentation: resampling dominates this
    script's runtime (see the module docstring's COST NOTE), so timing it
    file-by-file is what tells you whether it's genuinely this many files, a
    few slow ones, or (per already_on_template_grid's docstring) a
    template-grid mismatch forcing real interpolation everywhere it didn't
    used to.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_items = list(paths_by_key.items())
    index = {}
    cached_items, to_resample = [], []
    for key, path in all_items:
        cache_path = _cache_filename(cache_dir, key)
        if cache_path.exists():
            index[key] = cache_path
            cached_items.append((key, path))
        else:
            to_resample.append((key, path))

    total = len(all_items)
    print(
        f"Resampling {len(to_resample)}/{total} unique file(s) onto the template (--n-jobs={args.n_jobs}); "
        f"{len(cached_items)}/{total} already cached on disk from a previous run in {cache_dir}..."
    )
    t_start = time.time()
    per_file_timing = []

    if args.n_jobs <= 1:
        for i, (key, path) in enumerate(to_resample, 1):
            t0 = time.time()
            is_identity = already_on_template_grid(nib.load(path), template_img)
            data = resample_saliency(path, template_img)
            vec = apply_mask(data, brain_mask)
            cache_path = _cache_filename(cache_dir, key)
            np.save(cache_path, vec)
            index[key] = cache_path
            elapsed = time.time() - t0
            per_file_timing.append((path.name, elapsed, is_identity))
            if args.timing:
                print(f"  [{i}/{len(to_resample)}] {path.name} ({'identity' if is_identity else 'interpolated'}) -- {elapsed:.2f}s")
    elif to_resample:
        with ProcessPoolExecutor(
            max_workers=args.n_jobs, initializer=_init_worker, initargs=(args.template, args.brain_mask)
        ) as executor:
            tasks = [(str(path), key, str(cache_dir)) for key, path in to_resample]
            for i, (key, cache_path_str, elapsed, is_identity) in enumerate(executor.map(_resample_worker, tasks), 1):
                index[key] = Path(cache_path_str)
                per_file_timing.append((Path(key).name, elapsed, is_identity))
                if args.timing:
                    print(f"  [{i}/{len(to_resample)}] {Path(key).name} ({'identity' if is_identity else 'interpolated'}) -- {elapsed:.2f}s")

    total_elapsed = time.time() - t_start
    n_identity = sum(1 for _, _, ident in per_file_timing if ident)
    per_file_seconds = [t for _, t, _ in per_file_timing]
    print(
        f"Resampling done in {total_elapsed:.1f}s wall time "
        f"({sum(per_file_seconds):.1f}s summed across files, {'parallel' if args.n_jobs > 1 else 'sequential'}). "
        f"{n_identity}/{len(to_resample)} newly-resampled file(s) were already on the template's grid (no real "
        f"interpolation needed); {len(to_resample) - n_identity}/{len(to_resample)} needed real interpolation."
    )
    if per_file_seconds:
        slowest = sorted(per_file_timing, key=lambda row: -row[1])[:5]
        print(f"  avg {np.mean(per_file_seconds):.2f}s/file, slowest 5: "
              + ", ".join(f"{name} ({t:.2f}s)" for name, t, _ in slowest))
    return DiskVectorCache(cache_dir, index)


def csv_paths_for_tags(output_dir, stem, kind, tags, mask_suffix=""):
    """{tag: Path} for a family of per-tag CSVs sharing one {stem}_{kind}{mask_suffix}_{tag}.csv naming."""
    return {tag: output_dir / f"{stem}_{kind}{mask_suffix}_{tag}.csv" for tag in tags}


def csv_is_valid(path):
    """
    Whether `path` is a complete, parseable CSV -- used by --resume to
    decide a (metric, tag, comparison) combo is already done, rather than a
    bare path.exists(). Guards against the one way --resume could go wrong:
    if a previous run was killed (e.g. a SLURM timeout, exactly what
    prompted --resume) mid pd.to_csv(), the partially-written file exists on
    disk but is truncated/unparseable -- treating that as "done" would
    silently ship a missing or corrupt CSV into the figures instead of
    recomputing it.
    """
    if not path.exists():
        return False
    try:
        return not pd.read_csv(path, nrows=1).empty or path.stat().st_size > 0
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return False


def plan_remaining_work(args, pairwise_paths_by_metric, sanity_paths_by_metric, cross_model_paths_by_metric):
    """
    {metric: {tag: {"pairwise": bool, "sanity": bool, "cross_model": bool}}}
    -- whether each comparison still needs computing. Without --resume,
    everything is always True (unchanged behavior: full recompute every
    run). With --resume, a comparison is False (skip it) only if its CSV
    already exists and parses cleanly (see csv_is_valid) -- e.g. after a
    run that timed out partway through, already-written (metric, tag)
    combos are skipped and only the remainder is computed.

    "cross_model" is intentionally NOT folded into tags_with_remaining_work
    (see compute_correlations) -- it's derived from comparison #1's
    already-written pairwise CSV, not from vector_cache, so it never needs
    resampling even when it itself still needs computing.
    """
    plan = {}
    for metric in args.metrics:
        plan[metric] = {}
        for tag in args.attribution_tags:
            if args.resume:
                pairwise_done = csv_is_valid(pairwise_paths_by_metric[metric][tag])
                sanity_done = csv_is_valid(sanity_paths_by_metric[metric][tag])
                cross_model_done = csv_is_valid(cross_model_paths_by_metric[metric][tag])
            else:
                pairwise_done = sanity_done = cross_model_done = False
            plan[metric][tag] = {"pairwise": not pairwise_done, "sanity": not sanity_done, "cross_model": not cross_model_done}
    return plan


def draw_pairwise_heatmap(
    ax, df, title, metric, pair_annotations=None, no_title=False, tick_label_scale=1.0, annotation_scale=1.0,
):
    """
    Draws one tag's pairwise heatmap onto `ax`. Returns the imshow AxesImage
    (for the shared colorbar). Color scale/colormap come from METRIC_FIXED_RANGE/
    METRIC_CMAP: spearman/pearson get the fixed [-1, 1] diverging scale;
    mutual_information gets the fixed [0, 1] normalized-MI scale; ssd (no
    universal upper bound) gets a data-driven [0, 99th percentile] scale
    instead, computed off-diagonal only (the diagonal is trivially 0 for SSD
    -- self vs. self -- and would otherwise compress the scale for everything
    else).

    `pair_annotations` (optional): {(model_a, model_b): {"fisher", "perm"}}
    from combine_significance -- if given, each OFF-diagonal model-vs-model
    block gets both tests' significance stars overlaid at its center (see
    draw_pair_annotation for how the two are visually distinguished).
    Same-model (diagonal) blocks are skipped -- a model's trivial agreement
    with itself isn't a "model pair" comparison.
    """
    labels = sorted(df.index, key=lambda label: tuple(label.split("__", 1)))
    df = df.loc[labels, labels]
    models_in_order = [label.split("__", 1)[0] for label in labels]

    fixed_range = METRIC_FIXED_RANGE[metric]
    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        off_diagonal = df.values[~np.eye(df.shape[0], dtype=bool)]
        vmin = 0.0
        vmax = float(np.percentile(off_diagonal, 99)) if off_diagonal.size else 1.0
    # Upper triangle + diagonal only -- the matrix is symmetric (df.loc[a, b]
    # == df.loc[b, a] always), so the lower triangle is a pure duplicate;
    # masked to NaN here (rendered blank/transparent by imshow) rather than
    # dropped from `df` so labels/ticks/annotation positions below are
    # unaffected.
    masked_values = np.where(np.triu(np.ones(df.shape, dtype=bool)), df.values, np.nan)
    im = ax.imshow(masked_values, cmap=METRIC_CMAP[metric], vmin=vmin, vmax=vmax)

    boundaries = [0] + [i for i in range(1, len(labels)) if models_in_order[i] != models_in_order[i - 1]] + [len(labels)]
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.8)
        ax.axvline(b - 0.5, color="black", linewidth=0.8)
    tick_pos = [(boundaries[i] + boundaries[i + 1] - 1) / 2 for i in range(len(boundaries) - 1)]
    tick_labels = [models_in_order[boundaries[i]] for i in range(len(boundaries) - 1)]
    ax.set_xticks(tick_pos, labels=tick_labels, rotation=45, ha="right", fontsize=8 * tick_label_scale)
    ax.set_yticks(tick_pos, labels=tick_labels, fontsize=8 * tick_label_scale)
    if not no_title:
        ax.set_title(title, fontsize=11)

    if pair_annotations:
        n_blocks = len(tick_labels)
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                model_a, model_b = sorted((tick_labels[i], tick_labels[j]))
                annotation = pair_annotations.get((model_a, model_b))
                if not annotation:
                    continue
                # Upper triangle only (i < j here, matching the masked
                # imshow above) -- no mirrored lower-triangle draw.
                draw_pair_annotation(ax, tick_pos[j], tick_pos[i], annotation, annotation_scale=annotation_scale)
    return im


def extract_cross_model_pairs(pairwise_df):
    """
    From comparison #1's full (model, subject) x (model, subject) matrix,
    extract only SAME-subject, DIFFERENT-model pairs -- "how similar are
    two different models' saliency maps for the same person's scan",
    distinct from the full matrix (which also mixes in different-subject
    pairs) and from the sanity-check comparison (same model, real vs.
    randomized). Returns a long-form DataFrame, one row per (subject,
    model_a, model_b, score), model_a < model_b alphabetically so each
    unordered pair appears exactly once per subject (not twice, and never
    a model against itself).
    """
    labels = pairwise_df.index.tolist()
    by_subject = defaultdict(list)
    for label in labels:
        model, subject = label.split("__", 1)
        by_subject[subject].append((model, label))

    rows = []
    for subject, model_labels in by_subject.items():
        model_labels.sort()  # alphabetical by model name -> stable, deduplicated pair order
        for i in range(len(model_labels)):
            for j in range(i + 1, len(model_labels)):
                model_a, label_a = model_labels[i]
                model_b, label_b = model_labels[j]
                rows.append(
                    {"subject": subject, "model_a": model_a, "model_b": model_b, "score": pairwise_df.loc[label_a, label_b]}
                )
    return pd.DataFrame(rows)


def compute_aggregate_pairwise_matrix(aggregate_vectors, metric, mi_bins=DEFAULT_MI_BINS):
    """
    Aggregate-level counterpart of compute_pairwise_matrix: a small
    n_models x n_models agreement DataFrame (index/columns = model names)
    over `aggregate_vectors` (see compute_aggregate_baseline_vectors) --
    one row per model instead of one per (model, subject) baseline map.
    n_models is always small (single digits, see MODELS), so unlike the
    subject-level matrix this is computed directly in one pass, with no
    need for block_size/n_jobs/on-disk prepared-vector caching.
    """
    labels = sorted(aggregate_vectors)
    prepared = {label: prepare_vector(aggregate_vectors[label], metric, mi_bins) for label in labels}
    n = len(labels)
    corr = np.empty((n, n), dtype=np.float64)
    for i, label_i in enumerate(labels):
        for j in range(i, n):
            label_j = labels[j]
            score = score_pair(prepared[label_i], prepared[label_j], metric)
            corr[i, j] = score
            corr[j, i] = score
    return pd.DataFrame(corr, index=labels, columns=labels)


def extract_cross_model_pairs_aggregate(pairwise_df):
    """
    Aggregate-level counterpart of extract_cross_model_pairs: every
    DIFFERENT-model pair's score from an aggregate (model x model) matrix
    (see compute_aggregate_pairwise_matrix) -- i.e. that matrix's upper
    triangle, model_a < model_b so each unordered pair appears exactly once.
    There's no subject dimension to match same-subject pairs on here (each
    row/column is already a per-model mean across every subject), unlike
    extract_cross_model_pairs's per-subject grouping -- so this returns one
    row per model pair, not one row per (subject, model pair).
    """
    labels = pairwise_df.index.tolist()
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            model_a, model_b = sorted((labels[i], labels[j]))
            rows.append({"model_a": model_a, "model_b": model_b, "score": pairwise_df.loc[model_a, model_b]})
    return pd.DataFrame(rows)


def fisher_z_significance(cross_df, metric):
    """
    One-sample t-test vs. 0 on Fisher-z-transformed scores (z = arctanh(r)),
    per model pair, across that pair's per-subject scores in `cross_df` (see
    extract_cross_model_pairs). Testing z rather than the raw score directly
    fixes the normality violation of running a t-test on a bounded, skewed
    correlation coefficient: z's sampling distribution is approximately
    normal where r's is not, especially once |r| is large (exactly the
    regime this pipeline's unmasked-background inflation pushes scores
    into -- see the module docstring).

    Only valid for a score bounded in (-1, 1) that behaves like a signed
    correlation coefficient -- spearman/pearson directly, and ssim too
    (also [-1, 1], and METRIC_HAS_ZERO_BASELINE/METRIC_FIXED_RANGE/
    METRIC_CMAP already treat it the same way for exactly this reason, even
    though it's a local/windowed statistic rather than a global one -- see
    its own module-docstring entry). Returns {} (not a dict of nans) for
    mutual_information (bounded [0, 1], not a correlation) and ssd
    (unbounded dissimilarity), so callers render no "z:" line at all for
    those metrics rather than a misleading one.

    Returns {(model_a, model_b): {"mean_r", "t_stat", "p_value", "n"}}.
    Scores are clipped away from exactly +-1 before the transform
    (arctanh(+-1) = +-inf) -- a perfect +-1 correlation is vanishingly rare
    on real data but would otherwise blow up the whole test for that pair.
    n<2 -> nan t_stat/p_value (can't estimate a variance from 1 subject).
    """
    if metric not in ("spearman", "pearson", "ssim"):
        return {}
    stats = {}
    for (model_a, model_b), group in cross_df.groupby(["model_a", "model_b"]):
        scores = group["score"].dropna().to_numpy()
        if scores.size == 0:
            continue
        mean_r = float(np.mean(scores))
        z = np.arctanh(np.clip(scores, -1 + 1e-6, 1 - 1e-6))
        if scores.size < 2:
            t_stat, p_value = float("nan"), float("nan")
        else:
            result = ttest_1samp(z, popmean=0.0)
            t_stat, p_value = float(result.statistic), float(result.pvalue)
        stats[(model_a, model_b)] = {"mean_r": mean_r, "t_stat": t_stat, "p_value": p_value, "n": scores.size}
    return stats


def sanity_null_pool(sanity_df, model_a, model_b):
    """
    Pooled score_vs_arch/score_vs_data/score_vs_input values (see the
    sanity-check CSV built in compute_correlations's section 2b) for
    `model_a` and `model_b` -- an empirical stand-in for "how similar does a
    REAL baseline map look next to a RANDOMIZED (no genuine signal)
    counterpart of the SAME model", used as the null/chance-floor
    distribution cross-model agreement is tested against in
    permutation_null_significance. This reuses maps this pipeline already
    computes rather than testing against a literal 0, which real, unmasked
    anatomical maps clear trivially regardless of genuine cross-model
    agreement (see module docstring).
    """
    if sanity_df is None:
        return np.array([])
    cols = [c for c in ("score_vs_arch", "score_vs_data", "score_vs_input") if c in sanity_df.columns]
    if not cols:
        return np.array([])
    pooled = sanity_df.loc[sanity_df["model"].isin([model_a, model_b]), cols].to_numpy(dtype=float).ravel()
    return pooled[~np.isnan(pooled)]


def permutation_null_significance(cross_df, sanity_df, metric):
    """
    Monte-Carlo p-value per model pair: where the pair's REAL mean
    cross-model score falls relative to the pooled empirical null built
    from that same tag/metric's sanity-check maps (see sanity_null_pool) --
    "does this pair agree more than the noise floor either model shows
    against its own randomized counterpart", instead of "is this nonzero"
    (see module docstring for why 0 isn't a credible chance level here).
    Works for every metric, including mutual_information/ssd, unlike
    fisher_z_significance -- it makes no distributional assumption about
    the score itself, only about which direction counts as "more similar"
    (higher for spearman/pearson/mutual_information, lower for ssd -- see
    METRIC_HAS_ZERO_BASELINE's docstring for the same higher/lower-is-better
    split).

    p-value uses the standard add-one Monte Carlo correction (North et al.,
    2002), so it's never exactly 0 regardless of null-sample size:
    (# null values at least as extreme as the observed mean + 1) / (n_null + 1).

    Returns {(model_a, model_b): {"mean_score", "n_null", "p_value"}}.
    p_value is nan (not skipped) when the null pool is empty (no
    sanity-check maps found for either model in this tag) -- a caller needs
    to distinguish "tested and undefined" from "never attempted".
    """
    higher_is_more_similar = metric != "ssd"
    stats = {}
    for (model_a, model_b), group in cross_df.groupby(["model_a", "model_b"]):
        scores = group["score"].dropna().to_numpy()
        if scores.size == 0:
            continue
        observed = float(np.mean(scores))
        null_pool = sanity_null_pool(sanity_df, model_a, model_b)
        if null_pool.size == 0:
            stats[(model_a, model_b)] = {"mean_score": observed, "n_null": 0, "p_value": float("nan")}
            continue
        n_extreme = int(np.sum(null_pool >= observed)) if higher_is_more_similar else int(np.sum(null_pool <= observed))
        p_value = (n_extreme + 1) / (null_pool.size + 1)
        stats[(model_a, model_b)] = {"mean_score": observed, "n_null": int(null_pool.size), "p_value": p_value}
    return stats


def fdr_correct(p_values):
    """
    Benjamini-Hochberg FDR-adjusted p-values for one FAMILY of tests (e.g.
    every model pair's Fisher-z test within one tag/metric's heatmap panel)
    -- each subplot's (tag, metric, test-type) combination is its own
    family, corrected independently, never pooled across separate panels.
    Running dozens of per-pair tests per figure without this would let
    ~5% of truly-null pairs show a spurious "*" by chance alone at
    alpha=0.05 (see module docstring's significance-testing review).

    `p_values`: {key: p_value}, some possibly nan (an undefined test, e.g.
    n<2 subjects or an empty null pool) -- nans are excluded from the
    correction (an undefined test isn't a real trial in the family) and
    passed through unchanged. A no-op (returns `p_values` as-is) when 0 or 1
    valid p-values are given -- no multiple-comparisons problem exists for
    a single test.
    """
    keys = list(p_values.keys())
    if not keys:
        return dict(p_values)
    raw = np.array([p_values[k] for k in keys], dtype=float)
    valid = ~np.isnan(raw)
    if valid.sum() <= 1:
        return dict(p_values)
    idx = np.where(valid)[0]
    pvals = raw[idx]
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = ranked.size
    bh = ranked * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]  # BH step-up: enforce monotonicity from the largest rank down
    bh = np.clip(bh, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = bh
    out = dict(p_values)
    for i, p_adj in zip(idx, adjusted):
        out[keys[i]] = float(p_adj)
    return out


def significance_stars(p_value):
    """Conventional star notation for a p-value; "ns" if >= 0.05, "n/a" if p_value is nan (test attempted but undefined, e.g. n<2 subjects or an empty null pool)."""
    if p_value != p_value:  # nan
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


_FISHER_ANNOTATION_COLOR = "#08519c"  # dark blue -- Fisher-z one-sample t-test
_PERM_ANNOTATION_COLOR = "#a63603"  # dark orange -- permutation test vs. sanity-check null


def combine_significance(cross_df, sanity_df, metric):
    """
    Runs BOTH significance tests this pipeline supports for one tag/metric's
    model pairs and FDR-corrects each test's own family of p-values
    separately (see fdr_correct) before converting to stars. "fisher" and
    "perm" answer related but different questions -- is the Fisher-z-
    transformed mean nonzero, vs. does the real score clear this pipeline's
    own sanity-check noise floor -- and are shown side by side rather than
    picked between, since they can legitimately disagree (see
    fisher_z_significance/permutation_null_significance).

    Returns {(model_a, model_b): {"fisher": stars_or_None, "perm": stars_or_None}}.
    None means that test was never applicable for this metric/pair (skip
    that annotation line entirely) -- as opposed to the string "n/a", which
    means the test WAS run but came back undefined (e.g. n<2 subjects, or an
    empty sanity-check null pool).
    """
    fisher_raw = fisher_z_significance(cross_df, metric)
    perm_raw = permutation_null_significance(cross_df, sanity_df, metric) if sanity_df is not None else {}

    fisher_p_fdr = fdr_correct({pair: v["p_value"] for pair, v in fisher_raw.items()})
    perm_p_fdr = fdr_correct({pair: v["p_value"] for pair, v in perm_raw.items()})

    combined = {}
    for pair in set(fisher_raw) | set(perm_raw):
        combined[pair] = {
            "fisher": significance_stars(fisher_p_fdr[pair]) if pair in fisher_p_fdr else None,
            "perm": significance_stars(perm_p_fdr[pair]) if pair in perm_p_fdr else None,
        }
    return combined


def draw_pair_annotation(ax, x, y, annotation, annotation_scale=1.0):
    """
    Draws up to 2 stacked, color-coded lines at (x, y) on a heatmap axis --
    "z:{stars}" in _FISHER_ANNOTATION_COLOR for the Fisher-z test, "p:{stars}"
    in _PERM_ANNOTATION_COLOR for the permutation-null test (see
    combine_significance) -- distinct prefixes AND colors so the two tests
    are never conflated on the figure, even in grayscale print (the prefix
    survives even where the color doesn't).

    `annotation_scale` scales both the text size and the vertical spacing
    between the two lines. The spacing is applied in POINTS (offset points,
    display space), not data coordinates, on purpose: (x, y) is a heatmap
    block center whose size in data units can range from a handful of
    cells to thousands (--level subject's (model, subject) grid), so a
    fixed data-unit offset either overlaps illegibly on a large block or
    overshoots a small one -- an offset-points spacing stays visually
    consistent regardless of block size, and scales together with the text
    so raising annotation_scale doesn't just make bigger text sit closer
    together.
    """
    lines = []
    if annotation.get("fisher") is not None:
        lines.append((f"z:{annotation['fisher']}", _FISHER_ANNOTATION_COLOR))
    if annotation.get("perm") is not None:
        lines.append((f"p:{annotation['perm']}", _PERM_ANNOTATION_COLOR))
    if not lines:
        return
    fontsize = 6.5 * annotation_scale
    offsets = [-0.75 * fontsize, 0.75 * fontsize] if len(lines) == 2 else [0.0]
    for (text, color), dy in zip(lines, offsets):
        ax.annotate(
            text, xy=(x, y), xycoords="data", xytext=(0, dy), textcoords="offset points",
            ha="center", va="center", fontsize=fontsize, fontweight="bold", color=color,
        )


def draw_cross_model_heatmap(ax, cross_df, title, metric, pair_annotations=None):
    """
    Model x model heatmap of the MEAN across-subject score per model pair
    (see extract_cross_model_pairs) -- "which models tend to produce
    similar saliency patterns to which others", at a glance. The diagonal
    is filled in directly (1.0 for spearman/pearson/mutual_information,
    0.0 for ssd -- see METRIC_CMAP/METRIC_FIXED_RANGE) rather than
    computed, since a model's map is trivially identical to itself and
    extract_cross_model_pairs never returns a model_a == model_b row.

    `pair_annotations` (optional): see draw_pairwise_heatmap -- here each
    off-diagonal CELL already corresponds to exactly one model pair, so the
    annotation goes directly on that cell (no block-center math needed,
    unlike the pairwise heatmap's (model, subject) grid).
    """
    models = sorted(set(cross_df["model_a"]) | set(cross_df["model_b"]))
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    self_value = 0.0 if metric == "ssd" else 1.0
    mean_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(mean_matrix, self_value)

    for (model_a, model_b), mean_score in cross_df.groupby(["model_a", "model_b"])["score"].mean().items():
        mean_matrix[idx[model_a], idx[model_b]] = mean_score
        mean_matrix[idx[model_b], idx[model_a]] = mean_score

    fixed_range = METRIC_FIXED_RANGE[metric]
    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        off_diagonal = mean_matrix[~np.eye(n, dtype=bool)]
        off_diagonal = off_diagonal[~np.isnan(off_diagonal)]
        vmin = 0.0
        vmax = float(np.percentile(off_diagonal, 99)) if off_diagonal.size else 1.0

    # Upper triangle + diagonal only -- mean_matrix is symmetric (models are
    # sorted the same way into `idx` on both axes), so the lower triangle is
    # a pure duplicate; masked to NaN (blank/transparent) rather than
    # dropped from `mean_matrix` so tick labels/annotation positions are
    # unaffected.
    masked_matrix = np.where(np.triu(np.ones((n, n), dtype=bool)), mean_matrix, np.nan)
    im = ax.imshow(masked_matrix, cmap=METRIC_CMAP[metric], vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n), labels=models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels=models, fontsize=8)
    ax.set_title(title, fontsize=11)

    if pair_annotations:
        for (model_a, model_b), annotation in pair_annotations.items():
            if model_a not in idx or model_b not in idx or not annotation:
                continue
            # model_a < model_b alphabetically (see extract_cross_model_pairs),
            # and `models`/`idx` are alphabetically sorted too, so
            # idx[model_a] < idx[model_b] always -- upper triangle only, no
            # mirrored lower-triangle draw.
            draw_pair_annotation(ax, idx[model_b], idx[model_a], annotation)
    return im


def plot_cross_model_heatmap_grid(csv_paths_by_tag, output_path, kind, metric, sanity_paths_by_tag=None, show_significance=True):
    """
    One model x model mean-agreement heatmap subplot per attribution method
    with data (up to 2x2), sharing one colorbar. `sanity_paths_by_tag`
    (optional): that tag's already-written sanity-check CSV, used to build
    the permutation test's empirical chance floor (see
    sanity_null_pool/permutation_null_significance) -- without it, only the
    Fisher-z test (spearman/pearson only) can run; see combine_significance.
    `show_significance=False` skips both tests entirely (no annotations, no
    "n/a" clutter) -- used for --level aggregate, where each model pair has
    only one (mean-across-subjects) score and the Fisher-z test is undefined
    (n<2) by construction; see main().
    """
    available = [(tag, path) for tag, path in csv_paths_by_tag.items() if path.exists()]
    if not available:
        print(f"  no cross-model CSVs found for any attribution tag; skipping {output_path}")
        return

    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols + 1.5, 5.0 * nrows), squeeze=False)

    im = None
    any_annotations = False
    for i, (tag, path) in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        df = pd.read_csv(path)
        pair_annotations = {}
        if show_significance:
            sanity_df = None
            if sanity_paths_by_tag is not None:
                sanity_path = sanity_paths_by_tag.get(tag)
                if sanity_path is not None and sanity_path.exists():
                    sanity_df = pd.read_csv(sanity_path)
            pair_annotations = combine_significance(df, sanity_df, metric)
        any_annotations = any_annotations or bool(pair_annotations)
        im = draw_cross_model_heatmap(ax, df, _ATTRIBUTION_DISPLAY[tag], metric, pair_annotations)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    metric_display = METRIC_DISPLAY[metric]
    suffix = (
        " (z: Fisher-z one-sample t-test vs. 0; p: permutation test vs. sanity-check null; "
        "both FDR-corrected (BH) across model pairs; */**/*** = p<.05/.01/.001, ns = not significant, "
        "n/a = test undefined)"
    ) if any_annotations else ""
    fig.suptitle(
        f"Cross-model saliency {metric_display.lower()}, same subject, mean across subjects ({kind}){suffix}",
        fontsize=10 if suffix else 13, fontweight="bold",
    )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=f"mean {metric_display}")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def draw_cross_model_boxplot(ax, cross_df, title, metric):
    """One box+strip per model pair (points = subjects) -- the full across-subject spread the heatmap's mean collapses away."""
    cross_df = cross_df.copy()
    cross_df["pair"] = cross_df["model_a"] + " vs " + cross_df["model_b"]
    pairs = sorted(cross_df["pair"].unique())
    for i, pair in enumerate(pairs):
        values = cross_df.loc[cross_df["pair"] == pair, "score"].dropna().to_numpy()
        if values.size == 0:
            continue
        ax.boxplot(
            values, positions=[i], widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="#7570b3", alpha=0.4),
            medianprops=dict(color="black"), showfliers=False,
        )
        jitter = (np.random.RandomState(0).rand(values.size) - 0.5) * 0.3
        ax.scatter(np.full(values.size, i) + jitter, values, color="#7570b3", s=10, zorder=3)
    if METRIC_HAS_ZERO_BASELINE[metric]:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(pairs)), labels=pairs, rotation=60, ha="right", fontsize=7)
    ax.set_title(title, fontsize=11)


def plot_cross_model_boxplot_grid(csv_paths_by_tag, output_path, kind, metric):
    """One per-model-pair box+strip subplot per attribution method with data (up to 2x2)."""
    available = [(tag, path) for tag, path in csv_paths_by_tag.items() if path.exists()]
    if not available:
        print(f"  no cross-model CSVs found for any attribution tag; skipping {output_path}")
        return

    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    max_pairs = max(
        pd.read_csv(p).assign(pair=lambda d: d["model_a"] + " vs " + d["model_b"])["pair"].nunique() for _, p in available
    )
    panel_w = max(6.0, max_pairs * 0.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 4.5 * nrows), sharey=True, squeeze=False)

    metric_display = METRIC_DISPLAY[metric]
    for i, (tag, path) in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        df = pd.read_csv(path)
        draw_cross_model_boxplot(ax, df, _ATTRIBUTION_DISPLAY[tag], metric)
        if i % ncols == 0:
            ax.set_ylabel(metric_display)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Cross-model saliency {metric_display.lower()}, same subject, per model pair ({kind})",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_pairwise_heatmap_grid(
    csv_paths_by_tag, output_path, kind, metric, cross_model_paths_by_tag=None, sanity_paths_by_tag=None, show_significance=True,
    no_title=False, tick_label_scale=1.0, annotation_scale=1.0,
):
    """
    One pairwise-heatmap subplot per attribution method with data (up to
    2x2), sharing one colorbar. `cross_model_paths_by_tag` (optional): the
    same tag's already-written cross_model CSV (see extract_cross_model_pairs),
    the per-subject scores both significance tests need. `sanity_paths_by_tag`
    (optional): that tag's sanity-check CSV, the empirical null the
    permutation test needs (see combine_significance) -- without it, only
    the Fisher-z test (spearman/pearson only) can run. Both together
    overlay each off-diagonal model-vs-model block with both tests' stars
    (see draw_pairwise_heatmap/draw_pair_annotation). `show_significance=False`
    skips both tests entirely (no annotations, no "n/a" clutter) -- used for
    --level aggregate, where each model pair has only one (mean-across-subjects)
    score and the Fisher-z test is undefined (n<2) by construction; see main().
    """
    available = [(tag, path) for tag, path in csv_paths_by_tag.items() if path.exists()]
    if not available:
        print(f"  no pairwise CSVs found for any attribution tag; skipping {output_path}")
        return

    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)  # ceil
    max_labels = max(len(pd.read_csv(p, index_col=0).index) for _, p in available)
    # Capped at 40in/panel: pairwise_model_subject_* CSVs have one label per
    # (model, subject) baseline map (thousands, not just a handful of
    # models), so the uncapped max_labels * 0.16 blows the figure past
    # matplotlib's 2**16px Agg canvas limit at dpi=150 (e.g. 5746 labels ->
    # ~919in/panel -> ValueError: Image size ... too large). Individual
    # cells stay illegible past a few hundred labels either way -- draw_pairwise_heatmap
    # only labels model blocks, not each (model, subject) row/column -- so
    # capping loses no real information once max_labels is already this large.
    panel_size = min(max(4.0, max_labels * 0.16), 40.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols + 1.5, panel_size * nrows), squeeze=False)

    im = None
    any_annotations = False
    for i, (tag, path) in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        df = pd.read_csv(path, index_col=0)
        pair_annotations = {}
        if show_significance and cross_model_paths_by_tag is not None:
            cross_path = cross_model_paths_by_tag.get(tag)
            if cross_path is not None and cross_path.exists():
                cross_df = pd.read_csv(cross_path)
                sanity_df = None
                if sanity_paths_by_tag is not None:
                    sanity_path = sanity_paths_by_tag.get(tag)
                    if sanity_path is not None and sanity_path.exists():
                        sanity_df = pd.read_csv(sanity_path)
                pair_annotations = combine_significance(cross_df, sanity_df, metric)
        any_annotations = any_annotations or bool(pair_annotations)
        im = draw_pairwise_heatmap(
            ax, df, _ATTRIBUTION_DISPLAY[tag], metric, pair_annotations,
            no_title=no_title, tick_label_scale=tick_label_scale, annotation_scale=annotation_scale,
        )
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    metric_display = METRIC_DISPLAY[metric]
    suffix = (
        " (z: Fisher-z one-sample t-test vs. 0; p: permutation test vs. sanity-check null; "
        "both FDR-corrected (BH) per model pair; */**/*** = p<.05/.01/.001, ns = not significant, "
        "n/a = test undefined)"
    ) if any_annotations else ""
    if not no_title:
        fig.suptitle(
            f"Pairwise saliency {metric_display.lower()} across models x subjects ({kind}){suffix}",
            fontsize=10 if suffix else 13, fontweight="bold",
        )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=metric_display)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def draw_sanity_check_panel(ax, df, title, tests, colors, metric):
    """Draws one tag's per-model box+strip plot onto `ax`."""
    models = sorted(df["model"].unique())
    width = 0.25
    offsets = {test: (i - 1) * width for i, test in enumerate(tests)}

    for model_idx, model in enumerate(models):
        model_rows = df[df["model"] == model]
        for test in tests:
            col = f"score_vs_{test}"
            if col not in model_rows:
                continue
            values = model_rows[col].dropna().to_numpy()
            if values.size == 0:
                continue
            pos = model_idx + offsets[test]
            ax.boxplot(
                values,
                positions=[pos],
                widths=width * 0.85,
                patch_artist=True,
                boxprops=dict(facecolor=colors[test], alpha=0.4),
                medianprops=dict(color="black"),
                showfliers=False,
            )
            jitter = (np.random.RandomState(0).rand(values.size) - 0.5) * width * 0.5
            ax.scatter(np.full_like(values, pos) + jitter, values, color=colors[test], s=12, zorder=3)

    # 0 is only a meaningful "chance"/independent reference for
    # spearman/pearson/mutual_information -- SSD's uninformative baseline is
    # a large positive number, not 0 (0 means "identical", the opposite of
    # what randomization is expected to produce), so no reference line is
    # drawn for it (see METRIC_HAS_ZERO_BASELINE).
    if METRIC_HAS_ZERO_BASELINE[metric]:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(models)), labels=models, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)


def plot_sanity_check_summary_grid(csv_paths_by_tag, output_path, kind, metric):
    """One sanity-check boxplot subplot per attribution method with data (up to 2x2), one shared legend."""
    available = [(tag, path) for tag, path in csv_paths_by_tag.items() if path.exists()]
    if not available:
        print(f"  no sanity-check CSVs found for any attribution tag; skipping {output_path}")
        return

    tests = ["arch", "data", "input"]
    colors = {"arch": "#d62728", "data": "#1f77b4", "input": "#2ca02c"}
    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    max_models = max(len(pd.read_csv(p)["model"].unique()) for _, p in available)
    panel_w = max(6.0, max_models * 1.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 4.5 * nrows), sharey=True, squeeze=False)

    metric_display = METRIC_DISPLAY[metric]
    for i, (tag, path) in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        df = pd.read_csv(path)
        draw_sanity_check_panel(ax, df, _ATTRIBUTION_DISPLAY[tag], tests, colors, metric)
        if i % ncols == 0:
            ax.set_ylabel(f"{metric_display} vs. baseline")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Baseline vs. sanity-check maps, across subjects ({kind}, {metric_display.lower()})", fontsize=13, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[t], alpha=0.6, markersize=10, label=t) for t in tests]
    fig.legend(handles=handles, title="sanity check", loc="upper right")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_sanity_check_interactive(csv_paths_by_tag, output_path, kind, metric):
    """
    Plotly version of plot_sanity_check_summary_grid, saved as standalone
    HTML (not PNG): hovering a point shows exactly which subject/model/test
    it came from, which a static image can't do -- this is the "if it's not
    possible in PNG, use an interactive format" fallback, applied because
    matplotlib has no hover/tooltip mechanism at all (it's a static raster
    renderer), so this is a hard format constraint, not a style choice.

    One subplot panel per attribution method (same 2x2 layout as the PNG
    grid, via plotly's make_subplots), each with one box+all-points trace
    per test (arch/data/input) per model -- box for the distribution, points
    (customdata-tagged with the subject ID) for exactly which subject each
    score belongs to on hover.
    """
    available = [(tag, path) for tag, path in csv_paths_by_tag.items() if path.exists()]
    if not available:
        print(f"  no sanity-check CSVs found for any attribution tag; skipping {output_path}")
        return

    tests = ["arch", "data", "input"]
    colors = {"arch": "#d62728", "data": "#1f77b4", "input": "#2ca02c"}
    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    metric_display = METRIC_DISPLAY[metric]

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[_ATTRIBUTION_DISPLAY[tag] for tag, _ in available],
    )

    for i, (tag, path) in enumerate(available):
        row, col = i // ncols + 1, i % ncols + 1
        df = pd.read_csv(path)
        models = sorted(df["model"].unique())
        for test in tests:
            colname = f"score_vs_{test}"
            if colname not in df:
                continue
            sub = df.dropna(subset=[colname])
            if sub.empty:
                continue
            # One box+points trace per (test, model) so each model gets its
            # own x position (plotly boxplots auto-space by x category) --
            # legend/color grouped by test via legendgroup, shown once.
            for model in models:
                model_sub = sub[sub["model"] == model]
                if model_sub.empty:
                    continue
                fig.add_trace(
                    go.Box(
                        x=[f"{model} {test}"] * len(model_sub),
                        y=model_sub[colname],
                        name=test,
                        legendgroup=test,
                        marker_color=colors[test],
                        boxpoints="all",
                        pointpos=0,
                        jitter=0.6,
                        customdata=model_sub["subject"],
                        hovertemplate=(
                            f"model={model}<br>test={test}<br>subject=%{{customdata}}<br>{metric}=%{{y:.3f}}<extra></extra>"
                        ),
                        showlegend=(i == 0 and model == models[0]),
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(tickangle=45, row=row, col=col)
        # See METRIC_HAS_ZERO_BASELINE: 0 isn't SSD's meaningful reference point.
        if METRIC_HAS_ZERO_BASELINE[metric]:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

    fig.update_layout(
        title=f"Baseline vs. sanity-check maps, across subjects ({kind}, {metric_display}) -- hover a point for its subject",
        height=450 * nrows,
        width=max(900, 220 * len(MODELS)) * ncols,
        boxmode="group",
        legend_title="sanity check",
    )
    fig.write_html(str(output_path))
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pairwise + sanity-check agreement statistics across saliency maps (see --metrics).")
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Default: <repo>/<--saliency-root>/<target> (see --target/--saliency-root).",
    )
    parser.add_argument(
        "--saliency-root", choices=["saliency_outputs", "saliency_affine"], default="saliency_outputs",
        help="Which top-level saliency tree to read from (default: saliency_outputs, the original "
        "rigid-registration pipeline -- see NKI_RIGID_TURBOPREP_ROOT in slurm/env.sh). 'saliency_affine' is "
        "slurm/compute_saliency.sh's affine-registration pipeline (NKI_AFFINE_TURBOPREP_ROOT, "
        "checkpoints/heads/*_affine.pt -- see that script's own docstring); its <target>/<subject>/ layout and "
        "every model/file naming convention are otherwise identical -- only the top-level directory (and, by "
        "construction, the registration each file was produced from) differs, so no other flag needs to change "
        "to read it. Only used to resolve --saliency-dir's default -- ignored if --saliency-dir is passed "
        "explicitly. Output (--output-dir's default, <saliency-dir>/correlations) follows the same tree, so an "
        "affine run's CSVs/figures land under saliency_affine/<target>/correlations/, never mixed with a "
        "saliency_outputs run's.",
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default="age",
        help="Which target's <--saliency-root>/<target>/ subtree to analyze (default: age). Only used to "
        "resolve --saliency-dir's default -- ignored if --saliency-dir is passed explicitly.",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--brain-mask",
        type=Path,
        nargs="?",
        const=DEFAULT_BRAIN_MASK,
        default=None,
        help="Restrict correlations to brain-masked voxels (background/skull-stripped padding otherwise inflates "
        "agreement via trivially-tied zeros -- see the module docstring). Default: no mask, every voxel in the "
        "resampled volume is used. Pass --brain-mask with no value to use the template's own "
        f"desc-brain_mask.nii.gz ({DEFAULT_BRAIN_MASK}), or --brain-mask /path/to/mask.nii.gz for a specific one. "
        "Whenever a mask is applied, output CSV/PNG/HTML filenames get a '_masked' suffix (e.g. "
        "sanity_check_spearman_signed_masked.png) so masked and unmasked runs of the same combination of --metrics never "
        "overwrite each other.",
    )
    parser.add_argument("--kind", choices=["signed", "abs"], default="signed")
    parser.add_argument(
        "--level", choices=["aggregate", "subject"], default="aggregate",
        help="Granularity of the pairwise (#1) and cross-model (#3) comparisons -- see the module docstring's "
        "--level section. 'aggregate' (default): each model's baseline maps are averaged across every subject "
        "FIRST, then compared -- a small N_models x N_models matrix. 'subject': the original much larger "
        "(model, subject) x (model, subject) matrix (see compute_pairwise_matrix). The sanity-check comparison "
        "(#2) is always per-subject regardless of this flag. Output filenames encode the level ("
        "pairwise_model_aggregate_* / cross_model_aggregate_* vs. pairwise_model_subject_* / cross_model_*), so "
        "an aggregate run and a subject run never overwrite each other's CSVs/figures.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(METRICS),
        default=list(METRICS),
        help="Which agreement/dissimilarity statistic(s) to compute for both the pairwise and sanity-check "
        "comparisons (default: all 5). See module docstring for what each one measures, its range, and its "
        "performance characteristics. Every metric gets its own CSVs/figures (baked into the filename, e.g. "
        "sanity_check_mutual_information_signed.png), and -- like --attribution-tags -- every metric requested in "
        "one run shares the same resampled vector_cache (resampling, not scoring, is the dominant cost; see "
        "COST NOTE), so requesting several metrics in one invocation is far cheaper than one process per metric.",
    )
    parser.add_argument(
        "--mi-bins",
        type=int,
        default=DEFAULT_MI_BINS,
        help="--metrics mutual_information only: number of quantile bins each map is discretized into before "
        f"computing normalized mutual information (default: {DEFAULT_MI_BINS}).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--attribution-tags",
        nargs="*",
        default=list(ATTRIBUTION_TAGS),
        choices=list(ATTRIBUTION_TAGS),
        help="Which of calculate_gradcam.py's --attribution-method outputs to correlate (default: all 4). "
        "Each gets its own CSV and its own subplot panel in the combined figures.",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Skip resampling/correlation computation and just replot the CSVs already in --output-dir "
        "(e.g. after tweaking plot style, or to visualize a run from compute_correlations.sh).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip computing any (metric, attribution tag, pairwise/sanity-check) combination whose CSV already "
        "exists in --output-dir and parses cleanly -- for continuing a run that was killed partway through (e.g. "
        "a SLURM timeout) without recomputing everything already-written CSVs cover. Resampling is also narrowed "
        "to only the files still needed by the remaining work (see plan_remaining_work/collect_needed_paths). A "
        "truncated/corrupt CSV left by a mid-write kill is detected and recomputed, not mistaken for done (see "
        "csv_is_valid). Ignored (moot) together with --skip-compute, which skips computation entirely.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Compute the CSVs but don't generate the PNGs/HTML.")
    parser.add_argument(
        "--skip-cross-model-figures", action="store_true",
        help="Skip plot_cross_model_heatmap_grid/plot_cross_model_boxplot_grid (comparison #3's PNGs) -- the "
        "pairwise heatmap/sanity-check figures are still generated. The cross_model_*.csv itself is still written "
        "regardless (see compute_correlations's 2c: it's a cheap filter over the already-computed pairwise CSV, "
        "not a separate resampling/scoring pass), so a later --skip-compute run can still plot it by omitting "
        "this flag. Ignored (moot) together with --no-plot, which skips every figure.",
    )
    parser.add_argument(
        "--no-title", action="store_true",
        help="Pairwise heatmap grid only (plot_pairwise_heatmap_grid/draw_pairwise_heatmap): omit the figure "
        "suptitle and each panel's per-tag title. Combine with --skip-compute to restyle an already-computed run "
        "without touching its CSVs.",
    )
    parser.add_argument(
        "--tick-label-scale", type=float, default=1.0,
        help="Pairwise heatmap grid only: multiplies the x/y model-block tick label fontsize (base 8) by this "
        "factor (default: 1.0, unchanged). E.g. 10 for a 10x larger tick label size.",
    )
    parser.add_argument(
        "--annotation-scale", type=float, default=1.0,
        help="Pairwise heatmap grid only: multiplies the significance-annotation (z:*/p:* stars, see "
        "draw_pair_annotation) fontsize (base 6.5) by this factor, and scales the spacing between the two stacked "
        "lines to match -- unlike tick labels, this text sits inside heatmap blocks, so making it bigger without "
        "also widening the gap between the z:/p: lines would just make them overlap more, not less "
        "(default: 1.0, unchanged).",
    )
    parser.add_argument(
        "--models", nargs="*", default=MODELS, choices=MODEL_CHOICES,
        help=f"Subset of models to correlate (default: all {len(MODELS)}, with optim_uncnn's raw/rank/sobel "
        "channels collapsed into one optim_uncnn_avg -- a per-voxel mean of the 3, computed in-memory since "
        "they're heterogeneous units; see UNCNN_AVG_MODEL). Narrow this for a quick debugging run -- e.g. "
        "--models brainiac to see per-file timing on one model before committing to the full run. Pass an "
        "individual optim_uncnn_raw/optim_uncnn_rank/optim_uncnn_sobel channel instead of optim_uncnn_avg to "
        "inspect one channel in isolation.",
    )
    parser.add_argument(
        "--exclude-models", nargs="*", default=[], choices=MODEL_CHOICES,
        help="Drop these models from --models (default: none) -- e.g. --exclude-models anatcl_local to run "
        "everything else without having to spell out --models in full.",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Subset of subject IDs (subfolder names under --saliency-dir) to correlate (default: every subject "
        "folder found). Combine with --models for a fast debugging run, e.g. one subject x one model.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Resample files across this many worker processes (default: 1, sequential -- matches the original "
        "behavior exactly). Resampling (nilearn resample_to_img) is CPU-bound and independent per file, so this "
        "parallelizes close to linearly up to your allocation's core count -- e.g. compute_correlations.sh requests "
        "--cpus-per-task=4, so --n-jobs 4 there. Correlation math (corrcoef/spearmanr) stays single-process; it's "
        "cheap relative to resampling (see --timing) so parallelizing it wouldn't move the needle.",
    )
    parser.add_argument(
        "--timing", action="store_true",
        help="Print per-file resampling time (elapsed + whether it was an identity resample or a real "
        "interpolation) instead of just a filename per line. The end-of-run summary (unique file count, total/avg "
        "time, identity-vs-interpolated split, 5 slowest files) always prints regardless of this flag.",
    )
    parser.add_argument(
        "--block-size", type=int, default=200,
        help="Compute the pairwise (model, subject) x (model, subject) matrix (see comparison #1 in the module "
        "docstring) in square blocks of at most this many labels at a time, instead of ever holding every label's "
        "full-resolution vector in RAM together (see compute_pairwise_matrix). Peak RAM for that step is "
        "O(block_size * n_voxels) rather than O(n_labels * n_voxels) -- with n_labels in the thousands (one label "
        "per (model, subject) baseline map) and n_voxels ~7M at the default unmasked template grid, the old "
        "one-shot matrix was the single largest memory user in this script. Lower this if a run still runs out of "
        "memory; raise it (fewer, larger blocks) for a modest speedup if memory allows.",
    )
    parser.add_argument(
        "--vector-cache-dir", type=Path, default=None,
        help="Where to cache each unique file's resampled voxel vector on disk (default: "
        "<output-dir>/.correlations_cache/vectors/<kind><mask_suffix>) instead of holding all of them in RAM at "
        "once (see DiskVectorCache). Persists across runs -- re-running against the same cache dir (e.g. to add "
        "more --metrics, or after --resume picks up a killed run) skips re-resampling any file already cached. "
        "Point this at fast local/node scratch if available; it is not cleaned up automatically since it's "
        "reusable, so remove it yourself once you're done with a target/kind/mask combination.",
    )
    args = parser.parse_args()
    # DEFAULT_SALIENCY_DIR is <repo>/saliency_outputs (see visualize_saliency.py) --
    # .parent is <repo>, so swapping in --saliency-root reuses that same
    # repo-root resolution instead of hardcoding it again here.
    args.saliency_dir = args.saliency_dir or (DEFAULT_SALIENCY_DIR.parent / args.saliency_root / args.target)
    args.models = [m for m in args.models if m not in args.exclude_models]
    if len(args.models) < 2:
        raise SystemExit(f"--models/--exclude-models left fewer than 2 models ({args.models}); nothing to compare.")

    output_dir = args.output_dir or (args.saliency_dir / "correlations")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reflected in every output filename below so a masked run and an
    # unmasked run of the same --kind/metric never silently overwrite each
    # other.
    mask_suffix = "_masked" if args.brain_mask else ""
    # Baked into the pairwise/cross-model filenames (see --level's help)
    # so an aggregate run and a subject run never overwrite each other's
    # CSVs/figures. "pairwise_model_subject"/"cross_model" (no "_subject")
    # are the original, pre-existing names -- kept as-is for --level subject
    # so old on-disk CSVs from before --level existed stay valid/resumable.
    pairwise_stem = "pairwise_model_aggregate" if args.level == "aggregate" else "pairwise_model_subject"
    cross_model_stem = "cross_model_aggregate" if args.level == "aggregate" else "cross_model"
    # {metric: {tag: path}} -- one CSV family per requested metric.
    pairwise_paths_by_metric = {
        metric: csv_paths_for_tags(output_dir, f"{pairwise_stem}_{metric}", args.kind, args.attribution_tags, mask_suffix)
        for metric in args.metrics
    }
    sanity_paths_by_metric = {
        metric: csv_paths_for_tags(output_dir, f"sanity_check_{metric}", args.kind, args.attribution_tags, mask_suffix)
        for metric in args.metrics
    }
    cross_model_paths_by_metric = {
        metric: csv_paths_for_tags(output_dir, f"{cross_model_stem}_{metric}", args.kind, args.attribution_tags, mask_suffix)
        for metric in args.metrics
    }

    run_t0 = time.time()
    if args.skip_compute:
        if not any(
            p.exists()
            for paths in list(pairwise_paths_by_metric.values()) + list(sanity_paths_by_metric.values()) + list(cross_model_paths_by_metric.values())
            for p in paths.values()
        ):
            raise SystemExit(f"--skip-compute but no CSVs exist yet in {output_dir}; run without it first.")
    else:
        compute_correlations(args, output_dir, pairwise_paths_by_metric, sanity_paths_by_metric, cross_model_paths_by_metric)
    compute_elapsed = time.time() - run_t0

    plot_t0 = time.time()
    # --level aggregate collapses each model pair to a single (mean-across-
    # subjects) score -- the Fisher-z test needs >=2 per-subject scores to
    # estimate a variance (see fisher_z_significance) and is undefined by
    # construction there, and the permutation test's per-pair "is this one
    # number outside the null pool" read is a much weaker claim on one
    # aggregate score than on a real per-subject sample. Both are skipped
    # entirely (not just rendered as "n/a") for aggregate runs; --level
    # subject is unaffected.
    show_significance = args.level != "aggregate"
    if not args.no_plot:
        for metric in args.metrics:
            plot_pairwise_heatmap_grid(
                pairwise_paths_by_metric[metric], output_dir / f"{pairwise_stem}_{metric}_{args.kind}{mask_suffix}.png",
                args.kind, metric, cross_model_paths_by_metric[metric], sanity_paths_by_metric[metric],
                show_significance=show_significance,
                no_title=args.no_title, tick_label_scale=args.tick_label_scale, annotation_scale=args.annotation_scale,
            )
            plot_sanity_check_summary_grid(
                sanity_paths_by_metric[metric], output_dir / f"sanity_check_{metric}_{args.kind}{mask_suffix}.png",
                args.kind, metric,
            )
            plot_sanity_check_interactive(
                sanity_paths_by_metric[metric], output_dir / f"sanity_check_{metric}_{args.kind}{mask_suffix}.html",
                args.kind, metric,
            )
            # Separate files from the sanity-check figures above -- see the
            # module docstring's #3: this tests cross-model agreement, not
            # sanity-check robustness, so it shouldn't be mixed into or
            # overwrite either of the other two comparisons' outputs.
            # --skip-cross-model-figures only gates these two calls, not the
            # cross_model_*.csv computation above (2c) -- see that flag's
            # own help text for why.
            if not args.skip_cross_model_figures:
                plot_cross_model_heatmap_grid(
                    cross_model_paths_by_metric[metric], output_dir / f"{cross_model_stem}_heatmap_{metric}_{args.kind}{mask_suffix}.png",
                    args.kind, metric, sanity_paths_by_metric[metric],
                    show_significance=show_significance,
                )
                plot_cross_model_boxplot_grid(
                    cross_model_paths_by_metric[metric], output_dir / f"{cross_model_stem}_boxplot_{metric}_{args.kind}{mask_suffix}.png",
                    args.kind, metric,
                )
    plot_elapsed = time.time() - plot_t0

    print(
        f"\nTotal: {time.time() - run_t0:.1f}s "
        f"(compute: {compute_elapsed:.1f}s, plotting: {plot_elapsed:.1f}s)."
    )


# Populated once per worker process by _init_pairwise_worker (not per task --
# reused across every block that worker scores), mirroring _worker_template_
# img/_worker_brain_mask's pattern above for the resampling pool. Only used
# by the mutual_information/ssd/ssim parallel path -- spearman/pearson never
# touch this (see compute_pairwise_matrix).
_pw_labels = None
_pw_baseline_lookup = None
_pw_prepared_dir = None
_pw_tag = None
_pw_kind = None
_pw_metric = None


def _init_pairwise_worker(labels, baseline_lookup, prepared_dir, tag, kind, metric, ssim_template_shape, ssim_brain_mask):
    # See _init_worker's comment -- same oversubscription risk here: this
    # path only ever runs for mutual_information/ssd/ssim (compute_pairwise_
    # matrix never parallelizes the spearman/pearson matmul across
    # processes), but each of --n-jobs worker processes still independently
    # runs numpy ops (score_pair's SSD subtraction/sum, sklearn's
    # contingency-matrix construction for MI, skimage's SSIM filtering) that
    # can spin up their own multi-threaded BLAS pool otherwise.
    threadpoolctl.threadpool_limits(1)
    global _pw_labels, _pw_baseline_lookup, _pw_prepared_dir, _pw_tag, _pw_kind, _pw_metric
    _pw_labels = labels
    _pw_baseline_lookup = baseline_lookup
    _pw_prepared_dir = prepared_dir
    _pw_tag = tag
    _pw_kind = kind
    _pw_metric = metric
    # score_pair's "ssim" branch reads the module-global _SSIM_BRAIN_MASK --
    # a fresh worker process doesn't share the main process's globals unless
    # explicitly set here (never relies on fork inheriting them, same reason
    # every other cross-process value above is passed through initargs
    # rather than assumed).
    set_ssim_context(ssim_template_shape, ssim_brain_mask)


def _load_prepared_for_worker(label):
    """
    Reads one label's ALREADY-prepared vector straight off disk -- never
    computes/writes one (unlike load_prepared_baseline's closure in
    compute_correlations). Safe to call from multiple worker processes at
    once for the same label because compute_pairwise_matrix's parallel path
    pre-warms every label's prepared/*.npy via the serial closure BEFORE any
    worker starts (see its docstring) -- by the time this runs, the file this
    reads is guaranteed to already exist, so there's no compute-and-write
    race between workers.
    """
    model, subj_dir = _pw_baseline_lookup[label]
    key = baseline_prepare_key(model, subj_dir, _pw_tag, _pw_kind)
    return np.load(_cache_filename(_pw_prepared_dir, key), mmap_mode="r")


def _pairwise_block_worker(task):
    """
    Runs in a worker process (see _init_pairwise_worker): scores one (bi, bj)
    block of the pairwise matrix for a non-vectorizable metric (mutual_
    information/ssd) via the same per-pair score_pair loop
    compute_pairwise_matrix's serial path uses, and returns the small
    (block_size x block_size) score block itself -- not the input vectors
    (~7M elements each, far more expensive to pickle back through the pool;
    same "return the cheap thing, not the ~28MB thing" idiom _resample_worker
    uses for the same reason).
    """
    bi, bj, r0, r1, c0, c1 = task
    row_vecs = [_load_prepared_for_worker(label) for label in _pw_labels[r0:r1]]
    col_vecs = row_vecs if bj == bi else [_load_prepared_for_worker(label) for label in _pw_labels[c0:c1]]
    block = np.empty((len(row_vecs), len(col_vecs)))
    for i, prepared_i in enumerate(row_vecs):
        for j, prepared_j in enumerate(col_vecs):
            block[i, j] = score_pair(prepared_i, prepared_j, _pw_metric)
    return bi, bj, block


def compute_pairwise_matrix(labels, baseline_lookup, load_prepared_baseline, tag, metric, block_size, n_jobs=1, prepared_dir=None, kind=None):
    """
    Full n_labels x n_labels agreement matrix (comparison #1 -- see module
    docstring), computed in square blocks of at most `block_size` labels at
    a time instead of ever holding every label's full-resolution prepared
    vector in RAM together the way the old single-shot np.vstack/np.corrcoef
    (spearman/pearson) or full prepared-list (mutual_information/ssd)
    approach did. Peak RAM here is O(block_size * n_voxels), not O(n_labels
    * n_voxels) -- with n_labels one per (model, subject) baseline map
    (thousands, once every subject/model is included) and n_voxels ~7M at
    the default unmasked template grid, the old full matrix was this
    script's single largest memory user (see resample_all/DiskVectorCache
    for the other one, the raw resampled-vector cache).

    For spearman/pearson, prepare_vector already returns each vector
    mean-centered and L2-normalized (see _center_unit), so a block's
    pairwise correlations reduce to one matrix multiply (row_block @
    col_block.T) -- the same "vectorize the correlation across every pair
    at once" trick the old single np.corrcoef call used, just applied one
    block at a time so the full n_labels x n_voxels matrix never exists.
    Always serial (never worth parallelizing -- the matmul itself is already
    fast and vectorized).

    mutual_information/ssd have no equivalent single-call vectorization (see
    module docstring COST NOTE) and fall back to the explicit per-pair
    score_pair loop instead -- genuinely slow at thousands of labels (up to
    O(n_labels^2) individual score_pair calls, each on ~7M-element vectors).
    When n_jobs > 1, this runs across a ProcessPoolExecutor instead of
    serially: one (bi, bj) block per task (see _pairwise_block_worker),
    caller-side pre-warm on the labels' prepared vectors first (via the
    existing load_prepared_baseline closure -- serial, already-caching) so
    every worker only ever READS an already-cached prepared/*.npy, never
    computes+writes one itself (a cross-process write race otherwise, since
    two workers could easily need the same not-yet-cached label at once).
    `prepared_dir`/`kind` are only used on this parallel path (workers can't
    receive the load_prepared_baseline closure itself -- local closures
    aren't picklable across process boundaries -- so they independently
    reconstruct the same cache key via baseline_prepare_key instead).

    The final n_labels x n_labels matrix (float64) is comparatively tiny
    (e.g. ~5000x5000 is only ~200MB) and is built up directly in RAM here --
    only the intermediate per-label vectors are ever blocked.
    """
    n = len(labels)
    corr = np.empty((n, n), dtype=np.float64)
    is_corr_metric = metric in ("spearman", "pearson")
    n_blocks = -(-n // block_size)  # ceil

    if not is_corr_metric and n_jobs > 1:
        for label in labels:
            load_prepared_baseline(*baseline_lookup[label], tag)
        tasks = []
        for bi in range(n_blocks):
            r0, r1 = bi * block_size, min((bi + 1) * block_size, n)
            for bj in range(bi, n_blocks):
                c0, c1 = bj * block_size, min((bj + 1) * block_size, n)
                tasks.append((bi, bj, r0, r1, c0, c1))
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_pairwise_worker,
            # _SSIM_TEMPLATE_SHAPE/_SSIM_BRAIN_MASK: already set in THIS
            # (main) process by compute_correlations's set_ssim_context call
            # -- read directly here (module globals) rather than adding two
            # more parameters to this function/its one call site, and
            # re-set inside each worker process via initargs (see
            # _init_pairwise_worker) since a fresh process doesn't inherit
            # them otherwise.
            initargs=(labels, baseline_lookup, prepared_dir, tag, kind, metric, _SSIM_TEMPLATE_SHAPE, _SSIM_BRAIN_MASK),
        ) as pool:
            for bi, bj, block in pool.map(_pairwise_block_worker, tasks):
                r0, r1 = bi * block_size, min((bi + 1) * block_size, n)
                c0, c1 = bj * block_size, min((bj + 1) * block_size, n)
                corr[r0:r1, c0:c1] = block
                corr[c0:c1, r0:r1] = block.T
        return corr

    for bi in range(n_blocks):
        r0, r1 = bi * block_size, min((bi + 1) * block_size, n)
        row_vecs = [np.asarray(load_prepared_baseline(*baseline_lookup[label], tag)) for label in labels[r0:r1]]
        # float64 for the matmul itself (see score_pair's docstring-comment
        # on why float32 accumulation over ~7M-length vectors measurably
        # drifts from np.corrcoef's own float64 promotion) even though each
        # vector is stored as float32 on disk -- this only doubles one
        # block's worth of memory at a time (block_size * n_voxels), not the
        # whole matrix.
        row_mat = np.vstack(row_vecs).astype(np.float64) if is_corr_metric else None
        for bj in range(bi, n_blocks):
            c0, c1 = bj * block_size, min((bj + 1) * block_size, n)
            if bj == bi:
                col_vecs, col_mat = row_vecs, row_mat
            else:
                col_vecs = [np.asarray(load_prepared_baseline(*baseline_lookup[label], tag)) for label in labels[c0:c1]]
                col_mat = np.vstack(col_vecs).astype(np.float64) if is_corr_metric else None
            if is_corr_metric:
                block = row_mat @ col_mat.T
            else:
                block = np.empty((len(row_vecs), len(col_vecs)))
                for i, prepared_i in enumerate(row_vecs):
                    for j, prepared_j in enumerate(col_vecs):
                        block[i, j] = score_pair(prepared_i, prepared_j, metric)
            corr[r0:r1, c0:c1] = block
            corr[c0:c1, r0:r1] = block.T
    return corr


def compute_correlations(args, output_dir, pairwise_paths_by_metric, sanity_paths_by_metric, cross_model_paths_by_metric):
    template_img = load_canonical_template(args.template)
    brain_mask = load_brain_mask(args.brain_mask, template_img) if args.brain_mask else None
    print(f"Brain mask: {args.brain_mask if brain_mask is not None else 'none (using every voxel, unmasked)'}")
    set_ssim_context(template_img.shape, brain_mask)  # see prepare_vector/score_pair's "ssim" branches

    subject_dirs = sorted(
        p for p in args.saliency_dir.iterdir() if p.is_dir() and p.name not in ("aggregate", "correlations")
    )
    if args.subjects is not None:
        subject_dirs = [p for p in subject_dirs if p.name in set(args.subjects)]
        missing = set(args.subjects) - {p.name for p in subject_dirs}
        if missing:
            print(f"WARNING: --subjects entries not found under {args.saliency_dir}: {sorted(missing)}")
    print(f"Found {len(subject_dirs)} subject folder(s).")

    # --resume: figure out which (metric, tag, comparison) combos already
    # have a valid CSV on disk and can be skipped -- see
    # plan_remaining_work/csv_is_valid. Without --resume, `plan` marks
    # everything as remaining (unchanged full-recompute behavior).
    plan = plan_remaining_work(args, pairwise_paths_by_metric, sanity_paths_by_metric, cross_model_paths_by_metric)
    # cross_model deliberately excluded here -- it never needs resampling
    # (derived from the pairwise CSV, see below), so a tag needing ONLY
    # cross_model work still shouldn't force a resample pass.
    tags_with_remaining_work = [
        tag for tag in args.attribution_tags
        if any(plan[metric][tag]["pairwise"] or plan[metric][tag]["sanity"] for metric in args.metrics)
    ]
    if args.resume:
        n_total = len(args.metrics) * len(args.attribution_tags) * 3
        n_remaining = sum(
            plan[metric][tag]["pairwise"] + plan[metric][tag]["sanity"] + plan[metric][tag]["cross_model"]
            for metric in args.metrics for tag in args.attribution_tags
        )
        print(f"--resume: {n_total - n_remaining}/{n_total} (metric, tag, comparison) combo(s) already done, "
              f"{n_remaining} remaining. {len(args.attribution_tags) - len(tags_with_remaining_work)}/"
              f"{len(args.attribution_tags)} attribution tag(s) need no resampling at all.")
        if n_remaining == 0:
            print("Nothing left to compute.")
            return

    # ---- Phase 1: resample every unique file this run needs, up front ----
    # (see collect_needed_paths/resample_all docstrings -- this is what
    # actually dominates runtime; the scoring in phase 2 below is
    # comparatively cheap and reuses this cache with zero further resampling.
    # Shared across every --metrics entry (not just every --attribution-tags
    # entry) since which files are needed depends on kind/tags/models, never
    # on the scoring metric -- so requesting several metrics costs one
    # resample pass, not one per metric. Narrowed to tags_with_remaining_work
    # under --resume -- a tag with nothing left to compute for ANY metric
    # needs no resampling at all.)
    needed_paths = collect_needed_paths(args, subject_dirs, tags=tags_with_remaining_work)
    # Namespaced by kind + mask (not just --output-dir) since those change
    # what a given saliency file's resampled/masked vector actually is --
    # two runs with different --kind/--brain-mask must never share cache
    # entries under the same on-disk key. See DiskVectorCache/resample_all.
    mask_suffix = "_masked" if args.brain_mask else ""
    vector_cache_dir = args.vector_cache_dir or (output_dir / ".correlations_cache" / "vectors" / f"{args.kind}{mask_suffix}")
    vector_cache = resample_all(needed_paths, args, template_img, brain_mask, vector_cache_dir)

    def load_cached(path):
        return vector_cache[str(path.resolve())]

    corr_t0 = time.time()
    for metric in args.metrics:
        pairwise_paths = pairwise_paths_by_metric[metric]
        sanity_paths = sanity_paths_by_metric[metric]
        cross_model_paths = cross_model_paths_by_metric[metric]
        print(f"\n{'=' * 20} metric: {metric} {'=' * 20}")

        # scipy's spearmanr (and an equivalent per-pair MI/SSD call) redoes
        # its own per-vector prep work from scratch on every single call --
        # but the same baseline vector gets compared against 3 different
        # sanity-check targets per subject (arch/data/input), and the same
        # sanity-check target gets compared against every subject's baseline
        # (up to len(subject_dirs) times) -- so preparing the same
        # ~2M-voxel vector (rankdata for spearman, quantile-digitizing for
        # mutual_information) freshly for every comparison redoes that work
        # dozens of times over. Prepare each unique file's vector exactly
        # once per metric here instead (see prepare_vector for what
        # "prepared" means per metric) -- this is what actually made this
        # script slow for spearman, separately from (and likely bigger than)
        # the resampling cost above; the same caching benefits mutual_
        # information's digitization for the same reason.
        #
        # Backed by its own directory on disk (like vector_cache -- see
        # DiskVectorCache), not an in-RAM dict: a label's prepared vector is
        # read back at least twice by compute_pairwise_matrix's blocks (once
        # as a row, once as a column), and holding every label's prepared
        # vector resident for the whole metric would reintroduce the same
        # memory problem vector_cache had. Reset fresh for each metric (not
        # shared across metrics like vector_cache is) since what "prepared"
        # means differs per metric (ranks vs. raw vs. digitized bins) --
        # removed at the end of this metric's iteration (below) so disk use
        # never exceeds one metric's worth at a time.
        prepared_dir = output_dir / ".correlations_cache" / "prepared" / f"{metric}_{args.kind}{mask_suffix}"
        prepared_dir.mkdir(parents=True, exist_ok=True)

        def _cached_prepare(key, compute_fn):
            cache_path = _cache_filename(prepared_dir, key)
            if not cache_path.exists():
                np.save(cache_path, compute_fn())
            return np.load(cache_path, mmap_mode="r")

        def load_prepared(path):
            key = str(path.resolve())
            return _cached_prepare(key, lambda: prepare_vector(load_cached(path), metric, args.mi_bins))

        def load_prepared_baseline(model, subj_dir, tag):
            """
            Like load_prepared, but goes through resolve_baseline_vector so
            UNCNN_AVG_MODEL's in-memory per-voxel average (never file-backed,
            see its docstring-comment) gets prepared/cached exactly like a
            real file's vector -- keyed synthetically (not by resolved path,
            since there is no path) so repeated lookups within one metric
            still hit the cache instead of re-averaging.
            """
            if model != UNCNN_AVG_MODEL:
                return load_prepared(subj_dir / baseline_filename(model, tag, args.kind))
            key = baseline_prepare_key(model, subj_dir, tag, args.kind)
            return _cached_prepare(
                key, lambda: prepare_vector(resolve_baseline_vector(model, tag, args.kind, subj_dir, vector_cache), metric, args.mi_bins)
            )

        def load_prepared_sanity(model, tag, test):
            """Like load_prepared_baseline, for a sanity-check map (see resolve_sanity_vector)."""
            if model != UNCNN_AVG_MODEL:
                return load_prepared(args.saliency_dir / sanity_filenames(model, tag, args.kind)[test])
            key = f"AVG::sanity::{test}::{tag}::{args.kind}"
            return _cached_prepare(
                key,
                lambda: prepare_vector(
                    resolve_sanity_vector(model, tag, args.kind, test, args.saliency_dir, vector_cache), metric, args.mi_bins
                ),
            )

        for tag in tags_with_remaining_work:
            print(f"\n=== attribution method: {tag} ===")
            work = plan[metric][tag]

            # ---- 2a. Pairwise score, granularity per --level (see module docstring) ----
            if not work["pairwise"]:
                print(f"  --resume: {pairwise_paths[tag]} already computed, skipping pairwise comparison.")
            elif args.level == "aggregate":
                aggregate_vectors = compute_aggregate_baseline_vectors(args.models, tag, args.kind, subject_dirs, vector_cache)
                if len(aggregate_vectors) < 2:
                    print(f"  fewer than 2 models have an aggregate (mean-across-subjects) baseline map for {tag}; skipping that CSV.")
                else:
                    print(
                        f"  computing aggregate (per-model mean map) pairwise {METRIC_DISPLAY[metric]} "
                        f"across {len(aggregate_vectors)} model(s)..."
                    )
                    pairwise_df = compute_aggregate_pairwise_matrix(aggregate_vectors, metric, args.mi_bins)
                    pairwise_df.to_csv(pairwise_paths[tag])
                    print(f"  saved aggregate pairwise matrix ({len(aggregate_vectors)}x{len(aggregate_vectors)}) to {pairwise_paths[tag]}")
            else:  # args.level == "subject" -- every (model, subject) baseline map
                labels, baseline_lookup = [], {}
                for subj_dir in subject_dirs:
                    for model in args.models:
                        if resolve_baseline_vector(model, tag, args.kind, subj_dir, vector_cache) is None:
                            continue
                        label = f"{model}__{subj_dir.name}"
                        labels.append(label)
                        baseline_lookup[label] = (model, subj_dir)

                if len(labels) < 2:
                    print(f"  not enough baseline maps found for {tag} pairwise comparison; skipping that CSV.")
                else:
                    print(
                        f"  computing pairwise {METRIC_DISPLAY[metric]} across {len(labels)} baseline maps "
                        f"(block size {args.block_size})..."
                    )
                    corr = compute_pairwise_matrix(
                        labels, baseline_lookup, load_prepared_baseline, tag, metric, args.block_size,
                        n_jobs=args.n_jobs, prepared_dir=prepared_dir, kind=args.kind,
                    )
                    pairwise_df = pd.DataFrame(corr, index=labels, columns=labels)
                    pairwise_df.to_csv(pairwise_paths[tag])
                    print(f"  saved pairwise matrix ({len(labels)}x{len(labels)}) to {pairwise_paths[tag]}")

            # ---- 2b. Each subject's baseline vs. the shared sanity-check maps ----
            if not work["sanity"]:
                print(f"  --resume: {sanity_paths[tag]} already computed, skipping sanity-check comparison.")
                continue
            sanity_rows = []
            for model in args.models:
                tests = list(sanity_filenames(model, tag, args.kind).keys()) if model != UNCNN_AVG_MODEL else ["arch", "data", "input"]
                test_available = {
                    test: resolve_sanity_vector(model, tag, args.kind, test, args.saliency_dir, vector_cache) is not None
                    for test in tests
                }
                for test, available in test_available.items():
                    if not available:
                        print(f"  no {test} sanity-check map found for {model} ({tag}); that column will be blank.")

                for subj_dir in subject_dirs:
                    baseline_raw = resolve_baseline_vector(model, tag, args.kind, subj_dir, vector_cache)
                    if baseline_raw is None:
                        continue
                    baseline_prepared = load_prepared_baseline(model, subj_dir, tag)
                    row = {"model": model, "subject": subj_dir.name}
                    for test in tests:
                        if not test_available[test]:
                            row[f"score_vs_{test}"] = None
                            continue
                        row[f"score_vs_{test}"] = score_pair(baseline_prepared, load_prepared_sanity(model, tag, test), metric)
                    sanity_rows.append(row)

            sanity_df = pd.DataFrame(sanity_rows)
            sanity_df.to_csv(sanity_paths[tag], index=False)
            print(f"  saved sanity-check comparison table ({len(sanity_df)} rows) to {sanity_paths[tag]}")

        # ---- 2c. Cross-model comparison, granularity per --level. Derived
        # from each tag's pairwise CSV on disk (just written above, or
        # already there from a previous run), never from vector_cache -- see
        # module docstring #3 / extract_cross_model_pairs(_aggregate).
        for tag in args.attribution_tags:
            work = plan[metric][tag]
            if not work["cross_model"]:
                print(f"  --resume: {cross_model_paths[tag]} already computed, skipping cross-model comparison ({tag}).")
                continue
            if not pairwise_paths[tag].exists():
                print(f"  no pairwise matrix on disk for {tag} (needed to derive cross-model comparison); skipping.")
                continue
            pairwise_df_for_cross = pd.read_csv(pairwise_paths[tag], index_col=0)
            if args.level == "aggregate":
                cross_df = extract_cross_model_pairs_aggregate(pairwise_df_for_cross)
            else:
                cross_df = extract_cross_model_pairs(pairwise_df_for_cross)
            if cross_df.empty:
                print(f"  no cross-model pairs found for {tag} (need >=2 models); skipping.")
                continue
            cross_df.to_csv(cross_model_paths[tag], index=False)
            print(f"  saved cross-model comparison table ({len(cross_df)} rows) to {cross_model_paths[tag]}")

        # Free this metric's prepared-vector cache now that every comparison
        # using it has been written to CSV -- unlike vector_cache (the raw
        # resampled vectors, reused across every --metrics entry), a
        # prepared vector is metric-specific and worthless once this
        # metric's CSVs are done, so leaving it on disk would only grow
        # --output-dir's disk footprint by one metric's worth per iteration
        # for no benefit.
        shutil.rmtree(prepared_dir, ignore_errors=True)

    print(f"\nScoring math (all metrics/tags, using the pre-resampled cache): {time.time() - corr_t0:.1f}s")


if __name__ == "__main__":
    main()
