"""
Voxelwise normalize(aggregate) - normalize(arch) between each model's
across-subject ig_popmean aggregate saliency map and its own
slurm/compute_sanity_checks.sh --randomize-model ("arch_") sanity-check map
-- a spatial view of WHERE a model's explanation changes once its learned
weights are replaced with random ones (Adebayo et al., 2018), rather than
the single scalar Spearman/Pearson/MI correlation
compute_saliency_correlations.py already computes for the same pair, or the
side-by-side (not against-each-other) row layout
visualize_paper_nilearn.py --sanity-check arch already draws.

The diff is |normalize(aggregate) - normalize(arch)| (absolute value) for
--kind abs -- the source maps are already one-sided, so np.abs() there only
ever collapses "which of the two had the bigger magnitude." For --kind
signed, the diff KEEPS its true sign (see compute_abs_difference_map's own
docstring): which map's value was larger, not just how far apart they are.
Drawn accordingly -- one-sided YlOrRd (vmin=0) for abs, diverging coolwarm
(vmin=-vmax, symmetric) for signed -- see plot_difference_rows.

Why each map is normalized independently before differencing (never raw
values, by default): see docs/known_limitations.md ("Raw gradient
magnitudes are not comparable across models") and visualize_saliency.py's
own module docstring -- different backbones (and, here, a TRAINED vs.
RANDOMLY-INITIALIZED version of the very same backbone) have wildly
different raw gradient scales, so a raw voxelwise subtraction would mostly
just measure "which map had bigger numbers," not "which map's spatial
pattern of importance changed." A raw diff would also be dominated by
whichever of the two maps happens to have the larger scale, which is not a
meaningful "importance" signal (Ancona et al., 2018). --norm-method none is
still available as an explicit opt-out (see NORM_FUNCS below) for
exploratory/debugging use -- it is NOT a recommended way to compare models.

Normalization schemes (--norm-method)
--------------------------------------
- "percentile" / "zscore": visualize_saliency.py's own normalize_map,
  reused as-is (see that function's docstring). Rescale onto a shared
  [-1,1]/[0,1] range by a percentile-of-|value| or mean/std estimate of
  each map's own scale.
- "robust": median/MAD-based z-score (normalize_map_robust below) -- like
  "zscore" but robust to the heavy-tailed outliers visualize_saliency.py's
  own docstring already flags as a weakness of mean/std zscoring.
- "rank" (THE DEFAULT HERE, unlike normalize_map's own "percentile"
  default): each in-brain voxel is replaced by its own percentile rank
  within that single map's in-brain distribution (normalize_map_rank
  below). This is scale-invariant BY CONSTRUCTION, not just robust to
  outliers -- so it doesn't matter that a randomly-initialized backbone's
  gradients can have a completely different raw scale than the trained
  one's. It's also the scheme most consistent with how this repo already
  treats baseline-vs-sanity-check comparison: compute_saliency_correlations.py
  uses Spearman RANK correlation, not raw-magnitude correlation, as its
  primary metric for exactly this comparison, and docs/known_limitations.md
  documents Adebayo et al.'s finding that a map's SIGN/rank decorrelates
  fast under randomization while raw-magnitude structure can linger --
  a rank-based diff isolates "did the spatial importance PATTERN change"
  from "did the overall activity LEVEL change," which is the question the
  sanity check is actually asking. "percentile"/"zscore"/"robust" remain
  scale-preserving (a voxel that was merely a little bigger than its
  neighbors keeps looking that way after normalization); "rank" throws
  scale away entirely.

There is no single objectively-correct choice -- run this script's sibling
notebook (notebooks/compare_aggregate_arch_saliency.py) to compare all four
side by side on one model before trusting a --norm-method choice for a full
--target batch.

fs_roi is NOT supported and is actively rejected (see resolve_pair) even if
passed via --models: its arch_/aggregate files are per-ROI values painted
onto the DKT segmentation grid (calculate_gradcam_tabular.py's
paint_region_values), not true per-voxel gradients on the shared
turboprep/CAT12 grid the other 6 models share. nilearn would resample it
without erroring -- geometrically it's just another volume -- so a silent
diff between block-constant ROI regions and a true voxelwise map would look
like a normal result while being meaningless. See regionalize_saliency.py
if an ROI-space comparison is what's actually wanted.

slurm/compute_sanity_checks.sh currently has --randomize-model COMMENTED
OUT in both run_model/run_model_tabular, so the arch_* files this script
reads were produced by an earlier/other invocation, not guaranteed fresh --
this script treats a missing aggregate or arch file as a per-model WARN+skip
(never a hard failure), matching aggregate_saliency.py's own convention.

--randomize-model-variant: which --randomize-model-mode variant's arch_ file
to compare each model's aggregate against -- "all" (default, matches
--randomize-model-mode all) or "step{NN:02d}_{module}" for a
--randomize-model-mode cascade step, e.g. "step01_head" for
slurm/compute_sanity_checks_randomize_model.sh's head-only
--randomize-max-layers 1 run (aggregated via
RANDOMIZE_MODEL_VARIANT=linear slurm/aggregate_sanity_check_randomize_model.sh
-- see that script's own header comment). See sanity_prefix()'s docstring in
visualize_paper_nilearn.py for the full naming contract this relies on.

Usage (CLI):
    python compare_aggregate_arch_saliency.py --target age
    python compare_aggregate_arch_saliency.py --target sex --norm-method percentile --models brainiac swinbrain
    python compare_aggregate_arch_saliency.py --target age --randomize-model-variant step01_head --norm-method none

Usage (notebook/import):
    from compare_aggregate_arch_saliency import compare_and_plot
    compare_and_plot("age", norm_method="rank")
    compare_and_plot("age", norm_method="none", arch_variant="step01_head")

compare_and_plot_all_targets(): age/sex/bmi side by side, one figure
----------------------------------------------------------------------
A second entry point (notebook/import only -- no CLI wiring), for comparing
across targets rather than across models within one target. Lays out one
row per model x one column per target (age/sex/bmi, in that order) in a
SINGLE figure via plot_difference_grid(). Unlike compare_and_plot's single
GLOBAL vmax/colorbar for the whole figure, EVERY ROW gets its OWN vmax/
threshold/colorbar, computed from that model's own diff values across all
3 of its target columns combined -- so one model's much larger/smaller raw
scale (especially relevant for norm_method="none", see above) never washes
out another model's row by sharing one global scale, while a model's own
three target-columns stay on ONE consistent scale for genuine
age-vs-sex-vs-bmi comparison. A model missing a diff for any requested
target is dropped from every column (WARN printed), not left with a blank
cell.

Usage (notebook/import):
    from compare_aggregate_arch_saliency import compare_and_plot_all_targets
    compare_and_plot_all_targets(norm_method="none", kind="abs")
    compare_and_plot_all_targets(models=["brainiac", "swinbrain"], norm_method="rank")

compare_and_plot_all_targets_shared_scale(): same grid, ONE global scale
--------------------------------------------------------------------------
Third entry point, same age/sex/bmi x model grid as
compare_and_plot_all_targets, but computes ONE vmax/threshold pooled
across EVERY (target, model) cell in the figure -- not per row -- and uses
it everywhere, with a single figure-wide colorbar instead of one per row.
Color intensity is then directly comparable across the WHOLE figure, not
just within a row: the same color means the same absolute diff value in
every panel. The cost is the flip side of compare_and_plot_all_targets'
own tradeoff -- a model/target whose raw diff values are much smaller than
the figure's largest will render mostly dark/faint, honestly reflecting
that its values are small relative to the global scale (not a bug, not
hidden -- the printed global vmax/threshold make the scale explicit).

Usage (notebook/import):
    from compare_aggregate_arch_saliency import compare_and_plot_all_targets_shared_scale
    compare_and_plot_all_targets_shared_scale(norm_method="none", kind="abs")

compare_and_plot_all_targets_aggregate_shared_scale(): aggregate maps only
------------------------------------------------------------------------------
Fourth entry point. Same age/sex/bmi x model grid and single global-scale
colorbar as compare_and_plot_all_targets_shared_scale, but skips the
arch_ (randomized-weights) sanity-check entirely -- no diffing, no
norm_method, no arch_variant. Each cell is simply that model's own
ig_popmean AGGREGATE saliency map (from aggregate_saliency.py), resampled
onto the template grid and nothing else, so it answers "what does this
model's trained explanation look like" rather than "how much did the
explanation change under weight randomization" (the question every other
entry point in this file answers).

Usage (notebook/import):
    from compare_aggregate_arch_saliency import compare_and_plot_all_targets_aggregate_shared_scale
    compare_and_plot_all_targets_aggregate_shared_scale(kind="abs")
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from visualize_paper import DEFAULT_MODELS, default_aggregate_dir
from visualize_paper_nilearn import add_shared_colorbar, draw_model_row, montage_z_mm_coords, sanity_prefix
from visualize_saliency import (
    DEFAULT_BRAIN_MASK,
    DEFAULT_SALIENCY_DIR,
    DEFAULT_TEMPLATE,
    collect_kind_files,
    load_brain_mask,
    load_canonical_template,
    normalize_map,
    resample_saliency,
)

# Every add_shared_colorbar() call in this script (plot_difference_rows and plot_difference_grid,
# both the per-row and single-colorbar cases) passes this explicitly -- larger than
# add_shared_colorbar's own matplotlib-default tick label size, for readability at this file's
# typical figure sizes. Scoped to this script only: visualize_paper_nilearn.py's own main() calls
# add_shared_colorbar without it, so its ticks stay at the original default size.
COLORBAR_TICK_FONTSIZE = 18


# ---------------------------------------------------------------------------
# Normalization schemes
# ---------------------------------------------------------------------------

def normalize_map_robust(data, brain_mask_data, kind, mad_clip=3.0):
    """
    Median/MAD-based robust z-score -- same [-1,1]/[0,1] output convention
    as normalize_map, but scaled by the in-brain median absolute deviation
    (times the usual 1.4826 consistency constant, so MAD approximates std
    for a normal distribution) instead of normalize_map("zscore")'s
    mean/std, which a few extreme voxels can dominate.
    """
    vals = data[brain_mask_data] if brain_mask_data is not None else data.ravel()

    if kind == "abs":
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) * 1.4826
        upper = med + mad_clip * (mad if mad > 0 else 1.0)
        upper = upper if upper > 0 else 1.0
        return np.clip(data / upper, 0.0, 1.0)

    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    scale = mad_clip * (mad if mad > 0 else 1.0)
    scale = scale if scale > 0 else 1.0
    return np.clip((data - med) / scale, -1.0, 1.0)


def normalize_map_rank(data, brain_mask_data, kind):
    """
    Each in-brain voxel replaced by its own percentile rank (0..1) within
    that single map's in-brain value distribution -- scale-invariant BY
    CONSTRUCTION (unlike normalize_map/normalize_map_robust, which rescale
    but preserve relative magnitude), so a trained map and a
    randomly-initialized map's completely different raw gradient scales
    can never bias a rank-based diff. Out-of-brain voxels are left at 0.

    "abs": rank -> [0, 1] directly (already one-sided).
    "signed": rank computed on |value| (so magnitude, not sign, drives the
    rank), then the original sign is reapplied -> [-1, 1]. This keeps
    "rank" comparable in spirit to normalize_map's own signed convention
    (a rescaled magnitude that keeps its sign), rather than conflating a
    large negative value and a large positive value into the same rank.
    """
    out = np.zeros_like(data, dtype=np.float64)
    if brain_mask_data is not None:
        idx = brain_mask_data
    else:
        idx = np.ones_like(data, dtype=bool)

    if kind == "abs":
        vals = data[idx]
        if vals.size == 0:
            return out
        ranks = rankdata(vals, method="average") / vals.size
        out[idx] = ranks
        return out

    vals = data[idx]
    if vals.size == 0:
        return out
    abs_ranks = rankdata(np.abs(vals), method="average") / vals.size
    signs = np.sign(vals)
    out[idx] = abs_ranks * signs
    return out


NORM_FUNCS = {
    "percentile": lambda data, brain_mask_data, kind, **kw: normalize_map(
        data, brain_mask_data, kind, method="percentile", percentile=kw.get("norm_percentile", 99.0)
    ),
    "zscore": lambda data, brain_mask_data, kind, **kw: normalize_map(
        data, brain_mask_data, kind, method="zscore", zclip=kw.get("norm_zclip", 3.0)
    ),
    "robust": lambda data, brain_mask_data, kind, **kw: normalize_map_robust(
        data, brain_mask_data, kind, mad_clip=kw.get("mad_clip", 3.0)
    ),
    "rank": lambda data, brain_mask_data, kind, **kw: normalize_map_rank(data, brain_mask_data, kind),
    # Identity -- returns each map's raw values completely unchanged. NOT
    # recommended (see module docstring: a trained vs. randomly-initialized
    # version of the same backbone have wildly different raw gradient
    # scales, so this diff mostly measures "which map had bigger numbers").
    # Exists as an explicit opt-out for exploratory/debugging use, e.g.
    # sanity-checking that a normalization scheme isn't itself hiding a
    # difference that's obvious in raw units.
    "none": lambda data, brain_mask_data, kind, **kw: data,
}


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------

def resolve_aggregate_path(model, aggregate_dir, stat="mean", kind="abs", allow_fs_roi=False):
    """
    aggregate_path for `model`, or None with a printed WARN if
    missing/unsupported -- the aggregate-only half of resolve_pair's own
    lookup, factored out so an aggregate-only entry point (e.g.
    compare_and_plot_all_targets_aggregate_shared_scale, which never needs
    an arch_ sanity-check file to exist at all) doesn't drag in resolve_pair's
    arch_ lookup just to get here.

    `allow_fs_roi`: fs_roi is rejected by default for the same reason every
    other caller in this file rejects it -- its aggregate file is
    piecewise-constant per FreeSurfer ROI on the DKT segmentation grid, not
    smooth per-voxel data on the shared MNI-registered grid every other
    model uses, so MIXING it into a multi-model figure/pooled-vmax
    computation compares apples to oranges (see regionalize_saliency.py's
    module docstring for the full reasoning, and every other rejection
    site's own comment). That concern is specific to comparing it
    alongside other models, though -- it's still geometrically a normal
    NIfTI volume nilearn/resample_saliency can correctly resample onto the
    shared template on its own (see that function's docstring: "nilearn
    would resample it without erroring"), so a caller showing fs_roi BY
    ITSELF, with nothing else to compare it against, can pass True here --
    see compare_and_plot_fs_roi_shared_scale.
    """
    if model == "fs_roi" and not allow_fs_roi:
        print(f"WARN: fs_roi lives on the DKT-ROI grid, not the shared voxel grid -- skipping "
              f"(see regionalize_saliency.py if an ROI-space comparison is wanted instead)")
        return None

    agg_files = collect_kind_files(aggregate_dir, kind, [f"{model}_ig_reference_aggregate_{stat}"], None)
    if not agg_files:
        print(f"WARN: no ig_popmean aggregate '{kind}' file found for model {model!r} ({stat}) in {aggregate_dir} -- skipping")
        return None
    return agg_files[0]


def resolve_pair(model, saliency_dir, aggregate_dir, stat="mean", kind="abs", arch_variant="all"):
    """
    (aggregate_path, arch_path) for `model`, or (None, None) with a printed
    WARN if either is missing/unsupported -- mirrors
    visualize_paper_nilearn.py's own aggregate/sanity-check file-lookup
    exactly (see that script's main(), the collect_kind_files/sanity_prefix
    calls), so this script's file resolution can never silently drift from
    what the batch figure script already treats as correct.

    `arch_variant`: which --randomize-model-mode variant's arch_ file to
    look up -- see sanity_prefix()'s own docstring (visualize_paper_nilearn.py)
    for the "all" vs "step{NN:02d}_{module}" naming contract. Default "all"
    matches slurm/aggregate_sanity_check_randomize_model.sh's own default.
    """
    agg_path = resolve_aggregate_path(model, aggregate_dir, stat=stat, kind=kind)
    if agg_path is None:
        return None, None

    arch_path = saliency_dir / f"{sanity_prefix(model, 'arch', arch_variant)}ig_reference_{kind}_saliency.nii.gz"
    if not arch_path.exists():
        print(f"WARN: no arch sanity-check '{kind}' file found for model {model!r} ({arch_variant!r} variant) "
              f"at {arch_path} -- skipping")
        return None, None

    return agg_path, arch_path


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_abs_difference_map(agg_path, arch_path, template_img, brain_mask_data, kind, norm_method,
                                norm_percentile=99.0, norm_zclip=3.0, mad_clip=3.0):
    """
    Loads+resamples both maps onto template_img's grid, normalizes each
    independently via norm_method (see module docstring), and returns
    (diff, stats):
      diff  -- axial-first array, norm(agg) - norm(arch), masked to
               brain_mask_data (0 outside the brain). Despite this
               function's name (kept for API stability -- every caller
               already imports it by this name), the sign is only actually
               collapsed via np.abs() when `kind == "abs"`: a signed source
               (`kind == "signed"`) keeps its true sign, so which map's
               value was larger stays visible (e.g. "agg is more positive
               than arch here" vs. "agg is more negative"), not just how
               far apart they are. `kind == "abs"` inputs are already
               one-sided by construction (min 0), so np.abs() there only
               ever collapses "which of the two had the bigger magnitude"
               -- never a true sign -- same as before this docstring note.
      stats -- dict of in-brain mean/median (signed for kind=="signed", so
               they show NET directional bias -- e.g. a mean near 0 can
               mean "no difference" OR "differences cancel out", not
               distinguishable from this number alone) / p95 (always taken
               on |diff|, i.e. magnitude, even for a signed diff -- matches
               nilearn's own threshold semantics, see plot_difference_rows)
               of `diff`, plus the two RAW (pre-normalization, in-brain)
               maps' own Spearman correlation -- context tying this diff
               back to compute_saliency_correlations.py's own scalar metric
               for the same pair (a large diff should generally accompany a
               low |correlation|, not contradict it).
    """
    norm_fn = NORM_FUNCS[norm_method]
    kwargs = {"norm_percentile": norm_percentile, "norm_zclip": norm_zclip, "mad_clip": mad_clip}

    agg_raw = resample_saliency(agg_path, template_img)
    arch_raw = resample_saliency(arch_path, template_img)

    agg_norm = norm_fn(agg_raw, brain_mask_data, kind, **kwargs)
    arch_norm = norm_fn(arch_raw, brain_mask_data, kind, **kwargs)

    raw_diff = agg_norm - arch_norm
    diff = raw_diff if kind == "signed" else np.abs(raw_diff)
    if brain_mask_data is not None:
        diff = np.where(brain_mask_data, diff, 0.0)
        agg_vals, arch_vals, diff_vals = agg_raw[brain_mask_data], arch_raw[brain_mask_data], diff[brain_mask_data]
    else:
        agg_vals, arch_vals, diff_vals = agg_raw.ravel(), arch_raw.ravel(), diff.ravel()

    corr, _ = spearmanr(agg_vals, arch_vals)
    abs_diff_vals = np.abs(diff_vals)
    stats = {
        "diff_mean": float(diff_vals.mean()) if diff_vals.size else float("nan"),
        "diff_median": float(np.median(diff_vals)) if diff_vals.size else float("nan"),
        "diff_p95": float(np.percentile(abs_diff_vals, 95)) if diff_vals.size else float("nan"),
        "raw_spearman": float(corr),
    }
    return diff, stats


def compute_all_differences(target, models=DEFAULT_MODELS, stat="mean", kind="abs", norm_method="rank",
                             saliency_dir=None, aggregate_dir=None, template_path=DEFAULT_TEMPLATE,
                             brain_mask_path=DEFAULT_BRAIN_MASK, norm_percentile=99.0, norm_zclip=3.0, mad_clip=3.0,
                             arch_variant="all"):
    """
    Loops resolve_pair + compute_abs_difference_map over `models` for one
    `target`. Returns (diff_by_model, summary_df):
      diff_by_model -- {model: {"diff": array, "stats": {...}}}, skipping
                        any model resolve_pair/compute_abs_difference_map
                        couldn't resolve (WARN printed per skip).
      summary_df    -- one row per successfully-computed model, columns
                        diff_mean/diff_median/diff_p95/raw_spearman -- for a
                        quick side-by-side table in a notebook.

    `arch_variant`: see resolve_pair()'s own docstring -- which
    --randomize-model-mode variant's arch_ file to compare each model's
    aggregate against (default "all").
    """
    saliency_dir = Path(saliency_dir) if saliency_dir else DEFAULT_SALIENCY_DIR / target
    aggregate_dir = Path(aggregate_dir) if aggregate_dir else default_aggregate_dir(target)

    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)

    diff_by_model = {}
    rows = []
    for model in models:
        agg_path, arch_path = resolve_pair(model, saliency_dir, aggregate_dir, stat=stat, kind=kind, arch_variant=arch_variant)
        if agg_path is None:
            continue
        diff, stats = compute_abs_difference_map(
            agg_path, arch_path, template_img, brain_mask_data, kind, norm_method,
            norm_percentile=norm_percentile, norm_zclip=norm_zclip, mad_clip=mad_clip,
        )
        diff_by_model[model] = {"diff": diff, "stats": stats}
        rows.append({"model": model, **stats})

    summary_df = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame(
        columns=["diff_mean", "diff_median", "diff_p95", "raw_spearman"]
    )
    return diff_by_model, summary_df


def compute_all_aggregates(target, models=DEFAULT_MODELS, stat="mean", kind="abs", aggregate_dir=None,
                            template_path=DEFAULT_TEMPLATE, brain_mask_path=DEFAULT_BRAIN_MASK,
                            allow_fs_roi=False):
    """
    Aggregate-only sibling of compute_all_differences: loads+resamples each
    model's own ig_popmean aggregate saliency map for `target` -- NO arch_
    sanity-check file, no diffing, no NORM_FUNCS normalization. The map is
    shown exactly as aggregate_saliency.py wrote it, just resampled onto
    the template grid (resample_saliency) the same way every other map in
    this script is, and masked to zero outside brain_mask_data -- same
    convention compute_abs_difference_map's own `diff` already follows --
    so skull/scalp signal or resampling spillover near the brain boundary
    in the source .nii.gz never survives into the figure. See
    compare_and_plot_all_targets_aggregate_shared_scale.

    Returns (map_by_model, summary_df):
      map_by_model -- {model: {"map": array}}, skipping any model
                       resolve_aggregate_path couldn't resolve (WARN
                       printed per skip, same convention as
                       compute_all_differences).
      summary_df   -- one row per successfully-loaded model, columns
                       map_mean/map_median/map_p95 (in-brain) -- descriptive
                       stats only; there's no second map here to compute a
                       Spearman correlation against.
    """
    aggregate_dir = Path(aggregate_dir) if aggregate_dir else default_aggregate_dir(target)

    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)

    map_by_model = {}
    rows = []
    for model in models:
        agg_path = resolve_aggregate_path(model, aggregate_dir, stat=stat, kind=kind, allow_fs_roi=allow_fs_roi)
        if agg_path is None:
            continue
        agg_raw = resample_saliency(agg_path, template_img)
        if brain_mask_data is not None:
            vals = agg_raw[brain_mask_data]
            agg_raw = np.where(brain_mask_data, agg_raw, 0.0)
        else:
            vals = agg_raw.ravel()
        map_by_model[model] = {"map": agg_raw}
        rows.append({
            "model": model,
            "map_mean": float(vals.mean()) if vals.size else float("nan"),
            "map_median": float(np.median(vals)) if vals.size else float("nan"),
            "map_p95": float(np.percentile(np.abs(vals), 95)) if vals.size else float("nan"),
        })

    summary_df = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame(
        columns=["map_mean", "map_median", "map_p95"]
    )
    return map_by_model, summary_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _diff_to_ras_nifti(diff_axial_first, template_img):
    """
    Packages an already-computed axial-first diff array as a RAS
    nib.Nifti1Image on template_img's own grid/affine -- the same
    transpose-back step visualize_paper_nilearn.py's masked_stat_img does
    for a freshly-loaded map, factored out here since this diff array is
    already resampled/normalized/masked (no re-normalization needed).
    """
    data_ras = np.transpose(diff_axial_first, (2, 1, 0))
    return nib.Nifti1Image(data_ras, template_img.affine, template_img.header)


def plot_difference_rows(diff_by_model, template_img, brain_mask_data, n_montage=4, threshold_percentile=75.0,
                          cut_coords=None, title=None, output=None, label_scale=1.0, kind="abs"):
    """
    One row per model in diff_by_model, reusing visualize_paper_nilearn.py's
    draw_model_row/add_shared_colorbar/montage_z_mm_coords for the exact
    same ortho+montage figure styling every other cross-model figure in this
    repo uses.

    `kind`: the SAME kind compute_abs_difference_map was called with for
    these diffs -- "abs" (default): diff is non-negative by construction
    (see that function's docstring), drawn one-sided (vmin=0, YlOrRd,
    non-symmetric colorbar). "signed": diff keeps its true sign (which map
    was larger, not just how far apart), drawn diverging (vmin=-vmax,
    coolwarm, symmetric_cbar=True, matching how every other TRUE signed map
    in this repo is drawn -- see draw_model_row's own docstring) so the sign
    is actually visible instead of being thrown away. Passing a `kind` here
    that doesn't match what compute_abs_difference_map actually produced
    (e.g. "abs" diffs drawn as "signed") would silently mismatch the
    colorbar/thresholding below against the data -- always pass through the
    same `kind` you diffed with (compare_and_plot does this for you).

    threshold/vmax below are always computed on |diff| (magnitude),
    regardless of `kind` -- nilearn's own `threshold=` argument to
    plot_stat_map already hides voxels by |value| < threshold internally
    (not by raw value), so doing the SAME on our end keeps row_threshold's
    percentile consistent with what nilearn will actually do with it: for
    "signed" diffs this also means large-magnitude NEGATIVE voxels survive
    thresholding exactly like large positive ones do, instead of only the
    positive half surviving (which silently dropping to `vals > 0` here
    would have caused for a genuinely signed diff, right before nilearn's
    own -- correct -- |value| thresholding would have been applied anyway).
    """
    if not diff_by_model:
        raise SystemExit("plot_difference_rows: diff_by_model is empty -- nothing to plot.")

    cmap = "coolwarm" if kind == "signed" else "YlOrRd"

    stat_imgs = [(model, _diff_to_ras_nifti(d["diff"], template_img)) for model, d in diff_by_model.items()]

    def row_threshold(img):
        if threshold_percentile <= 0:
            return 1e-6
        vals = np.abs(img.get_fdata())
        vals = vals[vals > 0]
        if vals.size == 0:
            return 1e-6
        return max(float(np.percentile(vals, threshold_percentile)), 1e-6)

    vmax = max(float(np.percentile(np.abs(img.get_fdata())[np.abs(img.get_fdata()) > 0], 99.5)) for _, img in stat_imgs)
    vmax = vmax if vmax > 0 else 1.0

    montage_z_coords = montage_z_mm_coords(template_img, brain_mask_data, n_montage) if n_montage else []

    n_rows = len(stat_imgs)
    height_ratios = []
    for _ in range(n_rows):
        height_ratios += [1.5, 1.5 if n_montage else 0.0]
    fig = plt.figure(figsize=(14, 4.2 * n_rows), facecolor="black")

    MONTAGE_LEFT, MONTAGE_RIGHT = 0.05, 0.98
    ORTHO_PANEL_COUNT = 3
    montage_panel_count = n_montage or 4
    montage_tile_width = (MONTAGE_RIGHT - MONTAGE_LEFT) / montage_panel_count
    ORTHO_WIDTH_FRAC = montage_tile_width * ORTHO_PANEL_COUNT
    montage_center = (MONTAGE_LEFT + MONTAGE_RIGHT) / 2
    ortho_left = montage_center - ORTHO_WIDTH_FRAC / 2
    ortho_right = montage_center + ORTHO_WIDTH_FRAC / 2

    common_kwargs = dict(height_ratios=height_ratios, hspace=0.05, top=0.94 if title else 0.99, bottom=0.06)
    gs_ortho = fig.add_gridspec(2 * n_rows, 1, left=ortho_left, right=ortho_right, **common_kwargs)
    gs_montage = fig.add_gridspec(2 * n_rows, 1, left=ortho_left, right=ortho_right, **common_kwargs)

    for row, (model, stat_img) in enumerate(stat_imgs):
        draw_model_row(
            fig, gs_ortho[2 * row, 0], gs_montage[2 * row + 1, 0], stat_img, template_img, kind,
            cut_coords, montage_z_coords, vmax, row_threshold(stat_img), label=model, label_scale=label_scale,
            cmap=cmap,
        )

    cax = fig.add_axes([0.3, 0.015, 0.4, 0.02])
    add_shared_colorbar(fig, cax, cmap, kind, vmax, tick_fontsize=COLORBAR_TICK_FONTSIZE)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"Saved: {output}")
    return fig


def compare_and_plot(target, models=DEFAULT_MODELS, stat="mean", kind="abs", norm_method="rank",
                      saliency_dir=None, aggregate_dir=None, template_path=DEFAULT_TEMPLATE,
                      brain_mask_path=DEFAULT_BRAIN_MASK, norm_percentile=99.0, norm_zclip=3.0, mad_clip=3.0,
                      n_montage=4, threshold_percentile=75.0, cut_coords=None, title=None, output=None,
                      arch_variant="all"):
    """
    One-call convenience: compute_all_differences + plot_difference_rows,
    the main notebook entry point. Returns (fig, summary_df).

    `arch_variant`: see resolve_pair()'s own docstring -- which
    --randomize-model-mode variant's arch_ file to compare each model's
    aggregate against (default "all").
    """
    diff_by_model, summary_df = compute_all_differences(
        target, models=models, stat=stat, kind=kind, norm_method=norm_method, saliency_dir=saliency_dir,
        aggregate_dir=aggregate_dir, template_path=template_path, brain_mask_path=brain_mask_path,
        norm_percentile=norm_percentile, norm_zclip=norm_zclip, mad_clip=mad_clip, arch_variant=arch_variant,
    )
    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)
    norm_label = "un-normalized (raw)" if norm_method == "none" else f"{norm_method}-normalized"
    fig = plot_difference_rows(
        diff_by_model, template_img, brain_mask_data, n_montage=n_montage,
        threshold_percentile=threshold_percentile, cut_coords=cut_coords,
        title=title or f"|aggregate - arch| ({target}, {arch_variant} variant, {norm_label}, {kind})", output=output,
        kind=kind,
    )
    return fig, summary_df


# ---------------------------------------------------------------------------
# Multi-target (age/sex/bmi side by side) grid figure
# ---------------------------------------------------------------------------

DEFAULT_SALIENCY_AFFINE_ROOT = Path(__file__).resolve().parents[1] / "saliency_affine"
DEFAULT_TARGETS = ("age", "sex", "bmi")
# plot_difference_grid's disable_threshold uses this instead of threshold=None -- see that
# parameter's own docstring for why a real (if tiny) threshold, not "no threshold at all", is
# what actually disables VISIBLE thresholding without also breaking background transparency.
_NO_THRESHOLD_FLOOR = 1e-9


def plot_difference_grid(diff_by_target_model, template_img, brain_mask_data, targets, n_montage=4,
                          threshold_percentile=75.0, cut_coords=None, title=None, output=None,
                          label_scale=1.0, kind="abs", vmax=None, threshold=None, single_colorbar=False,
                          value_key="diff", disable_threshold=False, show_title=True, show_colorbar=True,
                          colorbar_width=0.4, colorbar_height=0.02):
    """
    Grid-layout sibling of plot_difference_rows: one row per model, one
    COLUMN per target (age/sex/bmi side by side, same ortho+montage detail
    level per cell). By default (vmax=None, threshold=None) EACH ROW gets
    its own vmax/threshold/colorbar -- computed from that model's own diff
    values pooled across all of its target columns -- instead of
    plot_difference_rows' single value shared by the whole figure. See
    module docstring for why (norm_method="none" in particular: raw scales
    differ wildly between models, so one global scale would wash out a
    weaker-scaled model's row).

    `value_key`: which key holds each cell's array in
    `diff_by_target_model[target][model][...]` -- "diff" (default, matches
    compute_all_differences' own output shape) for the usual diff figures,
    or e.g. "map" for a plain (non-diffed) map grid -- see
    compare_and_plot_all_targets_aggregate_shared_scale, which reuses this
    same function to draw raw aggregate saliency maps instead of diffs.
    Purely a dict-key switch; every other part of this function (layout,
    vmax/threshold logic, colorbar) is unaffected either way.

    `disable_threshold`: when True, skips ALL threshold computation and uses
    a tiny fixed floor (_NO_THRESHOLD_FLOOR, currently 1e-9) for every row
    instead -- NOT threshold=None. threshold=None would seem like the more
    literal "disabled," and is nilearn's own documented meaning for
    plot_stat_map's threshold param ("the image is not thresholded"), but
    it back fires here: everywhere outside brain_mask_data was zeroed by
    the caller (see compute_all_aggregates/compute_abs_difference_map), and
    with no threshold at all that exact 0.0 is just another real data value
    -- it gets colored in (the colormap's center color, for a diverging
    signed map) instead of staying transparent, painting a solid rectangle
    over the anatomical background for the whole non-brain region. A tiny
    real threshold keeps nilearn's actual transparency mechanism active
    (pixels with |value| < threshold are masked out, not just recolored) so
    the exact-zero background stays invisible while everything with any
    real signal (even very small) still renders -- functionally "no
    thresholding" without the transparency side effect. Also tried: NaN
    instead of 0.0 for the background -- doesn't work, nilearn's own image
    loading unconditionally replaces non-finite values with 0.0 before the
    plot/threshold logic ever sees them (nilearn/_utils/niimg.py
    ensure_finite_data), so it collapses to the exact same problem.
    Takes priority over `threshold`/`threshold_percentile`, which are both
    ignored when this is True. Not the same as threshold_percentile=0,
    which computes a real (if tiny) 1e-6 floor via the SAME per-row/global
    vmax pooling path as any other percentile -- this bypasses that pooling
    entirely.

    `vmax`/`threshold`: pass explicit floats (see
    compare_and_plot_all_targets_shared_scale, which computes ONE pooled
    value across every row and passes both here) to use the SAME value for
    every row instead of computing one per row -- the two are independent:
    you can override one without the other. `single_colorbar`: when True,
    draws ONE HORIZONTAL colorbar at the bottom of the whole figure (same
    fixed rect visualize_paper_nilearn.py's own single colorbar uses)
    instead of one VERTICAL colorbar per row in the right-margin strip --
    only meaningful when `vmax` is also fixed (per-row colorbars would
    otherwise each need their own value); sensible on its own too if you
    just want one colorbar regardless. The right-margin strip is only
    reserved (GRID_RIGHT narrows) for the per-row vertical case -- a
    single_colorbar figure uses the full grid width instead.

    `diff_by_target_model`: {target: {model: {"diff": array, "stats": {...}}}}
    -- one compute_all_differences() result per target (compare_and_plot_all_targets
    builds this). `targets`: the column order (left to right).

    Row selection: a model gets a row only if diff_by_target_model[t] has it
    for EVERY t in targets (set intersection) -- a model missing for even
    one target is dropped from every column (WARN printed), not left with a
    blank cell for that one target.

    vmax AND threshold are both per ROW (pooled across that model's target
    columns, one value each for the whole row) -- the point of this figure:
    one model's three targets stay on one consistent color SCALE, with the
    same "how much are we hiding" cutoff applied everywhere in that row.
    NOTE: when one target's raw scale is much larger than another's
    (common for norm_method="none" -- see module docstring), a pooled
    threshold gets calibrated against the larger target and can end up
    ABOVE most of a smaller target's own values, hiding a real (if smaller)
    signal as "below threshold" in that column. Each row's vmax/threshold
    are printed so this is always visible rather than a silent choice --
    lower `threshold_percentile` (or pass a per-call override) if a
    column looks suspiciously blank.

    `kind`: same meaning as plot_difference_rows' own `kind` -- picks the
    colormap (coolwarm for "signed", YlOrRd for "abs") and, via
    draw_model_row, whether each row is drawn one-sided (vmin=0) or
    diverging (vmin=-vmax, symmetric) -- see compute_abs_difference_map's
    docstring for what "signed" vs "abs" means for the diff's own sign.

    `show_title`/`show_colorbar`: styled after visualize_paper_nilearn.py's
    own --title/--no-colorbar CLI flags (single-target figure) -- set
    either to False to drop that element entirely (e.g. for a paper figure
    whose title/color scale is given in the caption instead), rather than
    just passing title=None (which here still computes and would draw a
    default title string -- see compare_and_plot_all_targets_aggregate_shared_scale,
    whose own `title` parameter always has a fallback). show_colorbar=False
    also reclaims the right-margin strip the colorbar(s) would have used
    (GRID_RIGHT widens to 0.98, matching visualize_paper_nilearn.py's own
    MONTAGE_RIGHT), same as that script widening its own bottom margin when
    --no-colorbar drops its bottom horizontal bar.

    `colorbar_width`/`colorbar_height`: single_colorbar's horizontal bar
    width/height, as figure fractions (defaults 0.4/0.02, matching
    visualize_paper_nilearn.py's own fixed [0.3, 0.015, 0.4, 0.02] rect) --
    the bar stays horizontally centered as `colorbar_width` grows/shrinks.
    Both ignored for the per-row vertical colorbars (fixed size via
    CBAR_LEFT/CBAR_RIGHT below) and when show_colorbar is False.
    """
    if not diff_by_target_model or not targets:
        raise SystemExit("plot_difference_grid: diff_by_target_model/targets is empty -- nothing to plot.")

    models_per_target = [set(diff_by_target_model[t].keys()) for t in targets]
    rows = [m for m in models_per_target[0] if all(m in s for s in models_per_target)]
    rows = sorted(rows)
    dropped = sorted(set().union(*models_per_target) - set(rows))
    if dropped:
        print(f"WARN: dropping model(s) missing a diff for at least one of {list(targets)}: {dropped}")
    if not rows:
        raise SystemExit(f"plot_difference_grid: no model has a diff for every target in {list(targets)}.")

    cmap = "coolwarm" if kind == "signed" else "YlOrRd"

    def in_brain_abs_vals(model, target):
        arr = diff_by_target_model[target][model][value_key]
        vals = np.abs(arr[brain_mask_data]) if brain_mask_data is not None else np.abs(arr).ravel()
        return vals[vals > 0]

    def percentile_or_floor(vals, pct):
        return max(float(np.percentile(vals, pct)), 1e-6) if vals.size and pct > 0 else 1e-6

    row_vmax = {}
    row_threshold_val = {}
    need_pooled_vals = vmax is None or (threshold is None and not disable_threshold)
    if disable_threshold:
        print(f"plot_difference_grid: threshold DISABLED (disable_threshold=True) -- every nonzero "
              f"in-brain voxel is shown (floored at {_NO_THRESHOLD_FLOOR:g} so the masked-to-zero "
              f"background outside the brain still renders transparent, not as a solid color).")
    if vmax is None or (threshold is None and not disable_threshold):
        computed_desc = []
        if vmax is None:
            computed_desc.append("vmax (99.5th pctile, halved)")
        if threshold is None and not disable_threshold:
            computed_desc.append(f"threshold ({threshold_percentile:.1f}th pctile)")
        print(f"plot_difference_grid: per-row {' and '.join(computed_desc)}, pooled across {list(targets)} "
              f"-- see this function's docstring for the tradeoff:")
    for model in rows:
        pooled_vals = np.concatenate([in_brain_abs_vals(model, t) for t in targets]) if need_pooled_vals else None
        # vmax's own /2 is an ad-hoc display gain applied ONLY when computing it per row here --
        # an explicit `vmax` override (e.g. from compare_and_plot_all_targets_shared_scale) is used
        # exactly as given, no hidden scaling.
        row_vmax[model] = vmax if vmax is not None else percentile_or_floor(pooled_vals, 99.5) / 2
        if disable_threshold:
            row_threshold_val[model] = _NO_THRESHOLD_FLOOR
        else:
            row_threshold_val[model] = threshold if threshold is not None else percentile_or_floor(pooled_vals, threshold_percentile)
        if pooled_vals is not None:
            print(f"  {model:20s} vmax={row_vmax[model]:.6f} threshold={row_threshold_val[model]:.6f}")

    n_rows = len(rows)
    n_cols = len(targets)
    montage_z_coords = montage_z_mm_coords(template_img, brain_mask_data, n_montage) if n_montage else []

    height_ratios = []
    for _ in range(n_rows):
        height_ratios += [1.5, 1.5 if n_montage else 0.0]
    fig = plt.figure(figsize=(13 * n_cols, 4.2 * n_rows), facecolor="black")

    # GRID_LEFT matches visualize_paper_nilearn.py's own MONTAGE_LEFT=0.05. GRID_RIGHT only needs
    # to narrow for the right-margin CBAR strip when drawing PER-ROW vertical colorbars -- a
    # single_colorbar figure now uses one horizontal bar at the bottom instead (matching that
    # script's own single colorbar), so it doesn't need the right margin reserved at all.
    GRID_LEFT = 0.05
    GRID_RIGHT = 0.90 if (show_colorbar and not single_colorbar) else 0.98
    CBAR_LEFT, CBAR_RIGHT = 0.93, 0.955
    ORTHO_PANEL_COUNT = 3
    montage_panel_count = n_montage or 4
    # COLUMN_GUTTER: fixed figure-fraction gap between adjacent task columns. Columns are sized to
    # their own actual content width (content_width, below) and packed with exactly this gap
    # between them -- NOT an equal 1/n_cols division of [GRID_LEFT, GRID_RIGHT] with the (narrower)
    # ortho/montage content then centered inside each division. That equal-division approach left
    # BOTH the ortho row AND the montage row confined to a width of only
    # montage_tile_width * ORTHO_PANEL_COUNT (3/4 of the column's own slice at the default
    # n_montage=4), centered -- so a quarter of every column's allotted width sat empty as margin
    # on each side, and adjacent columns' margins summed into a big visible gap between them.
    COLUMN_GUTTER = 0.02
    content_width = (GRID_RIGHT - GRID_LEFT - COLUMN_GUTTER * (n_cols - 1)) / n_cols

    # Two separate text rows share the top margin when a title is actually drawn (suptitle above,
    # column headers below it) -- top/header_y/suptitle's own y are spaced so they never collide
    # regardless of figure height (both are figure-fraction, not row-count-relative). `show_title`
    # gates this exactly like the `if title and show_title:` draw call below -- title=None and
    # show_title=False both mean "no suptitle, so tighten the top margin the same way".
    draw_title = bool(title) and show_title
    # bottom mirrors visualize_paper_nilearn.py's own bottom=0.01 if args.no_colorbar else 0.06 --
    # only shrinks when there's genuinely no colorbar at all; per-row vertical colorbars don't sit
    # in the bottom margin, but 0.06 was already the (untouched, working) value for that case too.
    # single_colorbar's horizontal bar DOES sit in this margin, though (at a fixed y=0.015 -- see
    # shared_cax below), so a taller colorbar_height needs a taller bottom margin to still leave
    # room below the bar for its tick labels -- 0.06 (colorbar_height's own old hardcoded 0.02
    # default) only barely fit them already.
    if show_colorbar and single_colorbar:
        bottom = max(0.06, 0.015 + colorbar_height + 0.025)
    elif show_colorbar:
        bottom = 0.06
    else:
        bottom = 0.01
    common_kwargs = dict(height_ratios=height_ratios, hspace=0.05, top=0.88 if draw_title else 0.95, bottom=bottom)

    montage_tile_width = content_width / montage_panel_count
    ortho_width_frac = montage_tile_width * ORTHO_PANEL_COUNT

    gs_pairs = []
    for col in range(n_cols):
        montage_left = GRID_LEFT + col * (content_width + COLUMN_GUTTER)
        montage_right = montage_left + content_width
        col_center = (montage_left + montage_right) / 2
        ortho_left = col_center - ortho_width_frac / 2
        ortho_right = col_center + ortho_width_frac / 2
        gs_ortho = fig.add_gridspec(2 * n_rows, 1, left=ortho_left, right=ortho_right, **common_kwargs)
        # Tried giving this the full content_width (montage_left:montage_right) instead of the
        # narrower ortho_left:ortho_right span, expecting the 4 montage tiles to pack edge to edge
        # and close up the inter-column gap further -- reverted: nilearn doesn't tile-pack into the
        # extra width, it inserts an uneven gap between tiles instead (verified visually), which
        # looks worse than the original centered-in-3/4-width layout. Keeping ortho_left:ortho_right
        # here, matching visualize_paper_nilearn.py's own single-column convention.
        gs_montage = fig.add_gridspec(2 * n_rows, 1, left=ortho_left, right=ortho_right, **common_kwargs)
        gs_pairs.append((gs_ortho, gs_montage))

        header_y = 0.925 if draw_title else 0.985
        fig.text(col_center, header_y, targets[col].upper(), fontsize=18, fontweight="bold",
                  ha="center", va="center", color="white")

    for row, model in enumerate(rows):
        row_ortho_ax = None
        for col, target in enumerate(targets):
            stat_img = _diff_to_ras_nifti(diff_by_target_model[target][model][value_key], template_img)
            gs_ortho, gs_montage = gs_pairs[col]
            ortho_ax = draw_model_row(
                fig, gs_ortho[2 * row, 0], gs_montage[2 * row + 1, 0], stat_img, template_img, kind,
                cut_coords, montage_z_coords, row_vmax[model], row_threshold_val[model],
                label=model if col == 0 else "", label_scale=label_scale, cmap=cmap,
            )
            row_ortho_ax = ortho_ax

        # single_colorbar's bar is drawn once, at a fixed bottom-of-figure position, after this
        # loop -- see below -- so nothing to do here for that case, just the per-row vertical one.
        if show_colorbar and not single_colorbar:
            pos = row_ortho_ax.get_position()
            row_cax = fig.add_axes([CBAR_LEFT, pos.y0, CBAR_RIGHT - CBAR_LEFT, pos.y1 - pos.y0])
            add_shared_colorbar(fig, row_cax, cmap, kind, row_vmax[model], orientation="vertical",
                                tick_fontsize=COLORBAR_TICK_FONTSIZE)

    if show_colorbar and single_colorbar:
        # Horizontal, fixed at the bottom of the whole figure -- same rect visualize_paper_nilearn.py's
        # own single shared colorbar uses (see its add_shared_colorbar call in main()), rather than
        # this function's previous vertical right-margin strip. Uses the last row's vmax -- with
        # `vmax` fixed (the intended use of single_colorbar), every row's row_vmax is identical, so
        # any row's value works here.
        shared_cax = fig.add_axes([0.5 - colorbar_width / 2, 0.015, colorbar_width, colorbar_height])
        add_shared_colorbar(fig, shared_cax, cmap, kind, row_vmax[rows[-1]], orientation="horizontal",
                            tick_fontsize=COLORBAR_TICK_FONTSIZE)

    if draw_title:
        # Explicit y (above header_y=0.925 above) -- default suptitle y would otherwise land close
        # enough to the column headers to visually overlap a long title string.
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=0.995)

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"Saved: {output}")
    return fig


def _gather_target_diffs(targets, models, stat, kind, norm_method, arch_variant, saliency_affine_root,
                          norm_percentile, norm_zclip, mad_clip, template_path, brain_mask_path):
    """
    Shared by compare_and_plot_all_targets and compare_and_plot_all_targets_shared_scale:
    loops compute_all_differences() once per target, returns
    (diff_by_target_model, summary_df) -- see compare_and_plot_all_targets's
    own docstring for the shape of both and what `saliency_affine_root`
    defaults to.
    """
    saliency_affine_root = Path(saliency_affine_root) if saliency_affine_root else DEFAULT_SALIENCY_AFFINE_ROOT

    diff_by_target_model = {}
    summary_frames = []
    for target in targets:
        saliency_dir = saliency_affine_root / target
        aggregate_dir = saliency_dir / "aggregate"
        diff_by_model, summary_df = compute_all_differences(
            target, models=models, stat=stat, kind=kind, norm_method=norm_method, saliency_dir=saliency_dir,
            aggregate_dir=aggregate_dir, template_path=template_path, brain_mask_path=brain_mask_path,
            norm_percentile=norm_percentile, norm_zclip=norm_zclip, mad_clip=mad_clip, arch_variant=arch_variant,
        )
        diff_by_target_model[target] = diff_by_model
        summary_df = summary_df.reset_index()
        summary_df.insert(0, "target", target)
        summary_frames.append(summary_df)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(
        columns=["target", "model", "diff_mean", "diff_median", "diff_p95", "raw_spearman"]
    )
    return diff_by_target_model, summary_df


def compare_and_plot_all_targets(targets=DEFAULT_TARGETS, models=DEFAULT_MODELS, stat="mean", kind="abs",
                                  norm_method="rank", arch_variant="all", saliency_affine_root=None,
                                  norm_percentile=99.0, norm_zclip=3.0, mad_clip=3.0, n_montage=4,
                                  threshold_percentile=75.0, cut_coords=None, title=None, output=None,
                                  template_path=DEFAULT_TEMPLATE, brain_mask_path=DEFAULT_BRAIN_MASK):
    """
    Multi-target sibling of compare_and_plot: age/sex/bmi (default `targets`
    order, matching every slurm/*.sh script's own TASK_TARGETS convention)
    side by side as columns in ONE figure, one row per model, each row on
    its own vmax/colorbar -- see plot_difference_grid's own docstring and
    the module docstring's "compare_and_plot_all_targets()" section for why.
    Returns (fig, summary_df).

    Colors are only comparable WITHIN a row here (each model's own scale) --
    see compare_and_plot_all_targets_shared_scale for a version where every
    panel in the whole figure shares ONE vmin/vmax, so color intensity is
    directly comparable across models too, at the cost of models with a
    much smaller raw scale rendering very faint.

    `saliency_affine_root`: defaults to DEFAULT_SALIENCY_AFFINE_ROOT
    (<repo>/saliency_affine -- the tree every slurm/*.sh sanity-check script
    actually writes to; NOT visualize_saliency.py's own DEFAULT_SALIENCY_DIR,
    which points at the stale saliency_outputs/ tree -- see resolve_pair's
    callers elsewhere for the same fix). Per target: saliency_dir =
    saliency_affine_root/target, aggregate_dir = saliency_dir/aggregate.

    `summary_df` here has an extra leading "target" column (vs.
    compare_and_plot's own summary_df), one row per (target, model) that
    resolved successfully -- NOT filtered down to the rows/models that
    survived plot_difference_grid's cross-target intersection, so it may
    list a model plot_difference_grid ends up dropping (see that WARN).
    """
    diff_by_target_model, summary_df = _gather_target_diffs(
        targets, models, stat, kind, norm_method, arch_variant, saliency_affine_root,
        norm_percentile, norm_zclip, mad_clip, template_path, brain_mask_path,
    )

    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)
    norm_label = "un-normalized (raw)" if norm_method == "none" else f"{norm_method}-normalized"
    fig = plot_difference_grid(
        diff_by_target_model, template_img, brain_mask_data, targets, n_montage=n_montage,
        threshold_percentile=threshold_percentile, cut_coords=cut_coords,
        title=title or f"|aggregate - arch| ({', '.join(targets)}, {arch_variant} variant, {norm_label}, {kind})",
        output=output, kind=kind,
    )
    return fig, summary_df


def compare_and_plot_all_targets_shared_scale(targets=DEFAULT_TARGETS, models=DEFAULT_MODELS, stat="mean",
                                               kind="abs", norm_method="rank", arch_variant="all",
                                               saliency_affine_root=None, norm_percentile=99.0, norm_zclip=3.0,
                                               mad_clip=3.0, n_montage=4, threshold_percentile=75.0,
                                               cut_coords=None, title=None, output=None,
                                               template_path=DEFAULT_TEMPLATE, brain_mask_path=DEFAULT_BRAIN_MASK):
    """
    Same age/sex/bmi x model grid as compare_and_plot_all_targets, but with
    ONE GLOBAL vmax/threshold/colorbar for the ENTIRE figure -- pooled
    across every (target, model) cell, not per row -- so color intensity is
    directly comparable EVERYWHERE: the same color means the same absolute
    diff value in every panel, not just within one row.

    The cost of that: models/targets whose raw diff values are much smaller
    than the largest one in the figure will render mostly dark/empty (their
    real values are honestly small relative to the global scale, not hidden
    by a bug -- see compare_and_plot_all_targets's own docstring for the
    opposite, per-row tradeoff). No arbitrary display gain is applied here
    (unlike compare_and_plot_all_targets's plot_difference_grid call, whose
    per-row vmax currently divides by 2) -- the printed vmax is the actual
    99.5th percentile, unscaled.

    Returns (fig, summary_df) -- summary_df has the same "target" + model
    columns as compare_and_plot_all_targets's own.
    """
    diff_by_target_model, summary_df = _gather_target_diffs(
        targets, models, stat, kind, norm_method, arch_variant, saliency_affine_root,
        norm_percentile, norm_zclip, mad_clip, template_path, brain_mask_path,
    )

    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)

    all_vals = np.concatenate([
        np.abs(diff_by_target_model[t][m]["diff"][brain_mask_data]) if brain_mask_data is not None
        else np.abs(diff_by_target_model[t][m]["diff"]).ravel()
        for t in targets for m in diff_by_target_model[t]
    ])
    all_vals = all_vals[all_vals > 0]
    global_vmax = max(float(np.percentile(all_vals, 99.5)), 1e-6)/3 if all_vals.size else 1.0
    global_threshold = (
        max(float(np.percentile(all_vals, threshold_percentile)), 1e-6)
        if all_vals.size and threshold_percentile > 0 else 1e-6
    )
    print(f"compare_and_plot_all_targets_shared_scale: ONE global vmax={global_vmax:.6f} "
          f"(99.5th pctile) and threshold={global_threshold:.6f} ({threshold_percentile:.1f}th pctile), "
          f"pooled across every (target, model) cell -- applied to the whole figure, not per row.")

    norm_label = "un-normalized (raw)" if norm_method == "none" else f"{norm_method}-normalized"
    fig = plot_difference_grid(
        diff_by_target_model, template_img, brain_mask_data, targets, n_montage=n_montage,
        threshold_percentile=threshold_percentile, cut_coords=cut_coords,
        title=title or (
            f"|aggregate - arch| ({', '.join(targets)}, {arch_variant} variant, {norm_label}, {kind}, "
            f"shared scale)"
        ),
        output=output, kind=kind, vmax=global_vmax, threshold=global_threshold, single_colorbar=True,
    )
    return fig, summary_df


def _gather_target_aggregates(targets, models, stat, kind, saliency_affine_root, template_path, brain_mask_path,
                               allow_fs_roi=False):
    """
    Aggregate-only sibling of _gather_target_diffs: loops
    compute_all_aggregates() once per target, returns
    (map_by_target_model, summary_df) -- {target: {model: {"map": array}}}
    and a "target"-prefixed summary_df, same shapes as _gather_target_diffs'
    own (with map_mean/map_median/map_p95 columns instead of
    diff_mean/diff_median/diff_p95/raw_spearman -- see compute_all_aggregates).

    `allow_fs_roi`: see resolve_aggregate_path's own docstring -- passed
    straight through to compute_all_aggregates.
    """
    saliency_affine_root = Path(saliency_affine_root) if saliency_affine_root else DEFAULT_SALIENCY_AFFINE_ROOT

    map_by_target_model = {}
    summary_frames = []
    for target in targets:
        aggregate_dir = saliency_affine_root / target / "aggregate"
        map_by_model, summary_df = compute_all_aggregates(
            target, models=models, stat=stat, kind=kind, aggregate_dir=aggregate_dir,
            template_path=template_path, brain_mask_path=brain_mask_path, allow_fs_roi=allow_fs_roi,
        )
        map_by_target_model[target] = map_by_model
        summary_df = summary_df.reset_index()
        summary_df.insert(0, "target", target)
        summary_frames.append(summary_df)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(
        columns=["target", "model", "map_mean", "map_median", "map_p95"]
    )
    return map_by_target_model, summary_df


def compare_and_plot_all_targets_aggregate_shared_scale(targets=DEFAULT_TARGETS, models=DEFAULT_MODELS,
                                                          stat="mean", kind="abs", saliency_affine_root=None,
                                                          n_montage=4, threshold_percentile=75.0, cut_coords=None,
                                                          title=None, output=None, template_path=DEFAULT_TEMPLATE,
                                                          brain_mask_path=DEFAULT_BRAIN_MASK, label_scale=1.0,
                                                          show_title=True, show_colorbar=True):
    """
    Aggregate-only sibling of compare_and_plot_all_targets_shared_scale:
    same age/sex/bmi x model grid, ONE global vmax/threshold/colorbar for
    the whole figure -- but displays each model's own ig_popmean AGGREGATE
    saliency map directly, NOT a diff against an arch_ (randomized-weights)
    sanity-check map. No norm_method/arch_variant here: there's nothing to
    normalize-then-difference and no sanity-check variant to pick, since
    this never touches the arch_ files at all (see resolve_aggregate_path).

    Use this to look at what a model's TRAINED explanation actually looks
    like across tasks, on one consistent scale -- as opposed to
    compare_and_plot_all_targets_shared_scale's "how much did the
    explanation change under weight randomization" question.

    `kind="signed"`: thresholding is effectively disabled -- no percentile
    computed, `threshold_percentile` is ignored (see plot_difference_grid's
    `disable_threshold`, which floors at a negligible constant rather than
    a literal threshold=None -- see its own docstring for why: nilearn
    treats an unthresholded exact 0.0, which is what every masked-out
    non-brain voxel has been set to, as a real value to color in rather
    than as transparent background). A signed map's negative values are
    just as meaningful as its positive ones (which sign a voxel "leans"),
    so a magnitude-based threshold would hide small-but-real values of BOTH
    signs near zero without a principled cutoff to pick; showing everything
    real and letting the diverging coolwarm scale/vmax do the visual
    filtering is the more honest default here. `kind="abs"` keeps the usual
    percentile-based threshold.

    `label_scale`/`show_title`/`show_colorbar`: styled after
    visualize_paper_nilearn.py's own --label-scale/--title/--no-colorbar
    CLI flags -- see plot_difference_grid's own docstring for exactly what
    each does. `label_scale` multiplies the row (model-name) label fontsize
    (base 11, same as that script); `show_title`/`show_colorbar` let you
    drop either element entirely rather than just leaving `title` empty.

    Returns (fig, summary_df) -- summary_df has target/model/map_mean/
    map_median/map_p95 columns (see compute_all_aggregates).
    """
    map_by_target_model, summary_df = _gather_target_aggregates(
        targets, models, stat, kind, saliency_affine_root, template_path, brain_mask_path,
    )

    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)

    all_vals = np.concatenate([
        np.abs(map_by_target_model[t][m]["map"][brain_mask_data]) if brain_mask_data is not None
        else np.abs(map_by_target_model[t][m]["map"]).ravel()
        for t in targets for m in map_by_target_model[t]
    ])
    all_vals = all_vals[all_vals > 0]
    global_vmax = max(float(np.percentile(all_vals, 99.5)), 1e-6)/4 if all_vals.size else 1.0

    disable_threshold = kind == "signed"
    if disable_threshold:
        print(f"compare_and_plot_all_targets_aggregate_shared_scale: ONE global vmax={global_vmax:.6f} "
              f"(99.5th pctile); threshold effectively disabled (kind='signed', floored at "
              f"{_NO_THRESHOLD_FLOOR:g} to keep the masked-out background transparent -- see "
              f"plot_difference_grid's disable_threshold docstring) -- pooled across every "
              f"(target, model) cell -- applied to the whole figure, not per row.")
        global_threshold = None
    else:
        global_threshold = (
            max(float(np.percentile(all_vals, threshold_percentile)), 1e-6)
            if all_vals.size and threshold_percentile > 0 else 1e-6
        )
        print(f"compare_and_plot_all_targets_aggregate_shared_scale: ONE global vmax={global_vmax:.6f} "
              f"(99.5th pctile) and threshold={global_threshold:.6f} ({threshold_percentile:.1f}th pctile), "
              f"pooled across every (target, model) cell -- applied to the whole figure, not per row.")

    fig = plot_difference_grid(
        map_by_target_model, template_img, brain_mask_data, targets, n_montage=n_montage,
        threshold_percentile=threshold_percentile, cut_coords=cut_coords,
        title=title or f"aggregate saliency ({', '.join(targets)}, {kind}, shared scale)",
        output=output, kind=kind, vmax=global_vmax, threshold=global_threshold, single_colorbar=True,
        value_key="map", disable_threshold=disable_threshold, label_scale=label_scale,
        show_title=show_title, show_colorbar=show_colorbar,
    )
    return fig, summary_df


def compare_and_plot_fs_roi_shared_scale(targets=DEFAULT_TARGETS, stat="mean", kind="abs",
                                          saliency_affine_root=None, n_montage=4, threshold_percentile=75.0,
                                          cut_coords=None, title=None, output=None, template_path=DEFAULT_TEMPLATE,
                                          brain_mask_path=DEFAULT_BRAIN_MASK, label_scale=1.0, show_title=True,
                                          show_colorbar=True, colorbar_width=0.4, colorbar_height=0.04):
    """
    fs_roi-only sibling of compare_and_plot_all_targets_aggregate_shared_scale:
    same age/sex/bmi-columns, ONE global vmax/threshold/colorbar figure --
    but always exactly ONE row, fs_roi's own aggregate map, never any other
    model. No `models` parameter -- unlike every other entry point in this
    file, fs_roi is normally REJECTED (see resolve_aggregate_path's own
    docstring): its aggregate file is piecewise-constant per FreeSurfer ROI
    on the DKT segmentation grid, not smooth per-voxel data on the shared
    MNI-registered grid every other model uses, so mixing it into a
    multi-model figure/pooled-vmax computation compares apples to oranges.
    That concern doesn't apply here -- there's nothing else in the figure
    to mix it with -- so this is the one entry point that passes
    allow_fs_roi=True through to resolve_aggregate_path. Geometrically,
    nilearn/resample_saliency can still resample it onto the shared
    template just fine on its own (see that docstring); it'll just look
    visibly blocky/piecewise-constant per ROI rather than smooth, which is
    the CORRECT appearance for fs_roi, not a bug.

    All other parameters match compare_and_plot_all_targets_aggregate_shared_scale
    exactly (targets/stat/kind/saliency_affine_root/n_montage/
    threshold_percentile/cut_coords/title/output/template_path/
    brain_mask_path/label_scale/show_title/show_colorbar) -- see that
    function's own docstring, plot_difference_grid's, and
    compute_all_aggregates' for what each does; `kind="signed"` still
    disables thresholding the same way (see disable_threshold below).
    `colorbar_width`/`colorbar_height` (see plot_difference_grid's own
    docstring) default to 0.4/0.04 -- `colorbar_width` matches that
    function's own default (unchanged, same as every other single_colorbar
    figure), but `colorbar_height` defaults TALLER here (0.04 vs. that
    function's own 0.02) -- with only one row, this figure has more spare
    room at the bottom to give the single colorbar than the other, denser
    multi-model figures do.

    Returns (fig, summary_df) -- summary_df has target/model/map_mean/
    map_median/map_p95 columns (see compute_all_aggregates), model always
    "fs_roi".
    """
    map_by_target_model, summary_df = _gather_target_aggregates(
        targets, ["fs_roi"], stat, kind, saliency_affine_root, template_path, brain_mask_path,
        allow_fs_roi=True,
    )

    template_img = load_canonical_template(template_path)
    brain_mask_data = load_brain_mask(brain_mask_path, template_img)

    all_vals = np.concatenate([
        np.abs(map_by_target_model[t]["fs_roi"]["map"][brain_mask_data]) if brain_mask_data is not None
        else np.abs(map_by_target_model[t]["fs_roi"]["map"]).ravel()
        for t in targets
    ])
    all_vals = all_vals[all_vals > 0]
    global_vmax = max(float(np.percentile(all_vals, 99.5)), 1e-6) if all_vals.size else 1.0

    disable_threshold = kind == "signed"
    if disable_threshold:
        print(f"compare_and_plot_fs_roi_shared_scale: ONE global vmax={global_vmax:.6f} "
              f"(99.5th pctile); threshold effectively disabled (kind='signed', floored at "
              f"{_NO_THRESHOLD_FLOOR:g} to keep the masked-out background transparent -- see "
              f"plot_difference_grid's disable_threshold docstring) -- pooled across every "
              f"target -- applied to the whole figure, not per row.")
        global_threshold = None
    else:
        global_threshold = (
            max(float(np.percentile(all_vals, threshold_percentile)), 1e-6)
            if all_vals.size and threshold_percentile > 0 else 1e-6
        )
        print(f"compare_and_plot_fs_roi_shared_scale: ONE global vmax={global_vmax:.6f} "
              f"(99.5th pctile) and threshold={global_threshold:.6f} ({threshold_percentile:.1f}th pctile), "
              f"pooled across every target -- applied to the whole figure, not per row.")

    fig = plot_difference_grid(
        map_by_target_model, template_img, brain_mask_data, targets, n_montage=n_montage,
        threshold_percentile=threshold_percentile, cut_coords=cut_coords,
        title=title or f"fs_roi aggregate saliency ({', '.join(targets)}, {kind}, shared scale)",
        output=output, kind=kind, vmax=global_vmax, threshold=global_threshold, single_colorbar=True,
        value_key="map", disable_threshold=disable_threshold, label_scale=label_scale,
        show_title=show_title, show_colorbar=show_colorbar, colorbar_width=colorbar_width,
        colorbar_height=colorbar_height,
    )
    return fig, summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--target", choices=["age", "sex", "bmi"], required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--stat", choices=["mean", "std"], default="mean")
    parser.add_argument("--kind", choices=["signed", "abs"], default="abs")
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Default: saliency_outputs/<target> (arch_* files live directly here).",
    )
    parser.add_argument(
        "--aggregate-dir", type=Path, default=None,
        help="Default: saliency_outputs/<target>/aggregate (aggregate_saliency.py's own output dir).",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument(
        "--norm-method", choices=list(NORM_FUNCS), default="rank",
        help="Default: rank (see module docstring for the four schemes). 'none' skips normalization "
        "entirely and diffs raw values -- NOT recommended for a real comparison (a trained vs. "
        "randomly-initialized backbone have wildly different raw gradient scales), but available for "
        "exploratory/debugging use.",
    )
    parser.add_argument(
        "--randomize-model-variant", default="all",
        help="Which --randomize-model-mode variant's arch_ file to compare each model's aggregate "
        "against -- 'all' (default, matches --randomize-model-mode all) or 'step{NN:02d}_{module}' "
        "for a --randomize-model-mode cascade step, e.g. 'step01_head' for "
        "slurm/compute_sanity_checks_randomize_model.sh's head-only --randomize-max-layers 1 run "
        "(aggregated via RANDOMIZE_MODEL_VARIANT=linear slurm/aggregate_sanity_check_randomize_model.sh). "
        "See sanity_prefix()'s docstring in visualize_paper_nilearn.py for the full naming contract.",
    )
    parser.add_argument("--norm-percentile", type=float, default=99.0)
    parser.add_argument("--norm-zclip", type=float, default=3.0)
    parser.add_argument("--mad-clip", type=float, default=3.0)
    parser.add_argument("--n-montage", type=int, default=4)
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--cut-coords", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="Optional path to also write the per-model diff_mean/diff_median/diff_p95/raw_spearman table.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(
            f"aggregate_arch_diff_{args.target}_{args.randomize_model_variant}_{args.norm_method}_{args.kind}.png"
        )

    fig, summary_df = compare_and_plot(
        args.target, models=args.models, stat=args.stat, kind=args.kind, norm_method=args.norm_method,
        saliency_dir=args.saliency_dir, aggregate_dir=args.aggregate_dir, template_path=args.template,
        brain_mask_path=args.brain_mask, norm_percentile=args.norm_percentile, norm_zclip=args.norm_zclip,
        mad_clip=args.mad_clip, n_montage=args.n_montage, threshold_percentile=args.threshold_percentile,
        cut_coords=args.cut_coords, title=args.title, output=args.output, arch_variant=args.randomize_model_variant,
    )
    print(summary_df)

    if args.summary_csv:
        summary_df.to_csv(args.summary_csv)
        print(f"Saved: {args.summary_csv}")


if __name__ == "__main__":
    main()
