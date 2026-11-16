"""
Overlay each model's saliency map on a common brain template so every model
is shown at identical dimensions, orientation, and slice indices -- i.e.
axial/coronal/sagittal mean the same thing for every model.

The default template is TemplateFlow's MNI152NLin2009cAsym (1mm, brain-
extracted). This isn't an arbitrary choice: turboprep normalizes every
subject onto this exact grid (verified byte-for-byte identical shape/affine
against NKI_hdbet_affine/*/normalized.nii.gz -- and against the original
turboprep_NKI/*/normalized.nii.gz, which shares the same grid), so for the
four turboprep-based
models (optim_uncnn, brainiac, swinbrain, simclr) resampling onto it is an
identity transform -- pixel-perfect, no interpolation. anatcl (CAT12 mwp1,
SPM's own MNI variant) still needs real resampling, but onto the template
its data was actually normalized towards, rather than FreeSurfer's unrelated
conformed/fsaverage space (e.g. brain.mgz), which is a different template
family and was leaving a systematic offset/FOV mismatch in the overlay.

Each model in this repo preprocesses its input differently (different resize
targets, crops, axis transposes -- see train_linear.py's per-model Dataset
classes), so the *.nii.gz files under saliency_outputs/ do not all share the
same voxel grid or, in general, the same affine as the template. Matching
array *shape* is not the same as matching *anatomical space*: this script
resamples every saliency map onto the template's grid using its own NIfTI
affine (nilearn.image.resample_to_img), which is the only way to guarantee
that a given array index corresponds to the same anatomical location across
models. The overlay is additionally restricted to the template's own brain
mask, so saliency in skull-stripped background/non-brain voxels isn't drawn.

After resampling, every array is reordered to an axial-first axis order
(axis 0 = inferior->superior), so `saliency[i]` is always the axial slice at
level i for every model, `coronal[i]` is always the coronal slice, etc.

--norm: put every model on one shared color scale
--------------------------------------------------
By default each model gets its own vmin/vmax (its own colorbar per row), because
raw vanilla-gradient magnitudes are NOT comparable across these architectures --
e.g. optim_uncnn's saliency spans ~[-2, 2] while simclr's spans ~[-0.005, 0.005].
That isn't a difference in "importance", it's a difference in each backbone's
input normalization, depth, and parameter scale (see Ancona et al., 2018,
"Towards Better Understanding of Gradient-Based Attribution Methods for Deep
Neural Networks", ICLR -- raw gradient magnitude reflects local function scale,
not attribution strength, and is not meaningfully comparable across networks).
This is also why the standard practice in both the saliency-map literature
(e.g. Adebayo et al., 2018, "Sanity Checks for Saliency Maps", NeurIPS) and
neuroimaging attribution studies (e.g. Levakov et al., 2020, brain-age CNN
relevance maps) is to rescale each map independently onto a common, unit-free
range *before* comparing it to others -- never to compare raw magnitudes
directly, and never to fit one shared vmin/vmax to the pooled raw values (a
single model with much larger raw gradients would dominate the color scale
and wash out the others).

Pass --norm to rescale each map this way and then display all rows with one
shared colorbar. Two rescaling methods are offered (--norm-method):
  - "percentile" (default): divide by each map's own P-th percentile |value|
    within the brain mask (--norm-percentile, default 99), clip to [-1, 1]
    (signed) or [0, 1] (abs). Robust to single-voxel outliers; this is the
    same clip-then-normalize convention used for statistical map display in
    FSL/SPM and in most saliency-visualization papers.
  - "zscore": standardize by the map's own in-brain mean/std, clip at
    +/-N standard deviations (--norm-zclip, default 3). More statistically
    interpretable (SPM-style Z-map units) but more sensitive to how skewed a
    given map's voxel distribution is.
Percentile is the recommended default: gradient saliency distributions here
are heavy-tailed (a few very large voxels), which z-scoring is sensitive to,
while percentile clipping stays robust regardless of the tail shape.

Usage:
    python visualize_saliency.py --kind signed --output overlay_signed.png
    python visualize_saliency.py --kind abs --models optim_uncnn brainiac
    python visualize_saliency.py --kind signed --norm --output overlay_norm.png
    python visualize_saliency.py --combine-kinds --norm --models optim_uncnn brainiac

--combine-kinds: signed and abs side by side, one row each
-------------------------------------------------------------
Normally --kind picks one of "signed"/"abs" and each row gets 3 columns
(axial/coronal/sagittal) for that kind alone. --combine-kinds instead gives
each row 6 columns: signed axial/coronal/sagittal, then abs axial/coronal/
sagittal for the SAME model -- so a model's signed and abs maps can be
compared directly without hunting between two separate figures. --kind is
ignored when this is set (both kinds are always shown). Rows are still
selected/ordered by --models against the *signed* filenames; each signed
file's matching abs file is found by replacing the kind suffix, and a row is
skipped (with a warning) if no matching abs file exists. Signed and abs
columns keep their own color scale (and colorbar) per the usual [-1,1]/[0,1]
convention -- they are never combined onto one shared scale.

--montage: an axial-slice strip below the 3 orthogonal views
------------------------------------------------------------------
--montage splits each kind's block across TWO physical rows sharing one row
label (see main()'s ylabel duplication), instead of the usual single row:
the 3 orthogonal views (axial/coronal/sagittal at the brain centroid) on
the first row, and an axial-only slice at each of --montage-axial-slices
(default 8) evenly spaced z-levels spanning the brain's own inferior->
superior extent (see axial_montage_indices) on the row directly below it --
the standard "orthogonal views + axial montage" layout used in typical
neuroimaging results figures. Pass --montage-axial-z-mm instead (e.g.
--montage-axial-z-mm -18 0 18) to show fixed world-space z levels rather
than N evenly spaced ones (see axial_indices_for_mm) -- useful when you want
the same anatomical levels shown across every figure instead of levels that
shift with each map's own brain extent. Combines with --kind or --combine-kinds
exactly as those already work (--combine-kinds still means "signed and abs
side by side, same rows", now with the montage strip beneath each): --kind
signed --montage gives 2 rows x 8 columns (the montage strip is wider than
the 3-column orthogonal row, which just leaves its extra columns blank),
--combine-kinds --montage gives 2 rows x 16 columns (8 signed, 8 abs). Every
panel in a kind's block -- orthogonal row and montage row alike -- shares
that one file's vmin/vmax/colormap and one colorbar, so the pair reads as
one consistent map split across two rows, not two unrelated ones.
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from targets import get_target_spec

# TemplateFlow's MNI152NLin2009cAsym (res-01, brain-extracted). Fetch it with
# `templateflow` (pip install templateflow; TemplateFlowAPI().get(...)) or
# point TEMPLATEFLOW_MNI_DIR at an existing local copy -- see
# config/paths.example.env.
_TEMPLATEFLOW_MNI = Path(
    # os.environ.get("TEMPLATEFLOW_MNI_DIR", str(Path.home() / ".cache" / "templateflow" / "tpl-MNI152NLin2009cAsym"))
    os.environ.get("TEMPLATEFLOW_MNI_DIR", str(Path.home() / ".cache" / "templateflow" / "tpl-MNI152Lin"))
)
# NOTE: desc-brain_T1w.nii.gz is sometimes only present in a templateflow
# cache as an empty (unfetched) placeholder, so we use the full head T1w as
# background and rely on desc-brain_mask.nii.gz (which does get fetched) to
# restrict the overlay.
# DEFAULT_TEMPLATE = _TEMPLATEFLOW_MNI / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
# DEFAULT_BRAIN_MASK = _TEMPLATEFLOW_MNI / "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
DEFAULT_TEMPLATE = _TEMPLATEFLOW_MNI / "tpl-MNI152Lin_res-01_T1w.nii.gz"
DEFAULT_BRAIN_MASK = _TEMPLATEFLOW_MNI / "tpl-MNI152Lin_res-01_desc-brain_mask.nii.gz"
DEFAULT_SALIENCY_DIR = Path(__file__).resolve().parents[1] / "saliency_outputs"


def load_canonical_template(template_path):
    """Load the template and reorient to canonical RAS (L->R, P->A, I->S)."""
    return nib.as_closest_canonical(nib.load(template_path))


def load_brain_mask(mask_path, template_img):
    """Resample the template's brain mask onto the template's own grid (nearest-neighbor)."""
    mask_img = nib.load(mask_path)
    resampled = resample_to_img(mask_img, template_img, interpolation="nearest")
    return to_axial_first(resampled.get_fdata()) > 0.5


def to_axial_first(data):
    """RAS array (X=L->R, Y=P->A, Z=I->S) -> axial-first (Z, Y, X)."""
    return np.transpose(data, (2, 1, 0))


def resample_saliency(saliency_path, template_img):
    """
    Resample a saliency map onto the template's exact grid/affine.

    Uses "linear" (trilinear), not nilearn's "continuous" (cubic spline)
    default: a cubic spline is not constrained to stay within the range of
    its input values, so at the sharp edge between exact-zero background and
    real tissue it overshoots/undershoots (ringing) -- verified concretely on
    an abs-valued saliency map, which can never be negative by definition,
    coming back with a negative minimum under "continuous". Trilinear
    interpolation is a convex combination of neighboring voxels, so it can
    never produce a value outside the range already present in the data.
    """
    sal_img = nib.load(saliency_path)
    resampled = resample_to_img(sal_img, template_img, interpolation="linear")
    return to_axial_first(resampled.get_fdata())


def brain_centroid_indices(volume, threshold=0.0):
    """Axial-first (z, y, x) centroid index of the nonzero (brain) voxels."""
    coords = np.array(np.nonzero(volume > threshold))
    return coords.mean(axis=1).astype(int)


def axial_montage_indices(brain_mask_data, n_slices):
    """
    `n_slices` evenly spaced axial-first z-indices spanning the brain's own
    inferior->superior extent -- the --montage-axial-slices strip shown
    below the 3 orthogonal views (see the module docstring's reference
    image: a row of axial slices at regularly spaced levels, not the single
    centroid slice brain_centroid_indices picks for the ortho view).

    `brain_mask_data` MUST be the real brain mask, resampled onto the
    template grid -- always load it for this purpose even when
    --no-brain-mask means the overlay itself won't be masked (see main()):
    DEFAULT_TEMPLATE's whole-head T1w is NOT skull-stripped and has a
    nonzero value at every single voxel in every slice (verified directly --
    "nonzero extent" there is the full FOV, not the brain), so a
    template-nonzero fallback silently picks slices well outside the head
    instead of narrowing to the brain.

    A small inset (5% of the z-range at each end) is trimmed before evenly
    spacing `n_slices` indices across what's left, so the outermost slices
    aren't right at the very edge of the brain where there's often almost no
    tissue to show.
    """
    z_present = np.where(brain_mask_data.any(axis=(1, 2)))[0]
    if z_present.size == 0:
        z_min, z_max = 0, brain_mask_data.shape[0] - 1
    else:
        z_min, z_max = int(z_present[0]), int(z_present[-1])
    inset = int(round(0.05 * (z_max - z_min)))
    z_min, z_max = z_min + inset, z_max - inset
    indices = np.linspace(z_min, z_max, n_slices).round().astype(int)
    return np.clip(indices, 0, brain_mask_data.shape[0] - 1).tolist()


def axial_indices_for_mm(template_img, z_mm_values, n_z):
    """
    Axial-first z-indices for explicit world-space z (mm) coordinates (e.g.
    [-18, 0, 18]), for callers that want fixed anatomical levels shown
    consistently across figures rather than axial_montage_indices' N evenly
    spaced levels (which shift with each map's own brain extent). Assumes
    template_img's affine is axis-aligned (true after
    load_canonical_template's as_closest_canonical reorientation -- no
    oblique rotation), so a world z only depends on voxel z and world x/y can
    be passed as 0 without affecting the result.
    """
    from nibabel.affines import apply_affine

    inv_affine = np.linalg.inv(template_img.affine)
    indices = []
    for z_mm in z_mm_values:
        voxel = apply_affine(inv_affine, np.array([0.0, 0.0, z_mm]))
        indices.append(int(round(voxel[2])))
    return np.clip(indices, 0, n_z - 1).tolist()


def voxel_to_mm(template_img, x_idx, y_idx, z_idx):
    """
    World-space (mm) coordinate for an axial-first (z_idx, y_idx, x_idx)
    voxel, via the template's own affine -- used to label montage slices
    with real coordinates (e.g. "z=15") the way standard neuroimaging
    figures do, rather than an arbitrary array index. to_axial_first
    transposes the canonical-RAS array (X, Y, Z) to (Z, Y, X), so the
    original voxel coordinate is simply (x_idx, y_idx, z_idx) in that order.
    """
    from nibabel.affines import apply_affine

    return apply_affine(template_img.affine, np.array([x_idx, y_idx, z_idx]))


def overlay_slice(ax, bg_slice, sal_slice, cmap, vmin, vmax, view_title, mask_thresh_frac=0.02, brain_mask_slice=None):
    """Draw one view (axial/coronal/sagittal); `view_title` is just that view's name."""
    ax.imshow(bg_slice, cmap="gray", origin="lower")
    mask_thresh = mask_thresh_frac * vmax if vmax > 0 else 0.0
    hide = np.abs(sal_slice) <= mask_thresh
    if brain_mask_slice is not None:
        hide = hide | ~brain_mask_slice
    masked = np.ma.masked_where(hide, sal_slice)
    im = ax.imshow(masked, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, alpha=0.85)
    ax.set_title(view_title, fontsize=9)
    # Hide ticks/frame but keep the axis "on" so a row-title set via
    # set_ylabel() (see main()) on the leftmost column stays visible --
    # ax.axis("off") would hide that ylabel along with everything else.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def model_name_from_filename(path, kind):
    return path.name.replace(f"_{kind}_saliency.nii.gz", "")


# aggregate_saliency.py ("{model}_aggregate_{stat}_{kind}_saliency.nii.gz")
# and bin_saliency.py ("{model}_{age,error}bin_{label}_{stat}_{kind}_saliency.nii.gz")
# both put the statistic immediately before "_{kind}_saliency.nii.gz" --
# checked longest-first so "median" can't be mistaken for a substring match
# against a shorter candidate.
_STATS = ("mean", "median", "std", "entropy")


def file_stat(path, kind):
    """
    Which aggregate/bin statistic (mean/median/std/entropy) `path`'s filename
    encodes, or None if it's not a stat-bearing file at all (a plain
    baseline or arch_/data_/input_ sanity-check map, neither of which has a
    stat in its name) -- see --mean/--std in main() for why this
    distinction matters: those flags must leave non-stat files alone and
    only filter files that actually claim to be one statistic or another.
    """
    for stat in _STATS:
        if path.name.endswith(f"_{stat}_{kind}_saliency.nii.gz"):
            return stat
    return None


_DISPLAY_PREFIX_MAP = (
    ("arch_", "random_weights_"),  # --randomize-model
    ("data_", "shuffled_labels_"),  # --randomize-data (was "shuffled_ages_" -- target-neutral now that this can be sex/bmi labels too)
    ("input_", "shuffled_voxels_"),  # --randomize-input
)


_DISPLAY_INFIX_MAP = (
    ("_aggregate_mean", "aggregate_mean_"),
    ("_aggregate_std", "aggregate_std_"),
    ("_aggregate_entropy", "aggregate_entropy_"),
    ("_agebin_", "agebin_"),
    ("_errorbin_", "errorbin_"),
    ("_valuebin_", "valuebin_"),
    ("_classbin_", "classbin_"),
)


# Filename-safe attribution-method tags calculate_gradcam.py's
# attribution_method_tag() inserts (see that function's docstring), longest
# names first so "ig_zero"/"ig_reference" can't be shadowed by a shorter
# false match. "gradient" (the default method) never inserts a tag in the
# first place, so it isn't in this list -- split_attribution_tag() below
# falls back to it when none of these match.
_ATTRIBUTION_TAGS = ("ig_reference", "ig_zero", "gxi")

# Display name per attribution method -- only "gradient" gets renamed
# (-> "vanillagrad", this repo's long-standing name for it); gxi/ig_zero/
# ig_reference already read fine as-is and are kept identical to their
# on-disk tag.
_ATTRIBUTION_DISPLAY = {"gradient": "vanillagrad", "gxi": "gxi", "ig_zero": "ig_zero", "ig_reference": "ig_popmean"}

_ATTRIBUTION_TAG_RE = re.compile(
    r"_(" + "|".join(_ATTRIBUTION_TAGS) + r")(?=_aggregate_mean|_aggregate_std|_aggregate_entropy|"
    r"_agebin_|_errorbin_|_valuebin_|_classbin_|$)"
)


def split_attribution_tag(model_name):
    """
    Extract calculate_gradcam.py's attribution-method tag from a raw,
    filename-derived model identifier, if present. compute_and_save_saliency
    inserts the tag right after the model/channel identifier and right
    before whatever comes next -- nothing (baseline/arch_/data_/input_ rows)
    or aggregate_saliency.py/bin_saliency.py's own "_aggregate_*"/"_agebin_"/
    "_errorbin_" infix (aggregate/bin rows, which are built from those same
    per-subject filenames) -- so the tag is always immediately followed by
    one of those markers or the end of the string.

    Returns (base_model_name, attribution_key): attribution_key is one of
    "gxi"/"ig_zero"/"ig_reference" with the tag removed from
    base_model_name, or "gradient" with base_model_name unchanged if no tag
    was found (the default method's filenames were never tagged to begin
    with -- see attribution_method_tag in calculate_gradcam.py).
    """
    match = _ATTRIBUTION_TAG_RE.search(model_name)
    if match is None:
        return model_name, "gradient"
    base_model_name = model_name[: match.start()] + model_name[match.end() :]
    return base_model_name, match.group(1)


def display_label(model_name):
    """
    Map the raw filename-derived model identifier onto a human-readable row
    label of the form "{attribution_method}_{rest}" -- e.g. "vanillagrad_
    optim_uncnn_raw", "gxi_brainiac", "ig_zero_random_weights_swinbrain" --
    so every row is legible at a glance about which attribution method
    produced it, never silently folded into "vanillagrad_..." regardless of
    its real method (see split_attribution_tag).

    "{rest}" follows the same relabeling as before attribution-method tags
    existed: arch_{M} -> random_weights_{M}, data_{M} -> shuffled_ages_{M},
    input_{M} -> shuffled_voxels_{M}, the real trained baseline is just {M}
    unchanged. Only affects the displayed label -- file lookup and --models
    filtering still match the real on-disk arch_/data_/input_ prefixes.

    Aggregate/bin outputs (e.g. "{model}_aggregate_mean", "{model}_agebin_0-20_mean")
    are relabeled with the model name moved after the descriptive infix, e.g.
    "aggregate_mean_{model}", "agebin_0-20_mean_{model}", so rows sort/scan by
    group first, model second -- with the attribution-method prefix in front
    of that, same as every other row.
    """
    base_model_name, attribution_key = split_attribution_tag(model_name)
    attribution_display = _ATTRIBUTION_DISPLAY[attribution_key]

    for old_prefix, new_prefix in _DISPLAY_PREFIX_MAP:
        if base_model_name.startswith(old_prefix):
            return f"{attribution_display}_{new_prefix}{base_model_name[len(old_prefix):]}"
    for infix, new_prefix in _DISPLAY_INFIX_MAP:
        if infix in base_model_name:
            model, _, rest = base_model_name.partition(infix)
            body = f"{new_prefix}{rest.lstrip('_')}_{model}" if rest else f"{new_prefix}{model}"
            return f"{attribution_display}_{body}"
    return f"{attribution_display}_{base_model_name}"


def load_subject_metadata(saliency_dir, target_spec):
    """
    {file_prefix: {"true_value", "predicted_value", "model"}} built from
    every *_sanity_checks.json in saliency_dir (written by
    calculate_gradcam.py), keyed by every prefix that JSON's baseline/tests
    recorded -- so a plotted file can be matched back to the run that
    produced it via startswith(). Reads target_spec.json_true_key/
    json_pred_key (e.g. true_age/predicted_age, true_sex/predicted_sex)
    from the on-disk JSON, but stores them under these generic in-memory
    keys -- this dict is never written back to disk, so renaming here has
    no external-compatibility cost.
    """
    prefix_to_meta = {}
    for jf in saliency_dir.glob("*_sanity_checks.json"):
        try:
            d = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        meta = {
            "true_value": d.get(target_spec.json_true_key),
            "predicted_value": d.get(target_spec.json_pred_key),
            "model": d.get("model"),
        }
        prefixes = []
        # "baseline" is a list of {"channel", "prefix", <json_pred_key>} entries
        # (1 entry for most models, 3 for optim_uncnn's raw/rank/sobel channels
        # -- see calculate_gradcam.py's compute_and_save_saliency).
        for b in d.get("baseline", []):
            if isinstance(b, dict) and "prefix" in b:
                prefixes.append(b["prefix"])
        for t in d.get("tests", []):
            if "prefix" in t:
                prefixes.append(t["prefix"])
        for p in prefixes:
            prefix_to_meta[p] = meta
    return prefix_to_meta


def normalize_map(data, brain_mask_data, kind, method="percentile", percentile=99.0, zclip=3.0):
    """
    Rescale a raw saliency map onto a fixed, unit-free, cross-model-comparable
    range: [-1, 1] for signed maps, [0, 1] for abs maps. See the module
    docstring (--norm) for why this -- not a shared raw-value scale -- is the
    right way to compare saliency across architectures with different
    gradient scales.

    Stats are computed only over in-brain voxels so background/skull-stripped
    padding (which is always ~0) can't bias the scale estimate.
    """
    vals = data[brain_mask_data] if brain_mask_data is not None else data.ravel()

    if kind == "abs":
        # One-sided (>=0) distribution: rescale so a robust upper bound -> 1.0.
        if method == "zscore":
            mu, sigma = vals.mean(), vals.std()
            upper = mu + zclip * (sigma if sigma > 0 else 1.0)
        else:
            upper = np.percentile(vals, percentile)
        upper = upper if upper > 0 else 1.0
        return np.clip(data / upper, 0.0, 1.0)

    # signed
    if method == "zscore":
        mu, sigma = vals.mean(), vals.std()
        scale = zclip * (sigma if sigma > 0 else 1.0)
    else:
        scale = np.percentile(np.abs(vals), percentile)
    scale = scale if scale > 0 else 1.0
    return np.clip(data / scale, -1.0, 1.0)


def collect_kind_files(saliency_dir, kind, models_filter, wanted_stat):
    """
    Sorted/filtered/ordered file list for one kind (signed or abs), applying
    --models (order-preserving prefix filter) and --mean/--std (stat filter)
    exactly as main() applied them inline before --combine-kinds existed.

    Matching is LONGEST-PREFIX-WINS, not "first prefix in models_filter that
    matches": calculate_gradcam.py inserts the attribution-method tag AFTER
    the arch_/data_/input_ sanity-check stem (right before the kind suffix),
    not before it, so a bare/untagged prefix like "arch_MODEL_all_CHANNEL_"
    is itself a genuine string-prefix of its own "arch_MODEL_all_CHANNEL_
    gxi_"/"..._ig_zero_"/"..._ig_reference_" siblings. A first-match-wins
    scan (the original implementation) let the bare prefix -- whichever
    entry in models_filter happened to be processed first -- silently
    swallow every tagged sibling too, scattering an attribution-method's
    sanity-check rows into whichever OTHER method's block was resolved
    first, rather than keeping each method's aggregate+arch+data+input rows
    together (concretely: visualize_aggregate.sh's *_sanity_check_*.png
    grouped every method's random_weights/shuffled_ages/shuffled_voxels rows
    together right after vanillagrad's, instead of under their own method).
    Assigning each file to whichever requested prefix is longest (most
    specific) fixes this regardless of models_filter's order, while leaving
    already-unambiguous callers (e.g. visualize_subjects.sh, which always
    passes the full filename including the kind suffix) unaffected, since
    exactly one prefix ever matches those anyway.
    """
    all_files = sorted(saliency_dir.glob(f"*_{kind}_saliency.nii.gz"))
    if models_filter:
        best_match_for_file = {}
        for f in all_files:
            best = None
            for m in models_filter:
                if f.name.startswith(m) and (best is None or len(m) > len(best)):
                    best = m
            if best is not None:
                best_match_for_file.setdefault(best, []).append(f)

        # Order follows the order --models was passed in, not alphabetical.
        files, seen = [], set()
        for m in models_filter:
            for f in best_match_for_file.get(m, []):
                if f not in seen:
                    files.append(f)
                    seen.add(f)
    else:
        files = all_files

    if wanted_stat is not None:
        files = [f for f in files if (stat := file_stat(f, kind)) is None or stat == wanted_stat]
    return files


def draw_kind_block(
    ortho_axes_row, col_offset, kind, file_path, template_img, template_data, brain_mask_data, z_c, y_c, x_c, args,
    col_title_prefix="", montage_z_indices=None, montage_axes_row=None,
):
    """
    Draws one kind's (signed/abs) panels for one file, split across up to
    two physical subplot rows so the same model/attribution-method's
    orthogonal views and axial-montage strip read as one group under one
    shared row label (see main()'s ylabel duplication), not as unrelated
    rows:
      - The 3 orthogonal views (axial/coronal/sagittal at the brain
        centroid) always go into ortho_axes_row[col_offset:col_offset+3].
      - When `montage_z_indices` is given (see --montage/
        axial_montage_indices), one additional axial-only panel per
        z-index goes into montage_axes_row[col_offset:col_offset+len(montage_z_indices)]
        -- a SEPARATE physical row from the orthogonal views, not appended
        after them in the same row. montage_axes_row is required whenever
        montage_z_indices is non-empty.

    Returns the image handle (for colorbar attachment). Called once per row
    for plain --kind mode, twice (col_offset 0 and block_width) for
    --combine-kinds. Every panel drawn -- orthogonal and montage alike --
    shares this one file's vmin/vmax/cmap, so the whole group reads as one
    consistent map split across two rows, not two independently-scaled
    halves.
    """
    print(f"Resampling {file_path.name} onto template grid...")
    data = resample_saliency(file_path, template_img)  # axial-first, same grid as template_data

    if args.norm:
        data = normalize_map(data, brain_mask_data, kind, args.norm_method, args.norm_percentile, args.norm_zclip)
        vmax = 1.0
    else:
        vals = data[brain_mask_data] if brain_mask_data is not None else data
        vmax = float(np.percentile(np.abs(vals), 99.5))
        vmax = vmax if vmax > 0 else 1.0
    # abs saliency is non-negative -- a vmin of -vmax would waste half the
    # colormap and give the colorbar negative ticks that can never occur.
    vmin = 0.0 if kind == "abs" else -vmax
    cmap = "coolwarm" if kind == "signed" else "YlOrRd"

    axial = data[z_c, :, :]
    coronal = data[:, y_c, :]
    sagittal = data[:, :, x_c]

    bg_axial = template_data[z_c, :, :]
    bg_coronal = template_data[:, y_c, :]
    bg_sagittal = template_data[:, :, x_c]

    if brain_mask_data is not None:
        mask_axial = brain_mask_data[z_c, :, :]
        mask_coronal = brain_mask_data[:, y_c, :]
        mask_sagittal = brain_mask_data[:, :, x_c]
    else:
        mask_axial = mask_coronal = mask_sagittal = None

    col_titles = [f"{col_title_prefix}axial", f"{col_title_prefix}coronal", f"{col_title_prefix}sagittal"]

    im = overlay_slice(
        ortho_axes_row[col_offset + 0], bg_axial, axial, cmap, vmin, vmax, col_titles[0], args.mask_thresh_frac, mask_axial
    )
    overlay_slice(
        ortho_axes_row[col_offset + 1], bg_coronal, coronal, cmap, vmin, vmax, col_titles[1], args.mask_thresh_frac, mask_coronal
    )
    overlay_slice(
        ortho_axes_row[col_offset + 2], bg_sagittal, sagittal, cmap, vmin, vmax, col_titles[2], args.mask_thresh_frac, mask_sagittal
    )

    if montage_z_indices:
        assert montage_axes_row is not None, "montage_axes_row is required when montage_z_indices is given"
        for i, z_idx in enumerate(montage_z_indices):
            mm_z = voxel_to_mm(template_img, x_c, y_c, z_idx)[2]
            mask_slice = brain_mask_data[z_idx, :, :] if brain_mask_data is not None else None
            im = overlay_slice(
                montage_axes_row[col_offset + i], template_data[z_idx, :, :], data[z_idx, :, :], cmap, vmin, vmax,
                f"z={mm_z:.0f}", args.mask_thresh_frac, mask_slice,
            )
    return im


def main():
    parser = argparse.ArgumentParser(description="Overlay saliency maps on a common brain template.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Default: <repo>/saliency_outputs/<target> (see --target).",
    )
    parser.add_argument("--kind", choices=["signed", "abs"], default="signed", help="Ignored when --combine-kinds is set.")
    parser.add_argument(
        "--combine-kinds",
        action="store_true",
        help="Show signed and abs side by side, 6 columns per row (signed axial/coronal/sagittal, then abs "
        "axial/coronal/sagittal), one row per model -- see module docstring. Overrides --kind.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Default: saliency_overlay_<target>.png, so different --target runs don't overwrite each other.",
    )
    parser.add_argument("--title", type=str, default=None, help="Super-title for the whole figure.")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Subset of model name prefixes to include (default: all found)."
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default="age",
        help="Which target the plotted sanity_checks.json files were produced for -- determines which JSON "
        "keys are read for the true/predicted-value suptitle (true_age/predicted_age, true_sex/predicted_sex, "
        "true_bmi/predicted_bmi).",
    )
    stat_group = parser.add_mutually_exclusive_group()
    stat_group.add_argument(
        "--mean",
        action="store_true",
        help="For aggregate_saliency.py/bin_saliency.py outputs (which encode a statistic -- mean/median/std -- in "
        "their filename), show only the mean rows; std/median rows for the same model are dropped. Plain baseline "
        "and arch_/data_/input_ sanity-check maps (no statistic in their name) are never affected by this. Default "
        "(neither --mean nor --std): show every statistic found, unchanged from before these flags existed.",
    )
    stat_group.add_argument(
        "--std",
        action="store_true",
        help="Same as --mean, but keep only std rows (mean/median rows for the same model are dropped).",
    )
    parser.add_argument(
        "--mask-thresh-frac",
        type=float,
        default=0.02,
        help="Fraction of each map's own 99.5th-percentile magnitude below which voxels are hidden.",
    )
    parser.add_argument(
        "--brain-mask",
        type=Path,
        default=DEFAULT_BRAIN_MASK,
        help="Binary mask restricting the overlay to brain voxels (matched to --template's grid).",
    )
    parser.add_argument(
        "--no-brain-mask", action="store_true", help="Disable brain-mask restriction; draw the overlay everywhere."
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Rescale each model's map onto a shared unit-free scale before display (see module docstring).",
    )
    parser.add_argument(
        "--norm-method",
        choices=["percentile", "zscore"],
        default="percentile",
        help="How to rescale each map when --norm is set (default: percentile, more robust to outlier voxels).",
    )
    parser.add_argument(
        "--norm-percentile",
        type=float,
        default=99.0,
        help="Percentile of each map's in-brain |value| that gets mapped to the scale's edge (--norm-method percentile).",
    )
    parser.add_argument(
        "--norm-zclip",
        type=float,
        default=3.0,
        help="Number of in-brain standard deviations mapped to the scale's edge (--norm-method zscore).",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Split each kind's block across two physical rows sharing one row label: the 3-panel axial/coronal/"
        "sagittal block on the first row, and a strip of --montage-axial-slices axial-only panels at evenly "
        "spaced z-levels spanning the brain (see axial_montage_indices) on the row below it -- the same layout as "
        "a typical neuroimaging results figure (3 orthogonal views + an axial slice montage). Combines with "
        "--kind (single-kind row pairs) or --combine-kinds (signed+abs side by side, same row pairs) exactly as "
        "those already work.",
    )
    parser.add_argument(
        "--montage-axial-slices",
        type=int,
        default=8,
        help="--montage only: number of additional evenly spaced axial panels per kind (default: 8; 7-10 is a "
        "typical range for this style of figure). Ignored when --montage-axial-z-mm is given.",
    )
    parser.add_argument(
        "--montage-axial-z-mm",
        type=float,
        nargs="+",
        default=None,
        help="--montage only: explicit world-space z levels (mm, e.g. -18 0 18) to show in the axial montage "
        "strip, instead of --montage-axial-slices evenly spaced levels -- fixed anatomical levels shown "
        "consistently across figures rather than levels that shift with each map's own brain extent.",
    )
    args = parser.parse_args()
    args.saliency_dir = args.saliency_dir or (DEFAULT_SALIENCY_DIR / args.target)
    args.output = args.output or Path(f"saliency_overlay_{args.target}.png")

    template_img = load_canonical_template(args.template)
    template_data = to_axial_first(template_img.get_fdata())
    z_c, y_c, x_c = brain_centroid_indices(template_data)
    brain_mask_data = None if args.no_brain_mask else load_brain_mask(args.brain_mask, template_img)
    # --montage needs the REAL brain extent regardless of --no-brain-mask
    # (which only controls whether the overlay itself gets masked) -- see
    # axial_montage_indices for why a template-nonzero fallback doesn't work
    # for this repo's whole-head (not skull-stripped) template.
    montage_z_indices = None
    if args.montage:
        if args.montage_axial_z_mm:
            montage_z_indices = axial_indices_for_mm(template_img, args.montage_axial_z_mm, template_data.shape[0])
        else:
            extent_mask_data = brain_mask_data if brain_mask_data is not None else load_brain_mask(args.brain_mask, template_img)
            montage_z_indices = axial_montage_indices(extent_mask_data, args.montage_axial_slices)

    wanted_stat = ("mean" if args.mean else "std") if (args.mean or args.std) else None
    # --mean/--std: keep non-stat files (baseline, arch_/data_/input_) as-is,
    # drop only the OTHER statistic's aggregate/bin rows for the same model
    # -- see file_stat's docstring for why a file with no stat in its name
    # is never dropped by this.

    if args.combine_kinds:
        signed_files = collect_kind_files(args.saliency_dir, "signed", args.models, wanted_stat)
        abs_files = collect_kind_files(args.saliency_dir, "abs", args.models, wanted_stat)
        abs_by_model = {model_name_from_filename(f, "abs"): f for f in abs_files}

        rows = []
        for f in signed_files:
            model_name = model_name_from_filename(f, "signed")
            abs_f = abs_by_model.get(model_name)
            if abs_f is None:
                print(f"WARN: no matching abs file for {model_name!r} -- skipping row")
                continue
            rows.append((model_name, f, abs_f))

        if not rows:
            raise SystemExit(f"No matching signed+abs file pairs found in {args.saliency_dir}")
    else:
        files = collect_kind_files(args.saliency_dir, args.kind, args.models, wanted_stat)
        if not files:
            raise SystemExit(f"No '{args.kind}' saliency files found in {args.saliency_dir}")
        rows = [(model_name_from_filename(f, args.kind), f, None) for f in files]

    target_spec = get_target_spec(args.target)
    metadata_by_prefix = load_subject_metadata(args.saliency_dir, target_spec)

    def metadata_for_file(fname):
        for prefix, meta in metadata_by_prefix.items():
            if fname.startswith(prefix):
                return meta
        return None

    # block_width is 3 (axial/coronal/sagittal) normally; with --montage, the
    # orthogonal views and the axial-montage strip go on two SEPARATE
    # physical rows (see draw_kind_block) rather than side by side in one
    # row, so block_width is instead however many columns the wider of the
    # two needs (usually the montage strip, since --montage-axial-slices
    # defaults to 8 > 3) -- the narrower row (usually the 3 orthogonal
    # views) just leaves its remaining columns blank. --combine-kinds always
    # means "two blocks side by side" (signed then abs), whether or not
    # --montage is set and regardless of how many physical rows each block
    # spans.
    n_montage = len(montage_z_indices) if montage_z_indices else 0
    block_width = max(3, n_montage) if args.montage else 3
    rows_per_group = 2 if args.montage else 1
    n_cols = block_width * 2 if args.combine_kinds else block_width
    n_physical_rows = len(rows) * rows_per_group
    col_width = 1.8 if args.montage else 3.0
    if args.montage:
        # Tightly group each model's orthogonal-views row with its
        # axial-montage row (rows_per_group == 2, same model, sharing one
        # ylabel below), while still visually separating different models'
        # groups from each other -- a single uniform hspace can't do both
        # (it's one fraction across the whole gridspec), so instead a short
        # blank spacer row is inserted BETWEEN groups only; hspace itself
        # stays small throughout, giving a tight in-group gap and a
        # visibly larger (spacer-row-sized) between-group gap.
        n_groups = len(rows)
        height_ratios = []
        gs_row_of_physical_row = []
        gs_row = 0
        for g in range(n_groups):
            for _ in range(rows_per_group):
                height_ratios.append(1.0)
                gs_row_of_physical_row.append(gs_row)
                gs_row += 1
            if g != n_groups - 1:
                height_ratios.append(0.45)  # between-group spacer, not mapped to any physical row
                gs_row += 1
        # Row height per unit: matplotlib's imshow keeps each slice's own
        # (height/width) aspect ratio, so if the gridspec cell's aspect
        # doesn't match, the slice only fills PART of its cell (letterboxed)
        # and the rest is blank space matplotlib puts inside the axes itself
        # -- invisible to hspace, which only controls space BETWEEN cells.
        # A flat "3.0 inches tall" (the non-montage default below) is much
        # taller than a col_width-wide axial slice actually needs, so nearly
        # all of the visible gap here was letterboxing, not hspace -- making
        # the in-group vs. between-group gap difference imperceptible.
        # Sizing the cell to the real axial-slice aspect (from the already-
        # loaded template) removes that slack, plus a fixed allowance for
        # the per-panel title text above the image, so hspace and the
        # spacer row are what actually set the visible gap.
        slice_h, slice_w = template_data[z_c, :, :].shape
        row_height_in = col_width * (slice_h / slice_w) + 0.3
        fig = plt.figure(figsize=(col_width * n_cols, row_height_in * sum(height_ratios)))
        gs = fig.add_gridspec(gs_row, n_cols, height_ratios=height_ratios, hspace=0.02)
        axes = np.empty((n_physical_rows, n_cols), dtype=object)
        for phys_row in range(n_physical_rows):
            target_row = gs_row_of_physical_row[phys_row]
            for c in range(n_cols):
                axes[phys_row, c] = fig.add_subplot(gs[target_row, c])
    else:
        fig, axes = plt.subplots(n_physical_rows, n_cols, figsize=(col_width * n_cols, 3 * n_physical_rows), squeeze=False)
    row_metas = []
    im_signed = im_abs = im = None  # last handle(s) drawn, for the --norm shared-colorbar case below

    def turn_off_unused(axes_row, col_offset, used_width):
        """Blanks the columns in a kind's block beyond what it actually drew (see block_width above)."""
        for c in range(col_offset + used_width, col_offset + block_width):
            axes_row[c].axis("off")

    def block_axes(ortho_axes_row, montage_axes_row, col_offset):
        """Every axis (both physical rows, if --montage) belonging to one kind's block -- for colorbar attachment."""
        axes_list = list(ortho_axes_row[col_offset:col_offset + block_width])
        if montage_axes_row is not None:
            axes_list += list(montage_axes_row[col_offset:col_offset + block_width])
        return axes_list

    for row, (model_name, primary_file, secondary_file) in enumerate(rows):
        ortho_row_idx = row * rows_per_group
        ortho_axes_row = axes[ortho_row_idx]
        montage_axes_row = axes[ortho_row_idx + 1] if args.montage else None

        if args.combine_kinds:
            im_signed = draw_kind_block(
                ortho_axes_row, 0, "signed", primary_file, template_img, template_data, brain_mask_data, z_c, y_c, x_c,
                args, col_title_prefix="signed ", montage_z_indices=montage_z_indices, montage_axes_row=montage_axes_row,
            )
            im_abs = draw_kind_block(
                ortho_axes_row, block_width, "abs", secondary_file, template_img, template_data, brain_mask_data, z_c, y_c, x_c,
                args, col_title_prefix="abs ", montage_z_indices=montage_z_indices, montage_axes_row=montage_axes_row,
            )
            turn_off_unused(ortho_axes_row, 0, 3)
            turn_off_unused(ortho_axes_row, block_width, 3)
            if montage_axes_row is not None:
                turn_off_unused(montage_axes_row, 0, n_montage)
                turn_off_unused(montage_axes_row, block_width, n_montage)
            if not args.norm:
                fig.colorbar(im_signed, ax=block_axes(ortho_axes_row, montage_axes_row, 0), shrink=0.7, label="signed saliency")
                fig.colorbar(im_abs, ax=block_axes(ortho_axes_row, montage_axes_row, block_width), shrink=0.7, label="abs saliency")
        else:
            im = draw_kind_block(
                ortho_axes_row, 0, args.kind, primary_file, template_img, template_data, brain_mask_data, z_c, y_c, x_c,
                args, montage_z_indices=montage_z_indices, montage_axes_row=montage_axes_row,
            )
            turn_off_unused(ortho_axes_row, 0, 3)
            if montage_axes_row is not None:
                turn_off_unused(montage_axes_row, 0, n_montage)
            if not args.norm:
                fig.colorbar(im, ax=block_axes(ortho_axes_row, montage_axes_row, 0), shrink=0.7, label=f"{args.kind} saliency")

        # Row title: the model name (relabeled per the sanity-check naming
        # scheme), as a horizontal label to the left of the row's first
        # (axial) subplot. Set on BOTH physical rows when --montage splits a
        # group across two rows, so the orthogonal-views row and the
        # axial-montage row read as one labeled group, not two separately
        # (and ambiguously) labeled rows.
        for r in ([ortho_row_idx, ortho_row_idx + 1] if args.montage else [ortho_row_idx]):
            axes[r, 0].set_ylabel(
                display_label(model_name), fontsize=11, fontweight="bold", rotation=0, ha="right", va="center", labelpad=10
            )
        row_metas.append(metadata_for_file(primary_file.name))

    if args.norm:
        if args.combine_kinds:
            fig.colorbar(
                im_signed, ax=axes[:, 0:block_width].ravel().tolist(), shrink=0.5,
                label=f"normalized signed saliency ({args.norm_method}, a.u.)",
            )
            fig.colorbar(
                im_abs, ax=axes[:, block_width:2 * block_width].ravel().tolist(), shrink=0.5,
                label=f"normalized abs saliency ({args.norm_method}, a.u.)",
            )
        else:
            label = f"normalized {args.kind} saliency ({args.norm_method}, a.u.)"
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label=label)

    # Single-subject suptitle: the true value is a property of the subject,
    # not the model, so it's shown whenever every plotted row traces back to
    # the same subject (checked below), regardless of how many distinct real
    # models are in the figure. The predicted value, unlike the true value,
    # genuinely differs per model -- for a single-model figure that's just
    # "predicted: X"; for a multi-model comparison it's "the" prediction that
    # would be ambiguous shown as one number, so every distinct model's own
    # prediction is listed instead (verified concretely: on one real
    # subject, 5 models predicted 24.0-41.5 from the same true age of 26.9).
    known_metas = [m for m in row_metas if m is not None]
    distinct_true_values = {m["true_value"] for m in known_metas if m["true_value"] is not None}
    age_note = None
    if len(distinct_true_values) == 1:
        true_value = next(iter(distinct_true_values))
        # {model: predicted_value}, not one entry per row -- every
        # attribution-method/sanity-check row for the same model shares one
        # forward pass and thus one prediction (see calculate_gradcam.py's
        # compute_direct_pipeline_saliency), so de-duplicate by model here
        # rather than listing that same number once per row.
        predicted_by_model = {
            m["model"]: m["predicted_value"] for m in known_metas if m["predicted_value"] is not None
        }
        true_display = target_spec.decode_display(true_value)
        if target_spec.task_type == "regression":
            if len(predicted_by_model) == 1:
                (model_name_, predicted_value), = predicted_by_model.items()
                diff = predicted_value - true_value
                age_note = (
                    f"true {target_spec.name}: {true_display}  |  predicted {target_spec.name}: "
                    f"{target_spec.decode_display(predicted_value)}  |  difference: {diff:+.1f}"
                )
            elif predicted_by_model:
                per_model = ", ".join(
                    f"{model_name_}={target_spec.decode_display(v)}" for model_name_, v in sorted(predicted_by_model.items())
                )
                age_note = f"true {target_spec.name}: {true_display}  |  predicted: {per_model}"
        else:
            # Classification (sex): predicted_value is a [0,1] probability
            # (calculate_gradcam.py sigmoids the raw logit before writing it
            # to JSON) -- no numeric "difference" against a categorical true
            # value, show the predicted class + probability instead.
            def _fmt_pred(v):
                return f"{target_spec.decode_display(v)} (p={v:.2f})"

            if len(predicted_by_model) == 1:
                (model_name_, predicted_value), = predicted_by_model.items()
                age_note = f"true {target_spec.name}: {true_display}  |  predicted: {_fmt_pred(predicted_value)}"
            elif predicted_by_model:
                per_model = ", ".join(f"{m}={_fmt_pred(v)}" for m, v in sorted(predicted_by_model.items()))
                age_note = f"true {target_spec.name}: {true_display}  |  predicted: {per_model}"

    suptitle_lines = [line for line in (args.title, age_note) if line]
    if suptitle_lines:
        fig.suptitle("\n".join(suptitle_lines), fontsize=14, fontweight="bold")

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
