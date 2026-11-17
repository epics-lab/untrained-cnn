"""
Cross-model ig_popmean comparison figure, styled like a standard
neuroimaging results figure (nilearn's own plot_stat_map look: black
background, "cold_hot" yellow/orange-vs-cyan/blue diverging colormap,
one row per model, ortho view + axial montage strip) -- a cleaner
alternative to visualize_paper.py's hand-rolled matplotlib overlay
(overlay_slice/draw_row there reimplement resampling, masking, and
slice compositing that nilearn's plot_stat_map already does internally,
which is also why this script's own code is much shorter).

Data prerequisites, template choice, and the ig_popmean/aggregate file
naming convention are identical to visualize_paper.py -- see that
module's docstring. This script reuses its DEFAULT_MODELS/
DEFAULT_SALIENCY_DIR/base_model_name and visualize_saliency.py's
template/mask loading and normalize_map, rather than redefining them.

--sanity-check {arch,data,input} switches to a different figure entirely:
one row per model of slurm/compute_sanity_checks.sh's single-reference-
subject Adebayo et al. sanity-check maps (arch_/data_/input_ randomization)
instead of the across-subject ig_popmean aggregate -- same nilearn styling,
same row-drawing machinery, just a different file-lookup (see
sanity_prefix()). There is no --stat dimension in that mode.

Usage:
    python visualize_paper_nilearn.py --output paper_ig_popmean_nilearn.png
    python visualize_paper_nilearn.py --kind abs --n-montage 8
    python visualize_paper_nilearn.py --models optim_uncnn_avg brainiac --cut-coords -17 1 15
    python visualize_paper_nilearn.py --sanity-check arch --kind abs --saliency-dir saliency_affine/age
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting

from visualize_paper import DEFAULT_MODELS, base_model_name, default_aggregate_dir
from visualize_saliency import (
    DEFAULT_BRAIN_MASK,
    DEFAULT_TEMPLATE,
    brain_centroid_indices,
    collect_kind_files,
    load_brain_mask,
    load_canonical_template,
    model_name_from_filename,
    normalize_map,
    resample_saliency,
    to_axial_first,
    voxel_to_mm,
)


def sanity_prefix(model, test, arch_variant="step01_head"):
    """
    arch_/data_/input_ filename prefix for (model, test in "arch"/"data"/
    "input"), matching slurm/compute_sanity_checks_randomize_model.sh's
    per-model case statement exactly (also duplicated in
    slurm/visualize_aggregate.sh's own sanity-check figure loop, and in
    aggregate_saliency.py's own deliberate nilearn-free duplicate -- keep
    all three in sync) -- anatcl_{level} and optim_uncnn_{channel} both need
    a differently-shaped prefix than a plain model name. Always followed by
    the "ig_reference_" tag (the only attribution method this whole script
    ever shows, see models_filter in main()) and the kind suffix by the
    caller.

    `arch_variant` (test == "arch" only): the segment calculate_gradcam.py's
    --randomize-model-mode bakes into the filename right after
    "arch_{model}_" -- "step{NN:02d}_{module}" for a --randomize-model-mode
    cascade step, default "step01_head" (the ONLY arch variant any current
    script can actually (re)produce -- slurm/compute_sanity_checks_
    randomize_model.sh's head-only --randomize-max-layers 1 run, resetting
    just the linear head and keeping the backbone). An "all" (full-model,
    backbone included) variant also exists on disk for NKI from an older,
    now-removed code path in slurm/compute_sanity_checks.sh's git history --
    those files are NOT reproducible by any current script, for any
    dataset, so "all" is deliberately not this function's default anymore;
    pass arch_variant="all" explicitly only to read those pre-existing NKI
    files. Ignored for test in ("data", "input"), which have no such
    variant.
    """
    if model.startswith("anatcl_"):
        level = model[len("anatcl_"):]
        return {
            "arch": f"arch_anatcl_{arch_variant}_{level}_",
            "data": f"data_anatcl_{level}_",
            "input": f"input_anatcl_permute_{level}_",
        }[test]
    if model.startswith("optim_uncnn_"):
        channel = model[len("optim_uncnn_"):]
        return {
            "arch": f"arch_optim_uncnn_{arch_variant}_{channel}_",
            "data": f"data_optim_uncnn_{channel}_",
            "input": f"input_optim_uncnn_permute_{channel}_",
        }[test]
    return {
        "arch": f"arch_{model}_{arch_variant}_",
        "data": f"data_{model}_",
        "input": f"input_{model}_permute_",
    }[test]


def masked_stat_img(file_path, template_img, brain_mask_data, kind, norm, norm_method, norm_percentile, norm_zclip):
    """
    A saliency map as a nib.Nifti1Image already resampled onto the
    template's own grid/affine (so plot_stat_map's own resampling is a
    no-op) and restricted to in-brain voxels, optionally rescaled onto
    the shared unit-free range via normalize_map -- see visualize_paper.py's
    module docstring for why per-map rescaling (not a shared raw-value
    scale) is the right way to compare saliency across architectures.
    """
    data = resample_saliency(file_path, template_img)
    if norm:
        data = normalize_map(data, brain_mask_data, kind, norm_method, norm_percentile, norm_zclip)
    if brain_mask_data is not None:
        data = np.where(brain_mask_data, data, 0.0)
    # Undo to_axial_first's (Z, Y, X) transpose -- plot_stat_map needs an
    # image whose array/affine agree with template_img's own (X, Y, Z) RAS
    # convention, not the axial-first layout visualize_saliency.py's
    # matplotlib-slicing helpers use.
    data_ras = np.transpose(data, (2, 1, 0))
    return nib.Nifti1Image(data_ras, template_img.affine, template_img.header)


def montage_z_mm_coords(template_img, brain_mask_data, n_montage, inset_frac=0.2):
    """
    n_montage z-levels (world mm) spanning the brain's own inferior->
    superior extent, evenly spaced within the CENTRAL (1 - 2*inset_frac)
    fraction of that extent -- i.e. inset_frac trimmed off each end before
    spacing, so with the default 0.2 the outermost slice is never closer
    than 20% of the brain's own z-extent to its true top/bottom edge.

    Deliberately not visualize_saliency.py's own axial_montage_indices,
    which only trims a fixed 5% -- fine for the 8+ slices it's normally
    used with, but with this script's much smaller default of 4 slices the
    outer two would still land almost exactly at the extremes (verified
    directly: z=-55/z=74 out of a roughly -60..80 brain extent). Picked
    directly from brain_mask_data (not each row's own thresholded stat
    map, as nilearn's find_cut_slices would) so every model gets the exact
    same z-levels -- required for a cross-model comparison figure.
    """
    x_c, y_c, z_c = brain_centroid_indices(brain_mask_data)[::-1]
    z_present = np.where(brain_mask_data.any(axis=(1, 2)))[0]
    z_min, z_max = int(z_present[0]), int(z_present[-1])
    inset = int(round(inset_frac * (z_max - z_min)))
    z_min, z_max = z_min + inset, z_max - inset
    z_indices = np.clip(np.linspace(z_min, z_max, n_montage).round().astype(int), 0, brain_mask_data.shape[0] - 1)
    return [voxel_to_mm(template_img, x_c, y_c, int(z))[2] for z in z_indices]


def simplify_uncnn_label(label):
    """
    "optim_uncnn_avg"/"optim_uncnn_raw"/"optim_uncnn_rank"/"optim_uncnn_sobel"
    (any channel) -> "unCNN" -- the untrained-CNN control's specific channel
    is an implementation detail (see PAPER_NL_MODELS' default of just the
    averaged channel), not something a reader comparing model architectures
    needs spelled out in a row label. Used behind --uncnn-label since the
    full channel-qualified name is still the right default elsewhere (e.g.
    matching --models values, file lookups).
    """
    return "unCNN" if label.startswith("optim_uncnn") else label


def draw_model_row(
    fig, gs_row, gs_row_montage, stat_img, template_img, kind, cut_coords, montage_z_coords, vmax, threshold, label,
    label_scale=1.0, cmap=None,
):
    """
    One model's 2-row block: ortho view on gs_row, axial montage strip on
    gs_row_montage. cold_hot (nilearn's own diverging colormap -- cyan/blue
    for negative, yellow/red for positive) only makes sense for signed maps:
    it colors by POSITION in [vmin, vmax], not by sign, so a one-sided abs
    map (whose values only ever occupy the upper half of that range once
    vmin=0) would still render its lower-magnitude survivors in the cyan/
    blue half -- misleadingly implying negative values that can't exist in
    an abs map. "hot" (matplotlib's plain black -> red -> yellow -> white
    sequential map, same convention visualize_saliency.py's own
    overlay_slice uses for abs via YlOrRd) is the one-sided equivalent --
    unlike nilearn's own *_black_bone maps, which blend two different
    colormaps together and are not a single sequential scale.

    `cmap`: overrides the kind-derived default colormap above when given
    (any matplotlib/nilearn colormap name or object) -- vmin/symmetric_cbar
    still follow `kind` either way, since those reflect whether the data
    being drawn is genuinely one-sided or symmetric, independent of which
    palette renders it. Default None keeps the kind-derived default
    (cold_hot/YlOrRd) for every existing caller.

    colorbar is always off here -- every row shares one colormap/vmax, so
    main() draws a single horizontal colorbar once at the bottom of the
    whole figure instead of nilearn's own per-row vertical one (see
    add_shared_colorbar), which also frees the right-hand margin nilearn
    would otherwise reserve for it on every row.
    """
    if cmap is None:
        cmap = plotting.cm.cold_hot if kind == "signed" else "YlOrRd"
    vmin = None if kind == "signed" else 0.0

    ortho_ax = fig.add_subplot(gs_row)
    plotting.plot_stat_map(
        stat_img, bg_img=template_img, display_mode="ortho", cut_coords=cut_coords, cmap=cmap, threshold=threshold,
        vmin=vmin, vmax=vmax, symmetric_cbar=(kind == "signed"), colorbar=False, black_bg=True, annotate=True,
        draw_cross=False, axes=ortho_ax,
    )

    montage_ax = fig.add_subplot(gs_row_montage)
    plotting.plot_stat_map(
        stat_img, bg_img=template_img, display_mode="z", cut_coords=montage_z_coords, cmap=cmap, threshold=threshold,
        vmin=vmin, vmax=vmax, symmetric_cbar=(kind == "signed"), colorbar=False, black_bg=True, annotate=True,
        draw_cross=False, axes=montage_ax,
    )

    fig.text(0.005, (ortho_ax.get_position().y0 + ortho_ax.get_position().y1) / 2, label, fontsize=11 * label_scale,
              fontweight="bold", ha="left", va="center", color="white")

    return ortho_ax


def add_shared_colorbar(fig, cax, cmap, kind, vmax, orientation="horizontal", tick_fontsize=None):
    """
    One colorbar for the whole figure (or, with a small `cax`, for one row
    of a larger figure -- see compare_aggregate_arch_saliency.py's
    plot_difference_grid), in cax (an axes sized/positioned by the caller --
    see main()), replacing the vertical one nilearn would otherwise draw on
    the first row only. Uses the same cmap/vmin/vmax every row was drawn
    with (kind=="signed" -> symmetric [-vmax, vmax], matching
    plot_stat_map's own symmetric_cbar=True there), so the color scale is
    identical to what's on screen -- the only thing not reproduced is
    nilearn's own per-row threshold notch, since each row can have a
    different threshold (see row_threshold in main()) but they all share
    this one colorbar's color scale regardless.

    `orientation`: "horizontal" (default, every existing caller) draws a
    full-width bar with ticks below it, for a single colorbar shared by an
    entire figure. "vertical" draws a narrow bar with ticks to its right,
    for a per-row colorbar placed in a margin strip alongside that row.

    `tick_fontsize`: None (default) keeps matplotlib's own default tick
    label size, unchanged for this function's own main() caller below. Pass
    a number to override it -- e.g. compare_aggregate_arch_saliency.py's
    plot_difference_grid does, for every colorbar it draws.
    """
    vmin = -vmax if kind == "signed" else 0.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation=orientation)
    tick_axis = cbar.ax.yaxis if orientation == "vertical" else cbar.ax.xaxis
    tick_kwargs = dict(color="white", labelcolor="white")
    if tick_fontsize is not None:
        tick_kwargs["labelsize"] = tick_fontsize
    tick_axis.set_tick_params(**tick_kwargs)
    cbar.outline.set_edgecolor("white")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Default: saliency_outputs/<target>/aggregate, aggregate_saliency.py's own output dir (see --target).",
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default="age",
        help="Which target's aggregate/ subtree to read (default: age). Only used to resolve --saliency-dir's "
        "and --output's defaults -- ignored if both are passed explicitly.",
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--stat", choices=["mean", "std"], default="mean")
    parser.add_argument("--kind", choices=["signed", "abs"], default="signed")
    parser.add_argument(
        "--sanity-check", choices=["arch", "data", "input"], default=None,
        help="Show the Adebayo et al. sanity-check battery (slurm/compute_sanity_checks.sh's single-reference-"
        "subject arch_/data_/input_ maps) instead of the across-subject ig_popmean aggregate -- one row per model, "
        "same nilearn styling. --stat is ignored in this mode (those maps have no stat dimension). Default: off, "
        "i.e. the original aggregate-comparison figure.",
    )
    parser.add_argument(
        "--arch-variant", default="step01_head",
        help="--sanity-check arch only: see sanity_prefix()'s own docstring -- default 'step01_head' (the "
        "linear-head-only cascade run, the only one any current script can reproduce). Pass 'all' to read the "
        "older, no-longer-reproducible full-model NKI files instead.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Default: paper_ig_popmean_nilearn_<target>.png (or paper_sanity_check_nilearn_<sanity-check>_<target>.png "
        "with --sanity-check), so different --target/--sanity-check runs don't overwrite each other.",
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--no-colorbar", action="store_true",
        help="Drop the shared horizontal colorbar at the bottom of the figure (see add_shared_colorbar) -- e.g. "
        "for a paper figure whose color scale is given in the caption instead.",
    )
    parser.add_argument(
        "--label-scale", type=float, default=1.0,
        help="Multiplies each row's model-name label fontsize (base 11) by this factor (default: 1.0, unchanged).",
    )
    parser.add_argument(
        "--uncnn-label", action="store_true",
        help="Show the untrained-CNN control's row label as plain 'unCNN' instead of "
        "'optim_uncnn_avg'/'optim_uncnn_raw'/'optim_uncnn_rank'/'optim_uncnn_sobel' (whichever channel was "
        "requested via --models) -- see simplify_uncnn_label. Default: off, i.e. the full channel-qualified name.",
    )
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--no-brain-mask", action="store_true")
    parser.add_argument("--norm", action="store_true", default=True)
    parser.add_argument("--no-norm", dest="norm", action="store_false", help="Use each row's own raw vmax instead of the shared [-1,1]/[0,1] rescaling (default: on).")
    parser.add_argument("--norm-method", choices=["percentile", "zscore"], default="percentile")
    parser.add_argument("--norm-percentile", type=float, default=99.0)
    parser.add_argument("--norm-zclip", type=float, default=3.0)
    parser.add_argument(
        "--cut-coords", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="Shared (x, y, z) mm cut for every model's ortho row (default: nilearn picks each row's own, from that row's own peak -- not comparable across rows).",
    )
    parser.add_argument("--n-montage", type=int, default=4, help="Axial slices in the montage strip below each ortho row (0 disables the strip).")
    parser.add_argument(
        "--threshold-percentile", type=float, default=75.0,
        help="Hide voxels below this in-brain |value| percentile (per row, computed on that row's own already-"
        "normalized/masked data) -- e.g. 75 keeps the top 25%% by magnitude. Lower = more voxels shown (less "
        "sparse); higher = fewer, higher-confidence clusters only. 0 disables thresholding (every nonzero "
        "in-brain voxel is drawn).",
    )
    args = parser.parse_args()
    args.saliency_dir = args.saliency_dir or default_aggregate_dir(args.target)
    if args.output is None:
        args.output = (
            Path(f"paper_sanity_check_nilearn_{args.sanity_check}_{args.target}.png")
            if args.sanity_check else Path(f"paper_ig_popmean_nilearn_{args.target}.png")
        )

    template_img = load_canonical_template(args.template)
    brain_mask_data = None if args.no_brain_mask else load_brain_mask(args.brain_mask, template_img)
    # axial_montage_indices (via montage_z_mm_coords) needs the real brain
    # extent even when --no-brain-mask means the overlay itself isn't
    # masked -- see that function's own docstring (in visualize_saliency.py)
    # for why the whole-head template can't stand in for it here.
    montage_extent_mask = brain_mask_data if brain_mask_data is not None else load_brain_mask(args.brain_mask, template_img)

    if args.sanity_check:
        # arch_/data_/input_ maps: one exact file per model (no stat
        # dimension, no ambiguous-prefix siblings to disambiguate), so a
        # direct per-model existence check is simpler and clearer than
        # collect_kind_files's longest-prefix-wins matching (built for the
        # aggregate case's genuinely ambiguous prefixes -- see that
        # function's docstring).
        rows, missing = [], []
        for model in args.models:
            f = args.saliency_dir / f"{sanity_prefix(model, args.sanity_check, args.arch_variant)}ig_reference_{args.kind}_saliency.nii.gz"
            if f.exists():
                rows.append((model, f))
            else:
                missing.append(model)
        if missing:
            print(f"WARN: no {args.sanity_check} sanity-check file found for model(s) {missing!r} -- skipping those rows")
        if not rows:
            raise SystemExit(f"No {args.sanity_check} sanity-check '{args.kind}' files found in {args.saliency_dir} for {args.models!r}")
    else:
        models_filter = [f"{model}_ig_reference_aggregate_{args.stat}" for model in args.models]
        files = collect_kind_files(args.saliency_dir, args.kind, models_filter, None)
        if not files:
            raise SystemExit(f"No ig_popmean aggregate '{args.kind}' files found in {args.saliency_dir} for {args.models!r}")
        rows = [(model_name_from_filename(f, args.kind), f) for f in files]

        found_models = {model_name for model_name, _ in rows}
        missing = [m for m in args.models if f"{m}_ig_reference_aggregate_{args.stat}" not in found_models]
        if missing:
            print(f"WARN: no ig_popmean aggregate file found for model(s) {missing!r} ({args.stat}) -- skipping those rows")

    print("Resampling and masking each model's map onto the template grid...")
    stat_imgs = [
        (model_name, masked_stat_img(f, template_img, brain_mask_data, args.kind, args.norm, args.norm_method, args.norm_percentile, args.norm_zclip))
        for model_name, f in rows
    ]

    def row_threshold(img):
        """
        A concrete numeric threshold for one row's already-masked data at
        --threshold-percentile, computed here (not passed as nilearn's own
        "X%" string threshold) because plot_stat_map's automatic
        (cut_coords=None) cut-finding path can't handle a string threshold --
        see find_xyz_cut_coords, which subtracts it from array values.
        """
        if args.threshold_percentile <= 0:
            return 1e-6
        vals = np.abs(img.get_fdata())
        vals = vals[vals > 0]
        if vals.size == 0:
            return 1e-6
        return max(float(np.percentile(vals, args.threshold_percentile)), 1e-6)

    vmax = 1.0 if args.norm else None
    if vmax is None:
        # One shared raw-value vmax across all rows, so panels stay visually
        # comparable when --no-norm is passed (plot_stat_map's own
        # symmetric_cbar="auto" would otherwise pick a different scale per
        # row, defeating the point of a multi-model comparison figure).
        vmax = max(float(np.percentile(np.abs(img.get_fdata()[img.get_fdata() != 0]), 99.5)) for _, img in stat_imgs)

    montage_z_coords = montage_z_mm_coords(template_img, montage_extent_mask, args.n_montage) if args.n_montage else []

    # Ortho and montage rows get the EXACT same height ratio (both 1.5) so a
    # panel's rendered size is driven only by whether its column is wide
    # enough to avoid being width-constrained below that shared height --
    # see ORTHO_WIDTH_FRAC below, which sizes the ortho row's columns to
    # match the montage row's per-tile width for exactly that reason.
    n_rows = len(stat_imgs)
    height_ratios = []
    for r in range(n_rows):
        height_ratios += [1.5, 1.5 if args.n_montage else 0.0]
    fig = plt.figure(figsize=(14, 4.2 * n_rows), facecolor="black")
    # bottom=0.06 reserves room below the last model row for the shared
    # horizontal colorbar added below (add_shared_colorbar) -- replacing
    # nilearn's own per-row vertical one, which used to eat into the right
    # margin on row 0 only.
    #
    # Two SEPARATE GridSpecs, not one: the ortho row's 3 differently-
    # oriented panels (sagittal/coronal/axial) only need a fraction of the
    # montage row's width before nilearn's own fixed inter-panel margin
    # starts reading as a big gap -- handing it the full width (like the
    # montage row below, whose 4 same-orientation axial tiles DO fill the
    # whole row tightly, since they're all the same view and pack edge to
    # edge) just stretches those 3 panels apart with dead space instead of
    # making them bigger (they're already height-limited by the row's own
    # height_ratio). ORTHO_WIDTH_FRAC is deliberately well under
    # MONTAGE_RIGHT - MONTAGE_LEFT so the ortho row's own inter-panel gaps
    # stay visibly tighter than this, but the montage tiles (already
    # touching edge to edge) are still the tighter-grouped of the two.
    #
    # gs_ortho is centered on gs_montage's own midpoint (not left-anchored
    # next to the row label like before) so the 3-panel block sits
    # symmetrically above the 4-panel montage strip -- a coherent, aligned
    # column composition per model row. Both GridSpecs share the exact same
    # top/bottom/height_ratios/hspace, so row i's vertical span lines up
    # between them even though their horizontal (left/right) extents
    # differ.
    #
    # ORTHO_WIDTH_FRAC = 3 panels' worth of the montage row's own per-tile
    # width (not a fixed guess) -- with the now-equal height_ratios above,
    # this is what makes each ortho panel render at the SAME size as a
    # montage tile: same row height, and a column just wide enough for that
    # height's natural (aspect-preserving) width, so nilearn draws it at
    # full size rather than shrinking it to fit a narrower column or
    # centering it in a wider one with dead space either side.
    MONTAGE_LEFT, MONTAGE_RIGHT = 0.05, 0.98
    ORTHO_PANEL_COUNT = 3  # sagittal, coronal, axial -- fixed, unlike the montage's --n-montage
    montage_panel_count = args.n_montage or 4
    montage_tile_width = (MONTAGE_RIGHT - MONTAGE_LEFT) / montage_panel_count
    ORTHO_WIDTH_FRAC = montage_tile_width * ORTHO_PANEL_COUNT
    montage_center = (MONTAGE_LEFT + MONTAGE_RIGHT) / 2
    ortho_left = montage_center - ORTHO_WIDTH_FRAC / 2
    ortho_right = montage_center + ORTHO_WIDTH_FRAC / 2

    common_kwargs = dict(
        height_ratios=height_ratios, hspace=0.05, top=0.94 if args.title else 0.99,
        bottom=0.01 if args.no_colorbar else 0.06,
    )
    gs_ortho = fig.add_gridspec(2 * n_rows, 1, left=ortho_left, right=ortho_right, **common_kwargs)
    # Rendered at the SAME [ortho_left, ortho_right] span as gs_ortho (not
    # the full [MONTAGE_LEFT, MONTAGE_RIGHT] that ORTHO_WIDTH_FRAC above was
    # itself computed from) so the montage row's total length matches the
    # ortho row's -- nilearn packs its 4 tiles into whatever width its own
    # axes is given, so narrowing this axes alone (without touching
    # ORTHO_WIDTH_FRAC/ortho_left/ortho_right or anything else) is enough to
    # shrink the row via tighter inter-tile spacing and/or smaller tiles.
    gs_montage = fig.add_gridspec(2 * n_rows, 1, left=ortho_left, right=ortho_right, **common_kwargs)

    for row, (model_name, stat_img) in enumerate(stat_imgs):
        # sanity-check rows already carry the plain model name (see the
        # args.sanity_check branch above) -- no "_ig_reference_aggregate_
        # {stat}" suffix to strip, unlike the aggregate-comparison rows.
        label = model_name if args.sanity_check else base_model_name(model_name, args.stat)
        if args.uncnn_label:
            label = simplify_uncnn_label(label)
        draw_model_row(
            fig, gs_ortho[2 * row, 0], gs_montage[2 * row + 1, 0], stat_img, template_img, args.kind,
            args.cut_coords, montage_z_coords, vmax, row_threshold(stat_img), label=label,
            label_scale=args.label_scale,
        )

    if not args.no_colorbar:
        cmap = plotting.cm.cold_hot if args.kind == "signed" else "YlOrRd"
        cax = fig.add_axes([0.3, 0.015, 0.4, 0.02])
        add_shared_colorbar(fig, cax, cmap, args.kind, vmax)

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold", color="white")

    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
