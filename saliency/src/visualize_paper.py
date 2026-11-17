"""
Cross-model comparison figure for the paper: ig_popmean (integrated
gradients, reference/population-mean baseline -- calculate_gradcam.py's
"ig_reference" attribution method) saliency, aggregated across subjects
(aggregate_saliency.py's across-subject mean/std), one row per model. This
is a narrower, paper-figure-focused sibling of visualize_saliency.py -- see
that module's docstring for the shared template/resampling/normalization
background this script reuses directly (imported below, not reimplemented):
only ig_popmean aggregate rows are ever shown here (no baseline/sanity-check
rows, no other attribution method), and there is one new normalization
option not present in visualize_saliency.py (see --norm-reference-model).

Expects saliency_outputs/aggregate/{model}_ig_reference_aggregate_{stat}_
{signed,abs}_saliency.nii.gz to already exist for every requested model --
i.e. slurm/visualize_aggregate.sh (or src/aggregate_saliency.py directly)
must have already been run with ig_reference among its aggregated
attribution methods. This script does not compute aggregates itself.

--norm-reference-model: a SECOND normalization strategy
--------------------------------------------------------
visualize_saliency.py's --norm always rescales each model's map onto the
shared unit-free range using THAT MAP'S OWN in-brain percentile/z-score (see
normalize_map there, reused as-is below) -- every row ends up independently
stretched to fill the same [-1,1]/[0,1] range, so the figure shows each
model's *relative* spatial pattern clearly, but two models whose raw
saliency magnitude genuinely differs by 10x look identical in the colorbar.

Passing --norm-reference-model MODEL instead computes that ONE scale from a
single reference model's map (--norm-percentile/--norm-zclip/--norm-method
still control how), and reuses that SAME scale for every other row -- so if
another model's raw saliency is much weaker than the reference, its row
stays visibly faint (or saturates at the colorbar's edge if much stronger)
instead of being re-stretched to look comparable. --norm-reference-model
implies --norm; the reference model itself must also be one of --models
(its own row is then normalized against its own scale, i.e. always maps
exactly onto +/-1 or [0,1]'s upper edge).

Usage:
    python visualize_paper.py --norm --output paper_ig_popmean_mean.png
    python visualize_paper.py --norm --norm-reference-model optim_uncnn_avg \\
        --output paper_ig_popmean_mean_refnorm.png
    python visualize_paper.py --norm --combine-kinds --montage --montage-axial-z-mm -18 0 18
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_saliency import (
    DEFAULT_BRAIN_MASK,
    DEFAULT_TEMPLATE,
    axial_indices_for_mm,
    axial_montage_indices,
    brain_centroid_indices,
    collect_kind_files,
    load_brain_mask,
    load_canonical_template,
    model_name_from_filename,
    normalize_map,
    overlay_slice,
    resample_saliency,
    to_axial_first,
    voxel_to_mm,
)

DEFAULT_SALIENCY_DIR = Path(__file__).resolve().parents[1] / "saliency_outputs"


def default_aggregate_dir(target):
    """<repo>/saliency_outputs/<target>/aggregate -- aggregate_saliency.py's per-target output location."""
    return DEFAULT_SALIENCY_DIR / target / "aggregate"

# CROSS_MODELS convention from slurm/visualize_aggregate.sh with
# AVERAGE_UNCNN_CHANNELS=1 -- optim_uncnn's raw/rank/sobel channels collapsed
# onto one row so a 6-model comparison isn't dominated by 3 optim_uncnn rows;
# override with --models to see the 3 channels separately (optim_uncnn_raw
# optim_uncnn_rank optim_uncnn_sobel) instead of optim_uncnn_avg.
DEFAULT_MODELS = ("optim_uncnn_avg", "brainiac", "swinbrain", "simclr", "anatcl_global", "anatcl_local")


def compute_scale(data, brain_mask_data, kind, method="percentile", percentile=99.0, zclip=3.0):
    """
    The scalar scale a map gets divided by to land on the shared unit-free
    range. visualize_saliency.py's normalize_map (imported and reused as-is
    below for the default per-map strategy) computes this internally but
    only returns the already-rescaled array, not the scale itself --
    --norm-reference-model needs the scale computed ONCE from a single
    reference map and then reused for every other row, so it's recomputed
    here rather than modifying that shared function. Mirrors normalize_map's
    own math exactly -- see that function's docstring for why percentile/
    z-score over in-brain voxels is the right way to pick this.
    """
    vals = data[brain_mask_data] if brain_mask_data is not None else data.ravel()
    if kind == "abs":
        if method == "zscore":
            mu, sigma = vals.mean(), vals.std()
            scale = mu + zclip * (sigma if sigma > 0 else 1.0)
        else:
            scale = np.percentile(vals, percentile)
        return scale if scale > 0 else 1.0

    if method == "zscore":
        mu, sigma = vals.mean(), vals.std()
        scale = zclip * (sigma if sigma > 0 else 1.0)
    else:
        scale = np.percentile(np.abs(vals), percentile)
    return scale if scale > 0 else 1.0


def apply_scale(data, scale, kind):
    """Divide `data` by an already-computed scale (see compute_scale) and clip onto the shared unit-free range."""
    if kind == "abs":
        return np.clip(data / scale, 0.0, 1.0)
    return np.clip(data / scale, -1.0, 1.0)


def base_model_name(model_name, stat):
    """
    model_name (e.g. "brainiac_ig_reference_aggregate_mean") -> the plain
    base model name a row was requested under in --models (e.g. "brainiac"),
    by stripping the fixed "_ig_reference_aggregate_{stat}" suffix every row
    here always has (see main()'s models_filter). Used for the row's ylabel
    instead of visualize_saliency.py's display_label, which is built for a
    figure mixing several attribution methods/stats and would otherwise
    repeat "ig_popmean_aggregate_{stat}_" on every single row here -- always
    the same tag, since this script only ever shows ig_popmean rows.
    """
    suffix = f"_ig_reference_aggregate_{stat}"
    return model_name[: -len(suffix)] if model_name.endswith(suffix) else model_name


def draw_row(
    ortho_axes_row, col_offset, kind, file_path, template_img, template_data, brain_mask_data, z_c, y_c, x_c, args,
    col_title_prefix="", montage_z_indices=None, montage_axes_row=None, ref_scale=None,
):
    """
    Draws one kind's (signed/abs) panels for one model's row -- the 3
    orthogonal views always, plus one axial-only panel per montage_z_indices
    entry on a second physical row when --montage is set. Deliberately a
    near-duplicate of visualize_saliency.py's draw_kind_block (rather than
    importing/patching it) so that function stays untouched: the one thing
    this script adds -- ref_scale, a precomputed scale reused across every
    row for --norm-reference-model instead of each row picking its own via
    normalize_map -- needed a hook that function doesn't expose.

    ref_scale=None (default): per-map --norm strategy, identical to
    draw_kind_block (normalize_map picks this file's own scale). A non-None
    ref_scale instead divides this file's data by that fixed scale via
    apply_scale, ignoring this file's own percentile/z-score entirely --
    see the module docstring's --norm-reference-model section.
    """
    print(f"Resampling {file_path.name} onto template grid...")
    data = resample_saliency(file_path, template_img)

    if args.norm:
        if ref_scale is not None:
            data = apply_scale(data, ref_scale, kind)
        else:
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


def build_figure_axes(n_logical_rows, rows_per_group, n_cols, col_width, montage, template_data, z_c):
    """
    Build the figure and its (n_logical_rows * rows_per_group, n_cols) axes
    grid. Duplicated from visualize_saliency.py's main() (not imported --
    that logic lives inline in that function, not exposed separately) rather
    than modifying that file. See its inline comments (same layout here) for
    why --montage tightly groups each model's orthogonal-views row with its
    axial-montage row while still visibly separating different models from
    each other.
    """
    n_physical_rows = n_logical_rows * rows_per_group
    if not montage:
        fig, axes = plt.subplots(n_physical_rows, n_cols, figsize=(col_width * n_cols, 3 * n_physical_rows), squeeze=False)
        return fig, axes

    height_ratios = []
    gs_row_of_physical_row = []
    gs_row = 0
    for g in range(n_logical_rows):
        for _ in range(rows_per_group):
            height_ratios.append(1.0)
            gs_row_of_physical_row.append(gs_row)
            gs_row += 1
        if g != n_logical_rows - 1:
            height_ratios.append(0.06)  # between-group spacer, not mapped to any physical row
            gs_row += 1

    slice_h, slice_w = template_data[z_c, :, :].shape
    row_height_in = col_width * (slice_h / slice_w) + 0.1
    fig = plt.figure(figsize=(col_width * n_cols, row_height_in * sum(height_ratios)))
    gs = fig.add_gridspec(gs_row, n_cols, height_ratios=height_ratios, hspace=0.0)
    axes = np.empty((n_physical_rows, n_cols), dtype=object)
    for phys_row in range(n_physical_rows):
        target_row = gs_row_of_physical_row[phys_row]
        for c in range(n_cols):
            axes[phys_row, c] = fig.add_subplot(gs[target_row, c])
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Directory holding {model}_ig_reference_aggregate_{stat}_{kind}_saliency.nii.gz files "
        "(default: saliency_outputs/<target>/aggregate, aggregate_saliency.py's own output dir -- see --target).",
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default="age",
        help="Which target's aggregate/ subtree to read (default: age). Only used to resolve --saliency-dir's "
        "and --output's defaults -- ignored if both are passed explicitly.",
    )
    parser.add_argument(
        "--models", nargs="*", default=list(DEFAULT_MODELS),
        help=f"Base model names (no ig_reference/aggregate/stat suffix) to show, one row each, in this order "
        f"(default: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument("--stat", choices=["mean", "std"], default="mean", help="Across-subject statistic to show (default: mean).")
    parser.add_argument("--kind", choices=["signed", "abs"], default="signed", help="Ignored when --combine-kinds is set.")
    parser.add_argument(
        "--combine-kinds", action="store_true",
        help="Show signed and abs side by side, 6 columns per row, one row per model -- see "
        "visualize_saliency.py's --combine-kinds (same behavior here).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Default: paper_ig_popmean_<target>.png, so different --target runs don't overwrite each other.",
    )
    parser.add_argument("--title", type=str, default=None, help="Super-title for the whole figure.")
    parser.add_argument(
        "--mask-thresh-frac", type=float, default=0.02,
        help="Fraction of each map's own 99.5th-percentile magnitude below which voxels are hidden.",
    )
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--no-brain-mask", action="store_true", help="Disable brain-mask restriction; draw the overlay everywhere.")
    parser.add_argument(
        "--norm", action="store_true",
        help="Rescale onto a shared unit-free scale before display -- see --norm-reference-model for which of the "
        "two normalization strategies is used (default here, when --norm-reference-model isn't given: per-map, "
        "identical to visualize_saliency.py's --norm).",
    )
    parser.add_argument(
        "--norm-reference-model", default=None,
        help="Compute the normalization scale from this ONE model's map (must also be in --models) and reuse that "
        "same scale for every row, instead of each row picking its own scale independently -- see the module "
        "docstring. Implies --norm.",
    )
    parser.add_argument("--norm-method", choices=["percentile", "zscore"], default="percentile")
    parser.add_argument("--norm-percentile", type=float, default=99.0)
    parser.add_argument("--norm-zclip", type=float, default=3.0)
    parser.add_argument("--montage", action="store_true", help="See visualize_saliency.py's --montage (same behavior here).")
    parser.add_argument("--montage-axial-slices", type=int, default=8)
    parser.add_argument("--montage-axial-z-mm", type=float, nargs="+", default=None)
    args = parser.parse_args()
    args.saliency_dir = args.saliency_dir or default_aggregate_dir(args.target)
    args.output = args.output or Path(f"paper_ig_popmean_{args.target}.png")

    if args.norm_reference_model:
        args.norm = True
        if args.norm_reference_model not in args.models:
            raise SystemExit(
                f"--norm-reference-model {args.norm_reference_model!r} must also be one of --models {args.models!r}."
            )

    template_img = load_canonical_template(args.template)
    template_data = to_axial_first(template_img.get_fdata())
    z_c, y_c, x_c = brain_centroid_indices(template_data)
    brain_mask_data = None if args.no_brain_mask else load_brain_mask(args.brain_mask, template_img)
    montage_z_indices = None
    if args.montage:
        if args.montage_axial_z_mm:
            montage_z_indices = axial_indices_for_mm(template_img, args.montage_axial_z_mm, template_data.shape[0])
        else:
            extent_mask_data = brain_mask_data if brain_mask_data is not None else load_brain_mask(args.brain_mask, template_img)
            montage_z_indices = axial_montage_indices(extent_mask_data, args.montage_axial_slices)

    # Only ever ig_popmean (ig_reference tag) aggregate rows -- the suffix is
    # fixed here rather than exposed as a CLI knob (contrast with
    # visualize_saliency.py's generic --models prefix filter), since that's
    # this whole script's reason to exist (see module docstring).
    models_filter = [f"{model}_ig_reference_aggregate_{args.stat}" for model in args.models]

    if args.combine_kinds:
        signed_files = collect_kind_files(args.saliency_dir, "signed", models_filter, None)
        abs_files = collect_kind_files(args.saliency_dir, "abs", models_filter, None)
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
            raise SystemExit(f"No matching signed+abs ig_popmean aggregate file pairs found in {args.saliency_dir}")
    else:
        files = collect_kind_files(args.saliency_dir, args.kind, models_filter, None)
        if not files:
            raise SystemExit(f"No ig_popmean aggregate '{args.kind}' files found in {args.saliency_dir} for {args.models!r}")
        rows = [(model_name_from_filename(f, args.kind), f, None) for f in files]

    found_models = {model_name for model_name, _, _ in rows}
    missing = [m for m in args.models if f"{m}_ig_reference_aggregate_{args.stat}" not in found_models]
    if missing:
        print(f"WARN: no ig_popmean aggregate file found for model(s) {missing!r} ({args.stat}) -- skipping those rows")

    # --norm-reference-model: compute ONE scale per kind shown, from the
    # reference row's own file(s), before the main drawing loop -- every
    # row's draw_row call below then reuses this same scale (ref_scale)
    # instead of picking its own (see module docstring).
    ref_scale_signed = ref_scale_abs = None
    if args.norm_reference_model:
        ref_model_name = f"{args.norm_reference_model}_ig_reference_aggregate_{args.stat}"
        ref_row = next((r for r in rows if r[0] == ref_model_name), None)
        if ref_row is None:
            raise SystemExit(f"--norm-reference-model {args.norm_reference_model!r}: no matching row found among {[r[0] for r in rows]!r}")
        _, ref_primary, ref_secondary = ref_row
        if args.combine_kinds:
            print(f"Computing --norm-reference-model scale from {ref_primary.name} (signed)...")
            ref_scale_signed = compute_scale(
                resample_saliency(ref_primary, template_img), brain_mask_data, "signed",
                args.norm_method, args.norm_percentile, args.norm_zclip,
            )
            print(f"Computing --norm-reference-model scale from {ref_secondary.name} (abs)...")
            ref_scale_abs = compute_scale(
                resample_saliency(ref_secondary, template_img), brain_mask_data, "abs",
                args.norm_method, args.norm_percentile, args.norm_zclip,
            )
        else:
            print(f"Computing --norm-reference-model scale from {ref_primary.name} ({args.kind})...")
            scale = compute_scale(
                resample_saliency(ref_primary, template_img), brain_mask_data, args.kind,
                args.norm_method, args.norm_percentile, args.norm_zclip,
            )
            if args.kind == "signed":
                ref_scale_signed = scale
            else:
                ref_scale_abs = scale

    n_montage = len(montage_z_indices) if montage_z_indices else 0
    block_width = max(3, n_montage) if args.montage else 3
    rows_per_group = 2 if args.montage else 1
    n_cols = block_width * 2 if args.combine_kinds else block_width
    col_width = 1.8 if args.montage else 3.0
    fig, axes = build_figure_axes(len(rows), rows_per_group, n_cols, col_width, args.montage, template_data, z_c)

    def turn_off_unused(axes_row, col_offset, used_width):
        for c in range(col_offset + used_width, col_offset + block_width):
            axes_row[c].axis("off")

    def block_axes(ortho_axes_row, montage_axes_row, col_offset):
        axes_list = list(ortho_axes_row[col_offset:col_offset + block_width])
        if montage_axes_row is not None:
            axes_list += list(montage_axes_row[col_offset:col_offset + block_width])
        return axes_list

    im_signed = im_abs = im = None
    for row, (model_name, primary_file, secondary_file) in enumerate(rows):
        ortho_row_idx = row * rows_per_group
        ortho_axes_row = axes[ortho_row_idx]
        montage_axes_row = axes[ortho_row_idx + 1] if args.montage else None

        if args.combine_kinds:
            im_signed = draw_row(
                ortho_axes_row, 0, "signed", primary_file, template_img, template_data, brain_mask_data, z_c, y_c, x_c,
                args, col_title_prefix="signed ", montage_z_indices=montage_z_indices, montage_axes_row=montage_axes_row,
                ref_scale=ref_scale_signed,
            )
            im_abs = draw_row(
                ortho_axes_row, block_width, "abs", secondary_file, template_img, template_data, brain_mask_data, z_c, y_c, x_c,
                args, col_title_prefix="abs ", montage_z_indices=montage_z_indices, montage_axes_row=montage_axes_row,
                ref_scale=ref_scale_abs,
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
            ref_scale = ref_scale_signed if args.kind == "signed" else ref_scale_abs
            im = draw_row(
                ortho_axes_row, 0, args.kind, primary_file, template_img, template_data, brain_mask_data, z_c, y_c, x_c,
                args, montage_z_indices=montage_z_indices, montage_axes_row=montage_axes_row, ref_scale=ref_scale,
            )
            turn_off_unused(ortho_axes_row, 0, 3)
            if montage_axes_row is not None:
                turn_off_unused(montage_axes_row, 0, n_montage)
            if not args.norm:
                fig.colorbar(im, ax=block_axes(ortho_axes_row, montage_axes_row, 0), shrink=0.7, label=f"{args.kind} saliency")

        label = base_model_name(model_name, args.stat)
        if args.montage:
            # ONE header for this model's whole 2-row group (ortho row +
            # montage row), not the same text repeated on both rows --
            # positioned at the vertical midpoint between the two rows'
            # own axes (figure-fraction coordinates, read AFTER both are
            # already placed by the gridspec, so this reflects their real,
            # final position) rather than attached to either row's own
            # ylabel.
            top_pos = ortho_axes_row[0].get_position()
            bottom_pos = montage_axes_row[0].get_position()
            mid_y = (top_pos.y0 + bottom_pos.y1) / 2
            fig.text(top_pos.x0 - 0.01, mid_y, label, fontsize=11, fontweight="bold", ha="right", va="center")
        else:
            axes[ortho_row_idx, 0].set_ylabel(
                label, fontsize=11, fontweight="bold", rotation=0, ha="right", va="center", labelpad=10
            )

    if args.norm:
        norm_kind_label = (
            f"reference-normalized ({args.norm_reference_model})" if args.norm_reference_model
            else f"normalized ({args.norm_method})"
        )
        if args.combine_kinds:
            fig.colorbar(
                im_signed, ax=axes[:, 0:block_width].ravel().tolist(), shrink=0.5,
                label=f"{norm_kind_label} signed saliency (a.u.)",
            )
            fig.colorbar(
                im_abs, ax=axes[:, block_width:2 * block_width].ravel().tolist(), shrink=0.5,
                label=f"{norm_kind_label} abs saliency (a.u.)",
            )
        else:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label=f"{norm_kind_label} {args.kind} saliency (a.u.)")

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold")

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
