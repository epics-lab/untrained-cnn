"""
Cross-model, cross-target Adebayo et al. (2018) "Sanity Checks for Saliency
Maps" comparison: every model (rows) x every target's 3 checks -- arch_
(--randomize-model), data_ (--randomize-data), input_ (--randomize-input) --
(columns, grouped by target) in ONE figure, ONE global vmax/threshold/
colorbar shared across the whole grid, so color intensity is directly
comparable everywhere (same convention as
compare_aggregate_arch_saliency.py's compare_and_plot_all_targets_
aggregate_shared_scale).

Showing all 3 checks x all 3 targets per model, each with the usual ortho
(3-view) + axial-montage layout src/visualize_paper_nilearn.py's own
--sanity-check mode draws, would make a figure `n_models` rows x `3 targets
* 3 checks * (3 ortho + 4 montage)` columns wide -- far too large to be
legible. Instead each (target, check) cell here shows only ONE anatomical
view (--view axial/coronal/sagittal, single nilearn plot_stat_map cut at
the shared brain-mask centroid), keeping the grid to a manageable
`n_models` rows x `n_targets * 3` columns.

Reuses this repo's existing plotting/file-lookup machinery rather than
reimplementing it:
  - sanity_prefix()/masked_stat_img()/add_shared_colorbar() from
    visualize_paper_nilearn.py -- the exact same arch_/data_/input_
    filename-prefix lookup and template-resample+normalize+mask logic that
    script's own --sanity-check mode already uses in production (see
    saliency_affine/<target>/aggregate/paper_sanity_check_nilearn_
    {arch,data,input}_abs.png). Sanity-check files are single-reference-
    subject maps written directly under saliency_affine/<target>/ (slurm/
    compute_sanity_checks.sh), NOT under <target>/aggregate/ -- unlike the
    ig_popmean aggregate files visualize_paper_nilearn.py's non-sanity-check
    mode reads.
  - load_canonical_template()/load_brain_mask()/brain_centroid_indices()/
    voxel_to_mm() from visualize_saliency.py.
  - DEFAULT_MODELS from visualize_paper.py.

A (model, target, check) cell whose file doesn't exist on disk (e.g. fs_roi
has no arch_/data_/input_ sanity-check maps at all; optim_uncnn_avg needs
average_uncnn_channels.py to have been run first) is left blank (WARN
printed), never a hard failure -- same graceful-degradation convention as
every other multi-model figure in this repo.

Usage (CLI):
    python visualize_sanity_check.py --view axial
    python visualize_sanity_check.py --view coronal --kind signed --models brainiac swinbrain simclr
    python visualize_sanity_check.py --view sagittal --targets age sex --checks arch data
    python visualize_sanity_check.py --view axial --no-norm

Usage (notebook/import):
    from visualize_sanity_check import compare_sanity_checks_all_targets
    compare_sanity_checks_all_targets(view="axial", kind="abs")
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting

from visualize_paper import DEFAULT_MODELS
from visualize_paper_nilearn import add_shared_colorbar, masked_stat_img, sanity_prefix
from visualize_saliency import (
    DEFAULT_BRAIN_MASK,
    DEFAULT_TEMPLATE,
    brain_centroid_indices,
    load_brain_mask,
    load_canonical_template,
    voxel_to_mm,
)

DEFAULT_SALIENCY_AFFINE_ROOT = Path(__file__).resolve().parents[1] / "saliency_affine"
DEFAULT_TARGETS = ("age", "sex", "bmi")
DEFAULT_CHECKS = ("arch", "data", "input")

# Column sub-header per check -- matches visualize_saliency.py's own
# _DISPLAY_PREFIX_MAP relabeling (arch_ -> random_weights, data_ ->
# shuffled_labels, input_ -> shuffled_voxels) so labels read the same way
# across every figure in this repo.
_CHECK_LABELS = {"arch": "random_weights", "data": "shuffled_labels", "input": "shuffled_voxels"}

# nilearn's plot_stat_map display_mode letters: x=sagittal, y=coronal, z=axial.
_VIEW_TO_DISPLAY_MODE = {"axial": "z", "coronal": "y", "sagittal": "x"}
# Index into voxel_to_mm's (x, y, z) mm tuple for each view's single cut.
_VIEW_TO_MM_INDEX = {"axial": 2, "coronal": 1, "sagittal": 0}


def centroid_mm(template_img, brain_mask_data):
    """
    (x_mm, y_mm, z_mm) at the brain mask's own centroid -- the one shared
    slice level used for every cell in the grid (so a given --view means
    the exact same anatomical level for every model/target/check).
    """
    x_c, y_c, z_c = brain_centroid_indices(brain_mask_data)[::-1]
    return voxel_to_mm(template_img, x_c, y_c, z_c)


def gather_sanity_check_imgs(
    targets, models, checks, kind, saliency_affine_root, template_img, brain_mask_data,
    norm_method="percentile", norm_percentile=99.0, norm_zclip=3.0, norm=True, arch_variant="step01_head",
    dataset=None,
):
    """
    {(target, model, check): nib.Nifti1Image}, one entry per (target, model,
    check) combination whose file exists on disk -- missing combinations are
    simply absent from the returned dict (see main()'s WARN for what's
    missing), rather than raising. Each image is already resampled onto
    template_img's grid and masked to in-brain voxels (see masked_stat_img);
    also rescaled onto the shared [-1,1]/[0,1] range UNLESS `norm=False`, in
    which case each map keeps its own raw values -- see
    compare_sanity_checks_all_targets's own docstring for why that's not
    the recommended default (raw scales aren't comparable across models).

    `arch_variant`: see sanity_prefix()'s own docstring (visualize_paper_
    nilearn.py) -- default "step01_head", the only arch variant any
    current script can reproduce. Ignored for check in ("data", "input").

    `dataset`: None (default) reads saliency_affine_root/<target>/, matching
    slurm/compute_sanity_checks.sh's own NKI-only output location. Set to
    "nki"/"ppmi"/"hbn" to instead read saliency_affine_root/<target>/
    <dataset>/ -- the pretrained-heads tree's layout (caller is responsible
    for pointing saliency_affine_root at saliency_affine_pretrained/ in
    that case, e.g. via --saliency-affine-root).
    """
    imgs = {}
    missing = []
    for target in targets:
        saliency_dir = saliency_affine_root / target / dataset if dataset else saliency_affine_root / target
        for model in models:
            for check in checks:
                f = saliency_dir / f"{sanity_prefix(model, check, arch_variant)}ig_reference_{kind}_saliency.nii.gz"
                if f.exists():
                    imgs[(target, model, check)] = masked_stat_img(
                        f, template_img, brain_mask_data, kind, norm, norm_method, norm_percentile, norm_zclip,
                    )
                else:
                    missing.append((target, model, check))
    if missing:
        print(f"WARN: {len(missing)} missing (target, model, check) sanity-check file(s), skipped: {missing}")
    return imgs


def compare_sanity_checks_all_targets(
    targets=DEFAULT_TARGETS, models=DEFAULT_MODELS, checks=DEFAULT_CHECKS, view="axial", kind="abs",
    saliency_affine_root=None, threshold_percentile=75.0, title=None, output=None,
    template_path=DEFAULT_TEMPLATE, brain_mask_path=DEFAULT_BRAIN_MASK, no_brain_mask=False, label_scale=1.0,
    show_title=True, show_colorbar=True, norm_method="percentile", norm_percentile=99.0, norm_zclip=3.0,
    norm=True, arch_variant="step01_head", dataset=None,
):
    """
    One figure: `models` as rows, `targets` x `checks` (grouped by target) as
    columns, each cell a single --view slice at the shared brain-centroid
    level, ONE global vmax/threshold/colorbar for the whole figure. Returns
    the matplotlib Figure.

    `norm`: True (default) rescales each map independently onto the shared
    [-1,1]/[0,1] range (norm_method/norm_percentile/norm_zclip control how)
    BEFORE pooling everything into the one global vmax/threshold above --
    same reasoning as compare_aggregate_arch_saliency.py's own module
    docstring: different models/backbones have wildly different raw
    gradient scales, so without this the global vmax would just measure
    "which map had bigger numbers." Pass norm=False (--no-norm on the CLI)
    to skip it and use each map's raw values instead -- an explicit
    opt-out for exploratory/debugging use, not the recommended way to
    compare models.
    """
    saliency_affine_root = Path(saliency_affine_root) if saliency_affine_root else DEFAULT_SALIENCY_AFFINE_ROOT

    template_img = load_canonical_template(template_path)
    brain_mask_data = None if no_brain_mask else load_brain_mask(brain_mask_path, template_img)
    # centroid_mm needs the REAL brain extent regardless of no_brain_mask
    # (which only controls whether the overlay itself is masked) -- same
    # reasoning as visualize_paper_nilearn.py's montage_extent_mask.
    centroid_mask = brain_mask_data if brain_mask_data is not None else load_brain_mask(brain_mask_path, template_img)

    imgs = gather_sanity_check_imgs(
        targets, models, checks, kind, saliency_affine_root, template_img, brain_mask_data,
        norm_method, norm_percentile, norm_zclip, norm=norm, arch_variant=arch_variant, dataset=dataset,
    )
    if not imgs:
        raise SystemExit(
            f"No sanity-check files found under {saliency_affine_root}/<target>/ for "
            f"models={list(models)!r} checks={list(checks)!r}"
        )

    all_vals = np.concatenate([np.abs(img.get_fdata()).ravel() for img in imgs.values()])
    all_vals = all_vals[all_vals > 0]
    global_vmax = max(float(np.percentile(all_vals, 99.5)), 1e-6)/4 if all_vals.size else 1.0 #0.01 for models
    # threshold_percentile <= 0 ("disable thresholding") floors at 1e-9, not 1e-6 -- matches
    # compare_aggregate_arch_saliency.py's own _NO_THRESHOLD_FLOOR constant, its dedicated "really
    # show everything" floor (used there for disable_threshold) as opposed to that script's
    # separate percentile_or_floor() helper's plain 1e-6 floor for an ordinary (>0) percentile
    # that happens to come out non-positive -- a different, unrelated case.
    global_threshold = (
        max(float(np.percentile(all_vals, threshold_percentile)), 1e-6)
        if all_vals.size and threshold_percentile > 0 else 1e-9
    )
    print(
        f"compare_sanity_checks_all_targets: ONE global vmax={global_vmax:.6f} (99.5th pctile) and "
        f"threshold={global_threshold:.6f} ({threshold_percentile:.1f}th pctile), pooled across every "
        f"(target, model, check) cell -- applied to the whole figure, not per row/column."
    )

    display_mode = _VIEW_TO_DISPLAY_MODE[view]
    cut_mm = centroid_mm(template_img, centroid_mask)[_VIEW_TO_MM_INDEX[view]]
    cmap = plotting.cm.cold_hot if kind == "signed" else "YlOrRd"
    vmin = None if kind == "signed" else 0.0

    models = list(models)
    targets = list(targets)
    checks = list(checks)
    columns = [(t, c) for t in targets for c in checks]
    n_rows, n_cols = len(models), len(columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows), facecolor="black", squeeze=False)
    # Pull the grid's own bottom margin in close to the last row (matplotlib's
    # default ~0.11 leaves a wide dead strip below the bottom row that
    # bbox_inches="tight" can't trim, since the colorbar axes added below is
    # itself a real artist sitting inside that strip, not empty margin) -- the
    # colorbar's own y-position (below) is then anchored just under this
    # tightened bottom edge instead of a fixed, layout-independent fraction.
    fig.subplots_adjust(bottom=0.04, top=0.92, hspace=0.15, wspace=0.05)

    for row, model in enumerate(models):
        for col, (target, check) in enumerate(columns):
            ax = axes[row, col]
            img = imgs.get((target, model, check))
            if img is None:
                ax.axis("off")
                continue
            plotting.plot_stat_map(
                img, bg_img=template_img, display_mode=display_mode, cut_coords=[cut_mm], cmap=cmap,
                threshold=global_threshold, vmin=vmin, vmax=global_vmax, symmetric_cbar=(kind == "signed"),
                colorbar=False, black_bg=True, annotate=False, draw_cross=False, axes=ax,
            )
        pos = axes[row, 0].get_position()
        fig.text(
            0.005, (pos.y0 + pos.y1) / 2, model, fontsize=11 * label_scale, fontweight="bold", ha="left",
            va="center", color="white",
        )

    # Two-level column headers: each check's own label directly above its
    # cell (set_title), and the target name spanning that check's whole
    # `len(checks)`-wide block above that.
    #
    # The target super-label's vertical gap above the axes top is expressed in INCHES (via
    # fig.get_figheight(), not a fixed figure-FRACTION like the "+ 0.05" this replaced -- the
    # check sub-titles below it (set_title) sit at a fixed point-based padding above the axes,
    # independent of figure height, so a fixed figure-fraction gap shrinks in absolute terms as
    # figsize=(2.0*n_cols, 2.2*n_rows) gets shorter with fewer models (n_rows) -- e.g. a single
    # model (n_rows=1) is 2.2in tall, giving old_gap=0.05*2.2=0.11in, not enough to clear the
    # sub-title text below it, unlike a 5-model figure (11in tall, 0.55in gap). Scaled by
    # label_scale too, since a larger label_scale needs proportionally more clearance for its
    # bigger fonts.
    super_label_gap_frac = (0.4 * label_scale) / fig.get_figheight()
    for t_idx, target in enumerate(targets):
        col_start, col_end = t_idx * len(checks), (t_idx + 1) * len(checks) - 1
        x0 = axes[0, col_start].get_position().x0
        x1 = axes[0, col_end].get_position().x1
        top = axes[0, col_start].get_position().y1
        fig.text(
            (x0 + x1) / 2, top + super_label_gap_frac, target.upper(), fontsize=13 * label_scale,
            fontweight="bold", ha="center", color="white",
        )
        for c_idx, check in enumerate(checks):
            axes[0, t_idx * len(checks) + c_idx].set_title(_CHECK_LABELS[check], fontsize=9 * label_scale, color="white")

    if show_colorbar:
        # Anchored just below the (already-tightened, see subplots_adjust
        # above) bottom row's own lowest edge, not a fixed figure-fraction y --
        # so the gap stays small regardless of n_rows/figsize.
        bottom_row_y0 = min(axes[-1, c].get_position().y0 for c in range(n_cols))
        cax_height = 0.015
        cax_y = max(bottom_row_y0 - 0.045, 0.005)
        cax = fig.add_axes([0.3, cax_y, 0.4, cax_height])
        add_shared_colorbar(fig, cax, cmap, kind, global_vmax)

    if show_title:
        norm_label = f"{norm_method}-normalized" if norm else "un-normalized (raw)"
        fig.suptitle(
            title or (
                f"Adebayo sanity checks -- {', '.join(t.upper() for t in targets)} "
                f"({kind}, {view}, {norm_label}, shared scale)"
            ),
            fontsize=14, fontweight="bold", color="white", y=1.02,
        )

    if output is not None:
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"Saved: {output}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--no-brain-mask", action="store_true")
    parser.add_argument(
        "--saliency-affine-root", type=Path, default=None,
        help="Default: <repo>/saliency_affine (each target's sanity-check files are read from "
        "<saliency-affine-root>/<target>/, matching slurm/compute_sanity_checks.sh's own output location). "
        "Pass saliency_affine_pretrained/ together with --dataset for the pretrained-heads tree.",
    )
    parser.add_argument(
        "--dataset", choices=["nki", "ppmi", "hbn", "nki_hbn"], default=None,  # nki_hbn: slurm/visualize_combined_datasets.sh's pooled tree
        help="Set together with --saliency-affine-root saliency_affine_pretrained/ to read "
        "<saliency-affine-root>/<target>/<dataset>/ instead of <saliency-affine-root>/<target>/. "
        "Unset (default): this script's original NKI-only layout.",
    )
    parser.add_argument("--targets", nargs="*", choices=list(DEFAULT_TARGETS), default=list(DEFAULT_TARGETS))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--checks", nargs="*", choices=list(DEFAULT_CHECKS), default=list(DEFAULT_CHECKS))
    parser.add_argument(
        "--arch-variant", default="step01_head",
        help="--checks arch only: see sanity_prefix()'s own docstring (visualize_paper_nilearn.py) -- default "
        "'step01_head' (the linear-head-only cascade run, the only one any current script can reproduce).",
    )
    parser.add_argument(
        "--view", choices=["axial", "coronal", "sagittal"], default="axial",
        help="Single anatomical view shown for every cell (keeps the grid to n_models x n_targets*3 columns "
        "instead of needing a full ortho+montage block per cell).",
    )
    parser.add_argument("--kind", choices=["signed", "abs"], default="abs")
    parser.add_argument(
        "--threshold-percentile", type=float, default=75.0,
        help="Hide voxels below this in-brain |value| percentile, pooled across the whole figure (0 disables thresholding).",
    )
    parser.add_argument("--norm", action="store_true", default=True)
    parser.add_argument(
        "--no-norm", dest="norm", action="store_false",
        help="Use each map's own raw values instead of the shared [-1,1]/[0,1] rescaling (default: on -- "
        "see compare_sanity_checks_all_targets's own docstring for why rescaling first is recommended).",
    )
    parser.add_argument("--norm-method", choices=["percentile", "zscore"], default="percentile")
    parser.add_argument("--norm-percentile", type=float, default=99.0)
    parser.add_argument("--norm-zclip", type=float, default=3.0)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-title", dest="show_title", action="store_false")
    parser.add_argument("--no-colorbar", dest="show_colorbar", action="store_false")
    parser.add_argument("--label-scale", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Default: sanity_check_all_models_all_targets_<view>_<kind>.png",
    )
    args = parser.parse_args()
    norm_suffix = "" if args.norm else "_raw"
    dataset_suffix = f"_{args.dataset}" if args.dataset else ""
    output = args.output or Path(f"sanity_check_all_models_all_targets_{args.view}_{args.kind}{dataset_suffix}{norm_suffix}.png")

    compare_sanity_checks_all_targets(
        targets=args.targets, models=args.models, checks=args.checks, view=args.view, kind=args.kind,
        saliency_affine_root=args.saliency_affine_root, threshold_percentile=args.threshold_percentile,
        title=args.title, output=output, template_path=args.template, brain_mask_path=args.brain_mask,
        no_brain_mask=args.no_brain_mask, label_scale=args.label_scale, show_title=args.show_title,
        show_colorbar=args.show_colorbar, norm_method=args.norm_method, norm_percentile=args.norm_percentile,
        norm_zclip=args.norm_zclip, norm=args.norm, arch_variant=args.arch_variant, dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
