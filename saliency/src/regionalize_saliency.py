"""
Region-averages each image-based model's aggregate saliency map onto the
same FreeSurfer DKT segmentation fs_roi's own saliency is painted onto --
so all models can be compared at IDENTICAL (ROI, not voxel) granularity in
a companion figure to visualize_paper_nilearn.py's usual voxel-level one.

fs_roi's saliency is fundamentally not per-voxel: calculate_gradcam_
tabular.py paints one value per FreeSurfer ROI onto a segmentation volume
(see roi_to_volume.py's paint_region_values). Every other model's aggregate
map, by contrast, is smooth per-voxel data on that model's own native
preprocessing grid (see docs/known_limitations.md's "Resampling pipeline"
section). Comparing them directly (as visualize_paper_nilearn.py's usual
figure does) compares apples (fine detail) to oranges (one flat value per
region) -- this script produces a second set of aggregate files where every
model is reduced to the SAME per-region granularity fs_roi is native to:
resample each model's map onto the segmentation's grid, then replace every
voxel with the MEAN of that model's values within its region.

fs_roi's own aggregate file needs none of this (it's already piecewise-
constant per region, on this exact grid, by construction) -- it's copied
through into --output-dir unchanged rather than reprocessed, so the new
directory is a complete, directly comparable set including fs_roi itself.

AXIS-ORDER PITFALL (read before touching this file): resample_saliency
(visualize_saliency.py) returns data in AXIAL-FIRST (Z, Y, X) array order,
not the source file's/template's own RAS array order. The segmentation
must be loaded and reoriented THE SAME WAY (load_canonical_template +
to_axial_first) before its labels are used to index/mask a resampled
saliency array -- NOT via roi_to_volume.load_segmentation, which applies no
reorientation at all and would silently misalign regions against the wrong
voxels (this exact class of bug -- mismatched axis/space conventions
producing wrong-but-plausible-looking output -- is the single most common
failure mode elsewhere in this project; see docs/known_limitations.md).
After region_mean_volume runs (both inputs correctly axial-first), the
result is transposed back to RAS (np.transpose(data, (2, 1, 0))) before
being saved -- the exact pattern visualize_paper_nilearn.py's own
masked_stat_img already uses for the same reason.

Output: {output_dir}/{model}_ig_reference_aggregate_{stat}_{kind}_saliency.
nii.gz, byte-for-byte the same NAMING CONVENTION visualize_paper_nilearn.py
already expects (collect_kind_files/model_name_from_filename) -- so that
script can render a "region-averaged" comparison figure with ZERO code
changes, just `--saliency-dir` pointed at --output-dir instead of the usual
saliency_outputs/aggregate/. Also writes one combined CSV,
{output_dir}/regionalized_saliency_means.csv (columns model,stat,kind,
region,mean_value), for sanity-checking against fs_roi's own per-region
values.

Usage:
    python regionalize_saliency.py
    python regionalize_saliency.py --models optim_uncnn_avg brainiac --stats mean --kinds abs
"""

import os
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from roi_to_volume import load_lut, region_mean_volume
from visualize_paper import DEFAULT_MODELS, default_aggregate_dir
from visualize_saliency import collect_kind_files, load_canonical_template, resample_saliency, to_axial_first

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEG_TEMPLATE = REPO_ROOT / "templates" / "aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz"
DEFAULT_LUT = Path(os.environ.get("FS_ROI_LUT", os.path.join(os.environ.get("FREESURFER_HOME", ""), "FreeSurferColorLUT.txt")))
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "saliency_outputs"


def regionalize_one_file(file_path, seg_img, seg_axial):
    """
    One (model, stat, kind) source file -> (nii.gz-ready RAS array,
    {region_name: mean_value}). seg_img is the segmentation loaded via
    load_canonical_template (its own affine/header are reused for the
    output image, since the output is now on the segmentation's grid, not
    the source file's own grid); seg_axial is to_axial_first(seg_img.
    get_fdata()) -- computed once by the caller and passed in, not
    recomputed per file.
    """
    data_axial = resample_saliency(file_path, seg_img)
    regionalized_axial, label_means = region_mean_volume(seg_axial, data_axial)
    regionalized_ras = np.transpose(regionalized_axial, (2, 1, 0))
    return regionalized_ras, label_means


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Default: saliency_outputs/<target>/aggregate, aggregate_saliency.py's own output dir (see --target).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: saliency_outputs/<target>/aggregate_roi (see --target).",
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default="age",
        help="Which target's aggregate/ subtree to read (default: age). Only used to resolve --saliency-dir's "
        "and --output-dir's defaults -- ignored if both are passed explicitly.",
    )
    parser.add_argument("--seg-template", type=Path, default=DEFAULT_SEG_TEMPLATE)
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--stats", nargs="*", default=["mean"])
    parser.add_argument("--kinds", nargs="*", default=["signed", "abs"], choices=["signed", "abs"])
    parser.add_argument(
        "--fs-roi-prefix", default="fs_roi",
        help="Model name copied through unchanged (no regionalization) rather than resampled+averaged -- its "
        "aggregate file is already piecewise-constant per region, on this exact grid, by construction.",
    )
    args = parser.parse_args()
    args.saliency_dir = args.saliency_dir or default_aggregate_dir(args.target)
    args.output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.target / "aggregate_roi")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lut_df = load_lut(args.lut)
    index_to_name = dict(zip(lut_df["Index"], lut_df["Label"]))

    seg_img = load_canonical_template(args.seg_template)
    seg_axial = to_axial_first(seg_img.get_fdata()).round().astype(np.int64)

    csv_rows = []
    models_to_regionalize = [m for m in args.models if m != args.fs_roi_prefix]

    for stat in args.stats:
        for kind in args.kinds:
            for model in models_to_regionalize:
                models_filter = [f"{model}_ig_reference_aggregate_{stat}"]
                files = collect_kind_files(args.saliency_dir, kind, models_filter, None)
                if not files:
                    print(f"WARN: no aggregate '{kind}' file found for {model} ({stat}) in {args.saliency_dir} -- skipping")
                    continue
                if len(files) > 1:
                    print(f"WARN: {len(files)} files matched {model} ({stat}, {kind}); using {files[0].name}")

                print(f"Regionalizing {files[0].name} onto {args.seg_template.name}...")
                regionalized_ras, label_means = regionalize_one_file(files[0], seg_img, seg_axial)

                out_path = args.output_dir / f"{model}_ig_reference_aggregate_{stat}_{kind}_saliency.nii.gz"
                nib.save(nib.Nifti1Image(regionalized_ras.astype(np.float32), seg_img.affine, seg_img.header), out_path)
                print(f"  -> saved {out_path}")

                for index, mean_value in label_means.items():
                    csv_rows.append(
                        {
                            "model": model, "stat": stat, "kind": kind,
                            "region": index_to_name.get(index, f"index_{index}"), "mean_value": mean_value,
                        }
                    )

            # fs_roi: copy through unchanged -- already on this exact grid,
            # already piecewise-constant per region (see module docstring).
            if args.fs_roi_prefix in args.models:
                fs_roi_filter = [f"{args.fs_roi_prefix}_ig_reference_aggregate_{stat}"]
                fs_roi_files = collect_kind_files(args.saliency_dir, kind, fs_roi_filter, None)
                if not fs_roi_files:
                    print(f"WARN: no aggregate '{kind}' file found for {args.fs_roi_prefix} ({stat}) -- skipping copy-through")
                else:
                    dest = args.output_dir / f"{args.fs_roi_prefix}_ig_reference_aggregate_{stat}_{kind}_saliency.nii.gz"
                    shutil.copy2(fs_roi_files[0], dest)
                    print(f"Copied {fs_roi_files[0].name} through unchanged -> {dest}")

    if csv_rows:
        import pandas as pd

        csv_path = args.output_dir / "regionalized_saliency_means.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
