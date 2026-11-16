"""
Paints a region-name -> scalar-value mapping (e.g. fs_roi saliency, one
value per ROI) onto a real FreeSurfer segmentation volume, via
FreeSurferColorLUT.txt label matching -- so a fundamentally non-spatial
per-ROI value (there is no per-voxel gradient here, only per-region) becomes
a genuine .nii.gz on a genuine anatomical grid, and can flow through the
existing NIfTI pipeline (aggregate_saliency.py, visualize_paper.py,
visualize_paper_nilearn.py) completely unmodified.

Ported from an earlier prototype's
load_lut/build_importance_volume (there specialized to one particular RF
permutation-importance CSV; generalized here to any region->value dict so
calculate_gradcam_tabular.py can reuse it for both signed and abs saliency).

The default segmentation template (--seg-template) is one specific
subject's real aparc+aseg.mgz recon-all output, not a population-averaged
atlas -- same approach the prototype above already took (one subject's
aparc.DKTatlas+aseg.deep.mgz as the painting canvas). Since this repo's
feature table uses aparc.DKTatlas (see build_freesurfer_roi_tables.py) while
aparc+aseg.mgz's own cortical labels come from the standard aparc
(Desikan-Killiany, not the "DKT" variant), a handful of DKT regions have no
LUT/segmentation match at all (DKT drops banks-superior-temporal-sulcus and
merges frontal-/temporal-pole into neighboring regions) -- expected, and
reported via the `unmatched` return value, not silently dropped.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def load_lut(lut_path):
    """FreeSurferColorLUT.txt -> DataFrame with columns Index, Label, R, G, B, A."""
    return pd.read_csv(
        lut_path,
        comment="#",
        sep=r"\s+",
        header=None,
        names=["Index", "Label", "R", "G", "B", "A"],
        engine="python",
    )


def paint_region_values(seg_data, lut_df, region_to_value):
    """
    seg_data: integer-labeled segmentation array (e.g. aparc+aseg.mgz's own
    data, one FreeSurferColorLUT index per voxel).
    lut_df: load_lut()'s output.
    region_to_value: {region_name (LUT Label, e.g. "ctx-lh-precentral" or
    "Left-Hippocampus"): scalar value}.

    Returns (volume (float32, same shape as seg_data, 0 outside any matched
    region), unmatched (sorted list of region_to_value keys with no LUT
    match) -- every voxel belonging to a given LUT index gets that region's
    single value (flat within the ROI, not smooth per-voxel gradient --
    genuinely the right granularity here, not an approximation of one).
    """
    volume = np.zeros(seg_data.shape, dtype=np.float32)
    matched = set()
    for row in lut_df.itertuples(index=False):
        value = region_to_value.get(row.Label)
        if value is None:
            continue
        volume[seg_data == row.Index] = value
        matched.add(row.Label)

    unmatched = sorted(set(region_to_value) - matched)
    return volume, unmatched


def load_segmentation(seg_path):
    """nib.load handles .mgz/.mgh/.nii(.gz) transparently -- returns the loaded image (not just the array), since callers need its own affine/header too."""
    return nib.load(str(seg_path))


def region_mean_volume(seg_data, data_array, background_label=0):
    """
    Inverse of paint_region_values: instead of painting a GIVEN per-region
    value onto voxels, COMPUTES the per-region value from a dense array
    (e.g. a smooth per-voxel saliency map, already resampled onto
    seg_data's own grid) and paints that mean back onto every voxel of the
    region -- turning a smooth map into a piecewise-constant per-ROI map on
    the same labeling scheme as seg_data.

    seg_data, data_array: both integer-labeled-segmentation-shaped arrays,
    already sharing one axis order (this function does no reorientation --
    same convention as paint_region_values, whose seg_data is likewise
    assumed pre-aligned with whatever it's compared against; getting two
    arrays into the same axis order is the caller's job, not this
    function's -- see regionalize_saliency.py for why that's easy to get
    wrong silently).

    Returns (volume (float32, same shape, 0 at background_label),
    label_means ({int label: float mean})).
    """
    assert seg_data.shape == data_array.shape, (
        f"seg_data and data_array must already share one shape/axis order, got {seg_data.shape} vs {data_array.shape}"
    )
    volume = np.zeros(seg_data.shape, dtype=np.float32)
    label_means = {}
    for label in np.unique(seg_data):
        if label == background_label:
            continue
        mask = seg_data == label
        mean_value = float(data_array[mask].mean())
        volume[mask] = mean_value
        label_means[int(label)] = mean_value
    return volume, label_means
