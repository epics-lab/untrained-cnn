"""
Averages optim_uncnn's 3 per-channel saliency maps (raw, rank, sobel -- see
UnCNNDataset in train_linear.py) into one synthetic "avg" map per (attribution
tag, kind[, stat]), so it can appear as a SINGLE row in a comparison figure
instead of 3 separate raw/rank/sobel rows.

CAVEAT (read before using this for anything beyond a compact overview
figure): raw/rank/sobel are heterogeneous units -- raw intensity, a
rank-filtered value, and a Sobel gradient magnitude, respectively (see
docs/known_limitations.md's "optim_uncnn: 3 separate per-channel maps"
section) -- averaging across them mixes those units the same way averaging
degrees Celsius with a percentile rank would. This script exists because
comparison figures can get crowded with 3 separate optim_uncnn rows next to
single-row models; it trades that crowding for a channel-mixing caveat on
the resulting row. The per-channel figures are unaffected and remain the
source of truth for optim_uncnn specifically.

All 3 channels share one voxel grid (they're 3 channels of the SAME
preprocessed volume, see UnCNNDataset.__getitem__), so this is a plain
per-voxel mean, no resampling needed -- a grid mismatch between channels
would indicate a real bug upstream, so it's treated as fatal, not skipped.

--input-template/--output-template: this script started out hardcoded to
visualize_aggregate.sh's across-subject aggregate filenames
("optim_uncnn_{channel}_{tag_prefix}aggregate_{stat}_{kind}_saliency.nii.gz"
-- still the default below, unchanged). compute_sanity_checks.sh's
reference-subject sanity-check maps (arch_optim_uncnn_all_{channel}_*,
data_optim_uncnn_{channel}_*, input_optim_uncnn_permute_{channel}_*) need the
exact same per-voxel averaging but have no "stat" component in their
filenames -- rather than duplicate average_one_combo's load/verify/mean/save
logic in a second script, both templates are configurable ({channel},
{tag_prefix}, {kind}, {stat}, {output_model_name} placeholders; a template
that omits {stat} is fine -- pass a single dummy --stats value so the stat
loop below still runs exactly once).

Usage:
    python average_uncnn_channels.py --aggregate-dir saliency_outputs/aggregate
    # compute_sanity_checks.sh's arch_/data_/input_ sanity-check maps (no stat in the name):
    python average_uncnn_channels.py --aggregate-dir saliency_outputs --stats x \\
        --input-template "arch_optim_uncnn_all_{channel}_{tag_prefix}{kind}_saliency.nii.gz" \\
        --output-template "arch_optim_uncnn_all_avg_{tag_prefix}{kind}_saliency.nii.gz"
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_CHANNELS = ("raw", "rank", "sobel")
DEFAULT_STATS = ("mean", "std")
DEFAULT_KINDS = ("signed", "abs")
DEFAULT_TAG_PREFIXES = ("", "gxi_", "ig_zero_", "ig_reference_")
DEFAULT_OUTPUT_MODEL_NAME = "optim_uncnn_avg"
DEFAULT_INPUT_TEMPLATE = "optim_uncnn_{channel}_{tag_prefix}aggregate_{stat}_{kind}_saliency.nii.gz"
DEFAULT_OUTPUT_TEMPLATE = "{output_model_name}_{tag_prefix}aggregate_{stat}_{kind}_saliency.nii.gz"


def average_one_combo(base_dir, channels, tag_prefix, stat, kind, output_model_name, input_template, output_template):
    paths = [
        base_dir / input_template.format(channel=channel, tag_prefix=tag_prefix, stat=stat, kind=kind, output_model_name=output_model_name)
        for channel in channels
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"  skipping {tag_prefix or '(gradient)'} {stat} {kind}: missing {[p.name for p in missing]}")
        return None

    imgs = [nib.load(p) for p in paths]
    ref_shape, ref_affine = imgs[0].shape, imgs[0].affine
    for img, path in zip(imgs, paths):
        if img.shape != ref_shape or not np.allclose(img.affine, ref_affine, atol=1e-3):
            raise RuntimeError(
                f"Grid mismatch between optim_uncnn channels for {tag_prefix or '(gradient)'} {stat} {kind}: "
                f"{path.name} has shape {img.shape} vs. reference {ref_shape}. This should never happen -- all 3 "
                f"channels come from the same preprocessed volume (see UnCNNDataset) -- investigate rather than "
                f"skip."
            )

    avg = np.mean(np.stack([img.get_fdata(dtype=np.float32) for img in imgs], axis=0), axis=0)
    out_path = base_dir / output_template.format(tag_prefix=tag_prefix, stat=stat, kind=kind, output_model_name=output_model_name)
    nib.save(nib.Nifti1Image(avg.astype(np.float32), ref_affine), out_path)
    print(f"  wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--aggregate-dir", type=Path, required=True, help="Directory holding both the input channel files and where the averaged output is written.")
    parser.add_argument("--channels", nargs="*", default=list(DEFAULT_CHANNELS))
    parser.add_argument(
        "--stats", nargs="*", default=list(DEFAULT_STATS),
        help="Values substituted for {stat} in --input-template/--output-template. Pass a single dummy value "
        "(e.g. `--stats x`) when using a template that has no {stat} placeholder, so this loop runs once.",
    )
    parser.add_argument("--kinds", nargs="*", default=list(DEFAULT_KINDS))
    parser.add_argument(
        "--tag-prefixes", nargs="*", default=list(DEFAULT_TAG_PREFIXES),
        help="Filename tag prefixes to average per attribution method (default: gradient/gxi/ig_zero/ig_reference "
        "-- '' means the untagged default method, gradient). Pass '' explicitly if overriding this to include it.",
    )
    parser.add_argument("--output-model-name", default=DEFAULT_OUTPUT_MODEL_NAME)
    parser.add_argument(
        "--input-template", default=DEFAULT_INPUT_TEMPLATE,
        help="Per-channel input filename, as a str.format() template with {channel}/{tag_prefix}/{stat}/{kind}/"
        "{output_model_name} placeholders (default: the across-subject aggregate filename shape).",
    )
    parser.add_argument(
        "--output-template", default=DEFAULT_OUTPUT_TEMPLATE,
        help="Averaged output filename, same placeholders as --input-template minus {channel} (there is no single "
        "channel for the averaged file).",
    )
    args = parser.parse_args()

    n_written = 0
    for tag_prefix in args.tag_prefixes:
        for stat in args.stats:
            for kind in args.kinds:
                if average_one_combo(
                    args.aggregate_dir, args.channels, tag_prefix, stat, kind, args.output_model_name,
                    args.input_template, args.output_template,
                ) is not None:
                    n_written += 1
    print(f"Wrote {n_written} averaged optim_uncnn map(s) to {args.aggregate_dir}.")


if __name__ == "__main__":
    main()
