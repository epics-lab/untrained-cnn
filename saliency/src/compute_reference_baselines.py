"""
Computes a population-mean reference (baseline) image per model, in that
model's own preprocessed input-tensor space -- after resize/crop/normalize,
exactly what the backbone actually sees, before any forward pass. Intended
as an Integrated Gradients baseline: per the literature (Distill's
"Visualizing the Impact of Feature Attribution Baselines", Erion et al.'s
"Expected Gradients"), a population/data-distribution-derived baseline is
better motivated than an arbitrary all-zero baseline for domains like this
one, where a z-scored/robust-scaled zero is just "this subject's own mean",
not anything population-meaningful. See docs/known_limitations.md.

For each model, every subject in --demographics is run through that model's
*actual* Dataset class from train_linear.py (UnCNNDataset, BrainIACDataset,
CAT12VBMDataset, SwinBrainDataset, TurboPrepDataset) -- the same
preprocessing the real backbone sees -- and averaged with a streaming sum
(accumulated in float64, not stack-then-mean), so memory stays at O(batch),
not O(cohort).

Backbones are never loaded (spec.model_factory is never called): computing
a mean *input* image only needs spec.dataset, so this runs on CPU with no
GPU, no pretrained checkpoints, and no AnatCL clone required.

Output: {output-dir}/{registration}_{model}_reference.npy (--registration
rigid|affine, default affine), one float32 array per model in that model's
own preprocessed (C, H, W, D) shape -- except optim_uncnn, which gets 3
separate single-channel files (raw/rank/sobel), one per channel, since those
channels are heterogeneous and shouldn't be pooled (same reasoning already
applied to optim_uncnn's saliency maps -- see docs/known_limitations.md).
anatcl's dataset/preprocessing doesn't depend on --anatcl-level (only the
encoder does, which we never load here), so anatcl_global and anatcl_local
share one reference image, {registration}_anatcl_reference.npy.

Usage:
    python compute_reference_baselines.py
    python compute_reference_baselines.py --models optim_uncnn anatcl
    python compute_reference_baselines.py --registration rigid
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_linear import (
    DEFAULT_DEMOGRAPHICS,
    DEFAULT_EXCLUDE_SUBJECTS,
    DEFAULT_NKI_BIDS_ROOT,
    DEFAULT_ROI_CSV,
    REPO_ROOT,
    make_feature_spec,
    parse_args as parse_args_train,
)
from calculate_gradcam import UNCNN_CHANNELS

DEFAULT_OUTPUT_DIR = REPO_ROOT / "reference_baselines"
DEFAULT_MODELS = ["optim_uncnn", "brainiac", "swinbrain", "simclr", "anatcl"]

# Same two env vars slurm/env.sh exports for every other script here --
# defined locally rather than imported from train_linear, which only ever
# exposes ITS OWN default (DEFAULT_TURBOPREP_ROOT, affine-only, read from
# NKI_AFFINE_TURBOPREP_ROOT) since train_linear.py has no rigid/affine switch of
# its own. --registration below picks between the two.
NKI_RIGID_TURBOPREP_ROOT = Path(os.environ.get("NKI_RIGID_TURBOPREP_ROOT", "/path/to/turboprep_NKI"))
NKI_AFFINE_TURBOPREP_ROOT = Path(os.environ.get("NKI_AFFINE_TURBOPREP_ROOT", "/path/to/NKI_hdbet_affine"))


def compute_mean_dataset_tensor(dataset, batch_size=4, num_workers=2, desc=""):
    """
    Streaming mean over every item in `dataset` (each shaped (C, H, W, D)),
    accumulated in float64 for numerical stability across a large cohort.
    Returns (mean_array (C, H, W, D) float32, n_subjects_used).
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    total = None
    n = 0
    for batch in loader:
        batch = batch.double()
        if total is None:
            total = torch.zeros(batch.shape[1:], dtype=torch.float64)
        total += batch.sum(dim=0)
        n += batch.shape[0]
        print(f"  {desc}: {n}/{len(dataset)}", end="\r")
    print()
    if n == 0:
        raise SystemExit(f"No subjects available for {desc!r} -- check paths/demographics.")
    return (total / n).float().numpy(), n


def build_spec(model, args):
    """Reconstructs a FeatureSpec via train_linear.make_feature_spec, exactly
    like calculate_gradcam.py's dummy-spec bootstrap -- but we only ever
    read spec.dataset here, never spec.model_factory, so no checkpoint or
    AnatCL clone is needed."""
    args_pipeline = parse_args_train(args=[])
    args_pipeline.model = "anatcl" if model == "anatcl" else model
    args_pipeline.turboprep_root = args.turboprep_root
    args_pipeline.nki_bids_root = args.nki_bids_root
    args_pipeline.roi_csv = args.roi_csv
    # anatcl's dataset/preprocessing doesn't depend on --anatcl-level (only
    # the encoder does, in model_factory, which we never call) -- leave the
    # default ("global") alone, it's irrelevant here.

    demographics = pd.read_csv(args.demographics, delimiter="\t", names=["Subject", "Sex", "Age"])
    demographics["Subject"] = demographics["Subject"].astype(str)
    if args.exclude_subject:
        demographics = demographics[~demographics["Subject"].isin(args.exclude_subject)].reset_index(drop=True)

    spec, kept_subjects, _ = make_feature_spec(args_pipeline, demographics)
    return spec, kept_subjects


def main():
    parser = argparse.ArgumentParser(
        description="Compute a population-mean reference (IG baseline) image per model, in that model's own preprocessed input space."
    )
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Subset of models to compute (default: all).")
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument(
        "--registration",
        choices=["rigid", "affine"],
        default="affine",
        help="Which registration pipeline's preprocessing to average over -- picks --turboprep-root's "
        "default (NKI_RIGID_TURBOPREP_ROOT or NKI_AFFINE_TURBOPREP_ROOT env var) when --turboprep-root isn't "
        "given explicitly, and tags every output filename ('{registration}_{model}_reference.npy') so "
        "rigid and affine baselines never collide in the same --output-dir.",
    )
    parser.add_argument(
        "--turboprep-root",
        type=Path,
        default=None,
        help="Overrides --registration's own default (NKI_RIGID_TURBOPREP_ROOT/NKI_AFFINE_TURBOPREP_ROOT env var).",
    )
    parser.add_argument("--nki-bids-root", type=Path, default=DEFAULT_NKI_BIDS_ROOT)
    parser.add_argument(
        "--roi-csv",
        type=Path,
        default=DEFAULT_ROI_CSV,
        help="--models fs_roi only: build_freesurfer_roi_tables.py's output CSV of per-subject FreeSurfer ROI volumes.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--exclude-subject", action="append", default=list(DEFAULT_EXCLUDE_SUBJECTS))
    args = parser.parse_args()

    if args.turboprep_root is None:
        args.turboprep_root = NKI_AFFINE_TURBOPREP_ROOT if args.registration == "affine" else NKI_RIGID_TURBOPREP_ROOT

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        print(f"=== {model} ===")
        spec, kept_subjects = build_spec(model, args)
        print(f"  {len(kept_subjects)} subject(s) with available input volumes")

        mean_arr, n = compute_mean_dataset_tensor(
            spec.dataset, batch_size=args.batch_size, num_workers=args.num_workers, desc=model
        )

        prefix = f"{args.registration}_"
        if model == "optim_uncnn":
            for idx, channel in enumerate(UNCNN_CHANNELS):
                out_path = args.output_dir / f"{prefix}optim_uncnn_{channel}_reference.npy"
                np.save(out_path, mean_arr[idx])
                print(f"  saved {out_path} (channel={channel}, shape={mean_arr[idx].shape}, n={n})")
        else:
            out_name = f"{prefix}anatcl_reference.npy" if model == "anatcl" else f"{prefix}{model}_reference.npy"
            out_path = args.output_dir / out_name
            np.save(out_path, mean_arr)
            print(f"  saved {out_path} (shape={mean_arr.shape}, n={n})")


if __name__ == "__main__":
    main()
