"""
Resolve which of the externally pre-trained linear heads
($HEADS_DIR/, one
heads_affine_linear_sklearn_summary_{NKI,PPMI,HBN}.csv manifest per dataset,
inside that same directory) apply to one (dataset, target) pair, and print,
one per usable checkpoint, the exact `--model ...` args calculate_gradcam.py
needs plus the checkpoint's absolute path -- so
slurm/compute_saliency_pretrained_heads.sh never has to hardcode a model-name
mapping or guess anatcl's --anatcl-folds value.

NOTE on which heads directory this is: the heads' author has two parallel checkpoint
sets, `heads_affine_linear/` (older, signed-sqrt applied to Untrained_CNN's
covariance-pooled features, matching this repo's own train_linear.py
convention) and `heads_affine_linear_sklearn/` (current default here --
Untrained_CNN's covariance features are fed in RAW, no signed-sqrt, straight
into sklearn's StandardScaler+Ridge/LogisticRegression). This script now
points at the sklearn/raw-feature set by default -- see
docs/known_limitations.md's "externally pre-trained heads" section for what
that changes about the resulting Untrained_CNN saliency maps.

Why read each .pt file instead of just the CSV: the CSV (columns dataset,
model, task, n_subjects, feature_dim, hyperparameter, value, refit_converged,
inner_nonconverged_fits, parity_max_abs_diff, path) names the checkpoint but
not the exact calculate_gradcam.py --model invocation. Each checkpoint's own
top-level dict carries a `train_linear_model` field that already IS that exact
spec string (e.g. "optim_uncnn", "anatcl --anatcl-level global --anatcl-folds
all" -- note "all", not this repo's own fold0 convention, which is why this
has to come from the checkpoint rather than a hand-written mapping). These
checkpoints are tiny (a linear head's weights, not a backbone), so loading all
of them for one (dataset, target) is cheap.

The 3 FreeSurfer-surface-feature models (FreeSurfer, FS_Schaefer, FS_aparc --
cortical thickness/surface-area features, keyed by feature_names like CTH_*/
SurfArea_*) have train_linear_model=None: there's no model_factory this
repo's calculate_gradcam.py/calculate_gradcam_tabular.py can build for them
(different feature set than the existing fs_roi DKT-volume pathway, no
Schaefer/aparc atlas-painting logic exists yet) -- skipped here, out of scope
until that's designed separately.

Usage:
    python list_pretrained_heads.py --dataset hbn --target age
    python list_pretrained_heads.py --dataset ppmi --target bmi   # -> nothing, PPMI has no bmi heads

Output: one TSV line per usable checkpoint to stdout --
    <label>\t<model-and-extra-args>\t<absolute-checkpoint-path>
e.g.:
    AnatCL_Global	anatcl --anatcl-level global --anatcl-folds all	/home/.../HBN_AnatCL_Global_age.pt
Missing/unusable rows (FreeSurfer-surface models, a checkpoint that's gone
missing from disk, or one with a surprising train_linear_model) are WARNed to
stderr and skipped, never a hard failure -- same graceful-degradation
convention as this repo's other multi-checkpoint scripts.
"""

import os
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

DEFAULT_HEADS_DIR = Path(os.environ["HEADS_DIR"]) if os.environ.get("HEADS_DIR") else None

# FreeSurfer-surface-feature models -- see module docstring for why these are
# excluded (no model_factory/atlas-painting logic exists for them here).
FREESURFER_SURFACE_MODELS = {"FreeSurfer", "FS_Schaefer", "FS_aparc"}


def list_pretrained_heads(heads_dir, dataset, target):
    """
    Returns [(label, model_args, checkpoint_path), ...] for every usable
    checkpoint matching (dataset, target) in `heads_dir`'s own
    heads_affine_linear_sklearn_summary_{DATASET}.csv -- see module docstring
    for what "usable" excludes and why.
    """
    heads_dir = Path(heads_dir)
    summary_csv = heads_dir / f"heads_affine_linear_sklearn_summary_{dataset.upper()}.csv"
    if not summary_csv.exists():
        raise SystemExit(f"no per-dataset summary CSV found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    # CSV's own dataset values are "NKI"/"PPMI"/"HBN" (uppercase); --dataset is
    # lowercase (nki/ppmi/hbn) to match this repo's existing slurm/env.sh
    # NKI_*/PPMI_*/HBN_* variable-name convention -- compare case-insensitively.
    rows = df[(df["dataset"].str.lower() == dataset.lower()) & (df["task"] == target)]

    # row["path"] is written relative to heads_dir's PARENT (e.g.
    # "heads_affine_linear_sklearn/PPMI_Untrained_CNN_age.pt"), not relative to
    # the CSV's own location -- the CSV lives *inside* heads_dir itself, one
    # level deeper than the root that path string assumes, so resolving
    # against heads_dir.parent (not heads_dir or summary_csv.parent) is what
    # avoids silently doubling the "heads_affine_linear_sklearn" directory
    # component.
    path_root = heads_dir.parent
    results = []
    for _, row in rows.iterrows():
        label = row["model"]
        if label in FREESURFER_SURFACE_MODELS:
            continue  # out of scope, see module docstring -- not a WARN, this is expected/common
        ckpt_path = (path_root / row["path"]).resolve()
        if not ckpt_path.exists():
            print(f"WARN: {label} ({dataset}/{target}): checkpoint listed in {summary_csv} but missing on disk: "
                  f"{ckpt_path} -- skipping", file=sys.stderr)
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_args = checkpoint.get("train_linear_model")
        if not model_args:
            print(f"WARN: {label} ({dataset}/{target}): checkpoint has no train_linear_model "
                  f"(got {model_args!r}) -- skipping {ckpt_path}", file=sys.stderr)
            continue
        results.append((label, model_args, str(ckpt_path)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--heads-dir", type=Path, default=DEFAULT_HEADS_DIR, required=DEFAULT_HEADS_DIR is None,
                        help="Directory of externally pre-trained linear heads (default: $HEADS_DIR).")
    parser.add_argument("--dataset", required=True, choices=["nki", "ppmi", "hbn"])
    parser.add_argument("--target", required=True, choices=["age", "sex", "bmi"])
    args = parser.parse_args()

    if not args.heads_dir.exists():
        raise SystemExit(f"--heads-dir not found: {args.heads_dir}")

    rows = list_pretrained_heads(args.heads_dir, args.dataset, args.target)
    for label, model_args, ckpt_path in rows:
        print(f"{label}\t{model_args}\t{ckpt_path}")


if __name__ == "__main__":
    main()
