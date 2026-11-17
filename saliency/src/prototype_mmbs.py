"""
Prototype: Monte Carlo Multi-Feature Baseline Shapley (MMBS) attribution,
computed against the real BrainAgePipeline (frozen backbone + linear RidgeCV
head) already used by train_linear.py/calculate_gradcam.py.

MMBS (Schut, van Liere & van Leeuwen, "Monte Carlo Multi-Feature Baseline
Shapley", TMLR 2026) generalizes Integrated Gradients (IG) and Baseline
Shapley (BShap) via a step-size parameter: divide all features into
`n_groups` = ceil(n_features / m) groups by a randomly sampled ordering,
then run one IG-style path integral *per group*, holding already-revealed
groups at the explicant x_bar and not-yet-revealed groups at the baseline
x0. n_groups=1 recovers plain BShap-style marginal-contribution attribution;
n_groups=n_features recovers IG. See docs/attribution_methods.md for this
repo's IG definition, which MMBS's per-group step directly reuses.

This is a STANDALONE PROTOTYPE, deliberately not wired into
calculate_gradcam.py's --attribution-method yet (see the repo's README/
docs/known_limitations.md for why new attribution methods here get
validated -- completeness check, convergence check -- before being trusted
enough to sit alongside "gradient"/"gradient_x_input"/"integrated_gradients"
in the shared per-subject SLURM pipeline). It only reads BrainAgePipeline/
make_feature_spec/load_reference_baseline from the existing modules; it
does not modify them.

Three modes, all operating over one or more --subject-nii volumes for one
model:

  completeness (default): compute one MMBS attribution map per subject and
    check it against the Completeness axiom, sum_v MMBS_v ~= F(x_bar) - F(x0)
    -- the fastest available signal that something in the estimator or the
    pipeline's autograd graph is broken, before trusting any map from it.

  convergence: repeat MMBS at increasing --convergence-samples sample
    counts (independent Monte Carlo draws each time, not reused/nested) and
    report each level's relative MSE against the largest sample count run,
    per subject -- the same diagnostic the MMBS paper's Figure 2 uses,
    except here the "reference" is a large-sample MMBS run rather than a
    converged MBShap run (this repo has no BShap/MBShap implementation to
    compare against; see docs/attribution_methods.md's Integrated Gradients
    section for the closest existing reference point). Treat this as a
    self-consistency check, not proof of convergence to true BShap.

  stepsize (optim_uncnn only): sweep --stepsize-grid group counts and, for
    each, report (a) the raw-channel attribution map's extreme-value/outlier
    statistics and (b) how close any covariance-pooled feature evaluated
    along the way came to the signed-sqrt derivative singularity at 0 (see
    BrainAgePipeline._signed_sqrt in train_linear.py and "optim_uncnn:
    signed-sqrt gradient amplification near zero" in
    docs/known_limitations.md). This directly tests whether MMBS's step
    size changes exposure to that known bug, rather than assuming it does.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_linear import (
    BrainAgePipeline,
    make_feature_spec,
    parse_args as parse_args_train,
    DEFAULT_DEMOGRAPHICS,
)
from calculate_gradcam import (
    UNCNN_CHANNELS,
    DEFAULT_REFERENCE_BASELINES_DIR,
    load_reference_baseline,
    lookup_true_age,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "saliency_outputs" / "mmbs_prototype"


# ---------------------------------------------------------------------------
# Pipeline/input loading -- mirrors calculate_gradcam.py's main() steps 1-4
# (spec construction, checkpoint loading into BrainAgePipeline, dataset
# ingestion of one subject's volume), factored out here so multiple subjects
# can reuse the same pipeline instance instead of rebuilding the backbone
# per subject.
# ---------------------------------------------------------------------------

def load_pipeline(model, checkpoint_path, seed, anatcl_level, anatcl_folds, anatcl_repo_root, device):
    """
    Reconstructs BrainAgePipeline exactly as calculate_gradcam.py's main()
    does (steps 1-3), without needing a real subject yet. The seed matters
    only for optim_uncnn: its backbone (CNN3D_CovPool) is a random init that
    was never checkpointed, so it must be reconstructed with the same seed
    train_linear.py used before fitting the ridge head, or this produces a
    different random network every run (see calculate_gradcam.py's --seed
    help text).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    args_pipeline = parse_args_train(args=[])
    args_pipeline.model = model
    args_pipeline.anatcl_level = anatcl_level
    args_pipeline.anatcl_folds = anatcl_folds
    args_pipeline.anatcl_repo_root = anatcl_repo_root

    import train_linear as _train_linear_mod
    _train_linear_mod.ANATCL_REPO_ROOT = anatcl_repo_root

    dummy_demo = pd.DataFrame([{"Subject": "dummy", "Sex": "M", "Age": 40.0}])
    spec, _, _ = make_feature_spec(args_pipeline, dummy_demo)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    feature_dim = state_dict["head.weight"].shape[1]

    backbone = spec.model_factory(device)
    head = nn.Linear(feature_dim, 1, bias=True)

    pipeline = BrainAgePipeline(
        backbone=backbone,
        head=head,
        cov_mask=np.zeros(feature_dim, dtype=bool),
        scaler_mean=np.zeros(feature_dim, dtype=np.float32),
        scaler_scale=np.ones(feature_dim, dtype=np.float32),
        postprocess_fn=spec.postprocess,
    )
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith("backbone.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch. Missing (non-backbone): {bad_missing}, Unexpected: {unexpected}"
        )
    pipeline.to(device)
    pipeline.eval()
    return pipeline, spec


def load_subject_input(spec, subject_nii, model):
    if model == "anatcl" and not Path(subject_nii).name.startswith("mwp1"):
        raise ValueError(
            f"--model anatcl expects a CAT12 modulated gray-matter VBM map (mwp1sub-*_T1w.nii), got: "
            f"{Path(subject_nii).name}."
        )
    spec.dataset.paths = [str(subject_nii)]
    return spec.dataset[0]  # (C, H, W, D)


def get_baseline_tensor(model, input_tensor, mmbs_baseline, reference_baselines_dir):
    if mmbs_baseline == "zero":
        return torch.zeros_like(input_tensor)
    baseline_arr = load_reference_baseline(model, reference_baselines_dir)
    if baseline_arr.shape != tuple(input_tensor.shape):
        raise ValueError(
            f"Reference baseline shape {baseline_arr.shape} does not match input tensor shape "
            f"{tuple(input_tensor.shape)} for model {model!r}. Was compute_reference_baselines.py run "
            f"with the same preprocessing this script uses?"
        )
    return torch.from_numpy(baseline_arr).float()


# ---------------------------------------------------------------------------
# MMBS core estimator (Eq. 6 of the MMBS paper)
# ---------------------------------------------------------------------------

def _feature_groups(shape, n_groups, seed, device):
    """
    Assigns every feature (voxel/channel entry) a group in [0, n_groups) by a
    random permutation, without ever materializing an n!-sized ordering set
    or a per-group index list -- just one torch.randperm over the flattened
    feature count, thresholded into n_groups equal-ish buckets. `shape` is
    (C, H, W, D); returned tensor has that same shape.
    """
    n = 1
    for s in shape:
        n *= s
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # randperm(n) is a bijection [0, n) -> [0, n); reading its output as
    # "rank assigned to position v" (rather than "position of the v-th
    # ranked feature") is equally valid since it's a uniformly random
    # permutation either way -- this avoids a second argsort.
    ranks = torch.randperm(n, generator=generator)
    groups = torch.div(ranks * n_groups, n, rounding_mode="floor").clamp_(max=n_groups - 1)
    return groups.reshape(shape).to(device)


def mmbs_attribution(
    pipeline,
    x_bar,
    x0,
    n_groups,
    n_samples,
    device,
    seed=0,
    capture_diagnostics=False,
):
    """
    Monte Carlo Multi-Feature Baseline Shapley, Eq. 6: average over
    n_samples independent draws of (ordering, per-group interpolation
    fraction z_k) of

      sum_{k=1}^{n_groups} (x_bar - x0)_{group k} * d F(x_hat_k) / dx |_{group k}

    where x_hat_k has already-revealed groups (< k) fixed at x_bar,
    not-yet-revealed groups (> k) fixed at x0, and group k itself at
    x0 + z_k*(x_bar - x0). Costs n_samples * n_groups forward+backward
    passes total (one per group per sample) -- --mmbs-steps/--mmbs-samples
    trade approximation quality for runtime, same tradeoff --ig-steps
    exposes for Integrated Gradients (docs/attribution_methods.md), but
    multiplied by an extra Monte Carlo sample dimension IG doesn't need
    (IG's Riemann sum is deterministic; MMBS's ordering draw isn't).

    Returns (attribution (C,H,W,D) float32 ndarray, predicted_age_x_bar,
    predicted_age_x0, diagnostics). diagnostics is None unless
    capture_diagnostics=True, in which case it's a dict with
    "min_abs_cov_feature" (float or None if the pipeline's cov_mask is
    all-False) -- the smallest |feature value| observed at any evaluated
    interpolation point, at a position that feeds BrainAgePipeline's
    signed-sqrt covariance normalization (see module docstring / "stepsize"
    mode).
    """
    pipeline.to(device)
    pipeline.eval()

    x_bar_d = x_bar.to(device).unsqueeze(0).double()
    x0_d = x0.to(device).unsqueeze(0).double()
    if x_bar_d.shape != x0_d.shape:
        raise ValueError(f"Baseline shape {tuple(x0_d.shape)} does not match input shape {tuple(x_bar_d.shape)}")
    delta = x_bar_d - x0_d
    shape = tuple(x_bar_d.shape[1:])

    with torch.no_grad():
        predicted_age_x_bar = float(pipeline(x_bar_d.float()).cpu().item())
        predicted_age_x0 = float(pipeline(x0_d.float()).cpu().item())

    attribution_sum = torch.zeros_like(x_bar_d)
    rng = np.random.default_rng(seed)
    min_abs_cov_feature = None
    has_cov_mask = bool(pipeline.cov_mask.any().item())

    for sample_idx in range(n_samples):
        sample_seed = seed * 1_000_003 + sample_idx
        groups = _feature_groups(shape, n_groups, sample_seed, device).unsqueeze(0)  # (1, C, H, W, D)

        for k in range(n_groups):
            mask_current = groups == k
            mask_revealed = groups < k
            z_k = float(rng.uniform(0.0, 1.0))

            interpolated = torch.where(mask_revealed, x_bar_d, x0_d)
            interpolated = torch.where(mask_current, x0_d + z_k * delta, interpolated)
            interpolated = interpolated.float()
            interpolated.requires_grad_(True)

            pipeline.zero_grad()
            if capture_diagnostics and has_cov_mask:
                age_prediction, cov_feat_values = _forward_with_cov_features(pipeline, interpolated)
                if cov_feat_values.numel() > 0:
                    batch_min = float(cov_feat_values.detach().abs().min().cpu().item())
                    if min_abs_cov_feature is None or batch_min < min_abs_cov_feature:
                        min_abs_cov_feature = batch_min
            else:
                age_prediction = pipeline(interpolated)
            age_prediction.backward()

            grad = interpolated.grad.double()
            contribution = torch.where(mask_current, delta * grad, torch.zeros_like(delta))
            attribution_sum += contribution

    attribution_mean = (attribution_sum / n_samples).squeeze(0).cpu().numpy().astype(np.float32)
    diagnostics = {"min_abs_cov_feature": min_abs_cov_feature} if capture_diagnostics else None
    return attribution_mean, predicted_age_x_bar, predicted_age_x0, diagnostics


def _forward_with_cov_features(pipeline, x):
    """
    Replicates BrainAgePipeline.forward (train_linear.py) step-by-step, the
    same way calculate_gradcam.py's audit_gradient_backpropagation already
    does for its own diagnostic purposes, so the pre-signed-sqrt covariance
    feature values (needed to measure proximity to the derivative
    singularity at 0 -- see "stepsize" mode) can be read out without a
    second forward pass. Mathematically identical to pipeline(x); backward()
    on the returned age_prediction flows through the same graph.
    """
    feat = pipeline.backbone(x)
    if pipeline.postprocess_fn is not None:
        feat = pipeline.postprocess_fn(feat)
    if isinstance(feat, tuple):
        feat = feat[0]
    feat = torch.flatten(feat, start_dim=1)

    cov_feat_values = feat[:, pipeline.cov_mask]

    cov_safe_input = torch.where(pipeline.cov_mask, feat, torch.ones_like(feat))
    sqrt_feat = pipeline._signed_sqrt(cov_safe_input)
    feat = torch.where(pipeline.cov_mask, sqrt_feat, feat)

    feat_scaled = (feat - pipeline.scaler_mean) / pipeline.scaler_scale
    age_prediction = pipeline.head(feat_scaled)
    return age_prediction, cov_feat_values


# ---------------------------------------------------------------------------
# Mode 1+2: completeness check (steps 1-2 of the plan)
# ---------------------------------------------------------------------------

def run_completeness(pipeline, spec, args, subject_nii):
    input_tensor = load_subject_input(spec, subject_nii, args.model)
    baseline_tensor = get_baseline_tensor(args.model, input_tensor, args.mmbs_baseline, args.reference_baselines_dir)

    attribution, pred_x_bar, pred_x0, _ = mmbs_attribution(
        pipeline, input_tensor, baseline_tensor, args.mmbs_steps, args.mmbs_samples, args.device, seed=args.seed
    )

    target = pred_x_bar - pred_x0
    actual = float(attribution.sum())
    abs_gap = abs(actual - target)
    rel_gap = abs_gap / max(abs(target), 1e-8)

    subject_id, true_age = lookup_true_age(Path(subject_nii), args.demographics)
    result = {
        "subject_id": subject_id,
        "true_age": true_age,
        "predicted_age_x_bar": pred_x_bar,
        "predicted_age_x0": pred_x0,
        "completeness_target": target,
        "completeness_actual": actual,
        "completeness_abs_gap": abs_gap,
        "completeness_rel_gap": rel_gap,
        "mmbs_steps": args.mmbs_steps,
        "mmbs_samples": args.mmbs_samples,
        "mmbs_baseline": args.mmbs_baseline,
    }
    print(
        f"[{subject_id}] F(x_bar)={pred_x_bar:.4f} F(x0)={pred_x0:.4f} target={target:.4f} "
        f"sum(MMBS)={actual:.4f} abs_gap={abs_gap:.4f} rel_gap={rel_gap:.4%}"
    )
    return result


# ---------------------------------------------------------------------------
# Mode: convergence sweep (step 3 of the plan)
# ---------------------------------------------------------------------------

def run_convergence(pipeline, spec, args, subject_nii):
    input_tensor = load_subject_input(spec, subject_nii, args.model)
    baseline_tensor = get_baseline_tensor(args.model, input_tensor, args.mmbs_baseline, args.reference_baselines_dir)
    subject_id, true_age = lookup_true_age(Path(subject_nii), args.demographics)

    sample_counts = sorted(args.convergence_samples)
    reference_n = sample_counts[-1]

    maps = {}
    completeness_by_n = {}
    for n_samples in sample_counts:
        attribution, pred_x_bar, pred_x0, _ = mmbs_attribution(
            pipeline, input_tensor, baseline_tensor, args.mmbs_steps, n_samples, args.device,
            seed=args.seed,
        )
        maps[n_samples] = attribution
        target = pred_x_bar - pred_x0
        actual = float(attribution.sum())
        completeness_by_n[n_samples] = abs(actual - target) / max(abs(target), 1e-8)

    reference_map = maps[reference_n]
    reference_norm_sq = float((reference_map.astype(np.float64) ** 2).sum())
    per_n = []
    for n_samples in sample_counts:
        diff_sq = float(((maps[n_samples] - reference_map).astype(np.float64) ** 2).sum())
        relative_mse = diff_sq / reference_norm_sq if reference_norm_sq > 0 else float("nan")
        per_n.append(
            {
                "n_samples": n_samples,
                "relative_mse_vs_largest": relative_mse,
                "completeness_rel_gap": completeness_by_n[n_samples],
            }
        )
        print(
            f"[{subject_id}] n_samples={n_samples:>5} relative_MSE_vs_n={reference_n}="
            f"{relative_mse:.4f} completeness_rel_gap={completeness_by_n[n_samples]:.4%}"
        )

    return {"subject_id": subject_id, "true_age": true_age, "mmbs_steps": args.mmbs_steps, "per_sample_count": per_n}


# ---------------------------------------------------------------------------
# Mode: step-size / signed-sqrt interaction test (step 4 of the plan)
# ---------------------------------------------------------------------------

def run_stepsize(pipeline, spec, args, subject_nii):
    if args.model != "optim_uncnn":
        raise ValueError(
            "--mode stepsize only applies to --model optim_uncnn: every other model has an all-False "
            "cov_mask and never touches BrainAgePipeline._signed_sqrt, so this test has nothing to "
            "measure for them (see docs/known_limitations.md)."
        )

    input_tensor = load_subject_input(spec, subject_nii, args.model)
    baseline_tensor = get_baseline_tensor(args.model, input_tensor, args.mmbs_baseline, args.reference_baselines_dir)
    subject_id, true_age = lookup_true_age(Path(subject_nii), args.demographics)

    raw_idx = UNCNN_CHANNELS.index("raw")
    per_stepsize = []
    for n_groups in sorted(args.stepsize_grid):
        attribution, pred_x_bar, pred_x0, diagnostics = mmbs_attribution(
            pipeline, input_tensor, baseline_tensor, n_groups, args.mmbs_samples, args.device,
            seed=args.seed, capture_diagnostics=True,
        )
        raw_map = attribution[raw_idx]
        mean, std = float(raw_map.mean()), float(raw_map.std())
        outlier_thresh = mean + 5 * std
        outlier_count = int(np.sum(np.abs(raw_map - mean) > 5 * std)) if std > 0 else 0
        min_abs_cov = diagnostics["min_abs_cov_feature"]
        worst_case_derivative = 1.0 / (2.0 * np.sqrt(min_abs_cov + 1e-6)) if min_abs_cov is not None else None

        entry = {
            "n_groups": n_groups,
            "raw_max_abs_attribution": float(np.abs(raw_map).max()),
            "raw_std_attribution": std,
            "raw_outlier_voxel_count_gt_5std": outlier_count,
            "min_abs_cov_feature_encountered": min_abs_cov,
            "worst_case_signed_sqrt_derivative": worst_case_derivative,
        }
        per_stepsize.append(entry)
        print(
            f"[{subject_id}] n_groups={n_groups:>4} max|attr|={entry['raw_max_abs_attribution']:.4e} "
            f"std={std:.4e} outliers(>5std)={outlier_count:>6} "
            f"min|cov_feat|={min_abs_cov if min_abs_cov is None else f'{min_abs_cov:.3e}'} "
            f"worst_case_d(signed_sqrt)={worst_case_derivative if worst_case_derivative is None else f'{worst_case_derivative:.1f}'}"
        )

    return {"subject_id": subject_id, "true_age": true_age, "per_stepsize": per_stepsize}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MMBS attribution prototype (see module docstring).")
    parser.add_argument("--model", choices=["optim_uncnn", "brainiac", "swinbrain", "anatcl", "simclr"], required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--subject-nii", type=Path, nargs="+", required=True, help="One or more subject NIfTI paths.")
    parser.add_argument("--mode", choices=["completeness", "convergence", "stepsize"], default="completeness")
    parser.add_argument("--mmbs-steps", type=int, default=8, help="Group count for completeness/stepsize modes (n_groups in the module docstring).")
    parser.add_argument("--mmbs-samples", type=int, default=8, help="Monte Carlo sample count for completeness/stepsize modes.")
    parser.add_argument("--convergence-samples", type=int, nargs="+", default=[4, 8, 16, 32], help="--mode convergence only: sample counts to sweep.")
    parser.add_argument("--stepsize-grid", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="--mode stepsize only: group counts to sweep.")
    parser.add_argument("--mmbs-baseline", choices=["zero", "reference"], default="reference")
    parser.add_argument("--reference-baselines-dir", type=Path, default=DEFAULT_REFERENCE_BASELINES_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anatcl-level", choices=["global", "local"], default=os.environ.get("anatcl_level", "global"))
    parser.add_argument("--anatcl-folds", default=os.environ.get("anatcl_folds", "all"))
    parser.add_argument("--anatcl-repo-root", type=Path, default=Path(os.environ.get("ANATCL_REPO_ROOT", "/path/to/AnatCL")))
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, spec = load_pipeline(
        args.model, args.checkpoint_path, args.seed, args.anatcl_level, args.anatcl_folds,
        args.anatcl_repo_root, args.device,
    )

    mode_fn = {"completeness": run_completeness, "convergence": run_convergence, "stepsize": run_stepsize}[args.mode]
    results = [mode_fn(pipeline, spec, args, subject_nii) for subject_nii in args.subject_nii]

    summary = {
        "model": args.model,
        "checkpoint_path": str(args.checkpoint_path),
        "mode": args.mode,
        "mmbs_baseline": args.mmbs_baseline,
        "seed": args.seed,
        "results": results,
    }
    out_path = args.output_dir / f"{args.model}_{args.mode}_mmbs_prototype.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
