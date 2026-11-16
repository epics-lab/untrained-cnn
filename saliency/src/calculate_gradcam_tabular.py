"""
Gradient-based saliency for the fs_roi model (a RidgeCV linear head trained
directly on FreeSurfer regional volumes -- see train_linear.py's "fs_roi"
FeatureSpec branch), for one subject at a time.

Why this is a SEPARATE script from calculate_gradcam.py, not a new branch
inside it: calculate_gradcam.py's compute_direct_pipeline_saliency assumes
a (C, H, W, D) image input throughout -- its channel-collapse logic treats
voxel_gradients.shape[0] as an image-channel count (special-cased for
optim_uncnn's 3 heterogeneous channels and swinbrain's repeated channels),
and compute_and_save_saliency/restore_native_space map the result onto a
NIfTI affine. A flat (D,) ROI-attribution vector has no "channel" concept at
all -- reusing that code path as-is would silently keep only the first
region's value (voxel_gradients[0]) and discard the rest. Its main() is
also deeply --subject-nii/NIfTI-specific (an anatcl mwp1*.nii filename
check, ref_img = nib.load(...), spec.dataset.paths = [str(subject_nii)]).
fs_roi's "subject input" is instead a single ROW of an already-cached
feature matrix, looked up by --subject-id -- no file load, no image shape,
no native-space restoration needed at all.

The gradient/gradient_x_input/Integrated-Gradients math itself (imported
from calculate_gradcam.py, not reimplemented) IS already shape-agnostic, so
this script reuses it directly on (D,) tensors -- only the channel-handling
wrapper and the NIfTI save path are replaced.

Output: two things per subject/attribution-method, both keyed to the same
feature-name ordering train_linear.py persisted at training time
({cache_name}_feature_names.json):
  - {output_dir}/{subject_id}/fs_roi_{method_tag}{signed,abs}_saliency.npy
    (raw (D,) arrays) and a companion ..._saliency.csv (region,signed,abs,
    sorted by |signed| descending) -- the human-readable/quick-inspection
    output, no template/LUT dependency.
  - {output_dir}/{subject_id}/fs_roi_{method_tag}{signed,abs}_saliency.nii.gz
    -- the SAME per-ROI values painted onto --seg-template via
    roi_to_volume.py's FreeSurferColorLUT matching. Because every subject's
    painted volume shares the same template grid/affine, these files are
    real, spatially valid NIfTI volumes on a shared grid -- so
    aggregate_saliency.py and visualize_paper.py/visualize_paper_nilearn.py
    can include "fs_roi" as one more --models entry with ZERO changes to
    either script. The map will visually look like flat-colored anatomical
    regions (one value per whole ROI), not a smooth per-voxel gradient --
    that's the real granularity of this attribution, not a rendering bug.

Usage:
    python calculate_gradcam_tabular.py --subject-id A00000000
    python calculate_gradcam_tabular.py --subject-id A00000000 \\
        --attribution-method integrated_gradients --ig-baseline reference
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_linear import BrainAgePipeline, DEFAULT_DEMOGRAPHICS, DEFAULT_ROI_CSV, make_feature_spec
from train_linear import parse_args as parse_args_train
from calculate_gradcam import (
    DEFAULT_REFERENCE_BASELINES_DIR,
    _integrated_gradients_and_prediction,
    attribution_method_tag,
    cascading_randomize_pipeline,
    fit_ridge_head_inmemory,
    load_reference_baseline,
    randomize_all_pipeline,
    report_similarity,
)
from roi_to_volume import load_lut, load_segmentation, paint_region_values
from targets import DEFAULT_PARTICIPANTS_TSV, get_target_spec, merge_bmi

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEG_TEMPLATE = Path(
    os.environ.get(
        "FS_ROI_SEG_TEMPLATE",
        str(REPO_ROOT / "templates" / "aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz"),
    )
)
DEFAULT_LUT = Path(os.environ.get("FS_ROI_LUT", os.path.join(os.environ.get("FREESURFER_HOME", ""), "FreeSurferColorLUT.txt")))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "saliency_outputs"
DEFAULT_FEATURES_DIR = REPO_ROOT / "artifacts"
DEFAULT_CACHE_NAME = "fs_roi_dktatlas_grayvol"


def compute_tabular_saliency(
    pipeline, input_tensor, device, attribution_method="gradient", baseline_tensor=None, ig_steps=50,
):
    """
    (D,) analogue of calculate_gradcam.py's compute_direct_pipeline_saliency,
    without the (C, H, W, D)-shaped channel-collapse block -- see this
    module's docstring for why that block doesn't apply here at all (there
    is exactly one "channel": the ROI vector itself).

    Returns (signed_map (D,), abs_map (D,), predicted_age, diagnostics)
    -- diagnostics is None for gradient/gradient_x_input, or Integrated
    Gradients' completeness dict (see _integrated_gradients_and_prediction).
    """
    if attribution_method not in ("gradient", "gradient_x_input", "integrated_gradients"):
        raise ValueError(f"Unknown attribution_method: {attribution_method!r}")
    if attribution_method == "integrated_gradients" and baseline_tensor is None:
        raise ValueError("attribution_method='integrated_gradients' requires baseline_tensor")

    pipeline.to(device)
    pipeline.eval()

    diagnostics = None
    if attribution_method == "integrated_gradients":
        grad, predicted_age, diagnostics = _integrated_gradients_and_prediction(
            pipeline, input_tensor, baseline_tensor, device, steps=ig_steps
        )
    else:
        x = input_tensor.to(device).unsqueeze(0).detach()  # (1, D)
        x.requires_grad_(True)
        pipeline.zero_grad()
        age_prediction = pipeline(x)
        predicted_age = float(age_prediction.detach().cpu().item())
        age_prediction.backward()
        if x.grad is None:
            raise RuntimeError("No gradient reached the input tensor after backward().")
        grad = x.grad.squeeze(0).cpu().numpy()  # (D,)

        if attribution_method == "gradient_x_input":
            input_values = x.detach().squeeze(0).cpu().numpy()
            grad = grad * input_values

    signed_map = grad.squeeze() if grad.ndim > 1 else grad
    abs_map = np.abs(signed_map)
    return signed_map, abs_map, predicted_age, diagnostics


def make_noise_input_tabular(input_tensor, dist="permute", seed=0):
    """
    (D,) analogue of calculate_gradcam.py's make_noise_input -- same
    rationale (a structure-free control input, same shape/scale as the
    real preprocessed input, computed through the real trained pipeline --
    see that function's docstring), adapted for a flat ROI-feature vector
    rather than a (C, H, W, D) image: there's no separate "channel" to loop
    over here (the whole vector already IS the one channel, see this
    module's own docstring), so "permute" shuffles the region VALUES
    themselves across regions (the tabular analogue of scrambling voxels
    within a channel), and "gaussian" draws fresh noise matched to the real
    vector's own mean/std.
    """
    g = torch.Generator().manual_seed(seed)
    if dist == "gaussian":
        mean, std = input_tensor.mean(), input_tensor.std()
        return mean + std * torch.randn(input_tensor.shape, generator=g)
    if dist == "permute":
        perm = torch.randperm(input_tensor.shape[0], generator=g)
        return input_tensor[perm]
    raise ValueError(f"Unknown dist: {dist!r}")


def save_region_csv(signed_map, abs_map, feature_names, output_path):
    df = pd.DataFrame({"region": feature_names, "signed": signed_map, "abs": abs_map})
    df = df.sort_values("abs", ascending=False)
    df.to_csv(output_path, index=False)


def save_roi_saliency(signed_map, abs_map, feature_names, output_dir, prefix, seg_img, seg_data, lut_df):
    """Writes both the .npy/.csv region-value output and the painted .nii.gz pair -- see module docstring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}signed_saliency.npy", signed_map)
    np.save(output_dir / f"{prefix}abs_saliency.npy", abs_map)
    save_region_csv(signed_map, abs_map, feature_names, output_dir / f"{prefix}saliency.csv")

    signed_region_to_value = dict(zip(feature_names, signed_map.tolist()))
    abs_region_to_value = dict(zip(feature_names, abs_map.tolist()))
    signed_vol, unmatched_signed = paint_region_values(seg_data, lut_df, signed_region_to_value)
    abs_vol, _ = paint_region_values(seg_data, lut_df, abs_region_to_value)
    if unmatched_signed:
        print(f"  NOTE: {len(unmatched_signed)} region(s) with no --seg-template/--lut match: {unmatched_signed}")

    nib.save(nib.Nifti1Image(signed_vol, seg_img.affine, seg_img.header), output_dir / f"{prefix}signed_saliency.nii.gz")
    nib.save(nib.Nifti1Image(abs_vol, seg_img.affine, seg_img.header), output_dir / f"{prefix}abs_saliency.nii.gz")


def lookup_true_value(subject_id, demographics_path, target_spec, participants_tsv_path=None):
    if not demographics_path.exists():
        print(f"WARNING: demographics file not found at {demographics_path}; true_{target_spec.name} will be null.")
        return None
    demographics = pd.read_csv(demographics_path, delimiter="\t", names=["Subject", "Sex", "Age"])
    demographics["Subject"] = demographics["Subject"].astype(str)
    if target_spec.name == "bmi":
        demographics = merge_bmi(demographics, participants_tsv_path)
    row = demographics[demographics["Subject"] == str(subject_id)]
    if row.empty:
        print(f"WARNING: subject {subject_id!r} not found in {demographics_path}; true_{target_spec.name} will be null.")
        return None
    return float(target_spec.encode(row.iloc[0][target_spec.demographics_column]))


def load_cached_row(features_dir, cache_name, subject_id):
    """
    Looks up subject_id's row directly in train_linear.py's cached
    {cache_name}_X_features_raw.npy/_subjects.npy/_feature_names.json --
    no per-subject file load needed (unlike every image-based model, whose
    saliency script re-reads a NIfTI per subject), since fs_roi's whole
    cohort already lives in one small in-memory matrix.
    """
    X = np.load(features_dir / f"{cache_name}_X_features_raw.npy")
    subjects = np.load(features_dir / f"{cache_name}_subjects.npy", allow_pickle=True).astype(str).tolist()
    with open(features_dir / f"{cache_name}_feature_names.json") as f:
        feature_names = json.load(f)

    if subject_id not in subjects:
        raise SystemExit(
            f"Subject {subject_id!r} not found in {features_dir / f'{cache_name}_subjects.npy'} -- "
            "was train_linear.py --model fs_roi run (and did it keep this subject)?"
        )
    row = X[subjects.index(subject_id)]
    return torch.from_numpy(row).float(), feature_names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--subject-id", required=True, help="NKI subject ID, e.g. A00000000.")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=None,
        help="Path to the trained head checkpoint (default: {--features-dir}/{--cache-name}_{suffix}.pt, "
        "where suffix depends on --target: brain_age_head/sex_head/bmi_head).",
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--cache-name", default=DEFAULT_CACHE_NAME)
    parser.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI_CSV)
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--target", choices=["age", "sex", "bmi"], default=os.environ.get("target", "age"))
    parser.add_argument(
        "--participants-tsv", type=Path, default=DEFAULT_PARTICIPANTS_TSV,
        help="--target bmi only: longitudinal participants.tsv BMI is joined in from (see src/targets.py).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seg-template", type=Path, default=DEFAULT_SEG_TEMPLATE,
        help="Segmentation volume (.mgz/.nii(.gz)) to paint per-ROI saliency onto -- any FreeSurfer aparc+aseg-"
        "style volume whose voxel labels match --lut. Default: one subject's real aparc+aseg.mgz recon-all "
        "output (not a population atlas -- same convention the existing visualize_roi_vs_aparc_surfarea_"
        "brain_maps.py prototype already used).",
    )
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT, help="FreeSurferColorLUT.txt matching --seg-template.")
    parser.add_argument("--attribution-method", choices=["gradient", "gradient_x_input", "integrated_gradients"], default="gradient")
    parser.add_argument("--ig-baseline", choices=["zero", "reference"], default="zero")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--reference-baselines-dir", type=Path, default=DEFAULT_REFERENCE_BASELINES_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--randomize-input", action="store_true",
        help="Run an input randomization control (see calculate_gradcam.py's --randomize-input): compute "
        "saliency for a structure-free control ROI-feature vector (same shape/scale as the real one, see "
        "--randomize-input-dist) using the real trained pipeline, and compare it to the baseline.",
    )
    parser.add_argument("--randomize-input-dist", choices=["permute", "gaussian"], default="permute")
    parser.add_argument(
        "--randomize-data", action="store_true",
        help="Run the data (label) randomization sanity check (Adebayo et al., 2018; see "
        "calculate_gradcam.py's fit_ridge_head_inmemory): refit the ridge head on this model's cached "
        "features with the target labels randomly shuffled, then compare its saliency map to the "
        "baseline. Requires cached {cache-name}_X_features_raw.npy/_y_{target}.npy from train_linear.py.",
    )
    parser.add_argument("--randomize-data-seeds", type=int, default=1)
    parser.add_argument(
        "--randomize-model", action="store_true",
        help="Run the model parameter randomization test (Adebayo et al., 2018): re-initialize the "
        "pipeline's parameterized modules and compare saliency to the baseline. fs_roi's backbone is "
        "nn.Identity (no parameters) so in practice only the head is ever reset -- see "
        "calculate_gradcam.py's randomize_all_pipeline/cascading_randomize_pipeline.",
    )
    parser.add_argument("--randomize-model-mode", choices=["cascade", "all"], default="cascade")
    parser.add_argument("--randomize-max-layers", type=int, default=None)
    parser.add_argument("--randomize-seed", type=int, default=0, help="Seed for all randomization tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ig_steps < 1:
        raise SystemExit("--ig-steps must be at least 1")
    if args.randomize_data_seeds < 1:
        raise SystemExit("--randomize-data-seeds must be at least 1")

    target_spec = get_target_spec(args.target)
    if args.checkpoint_path is None:
        args.checkpoint_path = args.features_dir / f"{args.cache_name}_{target_spec.checkpoint_suffix}.pt"

    # Scope every output (subject folders below, plus the arch_/data_/input_
    # sanity-check files written directly to args.output_dir) under a
    # per-target directory, so age/sex/bmi runs sharing the same caller-
    # supplied --output-dir never overwrite each other's files.
    args.output_dir = args.output_dir / target_spec.name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Reconstruct the fs_roi FeatureSpec purely for its model_factory
    # (nn.Identity) -- mirrors calculate_gradcam.py's own dummy-spec
    # bootstrap. roi_volume_rows_and_subjects reads --roi-csv here too, but
    # only spec.model_factory is actually used below.
    args_pipeline = parse_args_train(args=[])
    args_pipeline.model = "fs_roi"
    args_pipeline.roi_csv = args.roi_csv
    dummy_demo = pd.DataFrame([{"Subject": "__dummy__", "Sex": "M", "Age": 40.0}])
    spec, _, _ = make_feature_spec(args_pipeline, dummy_demo)

    # 2. Load the trained checkpoint.
    print(f"Extracting parameters from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    feature_dim = state_dict["head.weight"].shape[1]

    cov_mask = np.zeros(feature_dim, dtype=bool)
    backbone = spec.model_factory(device)
    head = nn.Linear(feature_dim, 1, bias=True)
    pipeline = BrainAgePipeline(
        backbone=backbone,
        head=head,
        cov_mask=cov_mask,
        scaler_mean=np.zeros(feature_dim, dtype=np.float32),
        scaler_scale=np.ones(feature_dim, dtype=np.float32),
    )
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith("backbone.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"Unexpected state_dict mismatch. Missing (non-backbone): {bad_missing}, Unexpected: {unexpected}")

    # 3. Look up this subject's cached feature row.
    input_tensor, feature_names = load_cached_row(args.features_dir, args.cache_name, args.subject_id)
    if len(feature_names) != feature_dim:
        raise RuntimeError(f"feature_names has {len(feature_names)} entries but the checkpoint's head expects {feature_dim}")
    print(f"Input tensor shape: {tuple(input_tensor.shape)} (D,)")

    baseline_tensor = None
    if args.attribution_method == "integrated_gradients":
        if args.ig_baseline == "zero":
            baseline_tensor = torch.zeros_like(input_tensor)
        else:
            baseline_arr = load_reference_baseline("fs_roi", args.reference_baselines_dir)
            if baseline_arr.shape != tuple(input_tensor.shape):
                raise ValueError(
                    f"Reference baseline shape {baseline_arr.shape} does not match input tensor shape "
                    f"{tuple(input_tensor.shape)}. Was compute_reference_baselines.py run with --models fs_roi?"
                )
            baseline_tensor = torch.from_numpy(baseline_arr).float()
        print(f"Integrated Gradients baseline: --ig-baseline={args.ig_baseline}, --ig-steps={args.ig_steps}")

    # 4. Compute saliency.
    signed_map, abs_map, predicted_age, diagnostics = compute_tabular_saliency(
        pipeline, input_tensor, device,
        attribution_method=args.attribution_method,
        baseline_tensor=baseline_tensor,
        ig_steps=args.ig_steps,
    )
    true_value = lookup_true_value(args.subject_id, args.demographics, target_spec, args.participants_tsv)
    reported_predicted_value = (
        predicted_age if target_spec.task_type != "classification" else float(1.0 / (1.0 + np.exp(-predicted_age)))
    )
    print(f"Subject {args.subject_id}: true {target_spec.name} = {true_value}, predicted {target_spec.name} = {reported_predicted_value}")
    if diagnostics is not None:
        print(
            "Integrated Gradients completeness: "
            f"F(x)-F(x0)={diagnostics['prediction_delta']:.6g}, sum(IG)={diagnostics['total_attribution']:.6g}, "
            f"error={diagnostics['completeness_error']:.6g}, relative_abs_error={diagnostics['relative_completeness_error']:.3e}"
        )

    # 5. Save: .npy/.csv region-value output + the painted .nii.gz pair.
    method_tag = attribution_method_tag(args.attribution_method, ig_baseline=args.ig_baseline)
    prefix = f"fs_roi_{method_tag}"
    subject_output_dir = args.output_dir / args.subject_id

    seg_img = load_segmentation(args.seg_template)
    seg_data = np.asarray(seg_img.dataobj, dtype=np.int32)
    lut_df = load_lut(args.lut)

    save_roi_saliency(signed_map, abs_map, feature_names, subject_output_dir, prefix, seg_img, seg_data, lut_df)
    print(f"Saved saliency ({prefix}*) to {subject_output_dir}")

    def report_value(raw_value):
        """See calculate_gradcam.py's identical closure: unchanged for a regression target, sigmoid'd for classification."""
        if raw_value is None or target_spec.task_type != "classification":
            return raw_value
        return float(1.0 / (1.0 + np.exp(-raw_value)))

    # Always written, even if no --randomize-* flag was passed ("tests" is
    # just empty then) -- matches calculate_gradcam.py's own summary
    # (subject_id, true/predicted value, tests: []), so bin_saliency.py's
    # load_meta can read fs_roi's true/predicted value for age/error binning
    # exactly like every other model (see visualize_bins.sh) -- previously
    # fs_roi had no such JSON at all. Written to subject_output_dir (the
    # per-subject baseline location) -- NOT args.output_dir directly -- so
    # visualize_subjects.sh's per-subject JSON glob (`"$SUBJECT_DIR/${JSON_STEM}"*`)
    # finds it the same way it already finds every other model's.
    sanity_summary = {
        "model": "fs_roi",
        "subject_id": args.subject_id,
        "checkpoint_path": str(args.checkpoint_path),
        "randomize_seed": args.randomize_seed,
        "attribution_method": args.attribution_method,
        "ig_baseline": args.ig_baseline if args.attribution_method == "integrated_gradients" else None,
        "ig_steps": args.ig_steps if args.attribution_method == "integrated_gradients" else None,
        "target": target_spec.name,
        target_spec.json_true_key: true_value,
        target_spec.json_pred_key: reported_predicted_value,
        "integrated_gradients_completeness": diagnostics,
        "tests": [],
    }

    # ---- Sanity checks (Adebayo et al., 2018) -- see calculate_gradcam.py's
    # --randomize-input/--randomize-data/--randomize-model for the image-
    # model versions this mirrors; only make_noise_input_tabular is genuinely
    # new here (fit_ridge_head_inmemory/randomize_all_pipeline/
    # cascading_randomize_pipeline/report_similarity are reused as-is, since
    # they're already generic over any BrainAgePipeline/feature-cache pair --
    # see this module's docstring for why the (C, H, W, D) image code path
    # itself doesn't generalize, but these particular helper functions do).
    # Written directly to args.output_dir (NOT subject_output_dir) with
    # arch_/data_/input_ prefixes -- the shared reference-subject convention
    # every other model's sanity-check battery already uses (see
    # compute_sanity_checks.sh), so sanity_filenames()'s generic (model, tag,
    # kind) construction in bin_saliency.py/compute_saliency_correlations.py/
    # visualize_subjects.sh resolves fs_roi's files with zero special-casing.
    if args.randomize_input:
        print("\n" + "=" * 60)
        print("CONTROL: INPUT RANDOMIZATION (structure-free input, real trained pipeline)")
        print("=" * 60)
        noise_input = make_noise_input_tabular(input_tensor, dist=args.randomize_input_dist, seed=args.randomize_seed)
        noise_signed, noise_abs, noise_predicted, noise_diag = compute_tabular_saliency(
            pipeline, noise_input, device,
            attribution_method=args.attribution_method, baseline_tensor=baseline_tensor, ig_steps=args.ig_steps,
        )
        corr = report_similarity(signed_map, noise_signed, abs_map, noise_abs, f"{args.randomize_input_dist} input")
        input_prefix = f"input_fs_roi_{args.randomize_input_dist}_{method_tag}"
        save_roi_saliency(noise_signed, noise_abs, feature_names, args.output_dir, input_prefix, seg_img, seg_data, lut_df)
        sanity_summary["tests"].append(
            {
                "test": "randomize_input",
                "dist": args.randomize_input_dist,
                "prefix": input_prefix,
                target_spec.json_pred_key: report_value(noise_predicted),
                "integrated_gradients_completeness": noise_diag,
                "spearman_vs_baseline_signed": float(corr["signed"]),
                "spearman_vs_baseline_abs": float(corr["abs"]),
            }
        )

    if args.randomize_data:
        print("\n" + "=" * 60)
        print("SANITY CHECK: DATA RANDOMIZATION TEST (Adebayo et al., 2018)")
        print("=" * 60)
        x_path = args.features_dir / f"{args.cache_name}_X_features_raw.npy"
        y_path = args.features_dir / f"{args.cache_name}_y_{target_spec.cache_suffix}.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"--randomize-data needs cached features from train_linear.py: expected {x_path} and {y_path}."
            )
        X = np.load(x_path)
        y = np.load(y_path)
        seeds = [args.randomize_seed + i for i in range(args.randomize_data_seeds)]
        signed_corrs, abs_corrs = [], []
        for seed_idx, seed in enumerate(seeds):
            print(f"Refitting ridge head on {args.cache_name} features with {len(y)} labels shuffled (seed={seed})...")
            noise_head, noise_scaler, ridge_diag = fit_ridge_head_inmemory(X, y, cov_mask, seed=seed, target_spec=target_spec)
            noise_pipeline = BrainAgePipeline(
                backbone=backbone, head=noise_head, cov_mask=cov_mask,
                scaler_mean=noise_scaler.mean_, scaler_scale=noise_scaler.scale_,
            ).to(device).eval()
            noise_signed, noise_abs, noise_predicted, noise_diag = compute_tabular_saliency(
                noise_pipeline, input_tensor, device,
                attribution_method=args.attribution_method, baseline_tensor=baseline_tensor, ig_steps=args.ig_steps,
            )
            corr = report_similarity(signed_map, noise_signed, abs_map, noise_abs, f"data-randomized head [seed={seed}]")
            signed_corrs.append(corr["signed"])
            abs_corrs.append(corr["abs"])
            data_prefix = (
                f"data_fs_roi_{method_tag}" if seed_idx == 0 else f"data_seed{seed}_fs_roi_{method_tag}"
            )
            save_roi_saliency(noise_signed, noise_abs, feature_names, args.output_dir, data_prefix, seg_img, seg_data, lut_df)
            sanity_summary["tests"].append(
                {
                    "test": "randomize_data",
                    "seed": seed,
                    "prefix": data_prefix,
                    "cache_name": args.cache_name,
                    "n_labels_shuffled": int(len(y)),
                    target_spec.json_pred_key: report_value(noise_predicted),
                    "integrated_gradients_completeness": noise_diag,
                    "spearman_vs_baseline_signed": float(corr["signed"]),
                    "spearman_vs_baseline_abs": float(corr["abs"]),
                    **ridge_diag,
                }
            )
        if len(seeds) > 1:
            signed_arr, abs_arr = np.array(signed_corrs), np.array(abs_corrs)
            sanity_summary["tests"].append(
                {
                    "test": "randomize_data_aggregate",
                    "seeds": seeds,
                    "spearman_signed_mean": float(signed_arr.mean()),
                    "spearman_signed_std": float(signed_arr.std()),
                    "spearman_abs_mean": float(abs_arr.mean()),
                    "spearman_abs_std": float(abs_arr.std()),
                }
            )

    if args.randomize_model:
        print("\n" + "=" * 60)
        print(f"SANITY CHECK: MODEL PARAMETER RANDOMIZATION TEST (mode: {args.randomize_model_mode})")
        print("=" * 60)
        if args.randomize_model_mode == "all":
            reset_names, randomized_pipeline = randomize_all_pipeline(pipeline, "fs_roi", seed=args.randomize_seed)
            print(f"  reinitialized {len(reset_names)} module(s): {reset_names}")
            all_signed, all_abs, all_predicted, all_diag = compute_tabular_saliency(
                randomized_pipeline, input_tensor, device,
                attribution_method=args.attribution_method, baseline_tensor=baseline_tensor, ig_steps=args.ig_steps,
            )
            corr = report_similarity(signed_map, all_signed, abs_map, all_abs, "all modules reset at once")
            arch_prefix = f"arch_fs_roi_all_{method_tag}"
            save_roi_saliency(all_signed, all_abs, feature_names, args.output_dir, arch_prefix, seg_img, seg_data, lut_df)
            sanity_summary["tests"].append(
                {
                    "test": "randomize_model",
                    "mode": "all",
                    "modules_reinitialized": reset_names,
                    "prefix": arch_prefix,
                    target_spec.json_pred_key: report_value(all_predicted),
                    "integrated_gradients_completeness": all_diag,
                    "spearman_vs_baseline_signed": float(corr["signed"]),
                    "spearman_vs_baseline_abs": float(corr["abs"]),
                }
            )
        else:  # cascade
            for step, module_name, randomized_pipeline in cascading_randomize_pipeline(
                pipeline, max_layers=args.randomize_max_layers, seed=args.randomize_seed
            ):
                print(f"  step {step}: reinitialized '{module_name}'")
                step_signed, step_abs, step_predicted, step_diag = compute_tabular_saliency(
                    randomized_pipeline, input_tensor, device,
                    attribution_method=args.attribution_method, baseline_tensor=baseline_tensor, ig_steps=args.ig_steps,
                )
                corr = report_similarity(signed_map, step_signed, abs_map, step_abs, f"step {step} ({module_name})")
                safe_module_name = module_name.replace(".", "-") or "pipeline"
                step_prefix = f"arch_fs_roi_step{step:02d}_{safe_module_name}_{method_tag}"
                # The consumer scripts' sanity_filenames() only ever look for
                # the "arch_fs_roi_all_..." shape (matching --randomize-model-
                # mode all), not a per-step cascade file -- this step file is
                # still written (for anyone inspecting the cascade by hand)
                # but won't be picked up by bin_saliency.py/visualize_subjects.sh.
                save_roi_saliency(step_signed, step_abs, feature_names, args.output_dir, step_prefix, seg_img, seg_data, lut_df)
                sanity_summary["tests"].append(
                    {
                        "test": "randomize_model",
                        "mode": "cascade",
                        "step": step,
                        "module_reinitialized": module_name,
                        "prefix": step_prefix,
                        target_spec.json_pred_key: report_value(step_predicted),
                        "integrated_gradients_completeness": step_diag,
                        "spearman_vs_baseline_signed": float(corr["signed"]),
                        "spearman_vs_baseline_abs": float(corr["abs"]),
                    }
                )

    summary_stem = "fs_roi" + (f"_{method_tag.rstrip('_')}" if method_tag else "")
    summary_path = subject_output_dir / f"{summary_stem}_sanity_checks.json"
    with open(summary_path, "w") as f:
        json.dump(sanity_summary, f, indent=2)
    print(f"Summary (true/predicted {target_spec.name}, correlations) saved to {summary_path}")


if __name__ == "__main__":
    main()
