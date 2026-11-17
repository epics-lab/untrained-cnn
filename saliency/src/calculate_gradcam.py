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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler

# Assuming your original script is accessible in the python path or local directory
# to load BrainAgePipeline, FeatureSpec, and the dataset/model factories.
from train_linear import (
    BrainAgePipeline,
    make_feature_spec,
    load_brainiac_vit,
    CNN3D_CovPool,
    load_swinbrain,
    load_simclr_encoder,
    load_anatcl_encoder,
    SwinBrainFeatureExtractor,
    AnatCLFoldAverager,
    signed_sqrt,
    DEFAULT_DEMOGRAPHICS,
)
from train_linear import parse_args as parse_args_train  # To get default args for the pipeline
from targets import DEFAULT_PARTICIPANTS_TSV, get_target_spec, merge_bmi

class _AgeEstimator:
    pass


# Channel order matches UnCNNDataset.__getitem__'s np.stack([...]) in
# train_linear.py: 0=raw robust-scaled intensity, 1=rank filter, 2=Sobel
# gradient magnitude. Used only for optim_uncnn -- every other model
# produces a single (unlabeled) map, see compute_direct_pipeline_saliency.
UNCNN_CHANNELS = ("raw", "rank", "sobel")

# Not imported from compute_reference_baselines.py: that module imports
# UNCNN_CHANNELS from *this* one, so importing anything back from it here
# would be circular. Recomputed the same way train_linear.py's REPO_ROOT is.
# The "nki" subfolder matches slurm/compute_reference_baselines.sh's
# --output-dir for NKI (PPMI/HBN get their own reference_baselines/{ppmi,hbn}
# sibling subfolders -- pass --reference-baselines-dir explicitly for those).
DEFAULT_REFERENCE_BASELINES_DIR = Path(__file__).resolve().parents[1] / "reference_baselines" / "nki"


def load_reference_baseline(model_name, reference_dir, registration="affine"):
    """
    Loads the population-mean reference tensor for `model_name` (see
    compute_reference_baselines.py), in that model's own preprocessed
    (C, H, W, D) shape -- for use as an Integrated Gradients baseline.

    `model_name` here is always the flat model identifier (e.g. "anatcl",
    never "anatcl_global"/"anatcl_local" -- see args_saliency.model's
    choices in main()), matching compute_reference_baselines.py's own
    DEFAULT_MODELS naming exactly, so no anatcl-level special-casing is
    needed here (anatcl_global/anatcl_local share one anatcl_reference.npy
    because their preprocessing -- unlike their encoders -- is identical).

    `registration` ("rigid" or "affine", matching compute_reference_
    baselines.py's own --registration) selects which of that script's
    filename-prefixed outputs to load. Every caller here hardwires the
    "affine" default since none of them expose a rigid/affine switch of
    their own yet -- pass it through explicitly once one does.

    optim_uncnn is the one exception: its 3 channels (raw/rank/sobel) were
    saved as 3 separate single-channel files (they're heterogeneous, not
    poolable -- same reasoning as its saliency maps), so they're loaded and
    restacked into one (3, H, W, D) array here.
    """
    prefix = f"{registration}_"
    if model_name == "optim_uncnn":
        arrays = []
        for channel in UNCNN_CHANNELS:
            p = reference_dir / f"{prefix}optim_uncnn_{channel}_reference.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing reference baseline: {p}. Run `python3 compute_reference_baselines.py "
                    f"--registration {registration} --models optim_uncnn` first, or pass --ig-baseline zero."
                )
            arrays.append(np.load(p))
        return np.stack(arrays, axis=0)  # (3, H, W, D)

    p = reference_dir / f"{prefix}{model_name}_reference.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing reference baseline: {p}. Run `python3 compute_reference_baselines.py "
            f"--registration {registration} --models {model_name}` first, or pass --ig-baseline zero."
        )
    return np.load(p)




def audit_gradient_backpropagation(pipeline: nn.Module, input_tensor: torch.Tensor, device: str = "cuda"):
    """
    Diagnostics suite that prints out gradient statistics at every structural stage
    of the BrainAgePipeline as the backward pass flows from the linear head to the voxels.
    """
    pipeline.to(device)
    pipeline.eval()

    # 1. Enforce gradient tracking on input
    input_tensor = input_tensor.to(device).unsqueeze(0)
    input_tensor.requires_grad = True

    # Storage dictionary to harvest backprop info dynamically
    grad_audit_trail = {}

    # Generic hook factory that captures gradient statistics on the fly
    def make_hook(stage_name):
        def hook_fn(grad):
            if grad is None:
                grad_audit_trail[stage_name] = "✘ DETACHED / NO GRADIENT RECEIVED"
            else:
                grad_audit_trail[stage_name] = {
                    "shape": list(grad.shape),
                    "max_abs": float(torch.max(torch.abs(grad)).cpu().item()),
                    "mean_abs": float(torch.mean(torch.abs(grad)).cpu().item()),
                    "has_nan": bool(torch.isnan(grad).any().cpu().item())
                }
            return grad
        return hook_fn

    print("\n" + "="*60)
    print("STAGE 1: EXECUTING INSTRUMENTED FORWARD PASS")
    print("="*60)

    # 2. Reconstruct forward execution trace step-by-step to intercept intermediate tensors
    # We copy the logic inside BrainAgePipeline but explicitly register tensor hooks

    # Step A: Backbone Feature Maps
    feat_raw = pipeline.backbone(input_tensor)
    if isinstance(feat_raw, tuple):
        feat_raw[0].register_hook(make_hook("1. Raw Backbone Feature Output"))
    else:
        feat_raw.register_hook(make_hook("1. Raw Backbone Feature Output"))

    # Step B: Model-specific Post-processing (e.g., CLS token isolation for BrainIAC)
    if pipeline.postprocess_fn is not None:
        feat_proc = pipeline.postprocess_fn(feat_raw)
        feat_proc.register_hook(make_hook("2. Postprocessed Feature Output"))
    else:
        feat_proc = feat_raw

    if isinstance(feat_proc, tuple):
        feat_proc = feat_proc[0]

    feat_flat = torch.flatten(feat_proc, start_dim=1)
    feat_flat.register_hook(make_hook("3. Flattened Embedding Vector"))

    # Step C: Non-linear Covariance Masking (mirrors BrainAgePipeline.forward's
    # sqrt-input guard -- see its docstring for why the input to sqrt needs
    # sanitizing before the where(), not just its output)
    cov_safe_input = torch.where(pipeline.cov_mask, feat_flat, torch.ones_like(feat_flat))
    sqrt_feat = pipeline._signed_sqrt(cov_safe_input)
    feat_trans = torch.where(pipeline.cov_mask, sqrt_feat, feat_flat)
    feat_trans.register_hook(make_hook("4. Non-Linear Masking Layer (where/signed_sqrt)"))

    # Step D: StandardScaler Z-Score Transformation
    feat_scaled = (feat_trans - pipeline.scaler_mean) / pipeline.scaler_scale
    feat_scaled.register_hook(make_hook("5. StandardScaler Layer"))

    # Step E: Regression Readout
    age_prediction = pipeline.head(feat_scaled)
    age_prediction.register_hook(make_hook("6. Linear Head Output (Age Prediction)"))

    print("✓ Forward pass tracked. All diagnostic anchors deployed successfully.")

    print("\n" + "="*60)
    print("STAGE 2: TRIGGERING BACKWARD PASS AUDIT TRAIL")
    print("="*60)

    # 3. Fire the backpropagation trigger
    pipeline.zero_grad()
    age_prediction.backward()

    # 4. Check the end-of-the-line input voxels manually
    if input_tensor.grad is None:
        grad_audit_trail["7. Raw Input Voxels (Final Destination)"] = "✘ DETACHED / NO GRADIENT RECEIVED"
    else:
        grad_audit_trail["7. Raw Input Voxels (Final Destination)"] = {
            "shape": list(input_tensor.grad.shape),
            "max_abs": float(torch.max(torch.abs(input_tensor.grad)).cpu().item()),
            "mean_abs": float(torch.mean(torch.abs(input_tensor.grad)).cpu().item()),
            "has_nan": bool(torch.isnan(input_tensor.grad).any().cpu().item())
        }

    # 5. Output summary metrics cleanly for visual inspection
    stages_sorted = sorted(grad_audit_trail.keys())
    for stage in stages_sorted:
        metrics = grad_audit_trail[stage]
        print(f"\n[Stage] {stage}")
        if isinstance(metrics, str):
            print(f"  {metrics}")
        else:
            print(f"  -> Tensor Shape: {metrics['shape']}")
            print(f"  -> Max Abs Gradient Magnitude: {metrics['max_abs']:.6e}")
            print(f"  -> Mean Abs Gradient Magnitude: {metrics['mean_abs']:.6e}")
            if metrics['has_nan']:
                print("  ⚠️ WARNING: NaN values detected inside this gradient matrix!")
            elif metrics['max_abs'] == 0.0:
                print("  ⚠️ WARNING: Gradient is exactly 0.0 (Dead Path / Vanished Gradient).")
            else:
                print("  ✓ Healthy gradient flow confirmed.")

    print("="*60 + "\n")

def _integrated_gradients_and_prediction(pipeline, input_tensor, baseline_tensor, device, steps):
    """
    Riemann-sum approximation of Integrated Gradients (Sundararajan et al.,
    2017): IG_v = (x_v - x0_v) * mean over `steps` points alpha=k/steps,
    k=1..steps, of d(age)/d(x_v) evaluated at x0 + alpha*(x - x0).

    Unlike "gradient"/"gradient_x_input" (one forward+backward pass total),
    this runs `steps` forward+backward passes -- one per interpolation
    point -- since that's what integrating along the path actually
    requires. Costs ~`steps`x more than the other two methods; tune
    --ig-steps down for faster iteration, at the cost of a coarser Riemann
    approximation.

    `input_tensor`/`baseline_tensor` are both (C, H, W, D), in the model's
    own preprocessed space (the same space the reference baselines in
    reference_baselines/ were computed in). predicted_age is captured at
    the final step (alpha=1.0, i.e. exactly the real input, no interpolation)
    so it matches what "gradient"/"gradient_x_input" report for the same
    input. The baseline prediction and finite-step completeness residual are
    also returned so convergence can be checked rather than assuming the
    theoretical exact-integral identity holds for this numerical sum.
    """
    if steps < 1:
        raise ValueError(f"Integrated Gradients requires steps >= 1, got {steps}")

    pipeline.to(device)
    pipeline.eval()

    x = input_tensor.to(device).unsqueeze(0).double()  # (1, C, H, W, D)
    x0 = baseline_tensor.to(device).unsqueeze(0).double()
    if x.shape != x0.shape:
        raise ValueError(f"Baseline shape {tuple(x0.shape)} does not match input shape {tuple(x.shape)}")
    delta = x - x0

    grad_sum = torch.zeros_like(x)
    predicted_age = None
    for k in range(1, steps + 1):
        alpha = k / steps
        interpolated = (x0 + alpha * delta).float()
        interpolated.requires_grad_(True)
        pipeline.zero_grad()
        age_prediction = pipeline(interpolated)
        if k == steps:
            predicted_age = float(age_prediction.detach().cpu().item())
        age_prediction.backward()
        if interpolated.grad is None:
            raise RuntimeError(
                f"No gradient reached the interpolated input at IG step k={k}/{steps}. "
                "Check that the pipeline does not detach or use torch.no_grad() in its forward path."
            )
        grad_sum += interpolated.grad.double()

    avg_grad = grad_sum / steps
    voxel_attribution_tensor = delta * avg_grad

    # Completeness is a property of the attribution over every model-input
    # channel and voxel. Compute it here, before repeated-channel collapse and
    # before restore_native_space changes the grid (and therefore the raw
    # voxel sum). The exact path integral satisfies a zero residual; this
    # finite right-endpoint Riemann sum generally only approximates it.
    with torch.no_grad():
        baseline_prediction = float(pipeline(x0.float()).detach().cpu().item())
    total_attribution = float(voxel_attribution_tensor.sum().detach().cpu().item())
    prediction_delta = predicted_age - baseline_prediction
    completeness_error = prediction_delta - total_attribution
    completeness = {
        "baseline_prediction": baseline_prediction,
        "prediction_delta": prediction_delta,
        "total_attribution": total_attribution,
        "completeness_error": completeness_error,
        "relative_completeness_error": abs(completeness_error) / (abs(prediction_delta) + 1e-12),
    }

    voxel_attribution = voxel_attribution_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # (C, H, W, D)
    return voxel_attribution, predicted_age, completeness


def compute_direct_pipeline_saliency(
    pipeline: nn.Module,
    input_tensor: torch.Tensor,
    model_name: str,
    device: str = "cuda",
    attribution_method: str = "gradient",
    baseline_tensor: torch.Tensor = None,
    ig_steps: int = 50,
    return_diagnostics: bool = False,
):
    """
    Computes voxel-level saliency maps by backpropagating directly from the linear head
    output through the standardization, non-linear masking, and 3D backbone.

    attribution_method:
      - "gradient" (default): raw voxelwise input-gradient sensitivity,
        S_v = d(age)/d(x_v) -- a *sensitivity* map, not a contribution map
        (see docs/known_limitations.md / the module's saliency-magnitude
        discussion). This is the method every caller used before
        --attribution-method existed; its behavior/output is unchanged.
      - "gradient_x_input": A_v = x_v * d(age)/d(x_v), the "gradient times
        input" attribution. For a linear model this exactly decomposes the
        non-bias part of the prediction (sum_v A_v == age - F(0)); for a
        nonlinear CNN it's only a local approximation, not an exact
        decomposition -- see docs/known_limitations.md. x_v is read from
        the same (preprocessed) input_tensor the gradient itself was taken
        with respect to, not raw pre-normalization image intensity.
      - "integrated_gradients": IG_v = (x_v - x0_v) * path-integral of
        d(age)/d(x_v) between a baseline x0 and x (Sundararajan et al.,
        2017); see _integrated_gradients_and_prediction. Requires
        `baseline_tensor` (same shape as input_tensor, in the model's own
        preprocessed space -- either an all-zero tensor or a population-mean
        reference from compute_reference_baselines.py/load_reference_baseline,
        see the literature review in docs/known_limitations.md for why the
        latter is generally better motivated than zero for the z-scored/
        robust-scaled models here).

    Returns (channel_maps, predicted_age), or
    (channel_maps, predicted_age, diagnostics) when return_diagnostics=True:
      - predicted_age: single float, the scalar prediction at the real input
        (the final IG path point for Integrated Gradients) -- shared across
        every channel_maps entry below, never duplicated per channel.
      - channel_maps: ordered list of dicts {"channel", "signed_map", "abs_map"}.
        For every model except optim_uncnn this is a single-entry list with
        "channel": None (identical values to what this function always
        returned before). For optim_uncnn it's 3 entries, one per
        UNCNN_CHANNELS, each with its own signed_map/abs_map.
      - diagnostics: None for gradient/gradient_x_input; for Integrated
        Gradients, predictions and the finite-step completeness residual,
        computed over all channels before any channel reduction.
    """
    if attribution_method not in ("gradient", "gradient_x_input", "integrated_gradients"):
        raise ValueError(f"Unknown attribution_method: {attribution_method!r}")
    if attribution_method == "integrated_gradients" and baseline_tensor is None:
        raise ValueError("attribution_method='integrated_gradients' requires baseline_tensor")

    pipeline.to(device)
    pipeline.eval()

    diagnostics = None
    if attribution_method == "integrated_gradients":
        voxel_gradients, predicted_age, diagnostics = _integrated_gradients_and_prediction(
            pipeline, input_tensor, baseline_tensor, device, steps=ig_steps
        )
    else:
        # Ensure the input volume tracks gradients.  detach() guarantees this is
        # a fresh leaf tensor (no prior grad_fn from .to()/.unsqueeze()), and
        # requires_grad_(True) enables gradient accumulation during backward().
        input_tensor = input_tensor.to(device).unsqueeze(0).detach()  # (1, C, H, W, D)
        input_tensor.requires_grad_(True)

        # 1. Clear any leftover gradients
        pipeline.zero_grad()

        # 2. Forward pass directly to the scalar age prediction
        # This automatically includes backbone + postprocess + signed_sqrt + scaler + linear head!
        age_prediction = pipeline(input_tensor)  # Shape: (1, 1)
        predicted_age = float(age_prediction.detach().cpu().item())

        # 3. Backward pass directly from the final prediction
        age_prediction.backward()

        # 4. Extract raw gradients from the input tensor image
        if input_tensor.grad is None:
            raise RuntimeError(
                "No gradient reached the input tensor after backward(). "
                "Check that the pipeline does not detach or use torch.no_grad() in its forward path."
            )
        voxel_gradients = input_tensor.grad.squeeze(0).cpu().numpy()  # Shape: (C, H, W, D)

        if attribution_method == "gradient_x_input":
            # x_v, in the same (preprocessed) space the gradient was computed in
            # -- not raw pre-normalization intensity. Elementwise multiply here,
            # before the channel-collapse/selection logic below, so every
            # downstream branch (optim_uncnn's per-channel split, the
            # repeated-channel sum, the single-channel passthrough) operates on
            # the attribution values exactly the way it already operates on raw
            # gradients -- no separate code path needed per model.
            input_values = input_tensor.detach().squeeze(0).cpu().numpy()  # Shape: (C, H, W, D)
            voxel_gradients = voxel_gradients * input_values

    # From here on, `voxel_gradients` holds whichever attribution method's
    # (C, H, W, D) array was computed above -- the per-model channel
    # handling below is identical regardless of which method produced it.
    if model_name == "optim_uncnn":
        # UnCNNDataset.__getitem__ (train_linear.py) stacks 3 *heterogeneous*
        # channels -- raw intensity, rank_filter(raw), and Sobel-magnitude(raw)
        # -- built with plain numpy/scipy calls outside the autograd graph.
        # input_tensor.requires_grad is only set afterward, on the already-
        # 3-channel tensor, so the single backward() above still produces a
        # real, valid gradient for all 3 channel slots at once -- no extra
        # forward/backward pass is needed. But channels 1/2's gradients are
        # d(age)/d(rank-filter value) and d(age)/d(Sobel value), treating
        # those already-computed values as their own independent inputs --
        # NOT d(age)/d(raw intensity) chained back through those nonlinear,
        # non-PyTorch transforms. That chain is genuinely unavailable (the
        # filters themselves were never in the traced graph), so instead of
        # collapsing all 3 into one map (which would mix unrelated units and
        # let opposite-signed gradients cancel) or keeping only channel 0,
        # each channel gets its own separately-labeled map: valid as a
        # sensitivity map in its own channel's units, but only channel 0's
        # map is interpretable as sensitivity to raw voxel intensity. See
        # docs/known_limitations.md.
        if voxel_gradients.shape[0] != len(UNCNN_CHANNELS):
            raise ValueError(
                f"optim_uncnn expects {len(UNCNN_CHANNELS)} input channels "
                f"{UNCNN_CHANNELS}, got attribution shape {voxel_gradients.shape}"
            )
        channel_maps = []
        for idx, channel in enumerate(UNCNN_CHANNELS):
            signed_map = voxel_gradients[idx]
            channel_maps.append({"channel": channel, "signed_map": signed_map, "abs_map": np.abs(signed_map)})
        result = (channel_maps, predicted_age)
        return (*result, diagnostics) if return_diagnostics else result

    if voxel_gradients.shape[0] > 1:
        # Other multi-channel models (e.g. swinbrain) repeat the *same* raw
        # image across channels. If X_c = x, the derivative/attribution with
        # respect to the original image is the chain-rule sum over c, not the
        # mean. Averaging preserves a normalized picture but divides physical
        # magnitudes, ROI totals, and IG completeness by the channel count.
        if model_name != "swinbrain":
            raise ValueError(
                f"Unexpected {voxel_gradients.shape[0]}-channel attribution for model "
                f"{model_name!r}; only swinbrain's repeated channels may be collapsed"
            )
        signed_map = np.sum(voxel_gradients, axis=0)
    else:
        signed_map = voxel_gradients[0]

    abs_map = np.abs(signed_map)
    result = ([{"channel": None, "signed_map": signed_map, "abs_map": abs_map}], predicted_age)
    return (*result, diagnostics) if return_diagnostics else result


def _undo_turboprep_transform(arr, ref_shape, target_shape=(150, 192, 192)):
    """
    Inverts TurboPrepDataset's forward transform (used by --model simclr):
    np.transpose(img, (2, 0, 1)) followed by a center crop to `target_shape`.

    A naive shape-matching zoom (as used for the other models) is wrong here:
    it neither restores the original axis order nor the true crop offset, so
    values from a small central patch get smeared across the full volume on
    the wrong axes. Instead we place the (cropped) map back at its exact
    offset inside a zero-padded volume in transposed space, then invert the
    transpose exactly -- no interpolation/resampling needed since transpose
    and crop are both shape-preserving, lossless-placement operations.
    """
    transposed_shape = (ref_shape[2], ref_shape[0], ref_shape[1])
    starts = [(s - t) // 2 for s, t in zip(transposed_shape, target_shape)]
    padded = np.zeros(transposed_shape, dtype=arr.dtype)
    padded[
        starts[0] : starts[0] + target_shape[0],
        starts[1] : starts[1] + target_shape[1],
        starts[2] : starts[2] + target_shape[2],
    ] = arr
    # transpose(2, 0, 1) is a 3-cycle; its inverse is transpose(1, 2, 0).
    return np.transpose(padded, (1, 2, 0))


def _nifti_with_qform(data, affine):
    """
    nib.Nifti1Image(data, affine) only populates the sform (sform_code=2);
    it leaves qform_code=0 ("unknown"). nibabel/nilearn-based code (this
    pipeline's own visualize_saliency.py included) correctly falls back to
    the sform, but several manual-inspection NIfTI viewers prefer/require a
    valid qform for registering two images against each other, and can
    silently display an unregistered/misplaced volume when qform_code=0 --
    manifesting as saliency and template appearing spatially offset even
    though the sform-based affine is correct. Set qform = sform explicitly
    so both code paths agree.
    """
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=2)
    return img


def restore_native_space(signed_map, abs_map, model_name, ref_shape, ref_affine):
    """
    Maps a saliency map computed in a model's resized/cropped/transposed
    input space back to a NIfTI affine that's spatially correct relative to
    the reference image loaded for that subject -- WITHOUT necessarily
    resampling the array up to `ref_shape` first. Returns
    (signed_map, abs_map, affine); the caller saves with this affine, not
    always `ref_affine`.

    - simclr (TurboPrepDataset): transposes axes and center-crops, both
      exactly invertible with no interpolation, so `_undo_turboprep_transform`
      already places the array losslessly onto `ref_shape`'s own grid --
      `ref_affine` is correct unchanged.
    - optim_uncnn / brainiac / swinbrain / anatcl: their datasets only
      resize (no crop, no transpose). Resizing back up to `ref_shape` with a
      second `ndi.zoom` here -- on top of whatever resize the forward
      Dataset class already did to shrink the image down to the model's
      input size -- is a redundant, lossy SECOND resample stacked before
      visualize_saliency.py's own (unavoidable) resample onto the display
      template. Two lossy resamples in a row measurably shrinks the visible
      saliency extent (confirmed empirically: up to ~17% bounding-box
      shrinkage for swinbrain, ~10% for optim_uncnn, from resampling alone,
      with no model/gradient computation involved -- see
      docs/known_limitations.md).

      Instead: keep the array at its own small native shape, and rescale
      the affine's spatial (rotation+scaling) block by the ratio of voxel
      sizes -- each voxel in the small array spans proportionally more
      physical space than a `ref_shape` voxel, by exactly the factor the
      forward Dataset shrank it by. The translation (world-space origin) is
      left unchanged: this is a corner-anchored rescale, matching how the
      forward zoom placed the small grid within the same physical
      field-of-view as the original image, rather than cropping any of it
      away. visualize_saliency.py's `resample_to_img` (nilearn) then does
      the one resample that's actually necessary -- straight from this
      small native grid onto the display template -- instead of
      small-native -> ref_shape -> template.
    """
    if model_name == "simclr":
        return (
            _undo_turboprep_transform(signed_map, ref_shape),
            _undo_turboprep_transform(abs_map, ref_shape),
            ref_affine,
        )

    if signed_map.shape == tuple(ref_shape):
        return signed_map, abs_map, ref_affine

    scale_factors = [r / s for r, s in zip(ref_shape, signed_map.shape)]
    affine = ref_affine.copy()
    affine[:3, :3] = ref_affine[:3, :3] @ np.diag(scale_factors)
    return signed_map, abs_map, affine


def attribution_method_tag(attribution_method, ig_baseline=None):
    """
    Filename-safe tag inserted into saved .nii.gz filenames and the summary
    JSON's filename, to distinguish attribution methods (and, for
    integrated_gradients, baseline choices) that would otherwise silently
    overwrite each other's output for the same subject/model. "" for the
    default "gradient" method, so its filenames stay byte-identical to
    before --attribution-method existed. Shared by compute_and_save_saliency
    and main()'s JSON-prefix bookkeeping so the two can't drift apart (the
    same class of bug fixed earlier between the optim_uncnn channel split
    and its JSON "prefix" field).
    """
    if attribution_method == "gradient":
        return ""
    if attribution_method == "gradient_x_input":
        return "gxi_"
    if attribution_method == "integrated_gradients":
        if ig_baseline not in ("zero", "reference"):
            raise ValueError(f"Unknown ig_baseline: {ig_baseline!r}")
        return f"ig_{ig_baseline}_"
    raise ValueError(f"Unknown attribution_method: {attribution_method!r}")


def compute_and_save_saliency(
    pipeline,
    input_tensor,
    model_name,
    ref_img,
    output_dir,
    prefix,
    device,
    attribution_method="gradient",
    baseline_tensor=None,
    ig_steps=50,
    ig_baseline=None,
    return_diagnostics=False,
):
    """
    Compute saliency for `pipeline`, map each channel's map back to `ref_img`'s
    native space, and save it. `predicted_age` is a single float, shared by
    every channel (one scalar prediction, regardless of channel count).

    attribution_method: see compute_direct_pipeline_saliency. Only affects
    the saved filenames when non-default -- see attribution_method_tag
    (e.g. "gradient_x_input" inserts "gxi_", "integrated_gradients" inserts
    "ig_zero_"/"ig_reference_" depending on ig_baseline, before "signed"/
    "abs": "optim_uncnn_raw_ig_reference_signed_saliency.nii.gz"). The
    default "gradient" method's filenames are completely unchanged from
    before this parameter existed, so existing downstream tooling
    (aggregate_saliency.py, bin_saliency.py, the visualize_*.sh scripts)
    keeps working without modification; it simply never sees the other
    methods' output files unless/until it's taught to.

    baseline_tensor/ig_steps: only used when attribution_method is
    "integrated_gradients" -- see compute_direct_pipeline_saliency. ig_baseline
    is only used to build the filename tag (must be "zero" or "reference",
    matching whichever baseline_tensor was actually passed).

    Returns (channel_results, predicted_age), or adds a third diagnostics
    item when return_diagnostics=True, where channel_results is
    {channel or "": (signed_map, abs_map)} -- for every model except
    optim_uncnn this dict has exactly one key ("" -- see
    compute_direct_pipeline_saliency), so callers can use uniform code
    regardless of model.
    """
    channel_maps, predicted_age, diagnostics = compute_direct_pipeline_saliency(
        pipeline,
        input_tensor,
        model_name,
        device=device,
        attribution_method=attribution_method,
        baseline_tensor=baseline_tensor,
        ig_steps=ig_steps,
        return_diagnostics=True,
    )
    method_tag = attribution_method_tag(attribution_method, ig_baseline=ig_baseline)

    channel_results = {}
    for entry in channel_maps:
        channel = entry["channel"]
        signed_map, abs_map, affine = restore_native_space(
            entry["signed_map"], entry["abs_map"], model_name, ref_img.shape, ref_img.affine
        )
        channel_prefix = f"{prefix}{channel}_{method_tag}" if channel else f"{prefix}{method_tag}"
        nib.save(_nifti_with_qform(signed_map, affine), output_dir / f"{channel_prefix}signed_saliency.nii.gz")
        nib.save(_nifti_with_qform(abs_map, affine), output_dir / f"{channel_prefix}abs_saliency.nii.gz")
        channel_results[channel or ""] = (signed_map, abs_map)

    result = (channel_results, predicted_age)
    return (*result, diagnostics) if return_diagnostics else result


def report_similarity(baseline_signed, candidate_signed, baseline_abs, candidate_abs, label):
    """
    Spearman rank correlation between a baseline saliency map and a candidate
    (randomized-model / randomized-data) map -- the correlation metric used
    by Adebayo et al. (2018) to score how much a saliency method's output
    changes under the sanity-check randomizations. A method that's actually
    explaining the *trained* model should decorrelate (drop toward 0) once
    the model or its training labels are randomized; a method that stays
    highly correlated with the baseline is failing the sanity check -- its
    explanation isn't sensitive to what the model actually learned.

    Reports BOTH the signed and the absolute-value correlation, not just
    signed -- Adebayo et al.'s own headline finding is that these two diverge:
    reinitialization/relabeling disrupts the *sign* of the map almost
    immediately (signed Spearman -> 0 fast), while the underlying input
    structure a method attends to often survives (abs-value Spearman and SSIM
    stay high) -- which is exactly the retained-structure failure mode that
    matters for gradient_x_input/integrated_gradients here. Reporting signed
    correlation alone can read as "passing" the sanity check for the wrong
    reason (sign decorrelation) while missing that failure mode entirely.
    """
    corr_signed, _ = spearmanr(baseline_signed.ravel(), candidate_signed.ravel())
    corr_abs, _ = spearmanr(baseline_abs.ravel(), candidate_abs.ravel())
    print(
        f"    Spearman rank correlation vs. baseline ({label}): "
        f"signed={corr_signed:.4f}, abs={corr_abs:.4f}"
    )
    return {"signed": corr_signed, "abs": corr_abs}


def cascading_randomize_pipeline(pipeline, max_layers=None, seed=0):
    """
    Model parameter randomization test (Adebayo et al., 2018): progressively
    re-initializes the pipeline's parameterized modules IN PLACE, in reverse
    registration order -- output-adjacent modules first (the head), then the
    backbone from its last-defined layers back toward its first -- and yields
    the pipeline after each additional module is randomized. Randomization is
    cumulative/cascading: earlier steps' randomized weights are kept as later
    modules are added, exactly as in the paper's cascading variant.

    This mutates `pipeline` destructively (each `reset_parameters()` call is
    in place, and there is no un-randomizing it), so only call this after any
    saliency computation you still need from the pristine/trained weights.

    named_modules() order approximates architectural depth via registration
    order; it isn't a guaranteed exact forward-pass depth ordering for every
    possible architecture (e.g. non-sequential attention/skip connections),
    but it's the standard practical approximation and holds well for the
    Linear/Conv3d/Norm-based backbones used here.
    """
    torch.manual_seed(seed)
    resettable = [(name, module) for name, module in pipeline.named_modules() if hasattr(module, "reset_parameters")]
    resettable.reverse()  # head (output side) first, early backbone layers (input side) last
    if max_layers is not None:
        resettable = resettable[:max_layers]

    for step, (name, module) in enumerate(resettable, start=1):
        module.reset_parameters()
        yield step, name, pipeline


def randomize_all_pipeline(pipeline, model_name, seed=0):
    """
    Model parameter randomization test (Adebayo et al., 2018), "all at once"
    variant: re-initializes every parameterized module IN PLACE in a single
    shot -- one fully-randomized pipeline compared once against the baseline
    -- rather than cascading module by module and comparing at every step.

    Exception: for optim_uncnn, only the head is reset. Its backbone is
    randomly initialized and frozen -- it is never trained -- so resetting it
    again just re-randomizes weights that were already random and tests
    nothing; the head is the only module with anything "learned" to
    randomize away.

    Mutates `pipeline` destructively, same caveat as cascading_randomize_pipeline.
    """
    torch.manual_seed(seed)
    if model_name == "optim_uncnn":
        candidates = [(name, module) for name, module in pipeline.named_modules() if name == "head" or name.startswith("head.")]
    else:
        candidates = list(pipeline.named_modules())

    reset_names = []
    for name, module in candidates:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
            reset_names.append(name)
    return reset_names, pipeline


_NKI_SUBJECT_ID_RE = re.compile(r"A\d{8}")


def lookup_true_value(subject_nii, demographics_path, target_spec, participants_tsv_path=None):
    """
    Find the subject's ground-truth target value from the demographics file
    (BMI-joined via targets.merge_bmi when target_spec.name == "bmi"), by
    pulling an NKI-style subject ID (e.g. A00000000) out of --subject-nii's
    path -- this works whether the path is turboprep's
    <subject_id>/normalized.nii.gz or CAT12's
    sub-<subject_id>/.../mwp1sub-<subject_id>_....nii, since the bare ID
    appears as a literal substring in both conventions. Returns None (with a
    warning printed) if the ID can't be found or isn't in the file, rather
    than failing the whole run over a missing/renamed demographics entry.
    """
    match = _NKI_SUBJECT_ID_RE.search(str(subject_nii))
    if match is None:
        print(f"WARNING: could not find an NKI subject ID (A########) in {subject_nii}; true_{target_spec.name} will be null.")
        return None, None

    subject_id = match.group(0)
    if not demographics_path.exists():
        print(f"WARNING: demographics file not found at {demographics_path}; true_{target_spec.name} will be null.")
        return subject_id, None

    demographics = pd.read_csv(demographics_path, delimiter="\t", names=["Subject", "Sex", "Age"])
    demographics["Subject"] = demographics["Subject"].astype(str)
    if target_spec.name == "bmi":
        demographics = merge_bmi(demographics, participants_tsv_path)
    row = demographics.loc[demographics["Subject"] == subject_id]
    if row.empty:
        print(f"WARNING: subject {subject_id} not found in {demographics_path}; true_{target_spec.name} will be null.")
        return subject_id, None

    return subject_id, float(target_spec.encode(row[target_spec.demographics_column].iloc[0]))


def fit_ridge_head_inmemory(X, y, cov_mask, seed=0, cv=5, target_spec=None):
    """
    Data randomization test, adapted from Adebayo et al. (2018) to a frozen
    backbone: refit a RidgeCV head on (X, y) entirely in memory (no
    checkpoint I/O, unlike train_linear.py's train_ridge_head). Mirrors that
    function's fitting recipe -- signed-sqrt on covariance features,
    StandardScaler, RidgeCV -- so a head fit on shuffled labels is
    architecturally comparable to the real saved checkpoint. Only the head is
    refit; the frozen backbone is untouched and reused as-is.

    This is NOT literally the paper's procedure -- Adebayo et al. retrain the
    *entire* network on permuted labels, and the test's validity there rests
    on Zhang et al. (2017)'s memorization premise: an overparameterized deep
    net can and does fit >95% of the *randomized* training labels, becoming a
    genuinely different, committed function. Here the backbone never sees
    labels at all (it's pretrained/self-supervised, or for optim_uncnn a
    frozen random init) and the only label-dependent parameters are this
    linear ridge head. A cross-validated linear model CANNOT memorize a
    shuffled *continuous* target the way a deep net memorizes shuffled
    *discrete* labels -- there is no linear direction in the frozen feature
    space that predicts noise, so RidgeCV's own regularization search
    typically selects a large alpha and shrinks the fitted weights toward
    (near-)zero, i.e. toward a constant mean-age predictor, rather than toward
    a committed memorized-random-labels function.

    Practically, gradient-based saliency (linear in the head weights, see
    report_similarity) still decorrelates from the baseline under this
    adaptation, so the sanity check still "runs" -- but what a low
    correlation demonstrates is narrower than in the paper: sensitivity to
    the label-learned linear readout direction, not sensitivity to a
    memorized alternative function. See the diagnostics this function
    returns (alpha_, train R² on the shuffled labels, ||w_shuffled||) --
    logged by the --randomize-data caller into sanity_summary -- so a
    near-zero-weight collapse (the expected, uninteresting reason a
    correlation is low) is visible rather than silently read as "passing"
    for the paper's reason.
    """
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)

    X_proc = X.copy()
    if cov_mask is not None and np.asarray(cov_mask).any():
        X_proc[:, cov_mask] = signed_sqrt(X_proc[:, cov_mask])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)

    is_classification = target_spec is not None and target_spec.task_type == "classification"
    head = nn.Linear(X.shape[1], 1, bias=True)
    if not is_classification:
        reg = RidgeCV(alphas=np.logspace(-2, 5, 30), cv=cv)
        reg.fit(X_scaled, y_shuffled)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(reg.coef_, dtype=torch.float32).unsqueeze(0))
            head.bias.copy_(torch.tensor([reg.intercept_], dtype=torch.float32))
        diagnostics = {
            "ridge_alpha": float(reg.alpha_),
            "shuffled_label_train_r2": float(reg.score(X_scaled, y_shuffled)),
            "shuffled_weight_norm": float(np.linalg.norm(reg.coef_)),
        }
    else:
        reg = LogisticRegressionCV(Cs=np.logspace(-5, 5, 30), cv=cv, max_iter=2000)
        reg.fit(X_scaled, y_shuffled)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(reg.coef_, dtype=torch.float32))
            head.bias.copy_(torch.tensor(reg.intercept_, dtype=torch.float32))
        diagnostics = {
            "logreg_C": float(reg.C_[0]),
            "shuffled_label_train_accuracy": float(reg.score(X_scaled, y_shuffled)),
            "shuffled_weight_norm": float(np.linalg.norm(reg.coef_)),
        }
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    return head, scaler, diagnostics


def make_noise_input(input_tensor, dist="permute", seed=0):
    """
    Input randomization control (complementary to, and distinct from, the
    paper's Data Randomization Test -- see the module docstring): build a
    structure-free version of the real preprocessed input, with the SAME
    shape and per-channel intensity scale, so any saliency difference
    reflects the loss of spatial/anatomical structure rather than a change
    in input scale. Uses the real, correctly-trained pipeline -- nothing
    about the model or its training is touched.

    - "permute" (default): randomly reshuffle each channel's own voxels.
      Preserves the exact per-channel intensity histogram, destroys all
      spatial structure. The tightest, most controlled version of this
      probe -- especially relevant for optim_uncnn, whose backbone is an
      untrained random CNN: if its map for spatially-scrambled noise looks
      like its map for a real brain, the map is dominated by the random
      architecture's fixed spatial biases, not by anatomy.
    - "gaussian": fresh per-channel Gaussian noise matched to the real
      input's mean/std. Simpler, but changes higher moments of the
      intensity distribution too (real MRI intensities aren't Gaussian).
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.empty_like(input_tensor)
    for c in range(input_tensor.shape[0]):
        channel = input_tensor[c]
        if dist == "gaussian":
            noise[c] = torch.randn(channel.shape, generator=g) * channel.std() + channel.mean()
        else:  # permute
            flat = channel.reshape(-1)
            perm = torch.randperm(flat.numel(), generator=g)
            noise[c] = flat[perm].reshape(channel.shape)
    return noise


def main():
    parser = argparse.ArgumentParser(description="Standalone BrainAge Pipeline Saliency Generator")
    parser.add_argument("--model", choices=["optim_uncnn", "brainiac", "swinbrain", "anatcl", "simclr"], required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to your saved _brain_age_head.pt asset")
    parser.add_argument("--subject-nii", type=Path, required=True, help="Path to the raw input NIfTI volume (.nii.gz)")
    parser.add_argument("--output-dir", type=Path, default=Path("./saliency_outputs"))
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Must match the --seed train_linear.py used to fit the checkpoint (default: 0, matching "
        "train_linear.py's own default). Only matters for optim_uncnn: its backbone (CNN3D_CovPool) has "
        "no checkpoint of its own -- it IS a random init, so without seeding it the same way "
        "train_linear.py did before caching features and fitting the ridge head, this script "
        "reconstructs a DIFFERENT random backbone every run, producing garbage, non-reproducible "
        "saliency maps and predictions. The other models load a fixed pretrained checkpoint via "
        "load_state_dict(), so this seed doesn't affect them.",
    )
    parser.add_argument(
        "--anatcl-level",
        choices=["global", "local"],
        default=os.environ.get("anatcl_level", "global"),
        help="--model anatcl only: must match the --anatcl-level the checkpoint/cache were built with "
        "(train_linear.py), so the right anatcl_{level}_cat12vbm_{fold_tag}_X_features_raw.npy is used.",
    )
    parser.add_argument(
        "--anatcl-folds",
        default=os.environ.get("anatcl_folds", "all"),
        help="--model anatcl only: 'all' or a single fold index -- must match how the checkpoint/cache "
        "were built (train_linear.py: fold_tag is 'allfold_mean' for 'all', else 'fold{N}').",
    )
    parser.add_argument(
        "--randomize-model",
        action="store_true",
        help="Run the model parameter randomization sanity check (Adebayo et al., 2018): cascade "
        "reset_parameters() through the pipeline from the head backward into the backbone, saving "
        "saliency at each step and comparing it to the baseline.",
    )
    parser.add_argument(
        "--randomize-data",
        action="store_true",
        help="Run the data (label) randomization sanity check (Adebayo et al., 2018): refit the ridge "
        "head on this model's cached features with the age labels randomly shuffled, then compare "
        "its saliency map to the baseline. Requires cached "
        "{cache_name}_X_features_raw.npy/_y_age.npy from train_linear.py (see --features-dir).",
    )
    parser.add_argument(
        "--randomize-model-mode",
        choices=["cascade", "all"],
        default="cascade",
        help="How --randomize-model randomizes the pipeline. 'cascade' (default): reset one module at "
        "a time, output-side first, saving+comparing saliency after each additional module is reset "
        "(--randomize-max-layers controls depth). 'all': reset every module in a single shot and "
        "compare once. For --model optim_uncnn, 'all' still only resets the head, for the same reason "
        "'cascade' defaults to depth 1 there -- its backbone is already random and frozen.",
    )
    parser.add_argument(
        "--randomize-max-layers",
        type=int,
        default=None,
        help="Cascade mode only: how many parameterized modules deep to cascade (default: all). "
        "For --model optim_uncnn this defaults to 1 (head only) unless set explicitly: its backbone "
        "is randomly initialized and frozen -- it is never trained -- so cascading further into it "
        "just re-randomizes weights that were already random and tests nothing.",
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Run an input randomization control (NOT the paper's data-randomization test -- see "
        "--randomize-data): compute saliency for a structure-free control input (same shape/scale as "
        "the real preprocessed input, see --randomize-input-dist) using the real trained pipeline, and "
        "compare it to the baseline.",
    )
    parser.add_argument(
        "--randomize-input-dist",
        choices=["permute", "gaussian"],
        default="permute",
        help="How to build the control input for --randomize-input (default: permute, see its help).",
    )
    parser.add_argument("--randomize-seed", type=int, default=0, help="Seed for all randomization tests.")
    parser.add_argument(
        "--randomize-data-seeds",
        type=int,
        default=1,
        help="--randomize-data only: number of permutation seeds (starting at --randomize-seed, "
        "incrementing by 1) to refit-and-compare, averaging the Spearman correlation over the draws "
        "(default: 1, matching Adebayo et al.'s own single-shot approach -- output filenames/summary "
        "are unchanged from before this flag existed at the default). >1 is a robustness upgrade: a "
        "linear ridge head's weights are only weakly determined by shuffled continuous labels (see "
        "fit_ridge_head_inmemory's docstring), so a single permutation draw's correlation is noisier "
        "than the paper's deep-net case. Only the first seed's saliency maps are saved to disk; extra "
        "seeds only contribute to the aggregate mean/std reported in the summary JSON "
        "('randomize_data_aggregate').",
    )
    parser.add_argument(
        "--attribution-method",
        choices=["gradient", "gradient_x_input", "integrated_gradients"],
        default="gradient",
        help="How to compute voxelwise attribution (default: gradient). 'gradient': raw voxelwise "
        "input-gradient sensitivity S_v = d(age)/d(x_v) -- the method this script has always used, "
        "unchanged. 'gradient_x_input': A_v = x_v * d(age)/d(x_v) -- for a linear model this exactly "
        "decomposes the non-bias part of the prediction, for a nonlinear CNN it's only a local "
        "approximation. 'integrated_gradients': IG_v = (x_v - x0_v) * path-integral of d(age)/d(x_v) "
        "between a baseline x0 and x (see --ig-baseline/--ig-steps below); ~--ig-steps times more "
        "expensive than the other two methods (one forward+backward pass per interpolation step). "
        "See docs/known_limitations.md for caveats on all three. Applies uniformly to the baseline "
        "and every sanity check requested below, so comparisons stay apples-to-apples. Output "
        "filenames get a tag when this is non-default ('gxi_' for gradient_x_input, "
        "'ig_zero_'/'ig_reference_' for integrated_gradients depending on --ig-baseline); the default "
        "'gradient' filenames are unchanged.",
    )
    parser.add_argument(
        "--ig-baseline",
        choices=["zero", "reference"],
        default="zero",
        help="--attribution-method integrated_gradients only: the baseline x0 to integrate from. "
        "'zero' (default): an all-zero tensor, always available, no prerequisite files -- but only "
        "weakly motivated for the z-scored/robust-scaled models here (their zero is just 'this "
        "subject's own mean', not anything population-meaningful); it IS well-motivated for anatcl "
        "(mwp1 zero means 'no gray matter'). 'reference': the population-mean preprocessed image from "
        "compute_reference_baselines.py (see --reference-baselines-dir) -- better motivated per the "
        "literature (Distill's 'Visualizing the Impact of Feature Attribution Baselines', Erion et "
        "al.'s Expected Gradients) for the other 5 models, but requires running "
        "compute_reference_baselines.py first. See docs/known_limitations.md.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="--attribution-method integrated_gradients only: number of Riemann-sum interpolation "
        "steps between the baseline and the input (default: 50, matching common IG practice/Captum's "
        "default). Directly multiplies runtime -- each step is one full forward+backward pass through "
        "the backbone. Lower for faster iteration at the cost of a coarser approximation.",
    )
    parser.add_argument(
        "--reference-baselines-dir",
        type=Path,
        default=DEFAULT_REFERENCE_BASELINES_DIR,
        help="--attribution-method integrated_gradients --ig-baseline reference only: directory with "
        "compute_reference_baselines.py's {model}_reference.npy output (default: reference_baselines/ "
        "at the repo root).",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Directory with train_linear.py's cached {cache_name}_X_features_raw.npy/_y_age.npy "
        "(default: this script's directory, matching train_linear.py's DEFAULT_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--demographics",
        type=Path,
        default=DEFAULT_DEMOGRAPHICS,
        help="Tab-delimited Subject/Sex/Age file (same one train_linear.py trains on) used to look up "
        "the subject's ground-truth target value for the saliency-check JSON summary, via an NKI-style "
        "subject ID (e.g. A00000000) found in --subject-nii's path.",
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default=os.environ.get("target", "age"),
        help="Which head this --checkpoint-path was trained for (must match train_linear.py's --target "
        "used to produce it) -- determines which demographics column is looked up and which JSON keys "
        "the sanity-check summary uses (true_age/predicted_age, true_sex/predicted_sex, true_bmi/predicted_bmi).",
    )
    parser.add_argument(
        "--participants-tsv", type=Path, default=DEFAULT_PARTICIPANTS_TSV,
        help="--target bmi only: longitudinal participants.tsv BMI is joined in from (see src/targets.py).",
    )
    parser.add_argument(
        "--anatcl-repo-root",
        type=Path,
        default=Path(os.environ.get("ANATCL_REPO_ROOT", "/path/to/AnatCL")),
        help="--model anatcl only: path to a cloned https://github.com/EIDOSLAB/AnatCL checkout.",
    )
    args_saliency = parser.parse_args()
    if args_saliency.ig_steps < 1:
        parser.error("--ig-steps must be at least 1")
    if args_saliency.randomize_data_seeds < 1:
        parser.error("--randomize-data-seeds must be at least 1")
    if args_saliency.features_dir is None:
        args_saliency.features_dir = Path(__file__).resolve().parent

    target_spec = get_target_spec(args_saliency.target)

    # Insert the target as a new path component directly under the
    # saliency output root, so age/sex/bmi runs never share an output
    # directory and silently overwrite each other's .nii.gz/sanity-check
    # files -- regardless of whether the caller's --output-dir already
    # embeds a per-subject subfolder (slurm/compute_saliency.sh's
    # "saliency_outputs/<SUBJECT>/" -> "saliency_outputs/<target>/<SUBJECT>/")
    # or points straight at the root (slurm/compute_sanity_checks.sh's
    # "saliency_outputs" -> "saliency_outputs/<target>"). The root is
    # identified by name prefix ("saliency...", e.g. saliency_outputs,
    # saliency_affine, saliency_rigid -- not a single hardcoded literal,
    # which used to silently mis-order saliency_affine/<SUBJECT>/ into
    # saliency_affine/<SUBJECT>/<target>/ instead of
    # saliency_affine/<target>/<SUBJECT>/, since it never matched the
    # "saliency_outputs" literal below). Falls back to a plain trailing
    # append if no path component starts with "saliency" (e.g. a
    # caller-renamed output root that doesn't follow this convention).
    output_parts = list(args_saliency.output_dir.parts)
    saliency_root_idx = next((i for i, part in enumerate(output_parts) if part.startswith("saliency")), None)
    if saliency_root_idx is not None:
        output_parts.insert(saliency_root_idx + 1, target_spec.name)
        args_saliency.output_dir = Path(*output_parts)
    else:
        args_saliency.output_dir = args_saliency.output_dir / target_spec.name

    def report_value(raw_value):
        """Raw pipeline output -> JSON-reportable value: unchanged for a
        regression target (age/bmi), sigmoid'd to a [0,1] probability for a
        classification target (sex) -- the pipeline's head never applies an
        activation itself (see BrainAgePipeline), so this is the one place
        that conversion happens, keeping the JSON human-readable."""
        if raw_value is None or target_spec.task_type != "classification":
            return raw_value
        return float(1.0 / (1.0 + np.exp(-raw_value)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Must happen before spec.model_factory(device) below. train_linear.py seeds
    # here too (same default) before extracting/caching features and fitting the
    # ridge head -- for optim_uncnn, whose backbone is never checkpointed (it IS
    # a random init), reconstructing it without this seed gives a completely
    # different random network than the one the head was trained on, and a
    # different one on every run. See --seed's help text.
    torch.manual_seed(args_saliency.seed)
    np.random.seed(args_saliency.seed)
    args_pipeline = parse_args_train(args=[]) # Get defaults from training script
    args_pipeline.model = args_saliency.model
    args_pipeline.anatcl_level = args_saliency.anatcl_level
    args_pipeline.anatcl_folds = args_saliency.anatcl_folds
    args_pipeline.anatcl_repo_root = args_saliency.anatcl_repo_root

    import train_linear as _train_linear_mod
    _train_linear_mod.ANATCL_REPO_ROOT = args_saliency.anatcl_repo_root

    # 1. Reconstruct dummy dataset/spec boundaries to fetch factory hooks
    dummy_demo = pd.DataFrame([{"Subject": "dummy", "Sex": "M", "Age": 40.0}])
    spec, _, _ = make_feature_spec(args_pipeline, dummy_demo)

    # 2. Load the pipeline parameters from the checkpoint
    print(f"Extracting parameters from {args_saliency.checkpoint_path}")
    checkpoint = torch.load(args_saliency.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Extract the Ridge weights to act as our substitute for group_diff_t_stats
    ridge_weights = state_dict["head.weight"].cpu().squeeze(0).numpy()
    feature_dim = len(ridge_weights)

    # 3. Instantiate underlying architecture
    backbone = spec.model_factory(device)
    head = nn.Linear(feature_dim, 1, bias=True)

    pipeline = BrainAgePipeline(
        backbone=backbone,
        head=head,
        cov_mask=np.zeros(feature_dim, dtype=bool),
        scaler_mean=np.zeros(feature_dim, dtype=np.float32),
        scaler_scale=np.ones(feature_dim, dtype=np.float32),
        postprocess_fn=spec.postprocess
    )

    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    # The backbone was never part of the saved checkpoint (it's frozen/reconstructed
    # fresh via model_factory), so missing backbone.* keys are expected. Anything
    # else missing/unexpected is a real problem and should fail loudly.
    bad_missing = [k for k in missing if not k.startswith("backbone.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch. Missing (non-backbone): {bad_missing}, "
            f"Unexpected: {unexpected}"
        )

    # AnatCL was trained on CAT12 modulated gray-matter VBM maps (mwp1*.nii),
    # not raw/turboprep-normalized T1 intensities. Feeding the wrong modality
    # in silently produces a saliency map for a meaningless forward pass.
    if args_saliency.model == "anatcl" and not args_saliency.subject_nii.name.startswith("mwp1"):
        raise ValueError(
            f"--model anatcl expects a CAT12 modulated gray-matter VBM map (mwp1sub-*_T1w.nii), "
            f"got: {args_saliency.subject_nii.name}. Pass the subject's mwp1*.nii from the CAT12VBM "
            f"NKI_BIDS derivatives (see cat12_paths_and_subjects() in train_linear.py), not the "
            f"turboprep normalized.nii.gz used by the other models."
        )

    # 4. Load and process the subject's 3D input volume using the spec's dataset wrapper
    print(f"Ingesting structural scan: {args_saliency.subject_nii.name}")
    # Injecting our runtime path directly into the dataset's tracking structures
    spec.dataset.paths = [str(args_saliency.subject_nii)]
    input_tensor = spec.dataset[0] # Returns transformed (C, H, W, D) array
    print(f"Input tensor shape: {input_tensor.shape} (C, H, W, D)")

    # Only built for --attribution-method integrated_gradients; every other
    # method ignores baseline_tensor entirely (see compute_direct_pipeline_saliency).
    baseline_tensor = None
    if args_saliency.attribution_method == "integrated_gradients":
        if args_saliency.ig_baseline == "zero":
            baseline_tensor = torch.zeros_like(input_tensor)
        else:
            baseline_arr = load_reference_baseline(args_saliency.model, args_saliency.reference_baselines_dir)
            if baseline_arr.shape != tuple(input_tensor.shape):
                raise ValueError(
                    f"Reference baseline shape {baseline_arr.shape} does not match input tensor shape "
                    f"{tuple(input_tensor.shape)} for model {args_saliency.model!r}. Was "
                    f"compute_reference_baselines.py run with the same preprocessing this script uses?"
                )
            baseline_tensor = torch.from_numpy(baseline_arr).float()
        print(f"Integrated Gradients baseline: --ig-baseline={args_saliency.ig_baseline}, --ig-steps={args_saliency.ig_steps}")

    # Check backprop pathways
    print("Checking backpropagation pathways...")
    audit_gradient_backpropagation(pipeline, input_tensor, device=device)

    # Mirrors compute_and_save_saliency's internal channel_prefix construction
    # exactly (both call attribution_method_tag()), so every "prefix" string
    # recorded into sanity_summary below matches the .nii.gz filenames
    # actually written to disk.
    def channel_prefix(base_prefix, channel):
        tag = attribution_method_tag(args_saliency.attribution_method, ig_baseline=args_saliency.ig_baseline)
        return f"{base_prefix}{channel}_{tag}" if channel else f"{base_prefix}{tag}"

    # 5. Baseline saliency: the real, trained pipeline. Always computed first
    # and preserved, since --randomize-model mutates `pipeline` in place below.
    print("Computing backpropagation pathways...")
    args_saliency.output_dir.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(args_saliency.subject_nii)
    anatcl_suffix = os.environ.get("ANATCL_SUFFIX", "")
    prefix = f"{args_saliency.model}_{anatcl_suffix}"
    baseline_channels, baseline_predicted_age, baseline_diagnostics = compute_and_save_saliency(
        pipeline, input_tensor, args_saliency.model, ref_img, args_saliency.output_dir, prefix, device,
        attribution_method=args_saliency.attribution_method,
        baseline_tensor=baseline_tensor,
        ig_steps=args_saliency.ig_steps,
        ig_baseline=args_saliency.ig_baseline,
        return_diagnostics=True,
    )
    subject_id, true_value = lookup_true_value(
        args_saliency.subject_nii, args_saliency.demographics, target_spec, args_saliency.participants_tsv
    )
    reported_predicted_value = report_value(baseline_predicted_age)
    print(f"Baseline saliency maps stored in {args_saliency.output_dir}")
    print(f"Subject {subject_id}: true {target_spec.name} = {true_value}, predicted {target_spec.name} = {reported_predicted_value}")
    if baseline_diagnostics is not None:
        print(
            "Integrated Gradients completeness: "
            f"F(x)-F(x0)={baseline_diagnostics['prediction_delta']:.6g}, "
            f"sum(IG)={baseline_diagnostics['total_attribution']:.6g}, "
            f"error={baseline_diagnostics['completeness_error']:.6g}, "
            f"relative_abs_error={baseline_diagnostics['relative_completeness_error']:.3e}"
        )

    # 6. Sanity checks (Adebayo et al., 2018, "Sanity Checks for Saliency Maps").
    # Every test's maps are saved with a {test}_{model}_..._{kind}_saliency.nii.gz
    # prefix -- same suffix convention visualize_saliency.py already globs on
    # (*_{kind}_saliency.nii.gz) -- so they drop straight into that script; point
    # its --saliency-dir at args_saliency.output_dir to plot them. Spearman
    # correlations vs. the baseline are also collected into a JSON summary
    # instead of only being printed, so results survive past the console.
    #
    # Input randomization and data randomization both run before model
    # randomization: they reuse `pipeline`/`backbone` as-is, and model
    # randomization destructively rewrites those weights in place.
    # One entry per channel_results key (1 for every model except
    # optim_uncnn, 3 for optim_uncnn) -- predicted_age is identical across
    # entries since they all come from the same single forward pass.
    baseline_prefixes = [
        {"channel": channel or None, "prefix": channel_prefix(prefix, channel), target_spec.json_pred_key: reported_predicted_value}
        for channel in baseline_channels
    ]

    sanity_summary = {
        "model": args_saliency.model,
        "subject_id": subject_id,
        "subject_nii": str(args_saliency.subject_nii),
        "checkpoint_path": str(args_saliency.checkpoint_path),
        "randomize_seed": args_saliency.randomize_seed,
        "attribution_method": args_saliency.attribution_method,
        "ig_baseline": args_saliency.ig_baseline if args_saliency.attribution_method == "integrated_gradients" else None,
        "ig_steps": args_saliency.ig_steps if args_saliency.attribution_method == "integrated_gradients" else None,
        "target": target_spec.name,
        target_spec.json_true_key: true_value,
        target_spec.json_pred_key: reported_predicted_value,
        "integrated_gradients_completeness": baseline_diagnostics,
        "baseline": baseline_prefixes,
        # optim_uncnn's saliency has two model-specific caveats -- see
        # docs/known_limitations.md for both:
        # 1. Its features go through covariance pooling + signed-sqrt
        #    normalization (BrainAgePipeline._signed_sqrt, train_linear.py);
        #    voxels feeding a near-zero covariance feature can get amplified
        #    saliency unrelated to their actual predictive importance. Every
        #    other model has an all-False cov_mask and isn't affected.
        # 2. Its saliency is reported as 3 separate per-channel maps (raw/
        #    rank/sobel -- see UNCNN_CHANNELS above); only the raw channel's
        #    map is interpretable as sensitivity to raw voxel intensity, the
        #    rank/sobel maps are sensitivity to that channel's own already-
        #    computed value, with no chain rule back through the
        #    non-differentiable filter that produced it.
        "known_limitations": (
            [
                "signed_sqrt gradient amplification near zero for covariance-pooled features -- "
                "see BrainAgePipeline._signed_sqrt in train_linear.py",
                "rank/sobel channel maps are sensitivity to that channel's own value, not to raw "
                "voxel intensity -- see docs/known_limitations.md",
            ]
            if args_saliency.model == "optim_uncnn"
            else []
        ),
        "tests": [],
    }

    if args_saliency.randomize_input:
        print("\n" + "=" * 60)
        print("CONTROL: INPUT RANDOMIZATION (structure-free input, real trained pipeline)")
        print("=" * 60)
        print(f"Building a '{args_saliency.randomize_input_dist}' control input...")
        noise_input = make_noise_input(
            input_tensor, dist=args_saliency.randomize_input_dist, seed=args_saliency.randomize_seed
        )
        prefix = f"input_{args_saliency.model}_{args_saliency.randomize_input_dist}_{anatcl_suffix}"
        noise_input_channels, noise_input_predicted_age, noise_input_diagnostics = compute_and_save_saliency(
            pipeline, noise_input, args_saliency.model, ref_img, args_saliency.output_dir, prefix, device,
            attribution_method=args_saliency.attribution_method,
            baseline_tensor=baseline_tensor,
            ig_steps=args_saliency.ig_steps,
            ig_baseline=args_saliency.ig_baseline,
            return_diagnostics=True,
        )
        for channel, (cand_signed, cand_abs) in noise_input_channels.items():
            corr = report_similarity(
                baseline_channels[channel][0], cand_signed,
                baseline_channels[channel][1], cand_abs,
                f"{args_saliency.randomize_input_dist} input" + (f" ({channel})" if channel else "")
            )
            sanity_summary["tests"].append(
                {
                    "test": "randomize_input",
                    "channel": channel or None,
                    "dist": args_saliency.randomize_input_dist,
                    "prefix": channel_prefix(prefix, channel),
                    target_spec.json_pred_key: report_value(noise_input_predicted_age),
                    "integrated_gradients_completeness": noise_input_diagnostics,
                    "spearman_vs_baseline_signed": float(corr["signed"]),
                    "spearman_vs_baseline_abs": float(corr["abs"]),
                }
            )

    if args_saliency.randomize_data:
        print("\n" + "=" * 60)
        print("SANITY CHECK: DATA RANDOMIZATION TEST (adapted from Adebayo et al., 2018 -- see "
              "fit_ridge_head_inmemory's docstring for how the memorization premise differs here)")
        print("=" * 60)
        x_path = args_saliency.features_dir / f"{spec.cache_name}_X_features_raw.npy"
        y_path = args_saliency.features_dir / f"{spec.cache_name}_y_{target_spec.cache_suffix}.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"--randomize-data needs cached features from train_linear.py: expected {x_path} and "
                f"{y_path}. Run `train_linear.py --model {args_saliency.model} ...` first, or pass "
                f"--features-dir."
            )
        X = np.load(x_path)
        y = np.load(y_path)
        cov_mask = spec.cov_mask if spec.cov_mask is not None else np.zeros(X.shape[1], dtype=bool)

        # Baseline (checkpoint) weight norm/scaler, captured once before the
        # loop below -- `pipeline` itself is never mutated by this block (a
        # fresh `noise_pipeline` is built per seed), so these stay valid for
        # every seed's comparison.
        baseline_weight_norm = float(np.linalg.norm(pipeline.head.weight.detach().cpu().numpy()))
        baseline_scaler_mean = pipeline.scaler_mean.detach().cpu().numpy()
        baseline_scaler_scale = pipeline.scaler_scale.detach().cpu().numpy()

        seeds = [args_saliency.randomize_seed + i for i in range(args_saliency.randomize_data_seeds)]
        per_channel_corrs = {}  # channel -> {"signed": [...], "abs": [...]}

        for seed_idx, seed in enumerate(seeds):
            print(f"Refitting ridge head on {spec.cache_name} features with {len(y)} labels randomly "
                  f"shuffled (seed={seed})...")
            noise_head, noise_scaler, ridge_diag = fit_ridge_head_inmemory(X, y, spec.cov_mask, seed=seed, target_spec=target_spec)

            # The checkpoint's own scaler (train_ridge_head, train_linear.py) was
            # fit on a train_test_split *subset* of this same cache, while this
            # refit uses the full cached X -- so it's fit on more rows and an
            # exact match isn't expected even with zero bugs. What this guards
            # against is a much larger mismatch: wrong/stale cache_name, a
            # feature-extraction change since the checkpoint was trained, etc.
            # -- anything that would silently make "data randomization changed
            # the map" partly reflect a scaler/feature mismatch rather than the
            # label shuffle. Threshold is generous on purpose (relative L2 norm
            # of the difference); tighten it if you know train/test split noise
            # is small for your cohort size.
            mean_reldiff = float(
                np.linalg.norm(noise_scaler.mean_ - baseline_scaler_mean)
                / max(np.linalg.norm(baseline_scaler_mean), 1e-8)
            )
            scale_reldiff = float(
                np.linalg.norm(noise_scaler.scale_ - baseline_scaler_scale)
                / max(np.linalg.norm(baseline_scaler_scale), 1e-8)
            )
            if seed_idx == 0 and (mean_reldiff > 0.5 or scale_reldiff > 0.5):
                print(
                    f"  WARNING: refit StandardScaler differs from the checkpoint's own scaler by "
                    f"{mean_reldiff:.2f}x (mean) / {scale_reldiff:.2f}x (scale) relative L2 norm -- "
                    f"larger than the train/test-split noise this is expected to have. Check that "
                    f"--features-dir/{spec.cache_name} actually matches the checkpoint at "
                    f"{args_saliency.checkpoint_path}."
                )

            weight_norm_ratio = ridge_diag["shuffled_weight_norm"] / max(baseline_weight_norm, 1e-8)
            diag_str = ", ".join(f"{k}={v:.4g}" for k, v in ridge_diag.items() if k != "shuffled_weight_norm")
            print(f"  {diag_str}, ||w_shuffled||/||w_true||={weight_norm_ratio:.4f}")
            if weight_norm_ratio < 0.05:
                print(
                    "  NOTE: shuffled head's weights collapsed to near-zero relative to the trained "
                    "head -- any low correlation below mostly reflects 'the model learned nothing' "
                    "(the expected outcome for a linear head on shuffled continuous labels), not "
                    "sensitivity to a memorized alternative function. See fit_ridge_head_inmemory's docstring."
                )

            noise_pipeline = BrainAgePipeline(
                backbone=backbone,
                head=noise_head,
                cov_mask=cov_mask,
                scaler_mean=noise_scaler.mean_,
                scaler_scale=noise_scaler.scale_,
                postprocess_fn=spec.postprocess,
            ).to(device).eval()

            # Only the first seed's maps are saved to disk under the original
            # "data_{model}_..." prefix (unchanged filenames/behavior when
            # --randomize-data-seeds is left at its default of 1); extra seeds
            # (>1) get a seed-suffixed prefix and only feed the aggregate
            # correlation stats below, so opting into multi-seed averaging
            # doesn't multiply .nii.gz output by --randomize-data-seeds.
            prefix = (
                f"data_{args_saliency.model}_{anatcl_suffix}" if seed_idx == 0
                else f"data_seed{seed}_{args_saliency.model}_{anatcl_suffix}"
            )
            noise_channels, noise_predicted_age, noise_diagnostics = compute_and_save_saliency(
                noise_pipeline, input_tensor, args_saliency.model, ref_img, args_saliency.output_dir, prefix, device,
                attribution_method=args_saliency.attribution_method,
                baseline_tensor=baseline_tensor,
                ig_steps=args_saliency.ig_steps,
                ig_baseline=args_saliency.ig_baseline,
                return_diagnostics=True,
            )
            for channel, (cand_signed, cand_abs) in noise_channels.items():
                corr = report_similarity(
                    baseline_channels[channel][0], cand_signed,
                    baseline_channels[channel][1], cand_abs,
                    "data-randomized head" + (f" ({channel})" if channel else "") + f" [seed={seed}]"
                )
                per_channel_corrs.setdefault(channel, {"signed": [], "abs": []})
                per_channel_corrs[channel]["signed"].append(corr["signed"])
                per_channel_corrs[channel]["abs"].append(corr["abs"])
                sanity_summary["tests"].append(
                    {
                        "test": "randomize_data",
                        "channel": channel or None,
                        "seed": seed,
                        "prefix": channel_prefix(prefix, channel),
                        "cache_name": spec.cache_name,
                        "n_labels_shuffled": int(len(y)),
                        target_spec.json_pred_key: report_value(noise_predicted_age),
                        "integrated_gradients_completeness": noise_diagnostics,
                        "spearman_vs_baseline_signed": float(corr["signed"]),
                        "spearman_vs_baseline_abs": float(corr["abs"]),
                        **ridge_diag,
                        "baseline_weight_norm": baseline_weight_norm,
                        "weight_norm_ratio": weight_norm_ratio,
                        "scaler_mean_reldiff": mean_reldiff,
                        "scaler_scale_reldiff": scale_reldiff,
                    }
                )

        # Robustness view across seeds (Adebayo et al. use a single permutation
        # draw; averaging several is a cheap upgrade given how much noisier a
        # single draw is once the head's weights are only weakly determined --
        # see fit_ridge_head_inmemory's docstring). Only emitted when
        # --randomize-data-seeds > 1, so the default single-seed run's summary
        # is unchanged.
        if len(seeds) > 1:
            for channel, corrs in per_channel_corrs.items():
                signed_arr = np.array(corrs["signed"])
                abs_arr = np.array(corrs["abs"])
                print(
                    f"  [{channel or 'aggregate'}] over {len(seeds)} seeds: "
                    f"signed={signed_arr.mean():.4f}+/-{signed_arr.std():.4f}, "
                    f"abs={abs_arr.mean():.4f}+/-{abs_arr.std():.4f}"
                )
                sanity_summary["tests"].append(
                    {
                        "test": "randomize_data_aggregate",
                        "channel": channel or None,
                        "seeds": seeds,
                        "spearman_signed_mean": float(signed_arr.mean()),
                        "spearman_signed_std": float(signed_arr.std()),
                        "spearman_abs_mean": float(abs_arr.mean()),
                        "spearman_abs_std": float(abs_arr.std()),
                    }
                )

    if args_saliency.randomize_model:
        print("\n" + "=" * 60)
        print("SANITY CHECK: MODEL PARAMETER RANDOMIZATION TEST (Adebayo et al., 2018)")
        print(f"mode: {args_saliency.randomize_model_mode}")
        print("=" * 60)

        if args_saliency.randomize_model_mode == "all":
            if args_saliency.model == "optim_uncnn":
                print(
                    "optim_uncnn's backbone is randomly initialized and frozen -- it is never trained, "
                    "it's the untrained CNN this project studies -- so resetting it again just "
                    "re-randomizes weights that were already random. Only the head is reset."
                )
            if args_saliency.randomize_max_layers is not None:
                print("--randomize-max-layers is ignored in --randomize-model-mode all (every module is reset).")
            reset_names, randomized_pipeline = randomize_all_pipeline(
                pipeline, args_saliency.model, seed=args_saliency.randomize_seed
            )
            print(f"  reinitialized {len(reset_names)} module(s): {reset_names}")
            prefix = f"arch_{args_saliency.model}_all_{anatcl_suffix}"
            all_channels, all_predicted_age, all_diagnostics = compute_and_save_saliency(
                randomized_pipeline, input_tensor, args_saliency.model, ref_img, args_saliency.output_dir, prefix, device,
                attribution_method=args_saliency.attribution_method,
                baseline_tensor=baseline_tensor,
                ig_steps=args_saliency.ig_steps,
                ig_baseline=args_saliency.ig_baseline,
                return_diagnostics=True,
            )
            for channel, (cand_signed, cand_abs) in all_channels.items():
                corr = report_similarity(
                    baseline_channels[channel][0], cand_signed,
                    baseline_channels[channel][1], cand_abs,
                    "all modules reset at once" + (f" ({channel})" if channel else "")
                )
                sanity_summary["tests"].append(
                    {
                        "test": "randomize_model",
                        "mode": "all",
                        "channel": channel or None,
                        "modules_reinitialized": reset_names,
                        "prefix": channel_prefix(prefix, channel),
                        target_spec.json_pred_key: report_value(all_predicted_age),
                        "integrated_gradients_completeness": all_diagnostics,
                        "spearman_vs_baseline_signed": float(corr["signed"]),
                        "spearman_vs_baseline_abs": float(corr["abs"]),
                    }
                )
        else:  # cascade
            max_layers = args_saliency.randomize_max_layers
            if max_layers is None and args_saliency.model == "optim_uncnn":
                max_layers = 1
                print(
                    "optim_uncnn's backbone is randomly initialized and frozen -- it is never trained, "
                    "it's the untrained CNN this project studies -- so cascading past the head just "
                    "re-randomizes weights that were already random. Defaulting --randomize-max-layers to 1."
                )
            for step, module_name, randomized_pipeline in cascading_randomize_pipeline(
                pipeline, max_layers=max_layers, seed=args_saliency.randomize_seed
            ):
                print(f"  step {step}: reinitialized '{module_name}'")
                safe_module_name = module_name.replace(".", "-") or "pipeline"
                prefix = f"arch_{args_saliency.model}_step{step:02d}_{safe_module_name}_{anatcl_suffix}"
                step_channels, step_predicted_age, step_diagnostics = compute_and_save_saliency(
                    randomized_pipeline, input_tensor, args_saliency.model, ref_img, args_saliency.output_dir, prefix, device,
                    attribution_method=args_saliency.attribution_method,
                    baseline_tensor=baseline_tensor,
                    ig_steps=args_saliency.ig_steps,
                    ig_baseline=args_saliency.ig_baseline,
                    return_diagnostics=True,
                )
                for channel, (cand_signed, cand_abs) in step_channels.items():
                    corr = report_similarity(
                        baseline_channels[channel][0], cand_signed,
                        baseline_channels[channel][1], cand_abs,
                        f"step {step} ({module_name})" + (f" [{channel}]" if channel else "")
                    )
                    sanity_summary["tests"].append(
                        {
                            "test": "randomize_model",
                            "mode": "cascade",
                            "step": step,
                            "channel": channel or None,
                            "module_reinitialized": module_name,
                            "prefix": channel_prefix(prefix, channel),
                            target_spec.json_pred_key: report_value(step_predicted_age),
                            "integrated_gradients_completeness": step_diagnostics,
                            "spearman_vs_baseline_signed": float(corr["signed"]),
                            "spearman_vs_baseline_abs": float(corr["abs"]),
                        }
                    )

    # Always write the summary (true/predicted age, baseline prefix) even if no
    # --randomize-* flag was passed -- "tests" is just empty in that case.
    # Filename includes anatcl_suffix (like the baseline .nii.gz files already
    # do) so --anatcl-level global and --anatcl-level local runs for the same
    # subject don't silently overwrite each other's summary -- and, for the
    # same reason, attribution_method_tag()'s tag (e.g. "_gxi", "_ig_zero",
    # "_ig_reference") when a non-default method/baseline was used, so that
    # run's JSON doesn't silently overwrite (or get overwritten by) a
    # different method's JSON for the same subject.
    _summary_tag = attribution_method_tag(args_saliency.attribution_method, ig_baseline=args_saliency.ig_baseline)
    summary_stem = f"{args_saliency.model}_{anatcl_suffix}".rstrip("_") + (
        f"_{_summary_tag.rstrip('_')}" if _summary_tag else ""
    )
    summary_path = args_saliency.output_dir / f"{summary_stem}_sanity_checks.json"
    with open(summary_path, "w") as f:
        json.dump(sanity_summary, f, indent=2)
    print(f"Summary (true/predicted age, correlations, file prefixes) saved to {summary_path}")

    print(f"Done. Saliency maps stored in {args_saliency.output_dir}")

if __name__ == "__main__":
    main()
