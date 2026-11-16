import argparse
import glob
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from targets import DEFAULT_PARTICIPANTS_TSV, get_target_spec, merge_bmi


REPO_ROOT = Path(__file__).resolve().parents[1]

# All of these are external to this repo -- the NKI Rockland Sample dataset
# (its own data-use agreement) and the pretrained 3D backbone checkpoints
# (BrainIAC is under a research-only license; AnatCL is MIT but lives in its
# own repo). None of that is vendored here; point these at wherever you have
# them via env vars (see config/paths.example.env) or the matching --flags.
DEFAULT_DEMOGRAPHICS = Path(os.environ.get("NKI_DEMOGRAPHICS", "/path/to/NKI/final_demographics.txt"))
DEFAULT_TURBOPREP_ROOT = Path(os.environ.get("NKI_AFFINE_TURBOPREP_ROOT", "/path/to/NKI_hdbet_affine"))
DEFAULT_NKI_BIDS_ROOT = Path(os.environ.get("NKI_BIDS_ROOT", "/path/to/NKI_BIDS"))
DEFAULT_CHECKPOINT_DIR = Path(os.environ.get("BACKBONE_CHECKPOINT_DIR", "/path/to/BrainIAC/checkpoints"))
ANATCL_REPO_ROOT = Path(os.environ.get("ANATCL_REPO_ROOT", "/path/to/AnatCL"))
# Trained linear-head checkpoints + feature caches this repo itself produces
# (as opposed to the pretrained backbones above). Feature caches are
# regenerable and gitignored; small trained heads live in checkpoints/heads/
# (tracked) but new runs default here to keep that directory hand-curated.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts"
# fs_roi only: build_freesurfer_roi_tables.py's output (in-repo, unlike the
# external-dataset paths above, since this is this repo's own generated
# artifact -- see that script's own module docstring for how it's built).
DEFAULT_ROI_CSV = REPO_ROOT / "artifacts" / "nki_freesurfer_roi.csv"
DEFAULT_EXCLUDE_SUBJECTS = ("A00058130", "A00059326")  # A00059326 has no BAS1 BMI row in participants.tsv


# Preprocessing used by the untrained covariance-pooling CNN.
def scale_robust(img, clip_low=2, clip_high=98, eps=1e-8):
    mask = img > np.percentile(img, 5)
    lo, hi = np.percentile(img[mask], [clip_low, clip_high])
    clipped = np.clip(img, lo, hi)
    brain = clipped[mask]
    q25, q75 = np.percentile(brain, [25, 75])
    iqr = float(q75 - q25)
    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (brain - float(np.median(brain))) / (iqr + eps)
    return out


def norm(ch):
    mn, mx = ch.min(), ch.max()
    return np.zeros_like(ch) if mx - mn < 1e-8 else (ch - mn) / (mx - mn)


def sobel_mag(img):
    return np.sqrt(ndi.sobel(img, 0) ** 2 + ndi.sobel(img, 1) ** 2 + ndi.sobel(img, 2) ** 2)


def rank_filter(img, size=3):
    r = ndi.percentile_filter(img, percentile=50, size=size)
    mn, mx = r.min(), r.max()
    return np.zeros_like(r) if mx - mn < 1e-8 else (r - mn) / (mx - mn)


class DepthwiseSepConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CNN3D_CovPool(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1a = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.norm1a = nn.GroupNorm(8, 64)
        self.conv1b = nn.Conv3d(64, 64, 3, padding=1)
        self.norm1b = nn.GroupNorm(8, 64)
        self.down1 = nn.AvgPool3d(2)
        self.conv2a = DepthwiseSepConv3d(64, 128)
        self.norm2a = nn.GroupNorm(8, 128)
        self.conv2b = DepthwiseSepConv3d(128, 128)
        self.norm2b = nn.GroupNorm(8, 128)
        self.down2 = nn.AvgPool3d(2)
        self.conv3a = DepthwiseSepConv3d(128, 256)
        self.norm3a = nn.GroupNorm(8, 256)
        self.conv3b = DepthwiseSepConv3d(256, 256)
        self.norm3b = nn.GroupNorm(8, 256)
        self.down3 = nn.AvgPool3d(2)
        self.conv4a = DepthwiseSepConv3d(256, 512)
        self.norm4a = nn.GroupNorm(8, 512)
        self.conv4b = DepthwiseSepConv3d(512, 512)
        self.norm4b = nn.GroupNorm(8, 512)
        self.down4 = nn.AvgPool3d(2)
        self.avg_pool = nn.AdaptiveAvgPool3d(2)

    def _cov_pool(self, x, max_ch=32):
        b, c = x.size(0), min(x.size(1), max_ch)
        mean_feats = self.avg_pool(x).view(b, -1)
        xc = x[:, :c]
        flat = xc.view(b, c, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (flat.size(2) - 1)
        idx = torch.triu_indices(c, c, offset=1, device=x.device)
        return torch.cat([mean_feats, cov[:, idx[0], idx[1]]], dim=1)

    def forward(self, x):
        x1 = F.relu(self.norm1a(self.conv1a(x)))
        x1 = self.down1(F.relu(self.norm1b(self.conv1b(x1))))
        x2 = F.relu(self.norm2a(self.conv2a(x1)))
        x2 = self.down2(F.relu(self.norm2b(self.conv2b(x2))))
        x3 = F.relu(self.norm3a(self.conv3a(x2)))
        x3 = self.down3(F.relu(self.norm3b(self.conv3b(x3))))
        x4 = F.relu(self.norm4a(self.conv4a(x3)))
        x4 = self.down4(F.relu(self.norm4b(self.conv4b(x4))))
        return torch.cat(
            [
                self._cov_pool(x1, 32),
                self._cov_pool(x2, 32),
                self._cov_pool(x3, 32),
                self._cov_pool(x4, 32),
            ],
            dim=1,
        )


MEAN_DIMS = [64 * 8, 128 * 8, 256 * 8, 512 * 8]
COV_CH = 32
COV_DIM = COV_CH * (COV_CH - 1) // 2
UNCNN_TOTAL_DIM = sum(MEAN_DIMS) + 4 * COV_DIM


def cov_feature_mask(mean_dims=MEAN_DIMS, cov_dim=COV_DIM):
    mask = np.zeros(sum(mean_dims) + len(mean_dims) * cov_dim, dtype=bool)
    pos = 0
    for md in mean_dims:
        pos += md
        mask[pos : pos + cov_dim] = True
        pos += cov_dim
    return mask


def signed_sqrt(x, eps=1e-6):
    return np.sign(x) * np.sqrt(np.abs(x) + eps)


class BrainAgePipeline(nn.Module):
    """
    Wraps a frozen feature extractor backbone along with conditional post-processing,
    covariance transformation masks, standardization constants, and a regularized linear head
    into an unified execution graph. This preserves autograd pathing for clean saliency mapping.
    """
    def __init__(self, backbone, head, cov_mask, scaler_mean, scaler_scale, postprocess_fn=None, eps=1e-6):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.postprocess_fn = postprocess_fn
        self.eps = eps

        # Store statistics as fixed architectural buffers
        self.register_buffer("cov_mask", torch.tensor(cov_mask, dtype=torch.bool))
        self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32))

    def _signed_sqrt(self, x):
        # KNOWN LIMITATION (optim_uncnn only -- cov_mask is all-False for every
        # other model, so this branch never affects their gradients): d/dx of
        # sign(x)*sqrt(|x|+eps) is 1/(2*sqrt(|x|+eps)), which blows up as x
        # approaches 0 -- e.g. ~500x at x=0 vs. 0.5x at x=1 (eps=1e-6). Since
        # this transform only applies to the covariance-pooled features (see
        # cov_feature_mask / CNN3D_CovPool._cov_pool), a covariance entry that
        # happens to sit near zero gets its saliency contribution amplified by
        # orders of magnitude purely from this derivative singularity, not
        # because it's actually predictive. There's no fix that doesn't also
        # change the forward value (which would invalidate the trained
        # RidgeCV weights -- see the max_abs_diff assert in main()), so this
        # is left as-is and should be treated as a caveat on optim_uncnn's
        # saliency maps specifically, not corrected numerically. See also
        # docs/known_limitations.md.
        return torch.sign(x) * torch.sqrt(torch.abs(x) + self.eps)

    def forward(self, x):
        feat = self.backbone(x)

        if self.postprocess_fn is not None:
            feat = self.postprocess_fn(feat)

        if isinstance(feat, tuple):
            feat = feat[0]

        feat = torch.flatten(feat, start_dim=1)

        # Guard sqrt's *input*, not just select on its output: torch.where's
        # backward computes gradients through both branches unconditionally,
        # even though only one is kept in the forward value. Feeding the
        # non-covariance positions' real values straight into a function with
        # a derivative singularity at 0 is a latent NaN trap the moment eps is
        # ever lowered or underflows (e.g. fp16/autocast). Substituting a safe
        # placeholder (1.0) at exactly the positions torch.where discards
        # anyway removes that possibility, with zero effect on the selected
        # forward value (feat is used unmodified at the False/mean positions
        # either way -- see the second torch.where below).
        cov_safe_input = torch.where(self.cov_mask, feat, torch.ones_like(feat))
        sqrt_feat = self._signed_sqrt(cov_safe_input)
        feat = torch.where(self.cov_mask, sqrt_feat, feat)

        # Apply standardization and execute prediction head
        feat_scaled = (feat - self.scaler_mean) / self.scaler_scale
        return self.head(feat_scaled)


class UnCNNDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype("float32")
        if img.shape != (91, 109, 91):
            img = ndi.zoom(img, [91 / img.shape[0], 109 / img.shape[1], 91 / img.shape[2]], order=1)
        brain_scaled = scale_robust(img)
        return torch.from_numpy(
            np.stack(
                [
                    brain_scaled,
                    rank_filter(brain_scaled),
                    norm(sobel_mag(brain_scaled)),
                ],
                axis=0,
            )
        ).float()


class BrainIACDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        if img.shape != (96, 96, 96):
            img = F.interpolate(
                torch.from_numpy(img[None, None]),
                size=(96, 96, 96),
                mode="trilinear",
                align_corners=False,
            ).squeeze().numpy()
        img = self.transform(torch.from_numpy(img))
        return img[None]


class CAT12VBMDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
        self.target_shape = (121, 128, 121)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        if img.shape != self.target_shape:
            img = F.interpolate(
                torch.from_numpy(img[None, None]),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze().numpy()
        return self.transform(img).unsqueeze(0)


class SwinBrainDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        img = F.interpolate(
            torch.from_numpy(img[None, None]),
            size=(128, 128, 64),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        img = (img - img.mean()) / (img.std() + 1e-6)
        return img.repeat(3, 1, 1, 1)


class TurboPrepDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.target_shape = (150, 192, 192)

    def __len__(self):
        return len(self.paths)

    def _center_crop(self, img):
        starts = [(s - t) // 2 for s, t in zip(img.shape, self.target_shape)]
        return img[
            starts[0] : starts[0] + self.target_shape[0],
            starts[1] : starts[1] + self.target_shape[1],
            starts[2] : starts[2] + self.target_shape[2],
        ]

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = self._center_crop(img)
        img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img[None]).float()


class ROIVolumeDataset(Dataset):
    """
    fs_roi's "dataset" -- unlike every other model here, there's no per-item
    file load: the feature matrix (FreeSurfer regional volumes, already
    built by build_freesurfer_roi_tables.py) is entirely in memory up front,
    so __getitem__ is just a row lookup.

    `paths` holds subject IDs (not filesystem paths) -- named `.paths`
    anyway, not `.subject_ids`, purely so verify_saved_pipelines's existing
    shallow-copy-swap trick (spec.dataset.paths = [...]) keeps working
    across every model uniformly. Because that trick reassigns `.paths` to
    an arbitrary sub-list without touching `feature_matrix`, and this
    dataset (unlike the file-path ones) indexes `feature_matrix` by
    POSITION, reassigning `.paths` alone would silently desync the two
    arrays. `subset(indices)` is the safe way to narrow both together --
    see its call site in verify_saved_pipelines.
    """

    def __init__(self, feature_matrix, paths):
        self.feature_matrix = feature_matrix
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.from_numpy(self.feature_matrix[idx]).float()

    def subset(self, indices):
        return ROIVolumeDataset(self.feature_matrix[indices], [self.paths[i] for i in indices])


class SwinBrainFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.extracted_features = None
        self.hook_handle = self.base_model.encoder10.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        self.extracted_features = output

    def forward(self, x):
        _ = self.base_model(x)
        if self.extracted_features is None:
            raise RuntimeError("SwinBrain encoder10 hook did not capture features.")
        pooled = self.pool(self.extracted_features)
        return torch.flatten(pooled, start_dim=1)

    def remove_hook(self):
        self.hook_handle.remove()


class _AgeEstimator:
    pass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    cache_name: str
    dataset: Dataset
    model_factory: Callable[[torch.device], nn.Module]
    postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    batch_size: int = 8
    cov_mask: Optional[np.ndarray] = None
    # fs_roi only: the feature vector's column order (region names), so
    # load_or_extract_features can persist it alongside the X/y/subjects
    # caches -- calculate_gradcam_tabular.py needs this to label saliency
    # output back to anatomical regions. None for every image-based model
    # (there's no equivalent "feature name" per covariance-pooled/CLS-token/
    # encoder-output dimension).
    feature_names: Optional[list] = None


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def require_monai():
    try:
        from monai.networks.nets import SwinUNETR, ViT, resnet18
    except ImportError as exc:
        raise ImportError("MONAI is required for BrainIAC, SwinBrain, and SimCLR backbones.") from exc
    return SwinUNETR, ViT, resnet18


def require_anatcl():
    if str(ANATCL_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(ANATCL_REPO_ROOT))
    try:
        from anatcl import AnatCL
        import anatcl.models as _anatcl_models
    except ImportError as exc:
        raise ImportError(
            "AnatCL is required for --model anatcl. Clone https://github.com/EIDOSLAB/AnatCL "
            "and point ANATCL_REPO_ROOT (env var) or --anatcl-repo-root at it."
        ) from exc

    estimators_mod = types.ModuleType("models.estimators")
    estimators_mod.AgeEstimator = _AgeEstimator
    sys.modules["models"] = _anatcl_models
    sys.modules["models.estimators"] = estimators_mod
    return AnatCL


def load_brainiac_vit(ckpt_path, device):
    _, ViT, _ = require_monai()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    vit = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=True,
    )
    pe_key = "backbone.patch_embedding.position_embeddings"
    state = ckpt.get("state_dict", ckpt)
    if pe_key in state:
        vit.patch_embedding.position_embeddings = nn.Parameter(state[pe_key].clone())
    backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
    vit.load_state_dict(backbone_state, strict=False)
    return vit.to(device).eval()


def brainiac_cls_token(out):
    return out[0][:, 0] if isinstance(out, tuple) else out[:, 0]


def load_swinbrain(ckpt_path, device):
    SwinUNETR, _, _ = require_monai()
    model = SwinUNETR(
        img_size=(128, 128, 64),
        in_channels=3,
        out_channels=3,
        spatial_dims=3,
        feature_size=24,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    )
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    clean_state = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(clean_state, strict=True)
    return model.to(device).eval()


def load_simclr_encoder(ckpt_path, device):
    _, _, resnet18 = require_monai()
    encoder = resnet18(spatial_dims=3, n_input_channels=1, num_classes=0)
    encoder.fc = nn.Identity()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {
        k.replace("module.encoder.", ""): v
        for k, v in ckpt["model_state_dict"].items()
        if "encoder" in k and "projector" not in k
    }
    encoder.load_state_dict(state_dict, strict=True)
    return encoder.to(device).eval()


def load_anatcl_encoder(ckpt_dir, level, fold, device):
    AnatCL = require_anatcl()
    ckpt_path = ckpt_dir / f"weights_{level}_fold{fold}.pth"
    encoder = AnatCL(descriptor=level, fold=fold, pretrained=False).to(device).eval()
    encoder.backbone.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    return encoder


class AnatCLFoldAverager(nn.Module):
    def __init__(self, ckpt_dir, level, folds, device):
        super().__init__()
        self.encoders = nn.ModuleList([load_anatcl_encoder(ckpt_dir, level, fold, device) for fold in folds])
        freeze_model(self)

    def forward(self, x):
        outputs = [encoder(x) for encoder in self.encoders]
        return torch.stack(outputs, dim=0).mean(dim=0)


def natural_first_path(paths: Iterable[str]) -> Optional[str]:
    paths = sorted(paths)
    return paths[0] if paths else None


def turboprep_paths(subjects, turboprep_root):
    return [str(turboprep_root / subject / "normalized.nii.gz") for subject in subjects]


def cat12_paths_and_subjects(subjects, bids_root):
    kept_subjects = []
    paths = []
    missing = []
    for subject in subjects:
        pattern = bids_root / f"sub-{subject}" / "ses-*" / "anat" / "mri" / "mwp1sub*.nii"
        path = natural_first_path(glob.glob(str(pattern)))
        if path is None:
            missing.append(subject)
            continue
        kept_subjects.append(subject)
        paths.append(path)
    if missing:
        print(f"Warning: dropped {len(missing)} subjects without CAT12 VBM files. First few: {missing[:5]}")
    return kept_subjects, paths


def existing_paths_and_subjects(subjects, paths):
    kept_subjects = []
    kept_paths = []
    missing = []
    for subject, path in zip(subjects, paths):
        if os.path.exists(path):
            kept_subjects.append(subject)
            kept_paths.append(path)
        else:
            missing.append(subject)
    if missing:
        print(f"Warning: dropped {len(missing)} subjects with missing input volumes. First few: {missing[:5]}")
    return kept_subjects, kept_paths


def roi_volume_rows_and_subjects(subjects, roi_csv_path):
    """
    Loads build_freesurfer_roi_tables.py's output CSV (one row per subject,
    one column per FreeSurfer ROI volume), aligns it to `subjects` (order-
    preserving, like existing_paths_and_subjects), and returns
    (kept_subjects, X (N, D) float32, feature_names (list[str])).

    Column order here (CSV column order, "Subject"/"atlas" bookkeeping
    columns dropped) is the PERMANENT feature-vector ordering: every cached
    array/checkpoint downstream (feature caches, the trained head's
    Linear.weight, calculate_gradcam_tabular.py's region labels) is only
    meaningful relative to this fixed order, which is why it's persisted
    verbatim as {cache_name}_feature_names.json in load_or_extract_features
    rather than re-derived from the CSV each time.
    """
    df = pd.read_csv(roi_csv_path)
    df["Subject"] = df["Subject"].astype(str)
    df = df.set_index("Subject")

    feature_names = [c for c in df.columns if c != "atlas"]

    kept_subjects = []
    rows = []
    missing = []
    for subject in subjects:
        if subject in df.index:
            kept_subjects.append(subject)
            rows.append(df.loc[subject, feature_names].to_numpy(dtype=np.float32))
        else:
            missing.append(subject)
    if missing:
        print(f"Warning: dropped {len(missing)} subjects without an fs_roi feature row. First few: {missing[:5]}")

    X = np.stack(rows, axis=0) if rows else np.zeros((0, len(feature_names)), dtype=np.float32)
    return kept_subjects, X, feature_names


def make_feature_spec(args, demographics):
    subjects = demographics["Subject"].astype(str).tolist()
    num_workers = args.num_workers

    if args.model == "optim_uncnn":
        kept_subjects, paths = existing_paths_and_subjects(subjects, turboprep_paths(subjects, args.turboprep_root))

        # optim_uncnn's backbone is intentionally untrained -- its random init
        # IS the feature extractor. Unlike every other model here, there's no
        # checkpoint pinning its weights, so model_factory gets called again
        # at verification time (verify_saved_pipelines/load_saved_pipeline)
        # against whatever the global RNG state happens to be by then, not
        # the state it had during the original feature extraction. Reset the
        # seed right before construction so the same weights come out both
        # times, regardless of how many other random draws happened between.
        def _make_uncnn(device):
            torch.manual_seed(args.seed)
            return freeze_model(CNN3D_CovPool(in_channels=3).to(device))

        return (
            FeatureSpec(
                name="optim_uncnn",
                cache_name="optim_uncnn_turboprep_covpool",
                dataset=UnCNNDataset(paths),
                model_factory=_make_uncnn,
                batch_size=args.batch_size,
                cov_mask=cov_feature_mask(),
            ),
            kept_subjects,
            num_workers,
        )

    if args.model == "brainiac":
        kept_subjects, paths = existing_paths_and_subjects(subjects, turboprep_paths(subjects, args.turboprep_root))
        transform = lambda x: (x - x.mean()) / (x.std() + 1e-6)
        return (
            FeatureSpec(
                name="brainiac",
                cache_name="brainiac_turboprep_96_cls",
                dataset=BrainIACDataset(paths, transform),
                model_factory=lambda device: freeze_model(load_brainiac_vit(args.brainiac_ckpt, device)),
                postprocess=brainiac_cls_token,
                batch_size=args.batch_size,
            ),
            kept_subjects,
            num_workers,
        )

    if args.model == "swinbrain":
        kept_subjects, paths = existing_paths_and_subjects(subjects, turboprep_paths(subjects, args.turboprep_root))
        return (
            FeatureSpec(
                name="swinbrain",
                cache_name="swinbrain_turboprep_128x128x64_encoder10_avgpool",
                dataset=SwinBrainDataset(paths),
                model_factory=lambda device: freeze_model(SwinBrainFeatureExtractor(load_swinbrain(args.swinbrain_ckpt, device))),
                batch_size=args.batch_size,
            ),
            kept_subjects,
            num_workers,
        )

    if args.model == "simclr":
        kept_subjects, paths = existing_paths_and_subjects(subjects, turboprep_paths(subjects, args.turboprep_root))
        return (
            FeatureSpec(
                name="simclr",
                cache_name="simclr_turboprep_transpose_crop_150x192x192",
                dataset=TurboPrepDataset(paths),
                model_factory=lambda device: freeze_model(load_simclr_encoder(args.simclr_ckpt, device)),
                batch_size=args.batch_size,
            ),
            kept_subjects,
            num_workers,
        )

    if args.model == "anatcl":
        kept_subjects, paths = cat12_paths_and_subjects(subjects, args.nki_bids_root)
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(x.copy()).float()),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
        folds = tuple(range(5)) if args.anatcl_folds == "all" else (int(args.anatcl_folds),)
        fold_tag = "allfold_mean" if args.anatcl_folds == "all" else f"fold{args.anatcl_folds}"
        return (
            FeatureSpec(
                name=f"anatcl_{args.anatcl_level}",
                cache_name=f"anatcl_{args.anatcl_level}_cat12vbm_{fold_tag}",
                dataset=CAT12VBMDataset(paths, transform),
                model_factory=lambda device: freeze_model(
                    AnatCLFoldAverager(args.anatcl_ckpt_dir, args.anatcl_level, folds, device)
                ),
                batch_size=args.batch_size,
            ),
            kept_subjects,
            num_workers,
        )

    if args.model == "fs_roi":
        # No image, no backbone -- the "features" already ARE the raw
        # FreeSurfer ROI volumes (build_freesurfer_roi_tables.py's CSV), so
        # model_factory is a frozen no-op. extract_features (format-
        # agnostic already) just passes each row through untouched.
        kept_subjects, X, feature_names = roi_volume_rows_and_subjects(subjects, args.roi_csv)
        return (
            FeatureSpec(
                name="fs_roi",
                cache_name="fs_roi_dktatlas_grayvol",
                dataset=ROIVolumeDataset(X, kept_subjects),
                model_factory=lambda device: freeze_model(nn.Identity().to(device)),
                batch_size=args.batch_size,
                feature_names=feature_names,
            ),
            kept_subjects,
            num_workers,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def extract_features(spec, device, num_workers):
    loader = DataLoader(spec.dataset, batch_size=spec.batch_size, num_workers=num_workers)
    model = spec.model_factory(device)
    features = []
    with torch.no_grad():
        for x in loader:
            out = model(x.to(device))
            if spec.postprocess is not None:
                out = spec.postprocess(out)
            if isinstance(out, tuple):
                out = out[0]
            features.append(torch.flatten(out, start_dim=1).cpu().numpy())
    if hasattr(model, "remove_hook"):
        model.remove_hook()
    return np.concatenate(features, axis=0)


def load_or_extract_features(args, spec, demographics, kept_subjects, device, num_workers, target_spec):
    cache_prefix = args.output_dir / spec.cache_name
    x_path = cache_prefix.with_name(f"{cache_prefix.name}_X_features_raw.npy")
    y_path = cache_prefix.with_name(f"{cache_prefix.name}_y_{target_spec.cache_suffix}.npy")
    subjects_path = cache_prefix.with_name(f"{cache_prefix.name}_subjects.npy")
    # Only meaningful for models with a fixed feature-name-per-column
    # ordering (currently fs_roi -- see FeatureSpec.feature_names); written
    # whenever spec.feature_names is set so calculate_gradcam_tabular.py can
    # later label saliency output back to region names without depending on
    # re-reading the source CSV (whose column order isn't itself guaranteed
    # stable across regenerations).
    feature_names_path = cache_prefix.with_name(f"{cache_prefix.name}_feature_names.json")

    if not args.no_cache and x_path.exists() and y_path.exists() and subjects_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        cached_subjects = np.load(subjects_path, allow_pickle=True).astype(str).tolist()
        # x_path/subjects_path are cache_name-scoped (shared across every
        # --target), while y_path is cache_name+target-scoped -- so a cache
        # built for one --target can go stale relative to a LATER run for a
        # DIFFERENT --target that re-extracted X/subjects (e.g. because the
        # underlying ROI CSV/demographics row count changed in between),
        # leaving this target's own y_path an orphaned, mismatched leftover.
        # Existence alone (the check above) can't catch that -- row counts
        # can. Treat a mismatch as a cache miss (fall through and
        # re-extract+overwrite all 3 files together) rather than silently
        # returning arrays that don't line up row-for-row (train_test_split
        # would otherwise fail deep in sklearn with a confusing traceback,
        # or worse, succeed with mismatched X/y sample never seen aligned
        # by the caller).
        if X.shape[0] == y.shape[0] == len(cached_subjects):
            print(f"Found cached features: {x_path}")
            return X, y, cached_subjects
        print(
            f"WARNING: cached features are inconsistent (X has {X.shape[0]} rows, y has {y.shape[0]}, "
            f"subjects has {len(cached_subjects)}) -- {y_path.name} is likely a stale leftover from an "
            f"earlier --target run whose subject pool didn't match this one's current "
            f"{x_path.name}/{subjects_path.name}. Re-extracting from scratch instead of trusting it."
        )

    demo_by_subject = demographics.set_index("Subject")
    aligned_demo = demo_by_subject.loc[kept_subjects]
    X = extract_features(spec, device, num_workers)
    y = aligned_demo[target_spec.demographics_column].map(target_spec.encode).to_numpy(dtype=np.float32)
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"Feature/label count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)
    np.save(subjects_path, np.asarray(kept_subjects))
    if spec.feature_names is not None:
        with open(feature_names_path, "w") as f:
            json.dump(spec.feature_names, f)
        print(f"Saved feature names: {feature_names_path}")
    print(f"Saved features: {x_path}")
    return X, y, kept_subjects


def train_ridge_head(args, X, y, sex, spec, device, target_spec):
    X_train_raw, X_test_raw, y_train, y_test, sex_train, sex_test = train_test_split(
        X,
        y,
        sex,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=sex,
    )

    print(f"Feature shape: {X.shape}")
    if target_spec.task_type == "regression":
        print(f"Train {target_spec.name} std/min/max: {y_train.std():.3f}/{y_train.min():.3f}/{y_train.max():.3f}")
        print(f"Test {target_spec.name} std/min/max:  {y_test.std():.3f}/{y_test.min():.3f}/{y_test.max():.3f}")
        chance_mae_train = mean_absolute_error(y_train, np.full_like(y_train, y_train.mean()))
        chance_mae_test = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
        print(f"Chance MAE (train mean predictor): train={chance_mae_train:.3f}  test={chance_mae_test:.3f}")
    else:
        print(f"Train {target_spec.name} class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test {target_spec.name} class counts:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    X_train_proc = X_train_raw.copy()
    X_test_proc = X_test_raw.copy()

    # Generate the boolean covariance tracking mask
    if spec.cov_mask is not None:
        cov_mask_arr = spec.cov_mask
        if X_train_proc.shape[1] != len(cov_mask_arr):
            raise RuntimeError(f"Covariance mask length {len(cov_mask_arr)} does not match feature dim {X_train_proc.shape[1]}")
        X_train_proc[:, cov_mask_arr] = signed_sqrt(X_train_proc[:, cov_mask_arr])
        X_test_proc[:, cov_mask_arr] = signed_sqrt(X_test_proc[:, cov_mask_arr])
    else:
        cov_mask_arr = np.zeros(X.shape[1], dtype=bool)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    if target_spec.task_type == "regression":
        alphas = np.logspace(0, 6, 30) if args.alt_alphas else np.logspace(-2, 5, 30)
        reg = RidgeCV(alphas=alphas, cv=args.cv)
        reg.fit(X_train_scaled, y_train)

        train_pred = reg.predict(X_train_scaled)
        test_pred = reg.predict(X_test_scaled)
        train_metric = mean_absolute_error(y_train, train_pred)
        test_metric = mean_absolute_error(y_test, test_pred)
        metric_name = "MAE"
        # Raw parity-comparable scores (== pipeline's un-activated output).
        raw_test_scores = test_pred
        print(f"Selected alpha: {reg.alpha_:.4g}")
        print(f"Train MAE: {train_metric:.3f}  |  Test MAE: {test_metric:.3f}")

        for s in np.unique(sex_test):
            m = sex_test == s
            print(f"  Test MAE ({s}): {mean_absolute_error(y_test[m], test_pred[m]):.3f}  (n={m.sum()})")
    else:
        reg = LogisticRegressionCV(Cs=np.logspace(-5, 5, 30), cv=args.cv, max_iter=2000)
        reg.fit(X_train_scaled, y_train)

        train_metric = accuracy_score(y_train, reg.predict(X_train_scaled))
        test_metric = accuracy_score(y_test, reg.predict(X_test_scaled))
        metric_name = "accuracy"
        # decision_function returns the raw pre-sigmoid logit, comparable to
        # the pipeline's un-activated output -- reg.predict() returns class
        # labels (0/1), which is NOT comparable for the parity check below.
        raw_test_scores = reg.decision_function(X_test_scaled)
        print(f"Selected C: {reg.C_[0]:.4g}")
        print(f"Train accuracy: {train_metric:.3f}  |  Test accuracy: {test_metric:.3f}")

        for s in np.unique(sex_test):
            m = sex_test == s
            m_acc = accuracy_score(y_test[m], reg.predict(X_test_scaled[m]))
            print(f"  Test accuracy ({s}): {m_acc:.3f}  (n={m.sum()})")

    # Assemble isolated head parameters. RidgeCV's coef_/intercept_ are 1-D
    # (a single continuous target); LogisticRegressionCV's binary coef_/
    # intercept_ are already shape (1, D)/(1,), so no unsqueeze/list-wrap.
    head = nn.Linear(X.shape[1], 1, bias=True)
    with torch.no_grad():
        if target_spec.task_type == "regression":
            head.weight.copy_(torch.tensor(reg.coef_, dtype=torch.float32).unsqueeze(0))
            head.bias.copy_(torch.tensor([reg.intercept_], dtype=torch.float32))
        else:
            head.weight.copy_(torch.tensor(reg.coef_, dtype=torch.float32))
            head.bias.copy_(torch.tensor(reg.intercept_, dtype=torch.float32))
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    # Instantiate unified deployment execution wrapper graph
    backbone_placeholder = lambda x: x  # Raw verification pass
    pipeline = BrainAgePipeline(
        backbone=backbone_placeholder,
        head=head,
        cov_mask=cov_mask_arr,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_
    ).to(device).eval()

    with torch.no_grad():
        test_inputs_torch = torch.tensor(X_test_raw, dtype=torch.float32, device=device)
        torch_preds = pipeline(test_inputs_torch).squeeze(1).cpu().numpy()

    max_abs_diff = np.max(np.abs(torch_preds - raw_test_scores))
    print(f"Max abs diff (sklearn vs BrainAgePipeline): {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "Unified BrainAgePipeline output does not match sklearn predictions"

    output_path = args.output_dir / f"{spec.cache_name}_{target_spec.checkpoint_suffix}_affine.pt"
    ckpt = {
        "state_dict": pipeline.state_dict(),
        "feature_dim": X.shape[1],
        "model": spec.name,
        "cache_name": spec.cache_name,
        "seed": args.seed,
        "test_size": args.test_size,
        "target": target_spec.name,
        "task_type": target_spec.task_type,
        f"train_{metric_name.lower()}": train_metric,
        f"test_{metric_name.lower()}": test_metric,
    }
    if target_spec.task_type == "regression":
        ckpt["alpha"] = float(reg.alpha_)
    else:
        ckpt["C"] = float(reg.C_[0])
    torch.save(ckpt, output_path)
    print(f"Saved deployment state asset to: {output_path}")


def load_saved_pipeline(ckpt_path: Path, spec: FeatureSpec, device: torch.device) -> BrainAgePipeline:
    """
    Reconstructs the full end-to-end BrainAgePipeline directly from a saved checkpoint asset.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing expected checkpoint file: {ckpt_path}")

    print(f"Loading checkpoint for validation: {ckpt_path.name}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Isolate feature dimensionality directly from the saved linear layer configuration
    feature_dim = state_dict["head.weight"].shape[1]

    # 1. Instantiate the frozen backbone and matching structural linear readout
    backbone = spec.model_factory(device)
    head = nn.Linear(feature_dim, 1, bias=True).to(device)

    # 2. Build dummy buffers with identical tracking shapes to receive the checkpoint data
    cov_mask = np.zeros(feature_dim, dtype=bool)
    scaler_mean = np.zeros(feature_dim, dtype=np.float32)
    scaler_scale = np.ones(feature_dim, dtype=np.float32)

    # 3. Assemble full pipeline graph and inject saved state
    pipeline = BrainAgePipeline(
        backbone=backbone,
        head=head,
        cov_mask=cov_mask,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        postprocess_fn=spec.postprocess
    ).to(device)

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
    pipeline.eval()

    return pipeline


def verify_saved_pipelines(args, specs_dict: dict, demographics, device, target_spec):
    """
    Automated QA suite that evaluates each trained checkpoint asset at inference.
    Compares runtime performance directly with saved evaluation metadata.
    """
    print("\n" + "="*50)
    print("RUNNING POST-TRAINING INFERENCE INTEGRITY SUITE")
    print("="*50)

    subjects = demographics["Subject"].astype(str).tolist()

    for model_name, spec in specs_dict.items():
        ckpt_path = args.output_dir / f"{spec.cache_name}_{target_spec.checkpoint_suffix}_affine.pt"
        if not ckpt_path.exists():
            print(f"Skipping verification for {model_name}: No saved checkpoint found.")
            continue

        print(f"\nEvaluating Model Architecture: {model_name}")
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        metric_key = "test_mae" if target_spec.task_type == "regression" else "test_accuracy"
        expected_test_metric = checkpoint.get(metric_key)
        seed = checkpoint.get("seed", args.seed)
        test_size = checkpoint.get("test_size", args.test_size)

        # 1. Isolate the exact sub-population split evaluated during training
        if model_name == "anatcl":
            kept_subjects, paths = cat12_paths_and_subjects(subjects, args.nki_bids_root)
        elif model_name == "fs_roi":
            kept_subjects, _, _ = roi_volume_rows_and_subjects(subjects, args.roi_csv)
            paths = kept_subjects  # ROIVolumeDataset indexes by subject id, not a filesystem path
        else:
            kept_subjects, paths = existing_paths_and_subjects(subjects, turboprep_paths(subjects, args.turboprep_root))

        aligned_demo = demographics.set_index("Subject").loc[kept_subjects]
        y_all = aligned_demo[target_spec.demographics_column].map(target_spec.encode).to_numpy(dtype=np.float32)
        sex_all = aligned_demo["Sex"].to_numpy()

        # Re-index the categorical positions using identical parameters to isolate the test split
        indices = np.arange(len(kept_subjects))
        _, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=sex_all
        )

        test_subjects = [kept_subjects[i] for i in test_idx]
        test_y_true = y_all[test_idx]

        # 2. Build verification data loader restricted purely to the test dataset
        # Use only the test-set paths without mutating spec.dataset, which may be
        # reused elsewhere in the same run.  A shallow copy is sufficient because
        # path entries are immutable (str or pathlib.Path).
        #
        # fs_roi's ROIVolumeDataset indexes feature_matrix POSITIONALLY, not
        # by subject id -- reassigning only .paths (as every other model's
        # Dataset supports, since their __getitem__ re-reads a file from
        # disk per path) would desync it from feature_matrix here. Build a
        # separate subsetted dataset for the loader instead (via
        # .subset()) rather than mutating spec.dataset in place -- FeatureSpec
        # is a frozen dataclass, so spec.dataset itself can't be reassigned
        # anyway, unlike the in-place .paths mutation the other branch uses.
        if model_name == "fs_roi":
            loader_dataset = spec.dataset.subset(test_idx)
        else:
            original_paths = list(spec.dataset.paths)
            spec.dataset.paths = [original_paths[i] for i in test_idx]
            loader_dataset = spec.dataset

        test_loader = DataLoader(
            loader_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # 3. Reconstruct integrated execution module and run inference pass
        pipeline = load_saved_pipeline(ckpt_path, spec, device)

        inferred_preds = []
        with torch.no_grad():
            for x in test_loader:
                preds = pipeline(x.to(device))
                inferred_preds.append(preds.squeeze(1).cpu().numpy())

        # Restore original paths so spec.dataset is not left in a partial state
        # (fs_roi never mutated spec.dataset in the first place -- see above).
        if model_name != "fs_roi":
            spec.dataset.paths = original_paths

        inferred_preds = np.concatenate(inferred_preds, axis=0)

        # 4. Perform assertion verification checks
        if target_spec.task_type == "regression":
            runtime_metric = mean_absolute_error(test_y_true, inferred_preds)
            print(f" -> Expected Saved MAE: {expected_test_metric:.4f}")
            print(f" -> Computed Runtime MAE: {runtime_metric:.4f}")
        else:
            # inferred_preds is the pipeline's raw (pre-sigmoid) logit.
            inferred_probs = 1.0 / (1.0 + np.exp(-inferred_preds))
            inferred_classes = (inferred_probs >= 0.5).astype(np.float32)
            runtime_metric = float((inferred_classes == test_y_true).mean())
            print(f" -> Expected Saved Accuracy: {expected_test_metric:.4f}")
            print(f" -> Computed Runtime Accuracy: {runtime_metric:.4f}")

        if expected_test_metric is not None:
            abs_discrepancy = abs(runtime_metric - expected_test_metric)
            print(f" -> Absolute Discrepancy: {abs_discrepancy:.6f}")
            assert abs_discrepancy < 1e-3, (
                f"Inference regression mismatch on {model_name}! "
                f"Runtime metric ({runtime_metric:.4f}) deviated from expected checkpoint metrics."
            )
            print(f"✓ {model_name} Pipeline Verified: Mathematical parity confirmed.")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a RidgeCV brain-age linear head on frozen 3D backbone features.")
    parser.add_argument(
        "--model",
        choices=["optim_uncnn", "brainiac", "swinbrain", "anatcl", "simclr", "fs_roi"],
        default=os.environ.get("model", "optim_uncnn"),
    )
    parser.add_argument("--anatcl-level", choices=["global", "local"], default=os.environ.get("anatcl_level", "global"))
    parser.add_argument("--anatcl-folds", default=os.environ.get("anatcl_folds", "all"), help="'all' or a single fold index.")
    parser.add_argument(
        "--roi-csv", type=Path, default=DEFAULT_ROI_CSV,
        help="--model fs_roi only: build_freesurfer_roi_tables.py's output CSV of per-subject FreeSurfer ROI volumes.",
    )
    parser.add_argument("--target", choices=["age", "sex", "bmi"], default=os.environ.get("target", "age"))
    parser.add_argument(
        "--participants-tsv", type=Path, default=DEFAULT_PARTICIPANTS_TSV,
        help="--target bmi only: longitudinal participants.tsv BMI is joined in from (see src/targets.py).",
    )
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--turboprep-root", type=Path, default=DEFAULT_TURBOPREP_ROOT)
    parser.add_argument("--nki-bids-root", type=Path, default=DEFAULT_NKI_BIDS_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--brainiac-ckpt", type=Path, default=DEFAULT_CHECKPOINT_DIR / "BrainIAC.ckpt")
    parser.add_argument("--swinbrain-ckpt", type=Path, default=DEFAULT_CHECKPOINT_DIR / "Brain_Swin_UNETR.pth")
    parser.add_argument("--simclr-ckpt", type=Path, default=DEFAULT_CHECKPOINT_DIR / "simclr_3d_brain_foundation.tar")
    parser.add_argument("--anatcl-ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--anatcl-repo-root", type=Path, default=ANATCL_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--alt-alphas", action="store_true", default=bool(os.environ.get("alt")))
    parser.add_argument("--exclude-subject", action="append", default=list(DEFAULT_EXCLUDE_SUBJECTS))
    return parser.parse_args(args)


def main():
    args = parse_args()
    global ANATCL_REPO_ROOT
    ANATCL_REPO_ROOT = args.anatcl_repo_root
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_spec = get_target_spec(args.target)

    demographics = pd.read_csv(args.demographics, delimiter="\t", names=["Subject", "Sex", "Age"])
    demographics["Subject"] = demographics["Subject"].astype(str)
    if args.exclude_subject:
        demographics = demographics[~demographics["Subject"].isin(args.exclude_subject)].reset_index(drop=True)
    if args.target == "bmi":
        demographics = merge_bmi(demographics, args.participants_tsv)

    spec, kept_subjects, num_workers = make_feature_spec(args, demographics)
    X, y, feature_subjects = load_or_extract_features(args, spec, demographics, kept_subjects, device, num_workers, target_spec)
    aligned_demo = demographics.set_index("Subject").loc[feature_subjects]
    sex = aligned_demo["Sex"].to_numpy()
    train_ridge_head(args, X, y, sex, spec, device, target_spec)

    specs_dict = {args.model: spec}
    verify_saved_pipelines(args, specs_dict, demographics, device, target_spec)


if __name__ == "__main__":
    main()
