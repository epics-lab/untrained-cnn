"""uncnn — an untrained (frozen, random-weight) 3D CNN feature extractor.

Quick start
-----------
    import uncnn

    # one image
    feats = uncnn.extract_features("subject01.nii.gz")     # -> (1, dim) array

    # many images
    import glob
    feats = uncnn.extract_features(glob.glob("data/*.nii.gz"))  # -> (N, dim)

Each input is any NIfTI volume (.nii / .nii.gz); it is resized to the MNI152 2mm
grid (91x109x91) and turned into a 3-channel image automatically.

Preprocessing configs (pass config=...)
----------------------------------------
    "robust"    : best overall age config            (default)
    "ranksobel" : best sex-accuracy config (clip 2/98)
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import scipy.ndimage as ndi

__all__ = [
    "extract_features",
    "load_model",
    "UnCNN",
    "DepthwiseSepConv3d",
    "DS_3ch_RobustScaler",
    "DS_3ch_RankSobel",
    "load_and_resize",
    "clip_minmax",
    "norm",
    "sobel_mag",
    "rank_filter",
    "scale_robust",
    "WEIGHT_SEED",
]

# Default fixed random-weight seed for the (untrained) CNN
WEIGHT_SEED = 0


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def load_and_resize(path):
    img = nib.load(path).get_fdata().astype(np.float32)
    if img.shape != (91, 109, 91):
        img = ndi.zoom(img, [91 / img.shape[0], 109 / img.shape[1],
                              91 / img.shape[2]], order=1)
    return img


def clip_minmax(img, lo=1, hi=99):
    p_lo, p_hi = np.percentile(img, [lo, hi])
    img = np.clip(img, p_lo, p_hi)
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + 1e-6)


def norm(ch):
    mn, mx = ch.min(), ch.max()
    if mx - mn < 1e-8:
        return np.zeros_like(ch)
    return (ch - mn) / (mx - mn)


def sobel_mag(img):
    sx = ndi.sobel(img, axis=0)
    sy = ndi.sobel(img, axis=1)
    sz = ndi.sobel(img, axis=2)
    return np.sqrt(sx**2 + sy**2 + sz**2)


def rank_filter(img, size=3):
    """Replace each voxel with its local median (percentile) value.
    Scanner-invariant: only relative ordering matters."""
    ranked = ndi.percentile_filter(img, percentile=50, size=size)
    mn, mx = ranked.min(), ranked.max()
    if mx - mn < 1e-8:
        return np.zeros_like(ranked)
    return (ranked - mn) / (mx - mn)


def scale_robust(img, clip_low=2, clip_high=98, eps=1e-8):
    """Brain-mask based robust scaling: mask background (>5th percentile),
    clip within brain mask, then center on median and scale by IQR.
    More resistant to outlier voxels and scanner-specific intensity
    distributions."""
    mask = img > np.percentile(img, 5)
    lo, hi = np.percentile(img[mask], [clip_low, clip_high])
    clipped = np.clip(img, lo, hi)
    brain = clipped[mask]
    q25, q75 = np.percentile(brain, [25, 75])
    iqr = float(q75 - q25)
    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (brain - float(np.median(brain))) / (iqr + eps)
    return out


class DS_3ch_RobustScaler(Dataset):
    """3ch: robust-scaled original + rank filter + Sobel magnitude.
    Best overall age config."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = scale_robust(load_and_resize(self.paths[idx]))
        ranked = rank_filter(img, size=3)
        edges = norm(sobel_mag(img))
        return torch.from_numpy(np.stack([img, ranked, edges], axis=0)).float()


class DS_3ch_RankSobel(Dataset):
    """3ch: original (clip 2/98) + rank filter + Sobel magnitude.
    Best sex-accuracy config."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = clip_minmax(load_and_resize(self.paths[idx]), lo=2, hi=98)
        ranked = rank_filter(img, size=3)
        edges = norm(sobel_mag(img))
        return torch.from_numpy(np.stack([img, ranked, edges], axis=0)).float()


# Map a friendly config name to its preprocessing Dataset
_CONFIGS = {
    "robust": DS_3ch_RobustScaler,
    "ranksobel": DS_3ch_RankSobel,
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DepthwiseSepConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class UnCNN(nn.Module):
    """Untrained 3D CNN feature extractor: DoubleConv backbone + covariance
    pooling readout. Weights are fixed random (seeded) and never trained.

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for the standard preprocessing configs).
    seed : int or None
        RNG seed applied before weight init for reproducible random weights.
        Pass None to skip seeding (use the ambient RNG state).
    """
    def __init__(self, in_channels=3, seed=WEIGHT_SEED):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

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
        """Mean features + upper triangle of covariance matrix."""
        b, c = x.size(0), min(x.size(1), max_ch)
        mean_feats = self.avg_pool(x).view(b, -1)
        xc = x[:, :c]
        flat = xc.view(b, c, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (flat.size(2) - 1)
        idx = torch.triu_indices(c, c, offset=1)
        cov_feats = cov[:, idx[0], idx[1]]
        return torch.cat([mean_feats, cov_feats], dim=1)

    def forward(self, x):
        x1 = F.relu(self.norm1a(self.conv1a(x)))
        x1 = self.down1(F.relu(self.norm1b(self.conv1b(x1))))
        x2 = F.relu(self.norm2a(self.conv2a(x1)))
        x2 = self.down2(F.relu(self.norm2b(self.conv2b(x2))))
        x3 = F.relu(self.norm3a(self.conv3a(x2)))
        x3 = self.down3(F.relu(self.norm3b(self.conv3b(x3))))
        x4 = F.relu(self.norm4a(self.conv4a(x3)))
        x4 = self.down4(F.relu(self.norm4b(self.conv4b(x4))))
        return torch.cat([
            self._cov_pool(x1, 32),
            self._cov_pool(x2, 32),
            self._cov_pool(x3, 32),
            self._cov_pool(x4, 32),
        ], dim=1)


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #
def _resolve_device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(seed=WEIGHT_SEED, device=None):
    """Build the frozen UnCNN, move it to `device`, and set eval mode."""
    device = _resolve_device(device)
    model = UnCNN(seed=seed).to(device)
    model.eval()
    return model


def extract_features(paths, config="robust", batch_size=8, num_workers=0,
                     device=None, seed=WEIGHT_SEED, model=None):
    """Run the frozen UnCNN on one or more NIfTI volumes and return features.

    Parameters
    ----------
    paths : str or list of str
        A single .nii/.nii.gz path, or a list of them.
    config : {"robust", "ranksobel"}
        Preprocessing pipeline. "robust" = best overall age config (default);
        "ranksobel" = best sex-accuracy config (clip 2/98).
    batch_size, num_workers : int
        DataLoader settings.
    device : str or None
        "cpu" / "cuda". Auto-detected if None.
    seed : int or None
        Random-weight seed (ignored if `model` is provided).
    model : UnCNN or None
        Reuse a prebuilt model instead of constructing one.

    Returns
    -------
    numpy.ndarray of shape (n_subjects, feature_dim)
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if config not in _CONFIGS:
        raise ValueError(f"config must be one of {list(_CONFIGS)}; got {config!r}")

    device = _resolve_device(device)
    if model is None:
        model = load_model(seed=seed, device=device)
    else:
        model = model.to(device)
    model.eval()

    dataset = _CONFIGS[config](paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    feats = []
    with torch.no_grad():
        for x in loader:
            feats.append(model(x.to(device)).cpu().numpy())
    return np.vstack(feats)