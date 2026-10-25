"""Frozen feature extractors of the main benchmark.

Each encoder has an input preparation, applied to one volume, and a forward
pass on a batch. ENCODERS gives, per model, the loader, the preparation, the
batch size used for feature extraction and the input: the TurboPrep affine
volume or the CAT12 gray matter map. All models run on CPU in eval mode.
"""

import os
import sys
import types

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import uncnn
from config import path


def load_nifti(file):
    return nib.load(file).get_fdata().astype(np.float32)


def crop_or_pad(img, target):
    """Center crop axes longer than the target and zero pad shorter ones."""
    slices = []
    for a, b in zip(img.shape, target):
        if a > b:
            s = (a - b) // 2
            slices.append(slice(s, s + b))
        else:
            slices.append(slice(0, a))
    out = img[tuple(slices)]
    pads = []
    for a, b in zip(out.shape, target):
        before = (b - a) // 2 if a < b else 0
        pads.append((before, b - a - before) if a < b else (0, 0))
    if any(p != (0, 0) for p in pads):
        out = np.pad(out, pads, mode="constant", constant_values=0)
    return out


# Input preparation, one volume, without the batch dimension

def prepare_uncnn(file):
    """Resize to 91 x 109 x 91, robust scaling, then intensity, rank filtered and Sobel magnitude channels."""
    return uncnn.DS_3ch_RobustScaler([file])[0]


def prepare_anatcl(file):
    img = load_nifti(file)
    if img.shape != (121, 128, 121):
        img = F.interpolate(torch.from_numpy(img[None, None]), size=(121, 128, 121),
                            mode="trilinear", align_corners=False).squeeze().numpy()
    return torch.from_numpy(img.copy()).float()[None]


def prepare_brainiac(file):
    img = load_nifti(file)
    if img.shape != (96, 96, 96):
        img = F.interpolate(torch.from_numpy(img[None, None]), size=(96, 96, 96),
                            mode="trilinear", align_corners=False).squeeze().numpy()
    img = torch.from_numpy(img)
    return ((img - img.mean()) / (img.std() + 1e-6))[None]


def prepare_swinbrain(file):
    img = F.interpolate(torch.from_numpy(load_nifti(file)[None, None]), size=(128, 128, 64),
                        mode="trilinear", align_corners=False).squeeze(0)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img.repeat(3, 1, 1, 1)


def prepare_simclr(file):
    img = crop_or_pad(np.transpose(load_nifti(file), (2, 0, 1)), (150, 192, 192))
    img = (img - img.mean()) / (img.std() + 1e-6)
    return torch.from_numpy(img[None]).float()


# Models

def load_uncnn():
    return uncnn.load_model(seed=0, device="cpu")


def _anatcl_class():
    # The AnatCL checkpoints pickle models.estimators.AgeEstimator; alias the module so they unpickle.
    import anatcl.models
    estimators = types.ModuleType("models.estimators")
    estimators.AgeEstimator = type("AgeEstimator", (), {})
    sys.modules["models"] = anatcl.models
    sys.modules["models.estimators"] = estimators
    from anatcl import AnatCL
    return AnatCL


def load_anatcl_fold(descriptor, fold):
    enc = _anatcl_class()(descriptor=descriptor, fold=fold, pretrained=False)
    ckpt = os.path.join(path("ANATCL_WEIGHTS_DIR"), f"weights_{descriptor}_fold{fold}.pth")
    enc.backbone.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return enc.eval()


class AnatCLFolds(nn.Module):
    """The five fold models of one AnatCL descriptor; the feature is their mean."""

    def __init__(self, descriptor):
        super().__init__()
        self.descriptor = descriptor
        self.folds = nn.ModuleList(load_anatcl_fold(descriptor, fold) for fold in range(5))

    def forward(self, x):
        return torch.stack([enc(x) for enc in self.folds]).mean(0)


class BrainIAC(nn.Module):
    """ViT backbone; the feature is the first output token, 768 dimensions."""

    def __init__(self):
        super().__init__()
        from monai.networks.nets import ViT
        ckpt = torch.load(path("BRAINIAC_CKPT"), map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        self.vit = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16), hidden_size=768,
                       mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True)
        pe_key = "backbone.patch_embedding.position_embeddings"
        if pe_key in sd:
            self.vit.patch_embedding.position_embeddings = nn.Parameter(sd[pe_key].clone())
        missing, _ = self.vit.load_state_dict(
            {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
        # MONAI 1.4 builds cross attention layers that forward never uses and the checkpoint does not contain.
        used_missing = [k for k in missing if "cross_attn" not in k]
        if used_missing:
            raise RuntimeError(f"BrainIAC: {len(used_missing)} used keys missing, e.g. {used_missing[:5]}")
        self.eval()

    def forward(self, x):
        out = self.vit(x)
        return (out[0] if isinstance(out, tuple) else out)[:, 0]


class SwinBrain(nn.Module):
    """SwinUNETR encoder; the feature is the average pooled encoder10 output, 384 dimensions."""

    def __init__(self):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        kw = dict(in_channels=3, out_channels=3, spatial_dims=3, feature_size=24, drop_rate=0.0, attn_drop_rate=0.0)
        try:
            self.model = SwinUNETR(img_size=(128, 128, 64), **kw)
        except TypeError:  # MONAI 1.5 and later dropped img_size
            self.model = SwinUNETR(**kw)
        sd = torch.load(path("SWINBRAIN_CKPT"), map_location="cpu", weights_only=False)
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self._encoder10 = {}
        self.model.encoder10.register_forward_hook(lambda m, i, o: self._encoder10.__setitem__("x", o))
        self.eval()

    def forward(self, x):
        self.model(x)
        return self.pool(self._encoder10["x"]).view(x.size(0), -1)


class SimCLR(nn.Module):
    """3D ResNet-18 encoder of 3D-Neuro-SimCLR, 512 dimensions."""

    def __init__(self):
        super().__init__()
        from monai.networks.nets import resnet18
        self.enc = resnet18(spatial_dims=3, n_input_channels=1, num_classes=0)
        self.enc.fc = nn.Identity()
        ckpt = torch.load(path("SIMCLR_CKPT"), map_location="cpu", weights_only=False)
        self.enc.load_state_dict({k.replace("module.encoder.", ""): v for k, v in ckpt["model_state_dict"].items()
                                  if "encoder" in k and "projector" not in k}, strict=True)
        self.eval()

    def forward(self, x):
        return self.enc(x)


# name: (loader, input preparation, batch size, input)
ENCODERS = {
    "Untrained_CNN": (load_uncnn, prepare_uncnn, 1, "affine"),
    "AnatCL_Global": (lambda: AnatCLFolds("global"), prepare_anatcl, 32, "cat12"),
    "AnatCL_Local": (lambda: AnatCLFolds("local"), prepare_anatcl, 32, "cat12"),
    "BrainIAC": (BrainIAC, prepare_brainiac, 16, "affine"),
    "SwinBrain": (SwinBrain, prepare_swinbrain, 8, "affine"),
    "3D-Neuro-SimCLR": (SimCLR, prepare_simclr, 8, "affine"),
}


class Volumes(Dataset):
    def __init__(self, files, prepare):
        self.files, self.prepare = files, prepare

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.prepare(self.files[i])


def n_params(model):
    # MONAI 1.4 cross attention layers in BrainIAC are unused and not part of the model.
    return sum(p.numel() for n, p in model.named_parameters() if "cross_attn" not in n)


def extract_features(name, files):
    """One feature vector per file. AnatCL runs each fold model in turn over all files, then averages."""
    load, prepare, batch_size, _ = ENCODERS[name]
    loader = DataLoader(Volumes(files, prepare), batch_size=batch_size, num_workers=0)
    with torch.no_grad():
        if name.startswith("AnatCL"):
            folds = []
            for fold in range(5):
                enc = load_anatcl_fold(name.split("_")[1].lower(), fold)
                folds.append(np.vstack([enc(x).numpy() for x in loader]))
                del enc
            return np.mean(folds, axis=0)
        model = load()
        return np.vstack([model(x).numpy() for x in loader])
