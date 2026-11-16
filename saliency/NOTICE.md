# Third-party notices

The code in this repository is MIT-licensed (see the repository's top-level `LICENSE`), but this
pipeline **depends on** several third-party assets that are not vendored
here and carry their own terms. Check each before publishing results,
sharing derived artifacts, or making this repo (or any fork) public.

## Data

**NKI Rockland Sample** (`NKI_DEMOGRAPHICS`, `TURBOPREP_ROOT`, `NKI_BIDS_ROOT`)
is distributed under its own data-use agreement (accessed via COINS/NITRC).
Nothing derived from it -- subject IDs, ages, feature caches, per-subject
saliency maps -- is included in this repository, and none of it should be
redistributed without checking that agreement. `saliency_outputs/` and any
`*_X_features_raw.npy` / `*_y_age.npy` / `*_subjects.npy` files are
gitignored for this reason, not just their size.

## Pretrained backbone checkpoints

Not vendored in this repo; see `config/paths.example.env` for where the
code expects to find them.

- **BrainIAC** (`BrainIAC.ckpt`): Mass General Brigham, **research-only
  license**. Confirm terms before any redistribution of the checkpoint
  itself or of weights fine-tuned from it.
- **AnatCL**: MIT-licensed, but lives in its own repository
  (https://github.com/EIDOSLAB/AnatCL) -- clone separately, not included
  here.
- **SwinBrain** (`Brain_Swin_UNETR.pth`) / **SimCLR**
  (`simclr_3d_brain_foundation.tar`): obtain from their respective sources;
  check their license terms independently.

## This repo's own trained artifacts

`checkpoints/heads/*_brain_age_head.pt` are small (a few KB to ~130KB)
linear-head weights this pipeline fits on top of the frozen backbones above
(RidgeCV coefficients + a `StandardScaler`'s per-feature mean/std). They do
not contain raw imaging data, but `scaler_mean`/`scaler_scale` are
aggregate statistics computed over the NKI cohort's extracted features --
confirm this is compatible with the NKI data-use agreement before treating
these as freely shareable.

## TemplateFlow

`TEMPLATEFLOW_MNI_DIR` points at TemplateFlow's MNI152NLin2009cAsym
template, a public, community-maintained resource
(https://www.templateflow.org/) -- no restriction beyond its own citation
request.
