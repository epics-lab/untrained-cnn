# Saliency maps and sanity checks for un-CNN

This directory of [`epics-lab/untrained-cnn`](../README.md) computes saliency maps for the un-CNN and
for five pretrained foundation-model backbones, and runs the sanity checks on them. It is separate
from the `uncnn` package and from `experiments/`: nothing here is imported by the rest of the repo.

Linear-probe regression/classification from frozen 3D backbone features,
plus vanilla-gradient saliency mapping and sanity-check analysis, across 6
backbone/feature variants: an **untrained** random-weight covariance-pooling
CNN (`optim_uncnn` -- the study's central comparison point), and 5
pretrained foundation-model backbones (`brainiac`, `swinbrain`, `simclr`,
`anatcl_global`, `anatcl_local`).

For each backbone, a linear head is fit on top of frozen features extracted
from NKI Rockland Sample T1w MRI, against one of three targets
(`train_linear.py --target`): **age** (continuous, `RidgeCV`, the original
and default target), **bmi** (continuous, `RidgeCV`, joined in from a
separate NKI phenotype file -- see `src/targets.py`), or **sex** (binary,
`LogisticRegressionCV`, a single logit -- reuses the same demographics
`Sex` column already used to stratify the age/bmi train/test split). Every
downstream stage (`--target`, default `age`) is target-generic. Saliency
maps (vanilla gradient, backpropagated through the full frozen-backbone +
linear-head pipeline) are computed per subject, then aggregated across
subjects (mean/std), binned by the target's value/class or by prediction
error, and validated against the sanity checks from Adebayo et al. (2018,
"Sanity Checks for Saliency Maps").

**Before trusting any saliency map from this pipeline, read
[docs/known_limitations.md](docs/known_limitations.md)** -- in particular,
`optim_uncnn` produces 3 separate per-channel maps (`optim_uncnn_raw`,
`optim_uncnn_rank`, `optim_uncnn_sobel`), only one of which
(`optim_uncnn_raw`) is interpretable as sensitivity to raw voxel intensity,
plus a gradient-amplification caveat from its covariance-feature
normalization -- neither applies to the other 5 models.

## Data and third-party dependencies

**This repo does not include any data or pretrained checkpoints.** See
[NOTICE.md](NOTICE.md) for what's required and its licensing:

- NKI Rockland Sample T1w MRI + demographics (own data-use agreement)
- BrainIAC, SwinBrain, SimCLR pretrained checkpoints (BrainIAC is
  research-only licensed)
- AnatCL (separate MIT-licensed repo, clone independently)
- TemplateFlow's MNI152NLin2009cAsym template (public)

## Setup

```bash
git clone https://github.com/epics-lab/untrained-cnn.git
cd untrained-cnn/saliency
pip install -r requirements.txt
cp config/paths.example.env config/paths.env   # edit with your paths
cp slurm/env.example.sh slurm/env.sh           # edit with your cluster/venv setup
```

All commands below run from this `saliency/` directory. The SLURM scripts use
`SLURM_SUBMIT_DIR` as the repo root, so submit them from here, and pass your
own account, e.g. `sbatch --account=<your-account> slurm/compute_saliency.sh`
(no account or email is hard-coded). A few scripts need extra paths that have no
default: `HEADS_DIR` (externally pre-trained heads), `NILEARN_VENV` (a venv with
matplotlib>=3.8 and nilearn) and `FREESURFER_DATASET_ROOT` (templates only); each
script's header says which ones it needs.

`slurm/env.sh` is what the SLURM scripts source; `config/paths.env` covers
the same variables for interactive use (`set -a; source config/paths.env; set +a`).
Both are gitignored -- they're machine/account-specific.

## Pipeline

Run in this order (SLURM wrappers in `slurm/`, calling the `src/` scripts):

1. **`slurm/train_all_models.sh`** -- single-shot job, fits all 6 models'
   linear heads on the full cohort for one target (`TARGET` env var, default
   `age`; `sex`/`bmi` also supported -- see above). Writes `checkpoints/heads/`
   inputs (copy the small `*_{brain_age,sex,bmi}_head.pt` files there once
   you're happy with them) and feature caches into `artifacts/` (gitignored).
2. **`slurm/compute_saliency.sh`** -- per-subject array job (`--array=0-N-1`,
   `N` = cohort size). Computes baseline saliency + the input-randomization
   sanity check for all 6 models per subject, into `saliency_outputs/<SUBJECT_ID>/`.
3. **`slurm/compute_sanity_checks.sh`** -- single job (not per-subject), runs
   the full Adebayo et al. (2018) sanity-check battery (`--randomize-model`,
   `--randomize-input`, `--randomize-data`) for all 6 models against one
   reference subject, into `saliency_outputs/` directly (`arch_*`/`data_*`/
   `input_*`, no subject-ID prefix) -- needs step 1's feature caches.
4. **`slurm/visualize_subjects.sh`** -- per-subject figures (baseline vs.
   sanity-check battery, cross-model comparison) plus one all-subjects
   figure per model.
5. **`slurm/visualize_aggregate.sh`** -- across-subject mean/std maps and
   figures.
6. **`slurm/visualize_bins.sh`** -- mean/std maps and figures binned by the
   target's true value (age/bmi) or class (sex) and, for the two continuous
   targets, by prediction error too (`TARGET` env var, default `age`) -- one
   figure per model per binning.
7. **`slurm/compute_correlations.sh`** -- pairwise and sanity-check Spearman
   correlation matrices/plots across every (model, subject) saliency map.

Every `src/*.py` script also runs standalone (`python3 src/train_linear.py
--help`, etc.) if you want to run a single model/subject outside SLURM.

`fs_roi` (a linear head trained directly on FreeSurfer regional volumes
instead of a deep-net backbone -- `train_linear.py --model fs_roi`,
`calculate_gradcam_tabular.py`) needs its saliency painted onto a
segmentation template in Turboprep's MNI152Lin space before it can be
compared against the other 6 models -- see
[docs/mni152_template_registration.md](docs/mni152_template_registration.md)
for how to build that template (`templates/build_mni152_dkt_templates.sh`)
and [docs/known_limitations.md](docs/known_limitations.md)'s `fs_roi`
section for why a native-space template doesn't work.

## Layout

```
src/            training, saliency computation, aggregation/binning, plotting
slurm/          SLURM job scripts (+ env.sh, gitignored, for cluster setup)
config/         paths.env (gitignored) for non-SLURM/interactive use
checkpoints/heads/  small trained linear-head checkpoints (gitignored -- dir must exist locally, see slurm/train_all_models.sh)
templates/      fs_roi segmentation templates (build script tracked; generated .nii.gz/.mgz + templates/_work/ gitignored -- see docs/mni152_template_registration.md)
docs/           known_limitations.md, attribution_methods.md, and other methodology notes
artifacts/      gitignored: feature caches + fresh checkpoints from training
saliency_outputs/  gitignored: per-subject/aggregate/bin saliency maps + figures
```

## Method notes

- **Why an untrained CNN at all?** `optim_uncnn`'s backbone
  (`CNN3D_CovPool`) is randomly initialized and never trained -- only its
  linear ridge head is fit. It's the central object of comparison against
  the pretrained foundation-model backbones: how much of a backbone's
  usefulness for this task comes from pretraining vs. from architecture
  alone.
- **Why covariance pooling for `optim_uncnn`?** Second-order/covariance
  pooling (mean + pairwise channel covariance per stage) captures
  texture-like co-activation structure a simple average discards. See
  `docs/known_limitations.md` for the saliency-side cost of this choice.
- **Why vanilla gradient, not Grad-CAM/etc.?** Vanilla gradient is the
  simplest attribution method with the fewest hyperparameters and
  assumptions, which matters for cross-architecture comparability across 5
  very different backbones -- but see the sanity-check caveat in
  `docs/known_limitations.md`. `--attribution-method` also offers
  `gradient_x_input` and `integrated_gradients` (with a zero or
  population-mean baseline) as alternatives; see
  [docs/attribution_methods.md](docs/attribution_methods.md) for formal
  definitions and references for all three.
