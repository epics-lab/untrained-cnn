# Known limitations

Methodology caveats worth knowing before interpreting saliency maps from
this pipeline. Each is also noted inline in the relevant code.

## `optim_uncnn`: signed-sqrt gradient amplification near zero

`optim_uncnn` is the only model whose features go through covariance
pooling (`CNN3D_CovPool._cov_pool` in `src/train_linear.py`) and the
associated signed-square-root normalization (`BrainAgePipeline._signed_sqrt`,
gated by `cov_mask` -- every other model has an all-`False` mask and never
touches this code path).

The derivative of `sign(x)*sqrt(|x|+eps)` is `1/(2*sqrt(|x|+eps))`, which
grows sharply as `x` approaches 0 -- roughly 500x at `x=0` vs. 0.5x at `x=1`
(`eps=1e-6`). Since this only applies to the covariance-pooled features, a
covariance entry that happens to sit near zero gets its saliency
contribution amplified by orders of magnitude purely from this derivative
singularity, not because it's actually predictive of the target (age, bmi,
or sex). This can show up as anatomically incoherent, high-magnitude voxels
in `optim_uncnn`'s saliency maps (both `signed` and `abs`) that reflect
numerical proximity to zero rather than genuine importance.

There's no fix that doesn't also change the forward value, which would
invalidate the trained head's weights (`train_ridge_head`'s
`max_abs_diff < 1e-3` assertion ties the deployed pipeline's raw output to
the fitted sklearn model's own raw output -- `reg.predict()` for a
regression target (age, bmi), `reg.decision_function()` for the
classification target (sex), since `reg.predict()` there returns class
labels, not a comparable raw score) -- so this is left as a documented
caveat rather than corrected numerically. Every `*_sanity_checks.json`
written for `optim_uncnn` includes a `known_limitations` field pointing
back here.

## Untrained_CNN features are raw in the externally pre-trained `*_sklearn` heads

`slurm/compute_saliency_pretrained_heads.sh` runs this pipeline's own
`calculate_gradcam.py` against linear heads trained independently by a
collaborator, not this repo's `train_linear.py`
(`$HEADS_DIR/`, manifest
`heads_affine_linear_sklearn_summary_{NKI,PPMI,HBN}.csv` -- see
`src/list_pretrained_heads.py`). Those checkpoints were fit with a different
training script (`benchmark_affine_linear.py`-style: `StandardScaler` +
`Ridge`/`LogisticRegression` inside an outer/inner cross-validated grid
search), not `train_linear.py`'s `train_ridge_head`.

The practical consequence for `Untrained_CNN` (`optim_uncnn`) specifically:
that script feeds `CNN3D_CovPool`'s covariance-pooled channels into
`StandardScaler`/`Ridge` **raw**, with no `signed_sqrt` normalization
applied first. This repo's own `optim_uncnn` head (`train_linear.py`'s
`train_ridge_head`) and the external older, non-`_sklearn` checkpoint set
(`heads_affine_linear/`) both apply `signed_sqrt` to those features before
fitting -- see the "signed-sqrt gradient amplification near zero" section
above. Every checkpoint under `heads_affine_linear_sklearn/` has `cov_mask`
all-`False` (confirmed via `torch.load`), so `BrainAgePipeline.forward`
skips the `signed_sqrt` nonlinearity entirely when running these heads
through `calculate_gradcam.py`.

Two things follow from this, and they are not interchangeable:

- **The fitted model itself differs**, not just its packaging: `Ridge`/
  `LogisticRegression` is fit on a different (unnormalized, more
  heavy-tailed) feature distribution than this repo's own `optim_uncnn`
  head sees, so the selected regularization strength, learned weights, and
  downstream saliency pattern are not directly comparable between the two
  -- a performance or saliency difference between them is confounded with
  this preprocessing choice, not purely a training-protocol difference.
- **The saliency maps don't show the signed-sqrt derivative-singularity
  artifact** described above (no near-zero covariance-feature gradient
  blowup), since that nonlinearity is never in this head's forward pass --
  at the cost of forgoing whatever variance-stabilizing benefit
  `signed_sqrt` gives the linear fit for genuinely covariance-structured
  features.

Every other model family under `heads_affine_linear_sklearn/` (AnatCL,
BrainIAC, SwinBrain, 3D-Neuro-SimCLR) has an all-`False` `cov_mask`
regardless of training script -- this caveat is specific to `Untrained_CNN`.

## `optim_uncnn`: three separate per-channel maps (raw/rank/Sobel), each with its own scope

`UnCNNDataset.__getitem__` (`src/train_linear.py`) stacks 3 channels: raw
intensity, a rank filter, and Sobel-gradient magnitude -- the latter two
computed with plain numpy/scipy calls *outside* the autograd graph, before
`input_tensor.requires_grad = True` is ever set. Autograd has no visibility
into how a leaf tensor's values were produced before it became a leaf, so
the single backward pass in `compute_direct_pipeline_saliency`
(`src/calculate_gradcam.py`) still produces a real, valid gradient for all 3
channel slots at once -- no extra forward/backward pass is needed to get
them. What differs is what each gradient *means*: channel 0's gradient is
`d(age)/d(raw intensity)`; channels 1/2's gradients are `d(age)/d(rank-filter
value)` and `d(age)/d(Sobel value)`, treating those already-computed values
as their own independent inputs. There is no valid chain rule from channels
1/2 back through the non-differentiable filters to raw voxel intensity --
that connection was never in the traced graph.

Rather than discard channels 1/2 (as earlier versions of this pipeline did,
keeping channel 0 only) or average all 3 together (which would mix unrelated
units and let opposite-signed gradients cancel),
`compute_direct_pipeline_saliency`/`compute_and_save_saliency` save all 3 as
separate, separately-labeled maps: `optim_uncnn_raw`, `optim_uncnn_rank`,
`optim_uncnn_sobel` (mirroring the `anatcl_global`/`anatcl_local` naming
convention -- one architecture, several named saliency-map variants). Each
is a valid sensitivity map *in its own channel's units*, but **only
`optim_uncnn_raw` is interpretable as sensitivity to raw voxel intensity**;
`optim_uncnn_rank`/`optim_uncnn_sobel` show sensitivity to that channel's
own filtered value, not to the raw image that produced it. Don't compare
`optim_uncnn_rank`/`optim_uncnn_sobel` directly against other models' raw-
intensity-space saliency maps as if they were in the same units.

Other multi-channel models (`swinbrain`) repeat the *same* raw image across
channels rather than deriving new ones from it, so this concern doesn't
apply there. Their channel attributions are summed by the chain rule to give
the attribution with respect to the original MRI. Averaging would produce
the same independently normalized spatial picture but incorrect raw
magnitudes, ROI totals, and Integrated Gradients completeness.

## Integrated Gradients is numerically, not exactly, complete

The exact Integrated Gradients path integral satisfies
`sum(IG) = F(x) - F(x0)`. This implementation uses a finite right-endpoint
Riemann sum (`--ig-steps`, default 50), so activation, pooling, or attention
changes along the path can leave a non-negligible numerical residual. Every
Integrated Gradients run records `baseline_prediction`, `prediction_delta`,
`total_attribution`, `completeness_error`, and
`relative_completeness_error` in its sanity-check JSON. The sum is computed
over all channels before repeated channels are collapsed and before spatial
resampling. Check convergence across increasing step counts before treating
voxel or ROI totals as a decomposition in predicted years.

## Across-subject signed means can hide inconsistent effects

`aggregate_saliency.py --stat mean` averages signed maps voxel by voxel, so
large positive and negative subject-level attributions can cancel. A near-zero
signed mean can therefore indicate either consistently negligible attribution
or strong attribution with inconsistent direction. The separately aggregated
`abs` maps answer a different question--mean attribution magnitude--and do not
preserve an additive prediction decomposition or Integrated Gradients
completeness. Interpret the signed and absolute aggregate panels together;
subject-level/ROI distributions and sign consistency are needed before making
population-level anatomical claims.

## Resampling pipeline: from training input to the displayed template

Every saliency map goes through up to three separate spatial transforms
before it reaches a figure, across two different files. Getting the *shape*
right at each stage is necessary but not sufficient -- getting the
*affine* (the voxel-to-physical-space mapping) right at each stage is what
actually keeps a given array index meaning the same anatomical location
throughout. This section documents all three stages per model, and a real
shrinkage bug that existed in stage 2 until it was fixed (see below).

**Stage 1 -- forward preprocessing** (`src/train_linear.py`, each model's
`Dataset.__getitem__`): resizes/crops the subject's native image down to
whatever fixed input size that model's backbone expects. This is what the
model actually sees, and therefore what the raw gradient is computed with
respect to.

| Model | Dataset class | Transform | Target shape |
|---|---|---|---|
| `optim_uncnn` | `UnCNNDataset` | `scipy.ndimage.zoom` (proportional resize) | `(91, 109, 91)` |
| `brainiac` | `BrainIACDataset` | `torch.nn.functional.interpolate` (trilinear, `align_corners=False`) | `(96, 96, 96)` |
| `swinbrain` | `SwinBrainDataset` | `F.interpolate` (trilinear, `align_corners=False`) | `(128, 128, 64)` |
| `anatcl` | `CAT12VBMDataset` | `F.interpolate` (trilinear, `align_corners=False`) | `(121, 128, 121)` |
| `simclr` | `TurboPrepDataset` | axis transpose + **center crop** (no interpolation at all) | `(150, 192, 192)` |

`simclr` is the odd one out: it never resizes, so no interpolation error is
possible for it -- but the center crop means voxels outside the crop window
are never seen by the model at all, and can never have any attribution (see
"simclr's field-of-view is smaller than the other models'" below).

**Stage 2 -- `restore_native_space`** (`src/calculate_gradcam.py`): maps the
model-resolution saliency array back to a spatially correct NIfTI, so a
saved `.nii.gz` file's array indices correspond to real anatomical
locations via its affine.

- `simclr`: `_undo_turboprep_transform` exactly inverts the transpose+crop
  (both lossless, shape-preserving placement operations -- no interpolation
  needed) and reuses the subject's own native affine unchanged.
- Every other model: **as of this pipeline, the array is kept at its own
  small native shape** (e.g. `(91,109,91)` for `optim_uncnn`), with a
  rescaled affine -- the affine's spatial (rotation+scaling) block is
  multiplied by the ratio of voxel sizes, translation (world-space origin)
  left unchanged. This is a corner-anchored rescale: it doesn't resample
  anything, it just correctly describes that each small-array voxel spans
  more physical space than a native-resolution voxel would.

  **This was not always the behavior.** Earlier versions of this function
  resampled the small array *up* to the subject's full native shape with a
  second `ndi.zoom` call, before stage 3 ever ran. That second resample was
  redundant with stage 3 (which has to resample onto the template grid
  regardless) and measurably shrank the visible saliency extent: verified
  empirically, with no model or gradient computation involved at all (pure
  image resampling), that two lossy resamples in a row shrank the
  thresholded "tissue" bounding box by up to ~10% for `optim_uncnn` and
  ~17% for `swinbrain` (worst on whichever axis was downsampled most
  aggressively in stage 1). Switching to the affine-rescale approach above
  cut that to ~1-2%, roughly what a single resample's interpolation blur
  costs on its own. If you have `saliency_outputs/` generated before this
  fix, regenerate it -- the old files are shape-`(193,229,193)`-style
  (upsampled) and carry the shrinkage; new files are shape-`(91,109,91)`-style
  (native small resolution) and don't.

  This shrinkage was invisible under the `gradient` attribution method,
  which tends to be nonzero almost everywhere (including background,
  since raw `d(age)/d(x)` sensitivity isn't constrained to respect
  anatomical boundaries), so the overlay visually washed over the whole
  image regardless of whether the underlying array was correctly sized.
  `gradient_x_input`/`integrated_gradients`, by contrast, are
  mathematically forced to exactly zero wherever the input (or its
  difference from the IG baseline) is exactly zero -- e.g. background
  voxels -- so those methods correctly reveal the true (masked) extent of
  the data. That's what made the pre-existing shrinkage visible for the
  first time: not a new bug those methods introduced, but a real one they
  happened to unmask.

**Stage 3 -- template resampling** (`src/visualize_saliency.py`,
`resample_saliency` -> `nilearn.image.resample_to_img`): the one resample
that's actually unavoidable, since every model's figure needs to land on
the same shared MNI template grid for a fair side-by-side comparison. This
uses each file's own affine (set in stage 2) to correctly place it,
regardless of what native shape/resolution it's actually stored at. For the
four `turboprep`-based models (`optim_uncnn`/`brainiac`/`swinbrain`/`simclr`),
this is close to an identity/near-lossless resample, since turboprep already
normalizes every subject onto that exact template grid before any
model-specific cropping/resizing happens (see
`src/visualize_saliency.py`'s module docstring). `anatcl` (CAT12's own,
different template family) genuinely needs a real geometric resample here,
not just a shape/resolution change.

**Side effect of the stage-2 fix worth knowing about**: going straight from
a coarse ~2mm-effective-resolution array (`optim_uncnn`, etc.) to the 1mm
template in one resample can show a faint grid/moiré pattern in the
displayed overlay that a pre-smoothing intermediate resample used to hide.
This is not a new bug -- it's a more honest display of the fact that
`optim_uncnn`'s backbone genuinely only operates at that coarser
resolution. Not currently corrected for; if it becomes visually
distracting, the fix would be display-only smoothing in
`visualize_saliency.py`, not another intermediate resample in
`calculate_gradcam.py` (which would reintroduce the shrinkage this section
just described fixing).

## simclr's field-of-view is smaller than the other models'

`TurboPrepDataset` (`src/train_linear.py`, used only by `simclr`) performs
a genuine spatial **center crop** to `(150, 192, 192)` out of the
transposed ~193x193x229 volume, discarding the margins entirely before the
image ever reaches the model. Unlike every other model (which only resize,
see above), there is no interpolation error possible here -- but by the
same token, whatever anatomy falls outside that crop window was **never
seen by the model at all**, and can never have any attribution, for any
attribution method. This shows up as a visibly smaller nonzero
region/field-of-view in `simclr`'s saliency figures compared to the other
5 models, most noticeable now that `gradient_x_input`/`integrated_gradients`
correctly reveal the true (zero-padded) extent rather than `gradient`'s
near-universal coverage masking it. Whether the crop actually excludes real
brain tissue (as opposed to just background margin) hasn't been fully
verified for every subject -- worth checking before treating `simclr`'s
saliency coverage as complete.

## Sanity checks are a necessary, not sufficient, check

`--randomize-model` / `--randomize-data` / `--randomize-input` implement the
sanity checks from Adebayo et al. (2018, "Sanity Checks for Saliency Maps",
NeurIPS): a saliency method that stays highly correlated with its baseline
after the model or its training labels are randomized is *not* explaining
what the trained model actually learned. Passing these checks (low
correlation after randomization) is necessary for the map to be considered
meaningful, but it is not sufficient evidence that the map correctly
localizes the anatomical regions actually driving the prediction --
vanilla gradient saliency is known to be noisy at the single-voxel level
even when it passes these checks.

## Raw gradient magnitudes are not comparable across models

Different backbones have very different parameter scales, depths, and
input normalizations, so raw vanilla-gradient magnitude is not a
meaningfully comparable "importance" scale across models (see
`src/visualize_saliency.py`'s module docstring for citations). Always use
`--norm` when comparing maps across models; never compare two models'
un-normalized colorbars directly.

## `fs_roi`: atlas choice, a dropped region, and one unpainted feature

`fs_roi` (`src/train_linear.py`'s `fs_roi` `FeatureSpec` branch) is trained
on FreeSurfer *regional volumes* rather than deep-net features, so its
"saliency" is a per-ROI vector, not a per-voxel gradient -- see
`src/calculate_gradcam_tabular.py`'s module docstring for why it needs its
own saliency script and `src/roi_to_volume.py` for how that vector gets
painted onto a `.nii.gz` for display. Three choices/artifacts in how its
feature table (`src/build_freesurfer_roi_tables.py`) is built are worth
knowing before interpreting its saliency maps or comparing them anatomically
against the segmentation template's own labels.

**Cortical atlas: `aparc.DKTatlas`, not `aparc`.** `regionsurfacestats.tsv`
carries ~30 different atlas parcellations per subject;
`load_regionsurfacestats_tsv` filters to exactly one, `aparc.DKTatlas` (the
"DKT" modification of Desikan-Killiany, 31 regions/hemisphere). The default
`--seg-template` (`aparc+aseg.mgz`, a real subject's `recon-all` output)
was, however, segmented with the standard **`aparc`** atlas, not DKT --
these are similar but not identical parcellations: DKT deliberately drops
temporal pole, frontal pole, and banks-of-superior-temporal-sulcus from its
ontology entirely (considered unreliable to segment consistently), regions
`aparc`'s LUT labels still carry. This mismatch is exactly why
`paint_region_values` reports `unmatched` labels rather than silently
zero-filling them -- if you swap in a `aparc.DKTatlas`-segmented template
via `--seg-template`, this specific gap goes away, but you'd then need to
verify its LUT indices match `FreeSurferColorLUT.txt`'s conventions instead.

**`ctx-rh-temporalpole` is dropped from the feature table entirely**
(`build_freesurfer_roi_tables.py`'s `min_present_frac` column-drop step).
Investigated directly: `aparc.DKTatlas` has no `lh_temporalpole` column at
all across the cohort (consistent with DKT's own ontology, see above), and
`rh_temporalpole` is present for only 4 of 958 subjects
(`A00019903`/`A00033834`/`A00038285`/`A00065995`), with `GrayVol` values of
`0.0`/`0.0`/`3.0`/`1.0` mm³ -- implausible for a real cortical region (every
other region here is in the hundreds-to-thousands mm³ range), so these look
like stray leftover rows in those 4 subjects' raw FreeSurfer output rather
than genuine DKT-atlas data. `RidgeCV` can't fit through a column that's
`NaN` for 954/958 subjects, and the 4 present values aren't trustworthy
anyway, so the column is dropped (not imputed) rather than dropping those 4
subjects' rows. Net effect: `fs_roi` has zero features for the temporal
pole on either hemisphere, in the training data, the trained head's
weights, and every saliency map -- it structurally cannot attribute
anything to that region, which is a real gap in coverage, not just a
missing label.

**The unlateralized `Cerebral-White-Matter` feature never gets painted.**
`brainmeasures.tsv` reports whole-brain white matter volume three ways: a
combined `Cerebral-White-Matter` total plus its two lateralized halves,
`Left-`/`Right-Cerebral-White-Matter` (the total is ≈ the sum of the two
halves). `FreeSurferColorLUT.txt` only defines segmentation-volume indices
for the lateralized pair -- there is no single voxel label meaning "both
hemispheres' white matter," since every voxel belongs to exactly one side.
`paint_region_values` therefore reports `Cerebral-White-Matter` as
`unmatched` and it renders as nothing in the `.nii.gz` output, even though
it **is** a real, trained feature with its own weight in the RidgeCV head
and its own entry in the `.csv`/`.npy` saliency output
(`calculate_gradcam_tabular.py`'s `save_roi_saliency`). Don't read a
spatial `fs_roi` figure as showing 100% of the model's attribution --
check the companion `.csv` if a specific run's total signed/abs attribution
needs to be accounted for exactly, since one feature's contribution is
invisible in the painted volume by construction, not by omission.
