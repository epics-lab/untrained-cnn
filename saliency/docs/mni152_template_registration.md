# Registering FreeSurfer segmentation templates to Turboprep's MNI152Lin space

`calculate_gradcam_tabular.py`'s `--seg-template` needs a discrete-label
segmentation volume that's actually in the same space every other model's
saliency map is displayed in (Turboprep's MNI152Lin grid -- see
`docs/known_limitations.md`'s `fs_roi` section for what goes wrong if it
isn't: painting per-ROI values onto a native-space FreeSurfer segmentation
and feeding that through the normal NIfTI pipeline produces a silently
misaligned brain, not an error).

`templates/build_mni152_dkt_templates.sh` builds two candidate templates in
that space -- one keeping the DKT atlas `fs_roi` was actually trained on but
based on a single subject's anatomy, one population-averaged but on a
different (DK/Destrieux) atlas -- see that script's own header comment for
the full design rationale. This page covers everything that needs to be in
place *before* running it.

## Prerequisites

### 1. Modules -- these module names/versions are Compute Canada (Alliance) specific

The exact module names and versions below (`ants/2.6.5`, `freesurfer/7.4.1`,
`git-annex/10.20231129`) are what's available on this Compute Canada
cluster's module system (`module spider <name>` to check versions on
whatever cluster you're actually on). On any other cluster or a local
install, install/load the equivalent tools (ANTs, FreeSurfer, git-annex)
however that system provides them -- the script only assumes the resulting
binaries (`antsRegistrationSyNQuick.sh`, `antsApplyTransforms`,
`mri_vol2vol`, `datalad`) are on `PATH` and `FREESURFER_HOME` is set; it
doesn't assume Compute Canada's module system itself.

```bash
module load ants/2.6.5 freesurfer/7.4.1 git-annex/10.20231129
source $EBROOTFREESURFER/FreeSurferEnv.sh   # sets FREESURFER_HOME, SUBJECTS_DIR, etc.
```

`source $EBROOTFREESURFER/FreeSurferEnv.sh` is required in addition to the
`module load` -- loading the module alone does not set `FREESURFER_HOME`,
which both `mri_vol2vol` and the script's `fsaverage` lookup need.

### 2. Python environment

```bash
source /path/to/your/venv/bin/activate
pip install templateflow   # not already in this venv; needed for step 4 below
```

### 3. FreeSurfer license

Loading the `freesurfer` module on this cluster fails at `module load` time
(before you even get to run anything) if it can't find a license:

```
Lmod has detected the following error: We did not find a suitable
license for FreeSurfer. [...] save it as freesurfer.lic file in
$HOME/.licenses/ directory.
```

Get a (free) license from
[the FreeSurfer registration page](https://surfer.nmr.mgh.harvard.edu/registration.html)
if you don't already have one, then:

```bash
mkdir -p ~/.licenses
cp /path/to/your/license.txt ~/.licenses/freesurfer.lic
chmod 644 ~/.licenses/freesurfer.lic
```

**Must be a real copy, not a symlink** -- this cluster's Lmod license check
does not follow symlinks and fails with the same "no suitable license
found" error even when a symlink correctly resolves to a valid license file
(confirmed directly: swapping the symlink for a plain `cp` was what fixed
it).

The license file itself must **never** be committed to this repo -- see the
`.gitignore` entry added alongside this doc. It lives in your home
directory (`~/.licenses/freesurfer.lic`), outside any repo, specifically so
there's no repo path to accidentally `git add`.

### 4. A real (non-placeholder) `tpl-MNI305_T1w.nii.gz`

The script fetches this itself (Step 0) via the `templateflow` package into
`templates/_work/.templateflow_cache/` -- but that fetch needs real internet
access, which compute nodes on this cluster (and most HPC clusters) do not
have. If you're running the script inside an interactive/batch job (i.e. on
a compute node), pre-fetch this file from a **login node** first, or the
script will hang/time out at Step 0:

```bash
mkdir -p templates/_work
curl -L --fail -o templates/_work/tpl-MNI305_T1w.nii.gz \
  https://templateflow.s3.amazonaws.com/tpl-MNI305/tpl-MNI305_T1w.nii.gz
```

Once that file exists and is non-empty, the script's own `if [ ! -s
"$MNI305_T1" ]` check skips the fetch entirely, so this step is safe to do
ahead of time regardless of where you eventually run the rest of the
script.

## Running it

```bash
module load ants/2.6.5 freesurfer/7.4.1 git-annex/10.20231129
source $EBROOTFREESURFER/FreeSurferEnv.sh
source /path/to/your/venv/bin/activate

cd /path/to/untrained-cnn/saliency
bash templates/build_mni152_dkt_templates.sh
```

### If running via an interactive job (`salloc`)

This is a light job -- affine-only registration on two ~180<sup>3</sup> 1mm
volumes, plus a couple of fast resamples. `--cpus-per-task=2 --mem=8G
--time=00:30:00` is generous headroom, not a tight estimate. The script does
not currently pass `-n <threads>` to `antsRegistrationSyNQuick.sh`, so
requesting more CPUs than that won't make it faster without also editing
the script (or exporting `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`) -- not
worth bothering with given how fast affine-only registration already is
here.

Do the network-dependent parts (prerequisite #4 above, and the `datalad
get` inside the script's Option 1) from a **login node**, not inside the
`salloc` session -- see that prerequisite section for why.

## What it produces

- `templates/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz` (Option 1 --
  single-subject, DKT atlas)
- `templates/aparc+aseg_fsaverage_MNI152Lin.nii.gz`,
  `templates/aparc.a2009s+aseg_fsaverage_MNI152Lin.nii.gz` (Option 2 --
  population-average, DK/Destrieux atlases)

Sanity-check whichever you use before trusting it (e.g.
`nilearn.plotting.plot_roi` overlaid on `tpl-MNI152Lin_res-01_T1w.nii.gz`)
-- affine-only registration can leave residual misalignment on atypical
anatomy, and Option 2's atlases don't match `fs_roi`'s own DKT training
atlas region-for-region (see the script's header comment).
