#!/usr/bin/env bash
#
# Builds DKT-labeled (or, for option 2, DK/Destrieux-labeled) segmentation
# volumes in Turboprep's MNI152Lin space, for use as calculate_gradcam_
# tabular.py's --seg-template -- so fs_roi's saliency can be painted onto a
# canvas that's actually in the same space every other model's map is
# displayed in (see docs/known_limitations.md's fs_roi section for the
# native-space misalignment problem this fixes, and the email/design
# discussion this script implements both halves of).
#
# TWO approaches, both produced by this script, so they can be compared:
#
#   OPTION 1 (subject-specific, keeps the DKT atlas): one individual
#   subject's (A00008326's) own aparc.DKTatlas+aseg.mgz, carried from
#   native space into MNI152Lin via TWO composed transforms:
#     native -> MNI305   (that subject's own already-computed talairach.xfm
#                          -- a real recon-all output, not something this
#                          script generates)
#     MNI305 -> MNI152Lin (a ONE-TIME affine registration between the two
#                          fixed reference templates -- computed once below,
#                          reused for option 2 too, NOT a per-subject fit)
#   Output: templates/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz
#   Keeps the exact atlas fs_roi was trained on (62 DKT regions), at the
#   cost of every region's shape/extent being based on one person's anatomy.
#
#   OPTION 2 (population-average, but NOT the DKT atlas): FreeSurfer's own
#   fsaverage subject's aparc+aseg.mgz (Desikan-Killiany, 68 regions) and
#   aparc.a2009s+aseg.mgz (Destrieux, 148 regions) -- already in MNi305
#   space by definition, so only the SAME MNI305->MNI152Lin correction is
#   needed, no subject-specific transform at all.
#   Output: templates/aparc+aseg_fsaverage_MNI152Lin.nii.gz,
#           templates/aparc.a2009s+aseg_fsaverage_MNI152Lin.nii.gz
#   Neither atlas matches fs_roi's training atlas (DKT) region-for-region --
#   using either as --seg-template alongside the existing fs_roi checkpoint
#   would need re-deriving/retraining on that atlas first, not a drop-in
#   swap (see the email discussion this script's design followed from).
#
# NEAREST-NEIGHBOR INTERPOLATION IS NOT OPTIONAL ANYWHERE BELOW: every
# antsApplyTransforms/mri_vol2vol call here carries discrete integer region
# labels -- any other interpolation (ANTs' own default is BSpline) would
# blend adjacent different-integer labels into meaningless fractional
# values. Every call explicitly requests it; do not "simplify" that away.
#
# PREREQUISITES (this script does not install/configure any of these):
#   - module load ants/2.6.5           (antsRegistrationSyNQuick.sh, antsApplyTransforms)
#   - module load freesurfer/7.4.1     (mri_vol2vol, $FREESURFER_HOME/subjects/fsaverage,
#                                        $FREESURFER_HOME/average/mni305.cor.mgz -- needs a
#                                        valid FreeSurfer license.txt, see
#                                        https://surfer.nmr.mgh.harvard.edu/registration.html)
#   - module load git-annex/10.20231129 + datalad installed in your active
#     Python env (see src/fetch_freesurfer_tsvs.py's own docstring)
#   - `pip install templateflow` in your active Python env, for fetching a
#     REAL tpl-MNI305_T1w.nii.gz -- the local TemplateFlow project cache
#     only has an empty, never-fetched placeholder for this file (verified
#     directly earlier), so this script fetches its own copy under
#     templates/.templateflow_cache rather than trusting that shared cache.
#   - Real network access (a LOGIN node, not a compute node) for that fetch.
#
# Usage:
#   module load ants/2.6.5 freesurfer/7.4.1 git-annex/10.20231129
#   source /path/to/your/venv/bin/activate
#   bash templates/build_mni152_dkt_templates.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_DIR/templates"
WORK_DIR="$OUT_DIR/_work"
mkdir -p "$OUT_DIR" "$WORK_DIR"

MNI152_T1="$REPO_DIR/tpl-MNI152Lin_res-01_T1w.nii.gz"
MNI305_T1="$WORK_DIR/tpl-MNI305_T1w.nii.gz"

FREESURFER_DATASET_ROOT="${FREESURFER_DATASET_ROOT:?set FREESURFER_DATASET_ROOT to the NKI_FreeSurfer datalad dataset root}"
SUBJECT_ID="A00008326"
SUBJECT_SES="ses-BAS1"
SUBJECT_DIR="sub-${SUBJECT_ID}_${SUBJECT_SES}"
SUBJECT_ARCHIVE="${FREESURFER_DATASET_ROOT}/freesurfer/${SUBJECT_DIR}/${SUBJECT_DIR}_freesurfer.tar.xz"

for cmd in antsRegistrationSyNQuick.sh antsApplyTransforms mri_vol2vol datalad python3; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "FATAL: '$cmd' not found on PATH -- see this script's PREREQUISITES comment block." >&2
        exit 1
    }
done
: "${FREESURFER_HOME:?FREESURFER_HOME is unset -- module load freesurfer/7.4.1 first}"
[ -f "$MNI152_T1" ] || { echo "FATAL: $MNI152_T1 not found." >&2; exit 1; }

# check_same_space A B -- fails loudly instead of silently producing a
# wrong-but-plausible result if two volumes we're ASSUMING share a grid/
# affine (rather than deriving one via an explicit registration step)
# actually don't. Cheap insurance given how many "should be the same
# space, but wasn't" surprises this project has already hit.
check_same_space() {
    python3 - "$1" "$2" <<'PYEOF'
import sys
import nibabel as nib
import numpy as np
a, b = nib.load(sys.argv[1]), nib.load(sys.argv[2])
if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-3):
    print(f"FATAL: {sys.argv[1]} and {sys.argv[2]} do not share a grid/affine "
          f"as assumed.\n  {sys.argv[1]}: shape={a.shape} affine=\n{a.affine}\n"
          f"  {sys.argv[2]}: shape={b.shape} affine=\n{b.affine}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

#############################################################
# STEP 0: a REAL tpl-MNI305_T1w.nii.gz (the shared project cache's copy is
# an empty placeholder -- fetch our own via templateflow's own downloader,
# which knows the correct source URL, rather than guessing one).
#############################################################
if [ ! -s "$MNI305_T1" ]; then
    echo "=== Fetching a real tpl-MNI305_T1w.nii.gz ==="
    python3 -c "import templateflow" 2>/dev/null || {
        echo "FATAL: the 'templateflow' package is not installed in this Python env (pip install templateflow)." >&2
        exit 1
    }
    TEMPLATEFLOW_HOME="$OUT_DIR/.templateflow_cache" python3 -c "
from templateflow.api import get
import shutil
p = get('MNI305', suffix='T1w', extension='.nii.gz')
shutil.copy(str(p), '$MNI305_T1')
print('Fetched:', p)
"
fi
[ -s "$MNI305_T1" ] || { echo "FATAL: $MNI305_T1 is still empty after fetch attempt." >&2; exit 1; }

#############################################################
# STEP 1: MNI305 -> MNI152Lin, ONCE. This is a registration between two
# fixed, population-level REFERENCE templates (not a per-subject fit) --
# both option 1 and option 2 reuse this exact same transform below.
#############################################################
MNI305_TO_MNI152_PREFIX="$WORK_DIR/mni305_to_mni152lin_"
MNI305_TO_MNI152_AFFINE="${MNI305_TO_MNI152_PREFIX}0GenericAffine.mat"
if [ ! -f "$MNI305_TO_MNI152_AFFINE" ]; then
    echo "=== Registering MNI305 -> MNI152Lin (one-time, reference-to-reference) ==="
    antsRegistrationSyNQuick.sh -d 3 -t a \
        -f "$MNI152_T1" \
        -m "$MNI305_T1" \
        -o "$MNI305_TO_MNI152_PREFIX"
fi

#############################################################
# OPTION 1: subject A00008326's own aparc.DKTatlas+aseg.mgz
#   native -> MNI305 (that subject's own talairach.xfm)
#   MNI305 -> MNI152Lin (Step 1's transform)
#############################################################
echo "=== Option 1: subject-specific aparc.DKTatlas+aseg -> MNI152Lin ==="

echo "Fetching subject $SUBJECT_ID's recon-all archive via datalad..."
datalad get -d "$FREESURFER_DATASET_ROOT" "$SUBJECT_ARCHIVE"

echo "Extracting native aparc.DKTatlas+aseg.mgz + talairach.xfm..."
tar -xJf "$SUBJECT_ARCHIVE" -O "${SUBJECT_DIR}/mri/aparc.DKTatlas+aseg.mgz" > "$WORK_DIR/native_dkt_aseg.mgz"
tar -xJf "$SUBJECT_ARCHIVE" -O "${SUBJECT_DIR}/mri/transforms/talairach.xfm" > "$WORK_DIR/talairach.xfm"

# native -> MNI305 via mri_vol2vol (FreeSurfer's own tool applying
# FreeSurfer's own transform format) -- NOT a hand conversion of
# talairach.xfm's matrix into an ITK/ANTs .mat. FreeSurfer's tkreg/
# scanner-RAS conventions differ from ITK's LPS convention in ways that are
# easy to get quietly wrong by manually re-typing/negating matrix entries;
# mri_vol2vol already knows how to apply its own native transform format
# correctly, so let it do that step instead of reimplementing it.
echo "Native -> MNI305 (mri_vol2vol, nearest-neighbor)..."
mri_vol2vol \
    --mov "$WORK_DIR/native_dkt_aseg.mgz" \
    --targ "${FREESURFER_HOME}/average/mni305.cor.mgz" \
    --xfm "$WORK_DIR/talairach.xfm" \
    --o "$WORK_DIR/dkt_aseg_mni305.nii.gz" \
    --interp nearest \
    --no-save-reg

echo "MNI305 -> MNI152Lin (Step 1's transform, nearest-neighbor)..."
antsApplyTransforms -d 3 \
    -i "$WORK_DIR/dkt_aseg_mni305.nii.gz" \
    -r "$MNI152_T1" \
    -o "$OUT_DIR/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz" \
    -t "$MNI305_TO_MNI152_AFFINE" \
    -n NearestNeighbor

echo "Wrote: $OUT_DIR/aparc.DKTatlas+aseg_subject_MNI152Lin.nii.gz"

#############################################################
# OPTION 2: fsaverage's population-average segmentations
#   already MNI305 by definition -> MNI152Lin (Step 1's transform)
#############################################################
echo "=== Option 2: fsaverage (population-average) segmentations -> MNI152Lin ==="

FSAVERAGE_MRI="${FREESURFER_HOME}/subjects/fsaverage/mri"
[ -d "$FSAVERAGE_MRI" ] || {
    echo "FATAL: $FSAVERAGE_MRI not found -- is FREESURFER_HOME set to a full install with fsaverage, not just a license dir?" >&2
    exit 1
}

# Sanity check the assumption fsaverage's own volumes and mni305.cor.mgz
# really do share one grid/affine (both are meant to be "the" MNI305/
# fsaverage space by FreeSurfer's own design) -- verify, don't just assume,
# given how many silent space mismatches this project has already hit.
check_same_space "$FSAVERAGE_MRI/orig.mgz" "${FREESURFER_HOME}/average/mni305.cor.mgz"

for ATLAS in "aparc+aseg" "aparc.a2009s+aseg"; do
    SRC="$FSAVERAGE_MRI/${ATLAS}.mgz"
    if [ ! -f "$SRC" ]; then
        echo "WARNING: $SRC not found -- skipping $ATLAS" >&2
        continue
    fi

    echo "Converting fsaverage $ATLAS.mgz (.mgz -> .nii.gz, still MNI305 space, no resampling)..."
    python3 -c "
import nibabel as nib
img = nib.load('$SRC')
nib.save(nib.Nifti1Image(img.get_fdata(), img.affine, img.header), '$WORK_DIR/${ATLAS}_mni305.nii.gz')
"

    echo "MNI305 -> MNI152Lin ($ATLAS, Step 1's transform, nearest-neighbor)..."
    antsApplyTransforms -d 3 \
        -i "$WORK_DIR/${ATLAS}_mni305.nii.gz" \
        -r "$MNI152_T1" \
        -o "$OUT_DIR/${ATLAS}_fsaverage_MNI152Lin.nii.gz" \
        -t "$MNI305_TO_MNI152_AFFINE" \
        -n NearestNeighbor

    echo "Wrote: $OUT_DIR/${ATLAS}_fsaverage_MNI152Lin.nii.gz"
done

echo "=== Done ==="
echo "Sanity-check each output before trusting it: overlay against"
echo "$MNI152_T1 (e.g. nilearn.plotting.plot_roi) and confirm ventricles/"
echo "cortex boundaries land in plausible locations -- affine-only"
echo "registration (both the MNI305->MNI152Lin step and turboprep's own"
echo "affine normalization) can leave residual misalignment on atypical"
echo "anatomy."
