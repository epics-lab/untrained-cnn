#!/usr/bin/env bash
# Cluster/environment setup sourced by every job script in this directory.
#
# Copy this file to env.sh (gitignored) and edit for your cluster/account --
# env.sh itself is never committed since it's account- and machine-specific.
# All the *.sh scripts in this directory `source "$(dirname
# "${BASH_SOURCE[0]}")/env.sh"` before anything else.

# Modules / virtualenv for this pipeline's Python dependencies (torch, monai,
# nibabel, nilearn, scipy, scikit-learn, pandas, numpy, matplotlib -- see
# ../requirements.txt).
module load gcc opencv/4.13.0
source /path/to/your/venv/bin/activate

# Data + external-checkpoint locations (see ../config/paths.example.env for
# the matching Python-side env vars -- keep both files in sync).
export NKI_DEMOGRAPHICS="/path/to/NKI/final_demographics.txt"
export NKI_PARTICIPANTS_TSV="/path/to/NKI/participants.tsv"
export NKI_RIGID_TURBOPREP_ROOT="/path/to/turboprep_NKI"
export NKI_AFFINE_TURBOPREP_ROOT="/path/to/NKI_hdbet_affine"
export NKI_BIDS_ROOT="/path/to/NKI_BIDS"

# PPMI reference-baseline cohort (see src/build_dataset_demographics.py and
# src/build_freesurfer_roi_tables.py, both --dataset ppmi).
export PPMI_TURBOPREP_ROOT="/path/to/PPMI_hdbet_affine"
export PPMI_SUBJECTS_LIST="/path/to/PPMI_hdbet_affine/common_subjects.txt"
export PPMI_COHORT_CSV="/path/to/PPMI_raw/processed_cohort_with_mri.csv"
export PPMI_FREESURFER_CTH_CSV="/path/to/FS7_APARC_CTH_23Oct2025.csv"
export PPMI_FREESURFER_SA_CSV="/path/to/FS7_APARC_SA_23Oct2025.csv"
export PPMI_CAT12_BIDS_ROOT="/path/to/ppmi_cat12_bids"
export PPMI_DEMOGRAPHICS="${REPO_DIR}/artifacts/demographics/ppmi_baseline_demographics.tsv"

# HBN reference-baseline cohort (see src/build_dataset_demographics.py and
# src/build_freesurfer_roi_tables.py, both --dataset hbn). CAT12 output is
# split across two non-overlapping BIDS trees -- build_freesurfer_roi_tables.py
# --build-cat12-merged-root merges them into one symlink tree since
# cat12_paths_and_subjects (train_linear.py) only takes a single root.
export HBN_TURBOPREP_ROOT="/path/to/HBN_hdbet_affine"
export HBN_SUBJECTS_LIST="/path/to/HBN_raw/final_subjects.txt"
export HBN_DEMOGRAPHICS_RAW="/path/to/HBN_raw/final_preprocessed_subjects_with_demographics.tsv"
export HBN_PARTICIPANTS_TSV="/path/to/HBN_raw/study-HBN_desc-participants.tsv"
export HBN_FREESURFER_ROOT="/path/to/HBN_FreeSurfer/freesurfer"
export HBN_CAT12_BIDS_ROOT_1="/path/to/HBN_BIDS"
export HBN_CAT12_BIDS_ROOT_2="/path/to/hbn_bids"
export HBN_DEMOGRAPHICS="${REPO_DIR}/artifacts/demographics/hbn_baseline_demographics.tsv"

export BACKBONE_CHECKPOINT_DIR="/path/to/BrainIAC/checkpoints"
export ANATCL_REPO_ROOT="/path/to/AnatCL"
export TEMPLATEFLOW_MNI_DIR="/path/to/templateflow/tpl-MNI152NLin2009cAsym"

# train_linear.py's cached {cache_name}_X_features_raw.npy/_y_age.npy (its own
# DEFAULT_OUTPUT_DIR is REPO_DIR/artifacts -- this only needs to be set
# explicitly because calculate_gradcam.py's own --features-dir default is
# its own script directory, not artifacts/, so anything using --randomize-data
# without passing --features-dir explicitly would silently look in the wrong
# place). REPO_DIR is set by the calling *.sh script before this is sourced.
export FEATURES_DIR="${REPO_DIR}/artifacts"

# NOTE: #SBATCH directives (including --account) are parsed by the scheduler
# at submission time, before this file is ever sourced, so --account can't be
# set here. None of the scripts in this directory hard-code one -- pass it at
# submission, e.g. `sbatch --account=<your-account> slurm/<script>.sh`
# (and `--mail-user=<you> --mail-type=END,FAIL` if you want email).
