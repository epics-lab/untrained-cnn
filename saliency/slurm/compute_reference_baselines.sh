#!/usr/bin/env bash
#SBATCH --job-name=compute_reference_baselines
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x.out
#
# Computes a population-mean reference (Integrated Gradients baseline) image
# per model, in that model's own preprocessed input-tensor space, by
# averaging every subject in a dataset's demographics file through that
# model's real Dataset preprocessing (see src/compute_reference_baselines.py
# for why a population-derived baseline, not an arbitrary all-zero one).
#
# CPU-only, no GPU needed -- backbones are never loaded, only preprocessing.
# Single-shot job, not a per-subject array: it needs the whole cohort at
# once to average over. Runs three datasets in sequence -- NKI (fs_roi only,
# unchanged from before this script grew PPMI/HBN support), then PPMI/HBN
# (fs_roi + every imaging model, since those two only have the turboprep-
# style *_hdbet_affine roots this pipeline's imaging models need, not the
# fs_roi-specific artifacts/nki_freesurfer_roi.csv table NKI already had --
# built here instead, see build_freesurfer_roi_tables.py). --time/--mem above
# are sized for all three; drop PPMI/HBN's imaging models (see --models
# below) if you only want fs_roi and don't need this headroom.
#
# Output: reference_baselines/{dataset}/{registration}_{model}_reference.npy
# (one array in that model's own preprocessed shape), except optim_uncnn
# which gets 3 (raw/rank/sobel), and anatcl_global/anatcl_local which share
# one ({registration}_anatcl_reference.npy -- their preprocessing is
# identical, only the encoder differs). --registration affine below matches
# NKI_AFFINE_TURBOPREP_ROOT/PPMI_TURBOPREP_ROOT/HBN_TURBOPREP_ROOT (env.sh,
# all three are affine-registered) and is the only pipeline every consumer
# (calculate_gradcam.py's load_reference_baseline etc.) currently reads by
# default; pass --registration rigid instead for NKI to compute
# rigid-pipeline baselines (NKI_RIGID_TURBOPREP_ROOT) into the same
# output-dir under a rigid_ prefix (PPMI/HBN have no rigid pipeline).
#
# calculate_gradcam.py/calculate_gradcam_tabular.py/prototype_mmbs.py all
# default --reference-baselines-dir to reference_baselines/nki -- pass it
# explicitly (reference_baselines/ppmi or reference_baselines/hbn) to use
# those cohorts' baselines instead.

set -uo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/compute_reference_baselines.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-2}"

# # --- NKI (unchanged scope: fs_roi only) -------------------------------------
# python3 src/compute_reference_baselines.py --num-workers "$NUM_WORKERS" \
#   --registration affine --models fs_roi \
#   --demographics "$NKI_DEMOGRAPHICS" --output-dir reference_baselines/nki

# --- PPMI/HBN cohort-table prep ---------------------------------------------
# Cheap (table-only), regenerated every run so a stale demographics/ROI CSV
# never silently goes unnoticed if the upstream raw files change.
python3 src/build_dataset_demographics.py --dataset ppmi
python3 src/build_dataset_demographics.py --dataset hbn
python3 src/build_freesurfer_roi_tables.py --dataset ppmi
python3 src/build_freesurfer_roi_tables.py --dataset hbn --build-cat12-merged-root

# --- PPMI --------------------------------------------------------------------
python3 src/compute_reference_baselines.py --num-workers "$NUM_WORKERS" \
  --registration affine --models fs_roi optim_uncnn brainiac swinbrain simclr anatcl \
  --demographics "$PPMI_DEMOGRAPHICS" --turboprep-root "$PPMI_TURBOPREP_ROOT" \
  --nki-bids-root "$PPMI_CAT12_BIDS_ROOT" --roi-csv artifacts/ppmi_freesurfer_roi.csv \
  --exclude-subject 3510 --output-dir reference_baselines/ppmi

# --- HBN -----------------------------------------------------------------
python3 src/compute_reference_baselines.py --num-workers "$NUM_WORKERS" \
  --registration affine --models fs_roi optim_uncnn brainiac swinbrain simclr anatcl \
  --demographics "$HBN_DEMOGRAPHICS" --turboprep-root "$HBN_TURBOPREP_ROOT" \
  --nki-bids-root artifacts/hbn_cat12_merged_bids --roi-csv artifacts/hbn_freesurfer_roi.csv \
  --exclude-subject NDAREF662XZT --exclude-subject NDARVN370YL8 \
  --exclude-subject NDARYC287UFV \
  --output-dir reference_baselines/hbn
