"""
datalad-get the two per-subject FreeSurfer TSVs the fs_roi model's feature
table (build_freesurfer_roi_tables.py) needs -- regionsurfacestats.tsv
(cortical GrayVol, many atlases) and brainmeasures.tsv (subcortical/aseg
Volume_mm3 columns) -- for every subject in $NKI_DEMOGRAPHICS, from the
datalad/git-annex dataset at DEFAULT_FREESURFER_DIR.

Uses a datalad-get pattern that fetches both TSVs per subject, from the
dataset location given by $NKI_FREESURFER_DIR.

PREREQUISITES (not solved by this script):
  - `datalad` must already be pip-installed into whichever Python
    environment this runs under (it is NOT installed in brainiac_env, this
    repo's usual env.sh venv -- confirmed present instead in a separate
    venv, which is what this script was validated against; point at whichever venv actually has it).
  - `module load git-annex/10.20231129` must be run before this script --
    datalad shells out to the `git-annex` binary that module provides.
  - `datalad get` resolves "the dataset" from the CURRENT WORKING
    DIRECTORY, not from the target path -- this script always passes an
    explicit `-d <dataset root>` (see dataset_root()) so it works regardless
    of the caller's cwd; do not drop that flag if this is ever refactored.
  - This needs real network access to the git-annex special remote to
    actually fetch content. On this cluster (and HPC clusters generally)
    that means running on a LOGIN node, not via `sbatch` -- compute nodes
    typically have no route to the remote at all, so a SLURM array job
    calling this would silently hang or fail per-task. Run this
    interactively (or under `nohup`/`tmux` for a long unattended fetch),
    not as a slurm/*.sh job.

Usage:
    python fetch_freesurfer_tsvs.py --dry-run
    python fetch_freesurfer_tsvs.py
    python fetch_freesurfer_tsvs.py --tsv-kind regionsurfacestats
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_FREESURFER_DIR = Path(
    os.environ.get("NKI_FREESURFER_DIR", "/path/to/NKI/NKI_FreeSurfer/freesurfer")
)
DEFAULT_SUBJECTS_FILE = Path(os.environ.get("NKI_DEMOGRAPHICS", "/path/to/NKI/final_demographics.txt"))

TSV_KINDS = ("regionsurfacestats", "brainmeasures")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read NKI subject IDs from a demographics file and datalad-get each subject's "
            "regionsurfacestats.tsv and/or brainmeasures.tsv (needed by build_freesurfer_roi_tables.py)."
        )
    )
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS_FILE)
    parser.add_argument(
        "--freesurfer-dir", type=Path, default=DEFAULT_FREESURFER_DIR,
        help=f"FreeSurfer datalad dataset directory. Default: {DEFAULT_FREESURFER_DIR}",
    )
    parser.add_argument("--session", default="BAS1", help="Session label to fetch. Default: BAS1")
    parser.add_argument(
        "--tsv-kind", choices=[*TSV_KINDS, "both"], default="both",
        help="Which TSV(s) to fetch per subject. Default: both (regionsurfacestats.tsv is cortical-only; "
        "brainmeasures.tsv has the subcortical/aseg Volume_mm3 columns build_freesurfer_roi_tables.py also needs).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the datalad commands without running them.")
    return parser.parse_args()


def read_subject_ids(subjects_file: Path):
    column_names = ["Subject", "Sex", "Age"]
    return list(pd.read_csv(subjects_file, delimiter="\t", names=column_names)["Subject"])


def build_tsv_path(freesurfer_dir: Path, subject_id: str, session: str, tsv_kind: str):
    basename = f"sub-{subject_id}_ses-{session}"
    return freesurfer_dir / basename / f"{basename}_{tsv_kind}.tsv"


def dataset_root(freesurfer_dir: Path):
    """
    The installed datalad dataset's own root is one level up from
    --freesurfer-dir (.../NKI_FreeSurfer/freesurfer -- .git/.datalad live in
    NKI_FreeSurfer/, not freesurfer/ itself, confirmed on this cluster's
    copy). `datalad get` without an explicit --dataset auto-detects from
    the CURRENT WORKING DIRECTORY, not from the target path -- if this
    script isn't run with cwd inside the dataset, that resolves to whatever
    (unrelated, non-dataset) directory it *was* run from and fails with
    "path not associated with dataset". Passing -d explicitly here avoids
    depending on the caller's cwd entirely.
    """
    return freesurfer_dir.parent


def main():
    args = parse_args()
    kinds = list(TSV_KINDS) if args.tsv_kind == "both" else [args.tsv_kind]

    subject_ids = read_subject_ids(args.subjects_file)
    if not subject_ids:
        raise SystemExit(f"No subject IDs found in {args.subjects_file}")
    print(f"Found {len(subject_ids)} unique subjects in {args.subjects_file} -- fetching {kinds}")

    success_count = failure_count = missing_count = 0

    for subject_id in subject_ids:
        for kind in kinds:
            target_path = build_tsv_path(args.freesurfer_dir, subject_id, args.session, kind)
            cmd = ["datalad", "get", "-d", str(dataset_root(args.freesurfer_dir)), str(target_path)]

            if args.dry_run:
                try:
                    print(" ".join(cmd))
                except BrokenPipeError:
                    sys.exit(0)
                continue

            if not target_path.parent.exists():
                print(f"Missing subject directory: {target_path.parent}")
                missing_count += 1
                continue

            print(f"Fetching {target_path}")
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                success_count += 1
            else:
                failure_count += 1
                print(f"datalad get failed for {target_path}")

    if args.dry_run:
        return

    print(f"Done. successes={success_count} failures={failure_count} missing={missing_count}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
