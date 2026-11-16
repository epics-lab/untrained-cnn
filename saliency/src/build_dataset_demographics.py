"""
Builds a PPMI or HBN Subject/Sex/Age demographics TSV in the exact format
compute_reference_baselines.py (and train_linear.py) expect for --demographics
-- tab-delimited, NO header, exactly 3 columns (Subject, Sex, Age), matching
$NKI_DEMOGRAPHICS's own format (see config/paths.example.env).
Neither dataset's raw cohort file is already in that shape:

- PPMI's raw cohort CSV ($PPMI_COHORT_CSV) is comma-delimited, has a header,
  and extra columns (COHORT, Visit code, Description, Group) -- this just
  narrows it to Subject/Sex/Age, restricted to $PPMI_SUBJECTS_LIST (the
  CAT12+FreeSurfer-intersected cohort), minus --exclude-subject.
- HBN's raw sex/age file ($HBN_DEMOGRAPHICS_RAW) has no BMI column; BMI has
  to be joined in from a separate longitudinal phenotype file
  ($HBN_PARTICIPANTS_TSV) and filtered to a plausible human range (10, 80)
  before intersecting with $HBN_SUBJECTS_LIST and dropping --exclude-subject.

Usage:
    python build_dataset_demographics.py --dataset ppmi
    python build_dataset_demographics.py --dataset hbn
"""

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "demographics"

# PPMI
PPMI_COHORT_CSV = Path(os.environ.get("PPMI_COHORT_CSV", "/path/to/PPMI_raw/processed_cohort_with_mri.csv"))
PPMI_SUBJECTS_LIST = Path(os.environ.get("PPMI_SUBJECTS_LIST", "/path/to/PPMI_hdbet_affine/common_subjects.txt"))
PPMI_DEFAULT_EXCLUDE = ["3510"]

# HBN
HBN_DEMOGRAPHICS_RAW = Path(
    os.environ.get("HBN_DEMOGRAPHICS_RAW", "/path/to/HBN_raw/final_preprocessed_subjects_with_demographics.tsv")
)
HBN_PARTICIPANTS_TSV = Path(os.environ.get("HBN_PARTICIPANTS_TSV", "/path/to/HBN_raw/study-HBN_desc-participants.tsv"))
HBN_SUBJECTS_LIST = Path(os.environ.get("HBN_SUBJECTS_LIST", "/path/to/HBN_raw/final_subjects.txt"))
HBN_DEFAULT_EXCLUDE = ["NDAREF662XZT", "NDARVN370YL8"]
HBN_BMI_MIN, HBN_BMI_MAX = 10, 80


def _strip_sub_prefix(subject_id):
    subject_id = str(subject_id).strip()
    return subject_id[len("sub-") :] if subject_id.startswith("sub-") else subject_id


def _load_subject_list(path):
    with open(path) as f:
        return {_strip_sub_prefix(line) for line in f if line.strip()}


def build_ppmi_demographics(args):
    df = pd.read_csv(args.cohort_csv)
    df["PATNO"] = df["PATNO"].astype(str)

    allowed = _load_subject_list(args.subjects_list)
    df = df[df["PATNO"].isin(allowed)]

    exclude = {_strip_sub_prefix(s) for s in args.exclude_subject}
    df = df[~df["PATNO"].isin(exclude)]

    df = df.rename(columns={"PATNO": "Subject"})[["Subject", "Sex", "Age"]].reset_index(drop=True)
    return df


def build_hbn_demographics(args):
    demographics = pd.read_csv(args.demographics_raw, sep="\t")
    demographics = demographics.rename(columns={"participant_id": "Subject", "sex": "Sex", "age": "Age"})
    demographics["Subject"] = demographics["Subject"].apply(_strip_sub_prefix)

    phenotype = pd.read_csv(args.participants_tsv, sep="\t")
    phenotype = phenotype.rename(columns={"participant_id": "Subject"})[["Subject", "bmi"]]
    phenotype["Subject"] = phenotype["Subject"].apply(_strip_sub_prefix)
    phenotype = phenotype.dropna(subset=["bmi"])
    phenotype = phenotype[(phenotype["bmi"] > args.bmi_min) & (phenotype["bmi"] < args.bmi_max)]

    df = demographics.merge(phenotype[["Subject"]], on="Subject", how="inner")

    allowed = _load_subject_list(args.subjects_list)
    df = df[df["Subject"].isin(allowed)]

    exclude = {_strip_sub_prefix(s) for s in args.exclude_subject}
    df = df[~df["Subject"].isin(exclude)]

    df = df[["Subject", "Sex", "Age"]].drop_duplicates(subset=["Subject"]).reset_index(drop=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset", choices=["ppmi", "hbn"], required=True)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--cohort-csv", type=Path, default=PPMI_COHORT_CSV, help="PPMI only.")
    parser.add_argument("--subjects-list", type=Path, default=None)
    parser.add_argument("--exclude-subject", action="append", default=None)

    parser.add_argument("--demographics-raw", type=Path, default=HBN_DEMOGRAPHICS_RAW, help="HBN only.")
    parser.add_argument("--participants-tsv", type=Path, default=HBN_PARTICIPANTS_TSV, help="HBN only.")
    parser.add_argument("--bmi-min", type=float, default=HBN_BMI_MIN, help="HBN only.")
    parser.add_argument("--bmi-max", type=float, default=HBN_BMI_MAX, help="HBN only.")

    args = parser.parse_args()

    if args.subjects_list is None:
        args.subjects_list = PPMI_SUBJECTS_LIST if args.dataset == "ppmi" else HBN_SUBJECTS_LIST
    if args.exclude_subject is None:
        args.exclude_subject = PPMI_DEFAULT_EXCLUDE if args.dataset == "ppmi" else HBN_DEFAULT_EXCLUDE
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"{args.dataset}_baseline_demographics.tsv"

    return args


def main():
    args = parse_args()
    df = build_ppmi_demographics(args) if args.dataset == "ppmi" else build_hbn_demographics(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", header=False, index=False)
    print(f"Wrote {len(df)} subjects to {args.output}")


if __name__ == "__main__":
    main()
