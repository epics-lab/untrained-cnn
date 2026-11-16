"""
Shared, dependency-light (no torch/monai) definition of the prediction
targets this pipeline supports: brain age (regression), sex (binary
classification), and BMI (regression). Every stage of the pipeline that
needs to know "which column to predict / how to encode it / what to name
files" imports from here instead of hardcoding "Age" or "age" locally.

TargetSpec.checkpoint_suffix and .cache_suffix for "age" are deliberately
kept identical to what train_linear.py/calculate_gradcam.py already wrote
before this module existed (`brain_age_head` / `age`) -- --target age (the
default everywhere) must keep producing byte-identical filenames to every
prior run, so nothing already on disk breaks.

BMI is not present in NKI_DEMOGRAPHICS (only Subject/Sex/Age) -- it's
joined in from a separate, longitudinal participants.tsv (one row per
subject per session; filtered here to the BAS1 baseline visit) via
load_bmi_lookup, mirroring the join done in the paper's permutation-importance analysis.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_PARTICIPANTS_TSV = Path(
    os.environ.get("NKI_PARTICIPANTS_TSV", "/path/to/NKI/participants.tsv")
)


def _encode_identity(value):
    return float(value)


def _encode_sex(value):
    s = str(value).strip().lower()
    if s in ("female", "f"):
        return 1.0
    if s in ("male", "m"):
        return 0.0
    raise ValueError(f"Unrecognized Sex value: {value!r} (expected Female/F or Male/M)")


def _decode_display_numeric(value):
    return f"{value:.1f}"


def _decode_display_sex(value):
    return "Female" if value >= 0.5 else "Male"


@dataclass(frozen=True)
class TargetSpec:
    name: str                  # "age" | "sex" | "bmi"
    task_type: str             # "regression" | "classification"
    demographics_column: str   # column read out of the (possibly BMI-joined) demographics frame
    encode: Callable           # raw demographics value -> float label
    cache_suffix: str          # train_linear.py's "{cache_name}_y_{cache_suffix}.npy"
    checkpoint_suffix: str     # "{cache_name}_{checkpoint_suffix}.pt"
    json_true_key: str         # sanity_checks.json key for the ground-truth value
    json_pred_key: str         # sanity_checks.json key for the model's prediction
    display_name: str          # human-readable name for prints/titles
    decode_display: Callable   # float label -> human-readable string


TARGET_SPECS = {
    "age": TargetSpec(
        name="age",
        task_type="regression",
        demographics_column="Age",
        encode=_encode_identity,
        cache_suffix="age",
        checkpoint_suffix="brain_age_head",
        json_true_key="true_age",
        json_pred_key="predicted_age",
        display_name="age",
        decode_display=_decode_display_numeric,
    ),
    "sex": TargetSpec(
        name="sex",
        task_type="classification",
        demographics_column="Sex",
        encode=_encode_sex,
        cache_suffix="sex",
        checkpoint_suffix="sex_head",
        json_true_key="true_sex",
        json_pred_key="predicted_sex",
        display_name="sex",
        decode_display=_decode_display_sex,
    ),
    "bmi": TargetSpec(
        name="bmi",
        task_type="regression",
        demographics_column="BMI",
        encode=_encode_identity,
        cache_suffix="bmi",
        checkpoint_suffix="bmi_head",
        json_true_key="true_bmi",
        json_pred_key="predicted_bmi",
        display_name="BMI",
        decode_display=_decode_display_numeric,
    ),
}


def get_target_spec(name: str) -> TargetSpec:
    if name not in TARGET_SPECS:
        raise ValueError(f"Unknown target {name!r}; choices: {sorted(TARGET_SPECS)}")
    return TARGET_SPECS[name]


def load_bmi_lookup(participants_tsv_path) -> pd.Series:
    """
    participants.tsv (external, longitudinal -- has a header, unlike
    NKI_DEMOGRAPHICS, and multiple rows per subject, one per session).
    Filters to session_id == "BAS1" (baseline visit) before returning a
    Subject-indexed Series of BMI values -- callers merge this into their
    own Subject/Sex/Age frame only when target == "bmi". Subject ID format
    here (e.g. A00000000) already matches NKI_DEMOGRAPHICS's Subject column,
    no reformatting needed.
    """
    df = pd.read_csv(participants_tsv_path, sep="\t")
    df = df[df["session_id"] == "BAS1"].copy()
    df["participant_id"] = df["participant_id"].astype(str)
    if df["participant_id"].duplicated().any():
        dupes = df.loc[df["participant_id"].duplicated(), "participant_id"].unique()
        print(
            f"WARNING: {len(dupes)} subject(s) have >1 BAS1 row in {participants_tsv_path}; "
            f"keeping first. e.g. {list(dupes[:5])}"
        )
        df = df.drop_duplicates("participant_id", keep="first")
    return df.set_index("participant_id")["bmi"]


def merge_bmi(demographics: pd.DataFrame, participants_tsv_path=None) -> pd.DataFrame:
    """
    Left-joins a "BMI" column onto a Subject/Sex/Age demographics frame,
    dropping subjects with no BAS1 BMI row (warns with the count). Returns a
    new DataFrame -- does not mutate the input.
    """
    participants_tsv_path = participants_tsv_path or DEFAULT_PARTICIPANTS_TSV
    bmi_lookup = load_bmi_lookup(participants_tsv_path)
    demographics = demographics.copy()
    demographics["Subject"] = demographics["Subject"].astype(str)
    demographics["BMI"] = demographics["Subject"].map(bmi_lookup)
    n_missing = int(demographics["BMI"].isna().sum())
    if n_missing:
        missing_ids = demographics.loc[demographics["BMI"].isna(), "Subject"].tolist()
        print(
            f"WARNING: {n_missing} subject(s) have no BAS1 BMI row in {participants_tsv_path}; "
            f"dropping. e.g. {missing_ids[:5]}"
        )
        demographics = demographics.dropna(subset=["BMI"]).reset_index(drop=True)
    return demographics
