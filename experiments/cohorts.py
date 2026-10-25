"""Subjects, labels, model inputs and FreeSurfer features of PPMI, NKI and HBN.

A cohort holds the subjects that have a TurboPrep affine volume, a CAT12 gray
matter map, complete FreeSurfer features and labels, sorted by subject ID. Sex
is coded 1 for female and 0 for male. BMI is kept when 10 < BMI < 80, for the
subset of subjects indexed by bmi_idx.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import optional_path, path

# Byte identical T1w images in the RBC HBN release (neurostars.org/t/36444): NDAREF662XZT and
# NDARVN370YL8 duplicate each other; NDARYC287UFV duplicates NDARVV704MND, which is not in the sample.
HBN_EXCLUDE = {"NDAREF662XZT", "NDARVN370YL8", "NDARYC287UFV"}
MIN_BMI_SUBJECTS = 20
SCHAEFER = "Schaefer2018_400Parcels_17Networks_order"


@dataclass
class Cohort:
    name: str
    subjects: list
    affine: list
    cat12: list
    sex: np.ndarray
    age: np.ndarray
    freesurfer: dict
    bmi_idx: np.ndarray = None
    bmi: np.ndarray = None


def load(name):
    return {"PPMI": load_ppmi, "NKI": load_nki, "HBN": load_hbn}[name]()


def describe(c):
    out = {"subjects": len(c.subjects), "male": int((c.sex == 0).sum()), "female": int((c.sex == 1).sum()),
           "age_min": c.age.min(), "age_max": c.age.max(), "age_mean": c.age.mean(), "age_sd": c.age.std(ddof=1)}
    if c.bmi is not None:
        out.update(bmi_subjects=len(c.bmi), bmi_mean=c.bmi.mean(), bmi_sd=c.bmi.std(ddof=1))
    return out


def strip_sub(pid):
    pid = str(pid).strip()
    return pid[4:] if pid.startswith("sub-") else pid


def read_subject_list(file):
    """Optional gate: None when the list is not given or absent."""
    if file is None or not os.path.exists(file):
        print(f"  [warn] subject list not found, skipping gate: {file}")
        return None
    with open(file) as f:
        return {strip_sub(line) for line in f if line.strip()}


def scan_affine_dir(affine_dir, valid_ids=None):
    """Map subject ID to <affine_dir>/<subject ID>/normalized.nii.gz."""
    out = {}
    for folder in sorted(os.listdir(affine_dir)):
        fp = os.path.join(affine_dir, folder)
        nf = os.path.join(fp, "normalized.nii.gz")
        if not (os.path.isdir(fp) and os.path.exists(nf)):
            continue
        if valid_ids is not None and folder not in valid_ids:
            continue
        out[folder] = nf
    return out


def read_bmi(df, keep=None):
    out = {}
    for _, row in df.iterrows():
        pid = strip_sub(row["participant_id"])
        if keep is not None and pid not in keep:
            continue
        if pd.notna(row.get("bmi")):
            try:
                b = float(row["bmi"])
                if 10 < b < 80:
                    out[pid] = b
            except (TypeError, ValueError):
                pass
    return out


def rbc_surface_features(tsv):
    """Schaefer 400: surface area then thickness, 800 features. aparc: thickness then surface area, 136 features."""
    df = pd.read_csv(tsv, sep="\t")
    out = {}
    sch = df[df["atlas"] == SCHAEFER].sort_values("StructName")
    if not sch.empty and "SurfArea" in sch.columns and "ThickAvg" in sch.columns:
        sa, th = sch["SurfArea"].values[:400], sch["ThickAvg"].values[:400]
        if not np.any(np.isnan(np.concatenate([sa, th]))) and len(sa) == 400:
            out["FS_Schaefer"] = np.concatenate([sa, th])
    apc = df[df["atlas"] == "aparc"].sort_values("StructName")
    if not apc.empty and "SurfArea" in apc.columns and "ThickAvg" in apc.columns:
        sa, th = apc["SurfArea"].values, apc["ThickAvg"].values
        if not np.any(np.isnan(np.concatenate([th, sa]))) and len(sa) == 68:
            out["FS_aparc"] = np.concatenate([th, sa])
    return out


def build_bmi_subset(common, id_bmi, label):
    """BMI subset of the cohort, as positions into its subject list."""
    s2i = {s: i for i, s in enumerate(common)}
    subj = sorted(set(common) & set(id_bmi))
    print(f"{label} BMI: {len(subj)} subjects")
    if len(subj) < MIN_BMI_SUBJECTS:
        print(f"  [warn] fewer than {MIN_BMI_SUBJECTS} BMI subjects, skipping BMI for {label}")
        return None, None
    return np.array([s2i[s] for s in subj]), np.array([id_bmi[s] for s in subj])


def _cohort(name, common, affine, cat12, sex, age, freesurfer, bmi=None):
    print(f"{name}: {len(common)} subjects after intersection")
    bmi_idx, bmi_y = build_bmi_subset(common, bmi, name) if bmi is not None else (None, None)
    return Cohort(name=name, subjects=common, affine=[affine[s] for s in common], cat12=[cat12[s] for s in common],
                  sex=np.array([sex[s] for s in common]), age=np.array([age[s] for s in common]),
                  freesurfer={k: np.array([v[s] for s in common]) for k, v in freesurfer.items()},
                  bmi_idx=bmi_idx, bmi=bmi_y)


def load_ppmi():
    labels = pd.read_csv(path("PPMI_LABELS"))
    sex, age = {}, {}
    for _, row in labels.iterrows():
        p = str(int(row["PATNO"]))
        sex[p] = 1 if row["Sex"].strip().upper() == "F" else 0
        age[p] = row["Age"]

    listed = read_subject_list(optional_path("PPMI_SUBJECT_LIST"))
    affine = scan_affine_dir(path("PPMI_AFFINE_DIR"), set(sex) if listed is None else set(sex) & listed)
    print(f"PPMI affine volumes: {len(affine)}")

    cat12 = {}
    for f in glob.glob(os.path.join(path("PPMI_CAT12_DIR"), "**", "mwp1*.nii*"), recursive=True):
        if os.path.isfile(f):
            p = next((part[4:] for part in f.split(os.sep) if part.startswith("sub-")), None)
            if p and p in sex:
                cat12[p] = f

    def baseline(csv):
        df = pd.read_csv(csv)
        df = df[df["EVENT_ID"] == "BL"].copy()
        df["PATNO"] = df["PATNO"].astype(str)
        return df, [c for c in df.columns if c not in ["PATNO", "EVENT_ID"]]

    (cth, cth_cols), (sa, sa_cols) = baseline(path("PPMI_FS_CTH")), baseline(path("PPMI_FS_SA"))
    fs = {}
    for p in affine:
        c, s = cth[cth["PATNO"] == p], sa[sa["PATNO"] == p]
        if len(c) > 0 and len(s) > 0:
            combo = np.concatenate([c[cth_cols].values.flatten(), s[sa_cols].values.flatten()])
            if not np.any(np.isnan(combo)):
                fs[p] = combo

    common = sorted(set(affine) & set(cat12) & set(fs))
    return _cohort("PPMI", common, affine, cat12, sex, age, {"FreeSurfer": fs})


def load_nki():
    sex, age = {}, {}
    with open(path("NKI_DEMOGRAPHICS")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                sid = strip_sub(parts[0])
                sex[sid] = 1 if parts[1].strip() == "Female" else 0
                age[sid] = float(parts[2])

    participants = pd.read_csv(path("NKI_PARTICIPANTS"), sep="\t")
    bmi = read_bmi(participants[participants["wave"] == "BAS1"])

    listed = read_subject_list(optional_path("NKI_SUBJECT_LIST"))
    affine = scan_affine_dir(path("NKI_AFFINE_DIR"), set(sex) if listed is None else set(sex) & listed)
    print(f"NKI affine volumes: {len(affine)}")

    cat12 = {}
    for sid in sex:
        files = glob.glob(os.path.join(path("NKI_BIDS_DIR"), f"sub-{sid}", "ses-BAS1", "anat", "mri", "mwp1*.nii"))
        if files:
            cat12[sid] = files[0]

    fs = {"FS_Schaefer": {}, "FS_aparc": {}}
    for sid in sex:
        tsv = os.path.join(path("NKI_FREESURFER_DIR"), f"sub-{sid}_ses-BAS1", f"sub-{sid}_ses-BAS1_regionsurfacestats.tsv")
        if os.path.exists(tsv):
            for k, v in rbc_surface_features(tsv).items():
                fs[k][sid] = v

    common = sorted(set(affine) & set(cat12) & set(fs["FS_Schaefer"]) & set(fs["FS_aparc"]))
    return _cohort("NKI", common, affine, cat12, sex, age, fs, bmi)


def load_nki_rigid():
    """Subjects of the architecture ablation: NKI with a rigid TurboPrep volume, a CAT12 map and both
    FreeSurfer feature sets. Returns (subjects, rigid volumes, sex, age)."""
    sex, age = {}, {}
    with open(path("NKI_DEMOGRAPHICS")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                sex[parts[0]] = 1 if parts[1].strip() == "Female" else 0
                age[parts[0]] = float(parts[2])

    rigid = {}
    for folder in os.listdir(path("NKI_RIGID_DIR")):
        fp = os.path.join(path("NKI_RIGID_DIR"), folder)
        nf = os.path.join(fp, "normalized.nii.gz")
        if os.path.isdir(fp) and os.path.exists(nf) and folder in sex:
            rigid[folder] = nf

    cat12 = {sid for sid in sex
             if glob.glob(os.path.join(path("NKI_BIDS_DIR"), f"sub-{sid}", "ses-BAS1", "anat", "mri", "mwp1*.nii"))}
    fs = set()
    for sid in sex:
        tsv = os.path.join(path("NKI_FREESURFER_DIR"), f"sub-{sid}_ses-BAS1", f"sub-{sid}_ses-BAS1_regionsurfacestats.tsv")
        if os.path.exists(tsv) and len(rbc_surface_features(tsv)) == 2:
            fs.add(sid)

    common = sorted(set(rigid) & cat12 & fs)
    print(f"NKI: {len(common)} subjects")
    return common, [rigid[s] for s in common], np.array([sex[s] for s in common]), np.array([age[s] for s in common])


def load_hbn():
    demo = pd.read_csv(path("HBN_DEMOGRAPHICS"), sep="\t")
    demo["participant_id"] = demo["participant_id"].astype(str).map(strip_sub)
    sex, age = {}, {}
    for _, row in demo.iterrows():
        pid = row["participant_id"]
        if pd.notna(row["sex"]) and row["sex"] in ["Male", "Female"]:
            sex[pid] = 1 if row["sex"] == "Female" else 0
        if pd.notna(row["age"]):
            age[pid] = float(row["age"])

    participants = pd.read_csv(path("HBN_PARTICIPANTS"), sep="\t")
    participants["participant_id"] = participants["participant_id"].astype(str).map(strip_sub)
    bmi = read_bmi(participants, keep=set(demo["participant_id"]))

    listed = read_subject_list(optional_path("HBN_SUBJECT_LIST"))
    valid = set(sex) if listed is None else set(sex) & listed
    print(f"HBN duplicated T1w subjects removed: {sorted(valid & HBN_EXCLUDE)}")
    affine = scan_affine_dir(path("HBN_AFFINE_DIR"), valid - HBN_EXCLUDE)
    print(f"HBN affine volumes: {len(affine)}")

    cat12 = {}
    for sid in sex:
        for base in path("HBN_CAT12_DIRS").split(os.pathsep):
            files = glob.glob(os.path.join(base, f"sub-{sid}", "ses-*", "anat", "mri", "mwp1sub*.nii"))
            if files:
                cat12[sid] = files[0]
                break

    fs = {"FS_Schaefer": {}, "FS_aparc": {}}
    for sid in sex:
        tsv = os.path.join(path("HBN_FREESURFER_DIR"), f"sub-{sid}", f"sub-{sid}_regionsurfacestats.tsv")
        if os.path.exists(tsv):
            for k, v in rbc_surface_features(tsv).items():
                fs[k][sid] = v

    common = sorted(set(affine) & set(cat12) & set(sex) & set(age) & set(fs["FS_Schaefer"]) & set(fs["FS_aparc"]))
    return _cohort("HBN", common, affine, cat12, sex, age, fs, bmi)
