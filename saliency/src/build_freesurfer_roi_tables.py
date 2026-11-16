"""
Builds the fs_roi model's per-dataset feature table: one row per subject, one
column per FreeSurfer ROI measurement, for whichever of --dataset {nki,ppmi,
hbn} is requested. NKI and HBN both go through the aparc.DKTatlas
regionsurfacestats.tsv pivot (cortical GrayVol) merged with each subject's
own brainmeasures.tsv (subcortical/aseg Volume_mm3) -- see
load_all_nki_freesurfer_tsvs/load_hbn_freesurfer_df, same column set and
harmonize_nki_freesurfer_roi_df mapping for both datasets, just different
on-disk layouts (NKI: sub-*_ses-*/ nested; HBN: flat sub-<ID>/). PPMI instead
reads its own pre-aggregated cortical-thickness/surface-area CSVs, since
that's the only FreeSurfer export available for it (see
build_ppmi_freesurfer_roi_df) -- no GrayVol/subcortical data exists for
PPMI, so its feature table has a different schema entirely and isn't
compatible with the fs_roi checkpoints trained on NKI's/HBN's.

NKI: ports an earlier prototype loader's logic near-verbatim (already correct: aparc.DKTatlas GrayVol pivot,
subcortical column harmonization dict) .

Only meaningful to run after fetch_freesurfer_tsvs.py has fetched the NKI
subjects you want included (PPMI/HBN's FreeSurfer exports are already
materialized wholesale, nothing to fetch) -- this script does not fetch
anything itself, it only reads whatever is already on disk.

--build-cat12-merged-root (HBN only) additionally builds a merged BIDS root
of per-subject symlinks under artifacts/hbn_cat12_merged_bids/, since HBN's
CAT12 mwp1 output is split across two non-overlapping BIDS trees
($HBN_CAT12_BIDS_ROOT_1/_2) but train_linear.py's cat12_paths_and_subjects
(and hence compute_reference_baselines.py's --nki-bids-root) only accepts a
single root.

Usage:
    python build_freesurfer_roi_tables.py --dataset nki
    python build_freesurfer_roi_tables.py --dataset ppmi
    python build_freesurfer_roi_tables.py --dataset hbn --build-cat12-merged-root
"""

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# NKI
DEFAULT_FREESURFER_ROOT = Path(
    os.environ.get("NKI_FREESURFER_DIR", "/path/to/NKI/NKI_FreeSurfer/freesurfer")
)
DEFAULT_SUBJECTS_FILE = Path(os.environ.get("NKI_DEMOGRAPHICS", "/path/to/NKI/final_demographics.txt"))

# HBN -- same regionsurfacestats.tsv schema as NKI's, but one flat
# sub-<ID>/sub-<ID>_regionsurfacestats.tsv per subject (no sub-*_ses-*/
# nesting, and no session suffix in the filename).
HBN_FREESURFER_ROOT = Path(
    os.environ.get("HBN_FREESURFER_ROOT", "/path/to/HBN_FreeSurfer/freesurfer")
)
HBN_CAT12_BIDS_ROOT_1 = Path(os.environ.get("HBN_CAT12_BIDS_ROOT_1", "/path/to/HBN_BIDS"))
HBN_CAT12_BIDS_ROOT_2 = Path(os.environ.get("HBN_CAT12_BIDS_ROOT_2", "/path/to/hbn_bids"))

# PPMI -- pre-aggregated cortical thickness (CTH) + surface area (SA) CSVs,
# one row per subject/visit, filtered to the BL (baseline) visit.
PPMI_FREESURFER_CTH_CSV = Path(
    os.environ.get("PPMI_FREESURFER_CTH_CSV", "/path/to/FS7_APARC_CTH_23Oct2025.csv")
)
PPMI_FREESURFER_SA_CSV = Path(os.environ.get("PPMI_FREESURFER_SA_CSV", "/path/to/FS7_APARC_SA_23Oct2025.csv"))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts"

# Which regionsurfacestats.tsv atlas to use for the cortical feature columns
# -- aparc.DKTatlas (62 regions: 31/hemisphere) per the user's decision to
# reuse the existing prototype's choice. Overridable via --atlas if a
# different granularity is ever wanted (e.g. plain "aparc", 68 regions).
DEFAULT_ATLAS = "aparc.DKTatlas"


def load_brainmeasures_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    volume_columns = [col for col in df.columns if "Volume_mm3" in col or "WhiteMatterVol" in col]
    columns_to_keep = ["subject_id", *volume_columns]

    missing = [col for col in ["subject_id"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {tsv_path}")

    return df.loc[:, columns_to_keep].copy()


def load_regionsurfacestats_tsv(tsv_path, atlas):
    df = pd.read_csv(tsv_path, sep="\t")

    columns_to_keep = ["subject_id", "atlas", "StructName", "hemisphere", "SurfArea", "GrayVol", "ThickAvg"]
    missing = [col for col in columns_to_keep if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {tsv_path}")

    df = df.loc[:, columns_to_keep].copy()
    df = df[df["atlas"].astype(str) == atlas].copy()
    return df


def reshape_regionsurfacestats_to_wide(regionsurfacestats_df):
    wide_df = regionsurfacestats_df.copy()
    wide_df["roi_name"] = wide_df["hemisphere"].astype(str) + "_" + wide_df["StructName"].astype(str)
    wide_df = wide_df.pivot_table(
        # Keep one row per subject/atlas; hemisphere is already encoded in roi_name.
        index=["subject_id", "atlas"],
        columns="roi_name",
        values="GrayVol",
        aggfunc="first",
    ).reset_index()
    wide_df.columns.name = None
    return wide_df


def load_subject_ids(subjects_file=DEFAULT_SUBJECTS_FILE):
    column_names = ["Subject", "Sex", "Age"]
    subject_ids = list(pd.read_csv(subjects_file, delimiter="\t", names=column_names)["Subject"])
    return {
        f"sub-{subject_id}" if not str(subject_id).startswith("sub-") else str(subject_id)
        for subject_id in subject_ids
    }


def _subject_id_from_tsv_path(tsv_path):
    return tsv_path.parent.name.split("_ses-", 1)[0]


def harmonize_nki_freesurfer_roi_df(df):
    """
    Maps brainmeasures.tsv's own *_Volume_mm3 column names onto the
    FreeSurferColorLUT.txt label convention (e.g. "Left_Hippocampus_Volume_mm3"
    -> "Left-Hippocampus") -- this is what src/roi_to_volume.py's LUT-based
    painting later matches directly against, so this mapping is load-bearing,
    not just cosmetic.
    """
    mapping = {
        "Left_Lateral_Ventricle_Volume_mm3": "Left-Lateral-Ventricle",
        "Left_Inf_Lat_Vent_Volume_mm3": "Left-Inf-Lat-Vent",
        "lhCerebralWhiteMatter_lhCerebralWhiteMatterVol": "Left-Cerebral-White-Matter",
        "Left_Cerebellum_White_Matter_Volume_mm3": "Left-Cerebellum-White-Matter",
        "Left_Cerebellum_Cortex_Volume_mm3": "Left-Cerebellum-Cortex",
        "Left_Thalamus_Proper_Volume_mm3": "Left-Thalamus",
        "Left_Caudate_Volume_mm3": "Left-Caudate",
        "Left_Putamen_Volume_mm3": "Left-Putamen",
        "Left_Pallidum_Volume_mm3": "Left-Pallidum",
        "Third_Ventricle_Volume_mm3": "3rd-Ventricle",
        "Fourth_Ventricle_Volume_mm3": "4th-Ventricle",
        "Brain_Stem_Volume_mm3": "Brain-Stem",
        "Left_Hippocampus_Volume_mm3": "Left-Hippocampus",
        "Left_Amygdala_Volume_mm3": "Left-Amygdala",
        "CSF_Volume_mm3": "CSF",
        "Left_Accumbens_area_Volume_mm3": "Left-Accumbens-area",
        "Left_VentralDC_Volume_mm3": "Left-VentralDC",
        "Left_vessel_Volume_mm3": "Left-vessel",
        "Left_choroid_plexus_Volume_mm3": "Left-choroid-plexus",
        "CerebralWhiteMatter_CerebralWhiteMatterVol": "Cerebral-White-Matter",
        "Right_Lateral_Ventricle_Volume_mm3": "Right-Lateral-Ventricle",
        "Right_Inf_Lat_Vent_Volume_mm3": "Right-Inf-Lat-Vent",
        "rhCerebralWhiteMatter_rhCerebralWhiteMatterVol": "Right-Cerebral-White-Matter",
        "Right_Cerebellum_White_Matter_Volume_mm3": "Right-Cerebellum-White-Matter",
        "Right_Cerebellum_Cortex_Volume_mm3": "Right-Cerebellum-Cortex",
        "Right_Thalamus_Proper_Volume_mm3": "Right-Thalamus",
        "Right_Caudate_Volume_mm3": "Right-Caudate",
        "Right_Putamen_Volume_mm3": "Right-Putamen",
        "Right_Pallidum_Volume_mm3": "Right-Pallidum",
        "Right_Hippocampus_Volume_mm3": "Right-Hippocampus",
        "Right_Amygdala_Volume_mm3": "Right-Amygdala",
        "Right_Accumbens_area_Volume_mm3": "Right-Accumbens-area",
        "Right_VentralDC_Volume_mm3": "Right-VentralDC",
        "Right_vessel_Volume_mm3": "Right-vessel",
        "Right_choroid_plexus_Volume_mm3": "Right-choroid-plexus",
        "Fifth_Ventricle_Volume_mm3": "5th-Ventricle",
        "WM_hypointensities_Volume_mm3": "WM-hypointensities",
        "Left_WM_hypointensities_Volume_mm3": "Left-WM-hypointensities",
        "Right_WM_hypointensities_Volume_mm3": "Right-WM-hypointensities",
        "non_WM_hypointensities_Volume_mm3": "non-WM-hypointensities",
        "Left_non_WM_hypointensities_Volume_mm3": "Left-non-WM-hypointensities",
        "Right_non_WM_hypointensities_Volume_mm3": "Right-non-WM-hypointensities",
        "Optic_Chiasm_Volume_mm3": "Optic-Chiasm",
        "CC_Posterior_Volume_mm3": "CC_Posterior",
        "CC_Mid_Posterior_Volume_mm3": "CC_Mid_Posterior",
        "CC_Central_Volume_mm3": "CC_Central",
        "CC_Mid_Anterior_Volume_mm3": "CC_Mid_Anterior",
        "CC_Anterior_Volume_mm3": "CC_Anterior",
    }

    for col in df.columns:
        if col.startswith("lh_"):
            mapping[col] = f"ctx-lh-{col[3:]}"
        elif col.startswith("rh_"):
            mapping[col] = f"ctx-rh-{col[3:]}"

    return df.rename(columns=mapping)


def load_all_nki_freesurfer_tsvs(
    freesurfer_root=DEFAULT_FREESURFER_ROOT,
    subjects_file=DEFAULT_SUBJECTS_FILE,
    session="BAS1",
    atlas=DEFAULT_ATLAS,
):
    freesurfer_root = Path(freesurfer_root)
    allowed_subject_ids = load_subject_ids(subjects_file)

    brain_pattern = f"sub-*_ses-{session}/sub-*_ses-{session}_brainmeasures.tsv"
    region_pattern = f"sub-*_ses-{session}/sub-*_ses-{session}_regionsurfacestats.tsv"
    brain_paths = sorted(freesurfer_root.glob(brain_pattern))
    region_paths = sorted(freesurfer_root.glob(region_pattern))

    brain_paths_by_subject = {
        _subject_id_from_tsv_path(path): path
        for path in brain_paths
        if path.exists() and _subject_id_from_tsv_path(path) in allowed_subject_ids
    }
    region_paths_by_subject = {
        _subject_id_from_tsv_path(path): path
        for path in region_paths
        if path.exists() and _subject_id_from_tsv_path(path) in allowed_subject_ids
    }

    shared_subject_ids = sorted(set(brain_paths_by_subject).intersection(region_paths_by_subject))
    print(
        f"{len(brain_paths_by_subject)} brainmeasures.tsv, {len(region_paths_by_subject)} "
        f"regionsurfacestats.tsv, {len(shared_subject_ids)} subjects with both (fetched + in demographics)"
    )

    brain_frames = [load_brainmeasures_tsv(brain_paths_by_subject[sid]) for sid in shared_subject_ids]
    region_frames = [load_regionsurfacestats_tsv(region_paths_by_subject[sid], atlas) for sid in shared_subject_ids]

    if not brain_frames:
        raise ValueError(f"No brainmeasures TSV files found under {freesurfer_root} for session {session}")
    if not region_frames:
        raise ValueError(f"No regionsurfacestats TSV files found under {freesurfer_root} for session {session}")

    brainmeasures_df = pd.concat(brain_frames, ignore_index=True).drop_duplicates(subset=["subject_id"]).copy()

    regionsurfacestats_df = pd.concat(region_frames, ignore_index=True)
    regionsurfacestats_df = reshape_regionsurfacestats_to_wide(regionsurfacestats_df)

    combined_df = regionsurfacestats_df.merge(brainmeasures_df, on="subject_id", how="left")

    df = harmonize_nki_freesurfer_roi_df(combined_df)
    df.rename(columns={"subject_id": "Subject"}, inplace=True)
    df["Subject"] = df["Subject"].str.replace("sub-", "", regex=False)
    df = df[df["atlas"] == atlas].dropna(axis=1, how="all")

    # A handful of regionsurfacestats.tsv/brainmeasures.tsv columns are only
    # present for a tiny minority of subjects (e.g. ctx-rh-temporalpole:
    # aparc.DKTatlas deliberately excludes temporalpole/frontalpole/banks-
    # STS from its ontology -- present in only 4/958 subjects here, verified
    # directly, presumably an upstream FreeSurfer-run inconsistency rather
    # than a real DKT-atlas region). RidgeCV can't handle NaN at all, and
    # these columns carry near-zero information anyway (>90% missing), so
    # drop the column rather than imputing or dropping those subjects'
    # entire rows. Any residual NaN after that (a column present for most
    # but not all subjects) drops the affected subject's row instead --
    # keeping every remaining column fully dense is worth losing a few
    # subjects over silently imputing FreeSurfer volumes.
    min_present_frac = 0.9
    df = df.dropna(axis=1, thresh=int(min_present_frac * len(df)))
    n_before = len(df)
    df = df.dropna(axis=0)
    if len(df) != n_before:
        print(f"Dropped {n_before - len(df)} subject(s) with residual NaNs in an otherwise-dense column")

    return df


def load_hbn_freesurfer_df(freesurfer_root=HBN_FREESURFER_ROOT, atlas=DEFAULT_ATLAS):
    """
    HBN's regionsurfacestats.tsv is byte-for-byte the same schema as NKI's
    (same columns, same "aparc.DKTatlas" atlas value -- confirmed directly),
    just laid out flat as sub-<ID>/sub-<ID>_regionsurfacestats.tsv (no
    sub-*_ses-*/ nesting). HBN DOES have a brainmeasures.tsv counterpart too
    (sub-<ID>/sub-<ID>_brainmeasures.tsv, same column-naming convention as
    NKI's own -- confirmed directly, e.g. "Left_Lateral_Ventricle_Volume_mm3"),
    just not consulted here until now -- this reuses load_brainmeasures_tsv
    and harmonize_nki_freesurfer_roi_df exactly as load_all_nki_freesurfer_tsvs
    does, so HBN's feature table gets the same subcortical columns NKI's has
    (previously cortical-only, missing those 48 columns entirely -- this is
    why the fs_roi checkpoint trained on NKI's 110-column table couldn't be
    reused against HBN's old 62-column one).
    """
    freesurfer_root = Path(freesurfer_root)
    all_region_paths = sorted(freesurfer_root.glob("sub-*/sub-*_regionsurfacestats.tsv"))
    all_brain_paths = sorted(freesurfer_root.glob("sub-*/sub-*_brainmeasures.tsv"))
    # This is a git-annex dataset (same as NKI's, see fetch_freesurfer_tsvs.py)
    # -- Path.exists() follows the symlink, so an un-fetched (broken-symlink)
    # subject is silently skipped here rather than crashing the whole batch.
    region_paths_by_subject = {
        _subject_id_from_tsv_path(path): path for path in all_region_paths if path.exists()
    }
    brain_paths_by_subject = {
        _subject_id_from_tsv_path(path): path for path in all_brain_paths if path.exists()
    }
    if not region_paths_by_subject:
        raise ValueError(f"No regionsurfacestats.tsv files found under {freesurfer_root}")
    if not brain_paths_by_subject:
        raise ValueError(f"No brainmeasures.tsv files found under {freesurfer_root}")

    # Same intersect-before-merge convention as load_all_nki_freesurfer_tsvs
    # -- a subject missing either file is dropped here rather than left-
    # joined into a row with subcortical columns silently NaN.
    shared_subject_ids = sorted(set(region_paths_by_subject).intersection(brain_paths_by_subject))
    print(
        f"{len(region_paths_by_subject)} regionsurfacestats.tsv, {len(brain_paths_by_subject)} "
        f"brainmeasures.tsv, {len(shared_subject_ids)} subjects with both"
    )

    region_frames = [load_regionsurfacestats_tsv(region_paths_by_subject[sid], atlas) for sid in shared_subject_ids]
    regionsurfacestats_df = pd.concat(region_frames, ignore_index=True)
    wide_df = reshape_regionsurfacestats_to_wide(regionsurfacestats_df)

    brain_frames = [load_brainmeasures_tsv(brain_paths_by_subject[sid]) for sid in shared_subject_ids]
    brainmeasures_df = pd.concat(brain_frames, ignore_index=True).drop_duplicates(subset=["subject_id"]).copy()

    combined_df = wide_df.merge(brainmeasures_df, on="subject_id", how="left")

    df = harmonize_nki_freesurfer_roi_df(combined_df)
    df.rename(columns={"subject_id": "Subject"}, inplace=True)
    df["Subject"] = df["Subject"].str.replace("sub-", "", regex=False)
    df = df[df["atlas"] == atlas].dropna(axis=1, how="all")

    min_present_frac = 0.9
    df = df.dropna(axis=1, thresh=int(min_present_frac * len(df)))
    n_before = len(df)
    df = df.dropna(axis=0)
    if len(df) != n_before:
        print(f"Dropped {n_before - len(df)} subject(s) with residual NaNs in an otherwise-dense column")

    return df


def load_ppmi_freesurfer_df(cth_csv=PPMI_FREESURFER_CTH_CSV, sa_csv=PPMI_FREESURFER_SA_CSV, event_id="BL"):
    """
    PPMI's own FreeSurfer export isn't per-subject regionsurfacestats.tsv --
    it's two wide, pre-pivoted CSVs (cortical thickness (CTH), surface area
    (SA)), one row per subject/visit already, so no pivoting is needed here,
    only filtering to one visit and joining the two on PATNO. Prefixes each
    CSV's region columns (cth_/sa_) since both files otherwise use the same
    bare region names (e.g. "lh_bankssts") and would collide on merge.
    """
    cth = pd.read_csv(cth_csv)
    sa = pd.read_csv(sa_csv)

    cth = cth[cth["EVENT_ID"] == event_id].drop(columns=["EVENT_ID"])
    sa = sa[sa["EVENT_ID"] == event_id].drop(columns=["EVENT_ID"])

    cth = cth.rename(columns={col: f"cth_{col}" for col in cth.columns if col != "PATNO"})
    sa = sa.rename(columns={col: f"sa_{col}" for col in sa.columns if col != "PATNO"})

    df = cth.merge(sa, on="PATNO", how="inner")
    df = df.rename(columns={"PATNO": "Subject"})
    df["Subject"] = df["Subject"].astype(str)
    df = df.drop_duplicates(subset=["Subject"]).dropna(axis=1, how="all").dropna(axis=0)
    return df


def build_hbn_cat12_merged_root(
    root_1=HBN_CAT12_BIDS_ROOT_1, root_2=HBN_CAT12_BIDS_ROOT_2, merged_root=None
):
    """
    HBN's CAT12 mwp1 output is split across two non-overlapping BIDS trees.
    train_linear.py's cat12_paths_and_subjects (and hence
    compute_reference_baselines.py's --nki-bids-root) only accepts a single
    root, so this builds one merged root of per-subject symlinks --
    sub-<ID> -> whichever of root_1/root_2 actually contains that subject --
    without touching cat12_paths_and_subjects's single-root assumption.
    """
    if merged_root is None:
        merged_root = REPO_ROOT / "artifacts" / "hbn_cat12_merged_bids"
    merged_root = Path(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)

    n_linked, n_skipped = 0, 0
    for root in (Path(root_1), Path(root_2)):
        for subject_dir in sorted(root.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            link_path = merged_root / subject_dir.name
            if link_path.exists() or link_path.is_symlink():
                n_skipped += 1
                continue
            link_path.symlink_to(subject_dir)
            n_linked += 1

    print(f"{merged_root}: {n_linked} subject symlink(s) created, {n_skipped} already present/skipped")
    return merged_root


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset", choices=["nki", "ppmi", "hbn"], default="nki")
    parser.add_argument("--freesurfer-root", type=Path, default=None, help="nki/hbn only.")
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS_FILE, help="nki only.")
    parser.add_argument("--session", default="BAS1", help="nki only.")
    parser.add_argument("--atlas", default=DEFAULT_ATLAS, help="nki/hbn only.")
    parser.add_argument("--cth-csv", type=Path, default=PPMI_FREESURFER_CTH_CSV, help="ppmi only.")
    parser.add_argument("--sa-csv", type=Path, default=PPMI_FREESURFER_SA_CSV, help="ppmi only.")
    parser.add_argument("--event-id", default="BL", help="ppmi only.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--skip-roi-table", action="store_true", help="hbn only, with --build-cat12-merged-root.")
    parser.add_argument(
        "--build-cat12-merged-root",
        action="store_true",
        help="hbn only -- also (or instead, with --skip-roi-table) merge HBN_CAT12_BIDS_ROOT_1/_2 into one symlink tree.",
    )
    parser.add_argument("--cat12-root-1", type=Path, default=HBN_CAT12_BIDS_ROOT_1, help="hbn only.")
    parser.add_argument("--cat12-root-2", type=Path, default=HBN_CAT12_BIDS_ROOT_2, help="hbn only.")
    parser.add_argument("--cat12-merged-root", type=Path, default=None, help="hbn only.")
    args = parser.parse_args()

    if args.build_cat12_merged_root and args.dataset != "hbn":
        parser.error("--build-cat12-merged-root is only meaningful for --dataset hbn")

    if args.freesurfer_root is None:
        args.freesurfer_root = DEFAULT_FREESURFER_ROOT if args.dataset == "nki" else HBN_FREESURFER_ROOT
    if args.output_csv is None:
        args.output_csv = DEFAULT_OUTPUT_DIR / f"{args.dataset}_freesurfer_roi.csv"

    return args


def main():
    args = parse_args()

    if args.dataset == "hbn" and args.build_cat12_merged_root:
        build_hbn_cat12_merged_root(args.cat12_root_1, args.cat12_root_2, args.cat12_merged_root)

    if args.dataset == "hbn" and args.skip_roi_table:
        return

    if args.dataset == "nki":
        df = load_all_nki_freesurfer_tsvs(
            freesurfer_root=args.freesurfer_root,
            subjects_file=args.subjects_file,
            session=args.session,
            atlas=args.atlas,
        )
    elif args.dataset == "hbn":
        df = load_hbn_freesurfer_df(freesurfer_root=args.freesurfer_root, atlas=args.atlas)
    else:
        df = load_ppmi_freesurfer_df(cth_csv=args.cth_csv, sa_csv=args.sa_csv, event_id=args.event_id)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} subjects x {len(df.columns)} columns to {args.output_csv}")


if __name__ == "__main__":
    main()
