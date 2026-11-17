"""
One-off migration: fixes the directory layout left behind by the
calculate_gradcam.py target-insertion bug (see git history/PR for the fix in
that script's --output-dir handling). Before the fix, slurm/compute_saliency.sh's
image-model runs (unCNN, BrainIAC, SIMCLR, SWINBRAIN, ANATCL GLOBAL, ANATCL
LOCAL) wrote saliency maps to

    saliency_affine/<SUBJECT>/<target>/...          (WRONG -- subject outside target)

instead of the intended

    saliency_affine/<target>/<SUBJECT>/...          (RIGHT -- matches saliency_outputs/
                                                       layout every downstream tool expects,
                                                       e.g. compute_saliency_correlations.py)

fs_roi (calculate_gradcam_tabular.py) was never affected by that bug -- it
already writes straight to saliency_affine/<target>/<SUBJECT>/, which means
EVERY <target>/<SUBJECT>/ destination directory this script writes into
already exists and already contains that subject's fs_roi_*  files. This
script only ever ADDS files into those directories (the 6 image models'
files, which never share a filename prefix with fs_roi_*), and never
deletes or overwrites anything -- see --dry-run below to preview first.

What it does, per top-level entry directly under --saliency-dir that is a
subject folder (i.e. NOT named "age"/"sex"/"bmi" -- those are already-correct
target roots and are left alone as move DESTINATIONS, never treated as
something to migrate):
  for each <SUBJECT>/<target>/ subdirectory (target in age/sex/bmi):
    for each file directly inside it:
      move it to <target>/<SUBJECT>/<same filename>
      -- if that destination file already exists, SKIP it (never overwrite)
         and record it as a conflict for you to look at by hand.
  after all files in a <SUBJECT>/<target>/ dir are moved out, remove that
  now-empty <target>/ dir; after all of a subject's target dirs are gone,
  remove the now-empty <SUBJECT>/ dir. Directories are only ever removed
  once they are verifiably empty (os.rmdir fails loudly otherwise) -- this
  script never recursively deletes a non-empty directory or any file.

Usage:
    # Preview only -- prints every planned move/conflict/removal, changes nothing.
    python3 src/reorganize_saliency_affine.py --dry-run

    # Actually perform the migration.
    python3 src/reorganize_saliency_affine.py

    # Point at a different root (default: <repo>/saliency_affine).
    python3 src/reorganize_saliency_affine.py --saliency-dir /path/to/saliency_affine --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path

TARGETS = ("age", "sex", "bmi")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SALIENCY_DIR = REPO_ROOT / "saliency_affine"


def is_subject_dir(path):
    """A top-level entry under --saliency-dir is treated as a subject folder
    to migrate iff it's a directory and its name isn't one of the target
    roots (age/sex/bmi) -- those are legitimate, already-correct destinations
    (see module docstring) and must never be walked as if they were subjects."""
    return path.is_dir() and path.name not in TARGETS


def plan_moves(saliency_dir):
    """
    Yields ("move", src_file, dst_file) for every file that needs migrating,
    and ("conflict", src_file, dst_file) for every one whose destination
    already exists (never auto-resolved -- see module docstring). Doesn't
    touch the filesystem itself; the caller decides what to do with each
    yielded action (see main()'s --dry-run branch vs. the real-move branch).
    """
    subject_dirs = sorted(p for p in saliency_dir.iterdir() if is_subject_dir(p))
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        for target in TARGETS:
            src_target_dir = subject_dir / target
            if not src_target_dir.is_dir():
                continue
            dst_target_dir = saliency_dir / target / subject
            # Recursive walk (not just a flat listdir) in case any file ever
            # ends up nested deeper than one level -- doesn't happen in the
            # current layout (every file sits directly in <SUBJECT>/<target>/),
            # but costs nothing to handle correctly rather than assume flat.
            for src_file in sorted(p for p in src_target_dir.rglob("*") if p.is_file()):
                rel = src_file.relative_to(src_target_dir)
                dst_file = dst_target_dir / rel
                if dst_file.exists():
                    yield ("conflict", src_file, dst_file)
                else:
                    yield ("move", src_file, dst_file)


def remove_empty_dirs(saliency_dir, subject_dirs, dry_run):
    """
    Removes now-empty <SUBJECT>/<target>/ and <SUBJECT>/ directories left
    behind after every file has been moved out of them. Only ever calls
    os.rmdir, which raises (and this function lets that propagate) if a
    directory isn't actually empty -- e.g. because a conflict left a file
    behind -- so a subject with unresolved conflicts simply keeps its
    (now near-empty) directory rather than being silently deleted.
    """
    for subject_dir in subject_dirs:
        for target in TARGETS:
            target_dir = subject_dir / target
            if target_dir.is_dir() and not any(target_dir.iterdir()):
                print(f"  rmdir {target_dir}" + (" (dry-run)" if dry_run else ""))
                if not dry_run:
                    os.rmdir(target_dir)
        if subject_dir.is_dir() and not any(subject_dir.iterdir()):
            print(f"  rmdir {subject_dir}" + (" (dry-run)" if dry_run else ""))
            if not dry_run:
                os.rmdir(subject_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate saliency_affine/<SUBJECT>/<target>/ (wrong) to saliency_affine/<target>/<SUBJECT>/ "
        "(right) -- see module docstring. Never overwrites or deletes an existing file."
    )
    parser.add_argument("--saliency-dir", type=Path, default=DEFAULT_SALIENCY_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every planned move/conflict/directory-removal without touching the filesystem.",
    )
    parser.add_argument(
        "--keep-empty-dirs", action="store_true",
        help="Don't remove <SUBJECT>/<target>/ and <SUBJECT>/ directories after their files are moved out "
        "(default: remove them once empty, since an empty leftover directory tree is itself confusing/misleading "
        "-- e.g. downstream tools that iterate saliency_affine/*/ expecting only subject dirs).",
    )
    args = parser.parse_args()

    if not args.saliency_dir.is_dir():
        parser.error(f"--saliency-dir {args.saliency_dir} does not exist or is not a directory")

    subject_dirs = sorted(p for p in args.saliency_dir.iterdir() if is_subject_dir(p))
    print(f"Found {len(subject_dirs)} subject folder(s) under {args.saliency_dir} to check.")
    if args.dry_run:
        print("--dry-run: no files or directories will actually be moved/removed.\n")

    n_moved = n_conflicts = 0
    conflicts = []
    for action, src, dst in plan_moves(args.saliency_dir):
        if action == "conflict":
            n_conflicts += 1
            conflicts.append((src, dst))
            print(f"  CONFLICT (skipped, destination already exists): {src} -> {dst}")
            continue
        n_moved += 1
        print(f"  move {src} -> {dst}" + (" (dry-run)" if args.dry_run else ""))
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    print(f"\n{n_moved} file(s) {'would be' if args.dry_run else ''} moved, "
          f"{n_conflicts} conflict(s) {'would be' if args.dry_run else ''} skipped.")

    if not args.keep_empty_dirs:
        print("\nRemoving now-empty <SUBJECT>/<target>/ and <SUBJECT>/ directories:")
        remove_empty_dirs(args.saliency_dir, subject_dirs, args.dry_run)

    if conflicts:
        print(f"\n{len(conflicts)} conflict(s) were skipped -- these destination files already existed and were "
              "left untouched, so their corresponding source files (see the CONFLICT lines above) are still sitting "
              "in the old <SUBJECT>/<target>/ location. Inspect these by hand -- this should not happen in practice "
              "(fs_roi's files use a fs_roi_* prefix, disjoint from every other model's), so a conflict here likely "
              "means this script was already run before, or something else wrote into the destination directory.")


if __name__ == "__main__":
    main()
