"""
Aggregate per-subject baseline saliency maps (from calculate_gradcam.py) across
subjects, per model, into a single map per model.

Expected layout:
    saliency_outputs/<SUBJECT_ID>/{model}_{signed,abs}_saliency.nii.gz

Sanity-check outputs (arch_/data_/input_-prefixed, from --randomize-model/
--randomize-data/--randomize-input) are skipped by default -- only the real
baseline maps are aggregated. Each subject's map for a given model already
lives on that model's shared native grid (turboprep's common template for
optim_uncnn/brainiac/swinbrain/simclr, CAT12's common template for anatcl --
verified identical shape/affine across subjects), so aggregation is a direct
voxel-wise statistic across subjects, no resampling needed. A grid-consistency
check is still run per file as a safety net; mismatched subjects are skipped
and reported rather than silently corrupting the aggregate.

Usage:
    python aggregate_saliency.py
    python aggregate_saliency.py --stat median --models brainiac simclr
    python aggregate_saliency.py --stat entropy --entropy-bins 10

--stat entropy computes, per voxel, the Shannon entropy of the across-subject
value histogram (see voxelwise_entropy) -- a disagreement metric distinct
from std: two voxels can share the same std but differ in entropy if one is
a single unimodal spread and the other is subjects splitting into distinct
clusters.

--sanity-check {arch,data,input}: aggregate sanity-check maps instead
------------------------------------------------------------------------
Switches from aggregating BASELINE maps to aggregating arch_/data_/input_
sanity-check maps -- specifically, slurm/compute_sanity_checks_randomize_model.sh's
per-subject "arch_..." model-randomization maps (one file per (subject,
model), all reset to the SAME randomly-initialized weights via a fixed
--randomize-seed), so averaging across subjects means "this randomly-
initialized model's average explanation across many real inputs," directly
analogous to how the trained baseline is already averaged across subjects
above -- rather than the single-reference-subject snapshot
slurm/compute_sanity_checks.sh alone produces.

Requires --models explicitly in this mode (no auto-discovery): unlike a
baseline filename, an arch_/data_/input_ filename isn't unambiguously
parseable back into a bare model name (calculate_gradcam.py inserts the
attribution-method tag AFTER the arch_/data_/input_ stem, and anatcl_{level}/
optim_uncnn_{channel} need their own differently-shaped prefix -- see
visualize_paper_nilearn.py's sanity_prefix()/collect_kind_files() docstrings
for the exact ambiguity this avoids). sanity_prefix() below is a deliberate
duplicate of that function (not an import) so this script stays nilearn-free
-- see its own docstring for why. Same duplication precedent as
slurm/visualize_aggregate.sh's own copy.

Output in this mode overwrites the SAME path slurm/compute_sanity_checks.sh's
single-subject run already writes to (saliency_affine/<target>/arch_..., no
"_aggregate_" suffix inserted) -- every downstream consumer
(visualize_paper_nilearn.py --sanity-check, compute_saliency_correlations.py)
already reads that exact filename regardless of whether it came from 1
subject or many, so no consumer needs to change. A ".json" sidecar records
how many subjects went into it, for provenance.

--randomize-model-variant {all,step{NN:02d}_{module},...}: which
--randomize-model-mode variant's filenames to collect/write (default "all",
matching --randomize-model-mode all). Pass e.g. "step01_head" to aggregate a
--randomize-model-mode cascade --randomize-max-layers 1 head-only run (see
slurm/compute_sanity_checks_randomize_model.sh) instead -- combined with
--saliency-dir pointing at that run's own persubject subtree (e.g.
arch_linear_persubject/), this keeps every variant's outputs in their own
filenames/directories so aggregating one variant never overwrites another's.

Usage:
    python aggregate_saliency.py --sanity-check arch --target age \\
        --saliency-dir saliency_affine/age/arch_persubject \\
        --output-dir saliency_affine/age --models brainiac swinbrain

    # head-only (linear layer) variant:
    python aggregate_saliency.py --sanity-check arch --target age \\
        --saliency-dir saliency_affine/age/arch_linear_persubject \\
        --output-dir saliency_affine/age --models brainiac swinbrain \\
        --randomize-model-variant step01_head
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

SANITY_PREFIXES = ("arch_", "data_", "input_")
DEFAULT_SALIENCY_DIR = Path(__file__).resolve().parents[1] / "saliency_outputs"


def sanity_prefix(model, test, arch_variant="all"):
    """
    arch_/data_/input_ filename prefix for (model, test in "arch"/"data"/
    "input") -- exact duplicate of visualize_paper_nilearn.py's own
    sanity_prefix (see module docstring for why this is a duplicate, not an
    import). Keep the two in sync if calculate_gradcam.py's/
    calculate_gradcam_tabular.py's naming ever changes.

    `arch_variant` (test == "arch" only): the segment calculate_gradcam.py's
    --randomize-model-mode bakes into the filename right after "arch_{model}_"
    -- "all" for --randomize-model-mode all (the default, matching every
    output produced before per-variant subdirectories existed), or
    "step{NN:02d}_{module}" for a --randomize-model-mode cascade step (e.g.
    "step01_head" for slurm/compute_sanity_checks_randomize_model.sh's
    --randomize-max-layers 1 head-only run -- see that script's own
    comments). Ignored for test in ("data", "input"), which have no such
    variant.
    """
    if model.startswith("anatcl_"):
        level = model[len("anatcl_"):]
        return {
            "arch": f"arch_anatcl_{arch_variant}_{level}_",
            "data": f"data_anatcl_{level}_",
            "input": f"input_anatcl_permute_{level}_",
        }[test]
    if model.startswith("optim_uncnn_"):
        channel = model[len("optim_uncnn_"):]
        return {
            "arch": f"arch_optim_uncnn_{arch_variant}_{channel}_",
            "data": f"data_optim_uncnn_{channel}_",
            "input": f"input_optim_uncnn_permute_{channel}_",
        }[test]
    return {
        "arch": f"arch_{model}_{arch_variant}_",
        "data": f"data_{model}_",
        "input": f"input_{model}_permute_",
    }[test]


def is_baseline_file(path):
    return path.name.endswith("_saliency.nii.gz") and not path.name.startswith(SANITY_PREFIXES)


def model_name_from_filename(path, kind):
    return path.name.replace(f"_{kind}_saliency.nii.gz", "")


def find_subject_dirs(root, exclude=()):
    """Subfolders of `root` that hold per-subject saliency maps.

    Excludes `exclude` (resolved paths) so the script's own --output-dir --
    by default a subfolder of `root` itself -- is never picked up as a
    "subject" on a rerun once it's populated with *_aggregate_* outputs.
    """
    exclude = {p.resolve() for p in exclude}
    return sorted(p for p in root.iterdir() if p.is_dir() and p.resolve() not in exclude)


def collect_baseline_files(subject_dirs, kind):
    """Returns {model_name: [(subject_id, Path), ...]} for baseline (non-sanity-check) files."""
    groups = defaultdict(list)
    for subj_dir in subject_dirs:
        for f in subj_dir.glob(f"*_{kind}_saliency.nii.gz"):
            if not is_baseline_file(f):
                continue
            groups[model_name_from_filename(f, kind)].append((subj_dir.name, f))
    return groups


def collect_sanity_check_files(subject_dirs, kind, models, sanity_check, attribution_tag, arch_variant="all"):
    """
    Returns {model_name: [(subject_id, Path), ...]} for arch_/data_/input_
    sanity-check files, one EXACT filename per model (via sanity_prefix +
    attribution_tag + kind) looked up in each subject dir -- not a glob, so
    there's no prefix-ambiguity to resolve (see module docstring).

    `arch_variant`: see sanity_prefix()'s own docstring -- ignored unless
    sanity_check == "arch".
    """
    groups = defaultdict(list)
    for model in models:
        filename = f"{sanity_prefix(model, sanity_check, arch_variant)}{attribution_tag}{kind}_saliency.nii.gz"
        for subj_dir in subject_dirs:
            f = subj_dir / filename
            if f.exists():
                groups[model].append((subj_dir.name, f))
    return groups


def voxelwise_entropy(data, bins=10):
    """
    Per-voxel Shannon entropy (in nats) of the across-subject value
    distribution, estimated by histogramming each voxel's `data.shape[0]`
    subject values into `bins` bins.

    Bin edges are the 1st/99th percentile of the *whole* stack (not
    per-voxel) so entropy stays comparable across voxels -- a per-voxel range
    would make every voxel's own min/max define its bins, silently inflating
    entropy for voxels whose subjects happen to agree tightly but not exactly
    (float noise still spans the full bin range). Values are clipped to that
    range before binning, so outlier subjects land in the edge bins rather
    than stretching everyone else's bins thin.

    Unlike `std`, this isn't a Gaussian-implied spread: a voxel where
    subjects split into two tight clusters (bimodal disagreement) gets higher
    entropy than a voxel with the same std but a single unimodal spread, so
    it surfaces a distinct kind of across-subject disagreement.

    `data`: stacked subject volumes, shape [n_subjects, D, H, W].
    Returns: entropy per voxel, shape [D, H, W], dtype float32.
    """
    n_subjects = data.shape[0]
    lo, hi = np.percentile(data, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6

    clipped = np.clip(data, lo, hi)
    bin_idx = np.clip(((clipped - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)

    entropy = np.zeros(data.shape[1:], dtype=np.float64)
    for b in range(bins):
        p = (bin_idx == b).sum(axis=0) / n_subjects
        nz = p > 0
        entropy[nz] -= p[nz] * np.log(p[nz])
    return entropy.astype(np.float32)


def aggregate_group(files, stat="mean", entropy_bins=10):
    """files: [(subject_id, Path), ...] all claiming to be the same model+kind."""
    ref_img = nib.load(files[0][1])
    ref_shape, ref_affine = ref_img.shape, ref_img.affine

    stack, used = [], []
    for i, (subject_id, path) in enumerate(files):
        img = ref_img if i == 0 else nib.load(path)
        if img.shape != ref_shape or not np.allclose(img.affine, ref_affine, atol=1e-3):
            print(f"    skipping {subject_id} ({path.name}): grid mismatch (shape {img.shape} vs {ref_shape})")
            continue
        stack.append(img.get_fdata(dtype=np.float32))
        used.append(subject_id)

    if not stack:
        return None, ref_affine, used

    data = np.stack(stack, axis=0)
    if stat == "mean":
        agg = data.mean(axis=0)
    elif stat == "median":
        agg = np.median(data, axis=0)
    elif stat == "entropy":
        agg = voxelwise_entropy(data, bins=entropy_bins)
    else:  # std
        agg = data.std(axis=0)
    return agg, ref_affine, used


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-subject baseline saliency maps across subjects, per model."
    )
    parser.add_argument(
        "--saliency-dir",
        type=Path,
        default=None,
        help="Directory containing per-subject subfolders (e.g. saliency_outputs/age/A00000000/...). "
        "Default: <repo>/saliency_outputs/<target>.",
    )
    parser.add_argument(
        "--target", choices=["age", "sex", "bmi"], default="age",
        help="Which target's saliency_outputs/<target>/ subtree to aggregate (default: age). Only used to "
        "resolve --saliency-dir's default -- ignored if --saliency-dir is passed explicitly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write {model}_aggregate_{kind}_saliency.nii.gz (default: <saliency-dir>/aggregate).",
    )
    parser.add_argument("--kinds", nargs="*", default=["signed", "abs"], choices=["signed", "abs"])
    parser.add_argument(
        "--stat",
        choices=["mean", "median", "std", "entropy"],
        default="mean",
        help="Aggregation statistic (default: mean).",
    )
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=10,
        help="Histogram bin count for --stat entropy (ignored otherwise). Default: 10.",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model names to aggregate. Default: all found (baseline mode only) -- REQUIRED when "
        "--sanity-check is passed, since arch_/data_/input_ filenames can't be auto-discovered (see module "
        "docstring).",
    )
    parser.add_argument(
        "--sanity-check", choices=["arch", "data", "input"], default=None,
        help="Aggregate arch_/data_/input_ sanity-check maps across per-subject subfolders instead of "
        "baseline maps -- see module docstring. Requires --models.",
    )
    parser.add_argument(
        "--attribution-tag", default="ig_reference_",
        help="--sanity-check only: the attribution-method tag baked into the sanity-check filename "
        "(calculate_gradcam.py's attribution_method_tag() -- '' for gradient, 'gxi_'/'ig_zero_'/"
        "'ig_reference_' otherwise). Default matches slurm/compute_sanity_checks*.sh's own default spec.",
    )
    parser.add_argument(
        "--randomize-model-variant", default="all",
        help="--sanity-check arch only: the --randomize-model-mode variant segment baked into the "
        "filename right after 'arch_{model}_' -- 'all' (default, matches --randomize-model-mode all) "
        "or 'step{NN:02d}_{module}' for a --randomize-model-mode cascade step (e.g. 'step01_head' for "
        "slurm/compute_sanity_checks_randomize_model.sh's head-only --randomize-max-layers 1 run). See "
        "sanity_prefix()'s docstring. Ignored for --sanity-check data/input.",
    )
    args = parser.parse_args()
    if args.sanity_check and not args.models:
        raise SystemExit("--sanity-check requires --models (see module docstring for why).")
    args.saliency_dir = args.saliency_dir or (DEFAULT_SALIENCY_DIR / args.target)

    output_dir = args.output_dir or (args.saliency_dir / "aggregate")

    subject_dirs = find_subject_dirs(args.saliency_dir, exclude=(output_dir,))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not subject_dirs:
        raise SystemExit(f"No subject subfolders found in {args.saliency_dir}")
    print(f"Found {len(subject_dirs)} subject folder(s): {[d.name for d in subject_dirs]}")

    for kind in args.kinds:
        if args.sanity_check:
            groups = collect_sanity_check_files(
                subject_dirs, kind, args.models, args.sanity_check, args.attribution_tag,
                arch_variant=args.randomize_model_variant,
            )
        else:
            groups = collect_baseline_files(subject_dirs, kind)
            if args.models:
                groups = {m: files for m, files in groups.items() if m in args.models}
        if not groups:
            label = f"{args.sanity_check} sanity-check" if args.sanity_check else "baseline"
            print(f"\nNo {label} '{kind}' files found.")
            continue

        for model_name, files in sorted(groups.items()):
            print(f"\n{model_name} ({kind}): {len(files)} subject(s) found [{', '.join(s for s, _ in files)}]")
            agg, affine, used = aggregate_group(files, stat=args.stat, entropy_bins=args.entropy_bins)
            if agg is None:
                print("    no subjects passed the grid-consistency check -- skipped.")
                continue
            if len(used) < len(files):
                print(f"    used {len(used)}/{len(files)} subjects after grid-consistency check")

            if args.sanity_check:
                out_path = output_dir / (
                    f"{sanity_prefix(model_name, args.sanity_check, args.randomize_model_variant)}"
                    f"{args.attribution_tag}{kind}_saliency.nii.gz"
                )
            else:
                out_path = output_dir / f"{model_name}_aggregate_{args.stat}_{kind}_saliency.nii.gz"
            nib.save(nib.Nifti1Image(agg.astype(np.float32), affine), out_path)
            print(f"    -> saved {out_path} ({args.stat} over {len(used)} subjects)")

            if args.sanity_check:
                sidecar = out_path.with_suffix("").with_suffix(".json")
                sidecar.write_text(json.dumps(
                    {
                        "sanity_check": args.sanity_check,
                        "randomize_model_variant": args.randomize_model_variant if args.sanity_check == "arch" else None,
                        "stat": args.stat,
                        "n_subjects": len(used),
                        "subjects": used,
                    },
                    indent=2,
                ))


if __name__ == "__main__":
    main()
