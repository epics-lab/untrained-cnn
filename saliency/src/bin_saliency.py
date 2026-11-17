"""
Aggregate per-subject baseline saliency maps (from calculate_gradcam.py), per
model, into per-bin mean/std maps -- like aggregate_saliency.py, but instead
of one mean/std map per model pooling *all* subjects, this produces one
mean/std map per (model, bin) group, where subjects are split into bins one
of two ways (--bin-by):

  age    Bins by the subject's true age (shared across models -- a subject's
         age doesn't depend on which model looked at them). Edges are set
         with --age-edges (default 0 20 40 60 80 100 -> "0-20", "21-40", ...).

  error  Bins by |predicted_age - true_age| for that specific model -- a
         subject can land in a different error bin for different models,
         since each model predicts differently. Edges are set with
         --error-edges (default 2 5 10 -> "within2", "2to5", "5to10",
         "over10").

True/predicted age come from the same {subject_dir}/{model}_sanity_checks.json
files calculate_gradcam.py already writes (see visualize_saliency.py's
load_subject_metadata for the same convention) -- or, for a non-default
--attribution-tags entry, {subject_dir}/{model}_{tag}_sanity_checks.json
(calculate_gradcam.py's attribution_method_tag()). true_age/predicted_age
are identical across every tag for a given (subject, model): they come from
the one shared forward pass, only the attribution/backward computation
differs -- so which tag's JSON gets read doesn't affect binning, only which
tag's saliency .nii.gz gets looked up and binned.

Output layout:
  {output-dir}/{model}_{age|error}bin_{label}_{mean|std}_{kind}_saliency.nii.gz
  {output-dir}/{model}_{tag}_{age|error}bin_{label}_{mean|std}_{kind}_saliency.nii.gz (non-default tag)

Usage:
    python bin_saliency.py --bin-by age
    python bin_saliency.py --bin-by error --error-edges 2 5 10 --models brainiac simclr
    python bin_saliency.py --bin-by age --attribution-tags gradient gxi
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from targets import get_target_spec

DEFAULT_SALIENCY_DIR = Path(__file__).resolve().parents[1] / "saliency_outputs"
DEFAULT_MODELS = [
    "optim_uncnn_raw",
    "optim_uncnn_rank",
    "optim_uncnn_sobel",
    "brainiac",
    "swinbrain",
    "simclr",
    "anatcl_global",
    "anatcl_local",
]

# optim_uncnn's 3 channel variants (raw/rank/sobel) are an internal fan-out
# inside a single calculate_gradcam.py run -- calculate_gradcam.py still
# writes one sanity_checks.json per subject named after the real model
# ("optim_uncnn"), not one per pseudo-model name. Map back to that stem.
_UNCNN_CHANNEL_SUFFIXES = ("_raw", "_rank", "_sobel")

# attribution_method_tag()'s tags (calculate_gradcam.py), plus "gradient" as
# the name for its untagged default so every method has an explicit,
# selectable name on this script's own --attribution-tags. Filename fragment
# is "" for gradient (calculate_gradcam.py never inserts a tag for it) and
# "{tag}_" for the other three, inserted directly before the kind
# ("abs"/"signed") in both the source saliency filename this script reads
# and the JSON sanity-checks filename it reads metadata from.
ATTRIBUTION_TAGS = ("gradient", "gxi", "ig_zero", "ig_reference")


def _filename_tag_prefix(tag):
    return "" if tag == "gradient" else f"{tag}_"


def sanity_json_stem(model, tag="gradient"):
    if model.startswith("optim_uncnn_"):
        for suffix in _UNCNN_CHANNEL_SUFFIXES:
            if model == f"optim_uncnn{suffix}":
                model = "optim_uncnn"
                break
    return model if tag == "gradient" else f"{model}_{tag}"


def find_subject_dirs(root, exclude_names=("aggregate", "age_bins", "error_bins", "value_bins", "class_bins")):
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in exclude_names)


def load_meta(subject_dir, model, target_spec, tag="gradient"):
    """
    Returns (true_value, predicted_value), or (None, None) if unavailable.
    true_value/predicted_value don't depend on `tag` (see module docstring),
    but the JSON is only ever written under one specific tag's filename per
    run, so the caller's tag still has to match an existing file.
    """
    jf = subject_dir / f"{sanity_json_stem(model, tag)}_sanity_checks.json"
    if not jf.exists():
        return None, None
    try:
        d = json.loads(jf.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    return d.get(target_spec.json_true_key), d.get(target_spec.json_pred_key)


def value_bin_label(value, edges):
    """Half-open bins [lo, hi) except the last, which is closed: [lo, hi].
    Labels use %g formatting (edges are floats) so "0-20", not "0.0-20.0" --
    must match the label format the calling shell script's bash arithmetic
    reproduces independently. Used for any continuous regression target
    (age, bmi) -- "age" is a backward-compatible --bin-by alias for this.
    """
    last = len(edges) - 2
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo <= value < hi or (i == last and value == hi):
            return f"{lo:g}-{hi:g}"
    return None


def error_bin_label(err, edges):
    """Half-open bins (prev, e] except the first, which starts at 0: [0, e]."""
    prev = None
    for e in edges:
        if err <= e:
            return f"within{e:g}" if prev is None else f"{prev:g}to{e:g}"
        prev = e
    return f"over{edges[-1]:g}"


def class_bin_label(value, target_spec):
    """Classification targets (sex): bin = the class label itself, no edges."""
    return target_spec.decode_display(value)


def aggregate_stack(paths, stat):
    """paths: [Path, ...], all the same model+kind. Skips grid-mismatched files."""
    ref_img = nib.load(paths[0])
    ref_shape, ref_affine = ref_img.shape, ref_img.affine

    stack, used = [], []
    for i, path in enumerate(paths):
        img = ref_img if i == 0 else nib.load(path)
        if img.shape != ref_shape or not np.allclose(img.affine, ref_affine, atol=1e-3):
            print(f"    skipping {path.name}: grid mismatch (shape {img.shape} vs {ref_shape})")
            continue
        stack.append(img.get_fdata(dtype=np.float32))
        used.append(path)

    if not stack:
        return None, ref_affine, used

    data = np.stack(stack, axis=0)
    agg = data.mean(axis=0) if stat == "mean" else data.std(axis=0)
    return agg, ref_affine, used


def main():
    parser = argparse.ArgumentParser(description="Bin per-subject saliency maps by age or prediction error, per model.")
    parser.add_argument(
        "--saliency-dir", type=Path, default=None,
        help="Default: <repo>/saliency_outputs/<target> (see --target).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <saliency-dir>/{value,error,class}_bins.")
    parser.add_argument(
        "--bin-by", choices=["age", "value", "error", "class"], required=True,
        help="'age' is a backward-compatible alias for 'value' with --target forced to age. 'value': bin by "
        "the target's true value (continuous regression targets: age, bmi). 'error': bin by |predicted-true| "
        "(regression targets only). 'class': bin by the true class label itself (classification targets: sex).",
    )
    parser.add_argument("--target", choices=["age", "sex", "bmi"], default="age")
    parser.add_argument("--age-edges", type=float, nargs="*", default=[0, 20, 40, 60, 80, 100], help="Edges for --bin-by value/age.")
    parser.add_argument("--error-edges", type=float, nargs="*", default=[2, 5, 10])
    parser.add_argument("--kinds", nargs="*", default=["abs"], choices=["signed", "abs"])
    parser.add_argument("--stats", nargs="*", default=["mean", "std"], choices=["mean", "std"])
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument(
        "--attribution-tags",
        nargs="*",
        default=list(ATTRIBUTION_TAGS),
        choices=list(ATTRIBUTION_TAGS),
        help="Which of calculate_gradcam.py's --attribution-method outputs to bin (default: all 4). "
        "'gradient' is the untagged default method (displayed as 'vanillagrad' by visualize_saliency.py); "
        "'gxi'/'ig_zero'/'ig_reference' match attribution_method_tag()'s tags exactly.",
    )
    parser.add_argument("--min-subjects", type=int, default=1, help="Skip a (model, tag, bin) group with fewer subjects than this.")
    args = parser.parse_args()

    # "age" is a backward-compatible alias for "value" binning + --target
    # forced to age -- dispatch logic uses bin_by_mode ("value"/"error"/
    # "class"), but args.bin_by itself is left as the literal the user
    # passed, so output filenames/dirs ("age_bins/", "_agebin_") for
    # existing `--bin-by age` invocations stay byte-for-byte unchanged.
    # `--bin-by value` (new) gets its own "_valuebin_"/"value_bins/" naming.
    if args.bin_by == "age":
        args.target = "age"
        bin_by_mode = "value"
    else:
        bin_by_mode = args.bin_by
    args.saliency_dir = args.saliency_dir or (DEFAULT_SALIENCY_DIR / args.target)
    target_spec = get_target_spec(args.target)
    if bin_by_mode == "error" and target_spec.task_type != "regression":
        parser.error(f"--bin-by error is only supported for regression targets (age, bmi), not {args.target!r}")

    output_dir = args.output_dir or (args.saliency_dir / f"{args.bin_by}_bins")
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = find_subject_dirs(args.saliency_dir)
    if not subject_dirs:
        raise SystemExit(f"No subject subfolders found in {args.saliency_dir}")
    print(f"Found {len(subject_dirs)} subject folder(s): {[d.name for d in subject_dirs]}")

    for kind in args.kinds:
        # {(model, tag, bin_label): [Path, ...]}
        groups = defaultdict(list)
        for model in args.models:
            for tag in args.attribution_tags:
                for subj_dir in subject_dirs:
                    true_value, predicted_value = load_meta(subj_dir, model, target_spec, tag)
                    if true_value is None:
                        continue
                    f = subj_dir / f"{model}_{_filename_tag_prefix(tag)}{kind}_saliency.nii.gz"
                    if not f.exists():
                        continue

                    if bin_by_mode == "value":
                        label = value_bin_label(true_value, args.age_edges)
                    elif bin_by_mode == "class":
                        label = class_bin_label(true_value, target_spec)
                    else:  # error
                        if predicted_value is None:
                            continue
                        label = error_bin_label(abs(predicted_value - true_value), args.error_edges)

                    if label is None:
                        continue
                    groups[(model, tag, label)].append(f)

        if not groups:
            print(f"\nNo baseline '{kind}' files found to bin.")
            continue

        for (model, tag, label), files in sorted(groups.items()):
            tag_note = "" if tag == "gradient" else f" ({tag})"
            if len(files) < args.min_subjects:
                print(f"\n{model}{tag_note} [{label}] ({kind}): {len(files)} subject(s) -- below --min-subjects, skipped.")
                continue
            print(f"\n{model}{tag_note} [{label}] ({kind}): {len(files)} subject(s)")
            # Tag sits right after `model`, before "_{age,error}bin_" -- same
            # position visualize_saliency.py's split_attribution_tag() (see
            # its lookahead over "_agebin_"/"_errorbin_") expects it in, so
            # non-gradient bin outputs get their own correctly-labeled row
            # instead of collapsing into "vanillagrad_...".
            model_tag_prefix = model if tag == "gradient" else f"{model}_{tag}"
            for stat in args.stats:
                agg, affine, used = aggregate_stack(files, stat)
                if agg is None:
                    print(f"    {stat}: no subjects passed the grid-consistency check -- skipped.")
                    continue
                out_path = output_dir / f"{model_tag_prefix}_{args.bin_by}bin_{label}_{stat}_{kind}_saliency.nii.gz"
                nib.save(nib.Nifti1Image(agg.astype(np.float32), affine), out_path)
                print(f"    {stat}: -> saved {out_path} (over {len(used)} subjects)")


if __name__ == "__main__":
    main()
