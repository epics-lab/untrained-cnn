"""
Diagnostic: compares NIfTI resolution (array shape, voxel size, affine) for
one or more subjects across 3 categories -- their turboprep-normalized T1
(normalized.nii.gz), their per-model saliency maps (each at its own native,
usually-small model resolution -- see docs/known_limitations.md's
"Resampling pipeline" section), and the display template
(visualize_saliency.py's DEFAULT_TEMPLATE) -- then resamples the normalized
T1 and each saliency map onto the template's grid exactly the way
visualize_saliency.py's resample_saliency() does, and reports the resolution
after that resample, to confirm concretely whether everything actually lands
on the template's own grid (rather than assuming it, per the module docstring
in visualize_saliency.py).

Motivated by DEFAULT_TEMPLATE recently being pointed at MNI152Lin instead of
MNI152NLin2009cAsym (turboprep's own normalization target, see slurm/env.sh):
visualize_saliency.py's module docstring claims resampling
normalized.nii.gz-derived (turboprep) saliency maps onto the template is "an
identity transform -- pixel-perfect, no interpolation" for the 4
turboprep-based models, a claim that only holds when the template IS
MNI152NLin2009cAsym. This script checks that claim concretely for whichever
template is actually configured, instead of assuming it.

Usage:
    python3 check_resolutions.py --subjects A00000000 A00000001
    python3 check_resolutions.py --subjects A00000000 --kind abs --models brainiac swinbrain
    python3 check_resolutions.py --subjects A00000000 --attribution-tag gxi
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from visualize_saliency import DEFAULT_SALIENCY_DIR, DEFAULT_TEMPLATE, load_canonical_template, resample_saliency

DEFAULT_TURBOPREP_ROOT = Path(os.environ.get("NKI_AFFINE_TURBOPREP_ROOT", "/path/to/NKI_hdbet_affine"))

MODELS = [
    "optim_uncnn_raw",
    "optim_uncnn_rank",
    "optim_uncnn_sobel",
    "brainiac",
    "swinbrain",
    "simclr",
    "anatcl_global",
    "anatcl_local",
]

# calculate_gradcam.py's attribution_method_tag() filename fragments -- see
# compute_saliency_correlations.py's identical mapping.
ATTRIBUTION_FILENAME_PREFIX = {"gradient": "", "gxi": "gxi_", "ig_zero": "ig_zero_", "ig_reference": "ig_reference_"}


def describe_nifti(img):
    """{shape, voxel_size (mm, from the affine, not just header zooms -- matches what nilearn/nibabel actually use for resampling), affine_diag}."""
    voxel_sizes = nib.affines.voxel_sizes(img.affine)
    return {
        "shape": "x".join(str(d) for d in img.shape),
        "voxel_size_mm": "x".join(f"{v:.3f}" for v in voxel_sizes),
        "affine_diag": ",".join(f"{img.affine[i, i]:.3f}" for i in range(3)),
    }


def baseline_saliency_path(saliency_dir, model, tag, kind):
    return saliency_dir / f"{model}_{ATTRIBUTION_FILENAME_PREFIX[tag]}{kind}_saliency.nii.gz"


def already_on_template_grid(img, template_img, atol=1e-3):
    """
    Whether `img` already sits on `template_img`'s exact grid (same shape,
    same affine within `atol`) -- i.e. resample_to_img would be a true
    no-op identity resample for it, not a real interpolation. This is the
    meaningful check here, NOT the post-resample shape (nilearn's
    resample_to_img *always* forces the output onto the target's shape by
    construction, so comparing shape after resampling is tautological and
    would never catch anything). visualize_saliency.py's module docstring
    claims this holds for the 4 turboprep-based models' saliency maps when
    DEFAULT_TEMPLATE is turboprep's own normalization target
    (MNI152NLin2009cAsym) -- that claim doesn't automatically carry over if
    DEFAULT_TEMPLATE points elsewhere (see this module's docstring).
    """
    return tuple(img.shape) == tuple(template_img.shape) and np.allclose(img.affine, template_img.affine, atol=atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs, e.g. A00000000 A00000001.")
    parser.add_argument("--saliency-dir", type=Path, default=DEFAULT_SALIENCY_DIR)
    parser.add_argument("--turboprep-root", type=Path, default=DEFAULT_TURBOPREP_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--kind", choices=["signed", "abs"], default="signed")
    parser.add_argument("--models", nargs="*", default=MODELS, help="Subset of models to check (default: all 8).")
    parser.add_argument(
        "--attribution-tag",
        choices=list(ATTRIBUTION_FILENAME_PREFIX),
        default="gradient",
        help="Which attribution method's saliency maps to check (default: gradient, i.e. vanillagrad's untagged "
        "baseline files). Every model's map is a genuinely different file per tag, but its native resolution "
        "doesn't depend on which attribution method produced it -- only on that model's own Dataset preprocessing "
        "(train_linear.py) -- so 'gradient' is representative for a pure resolution/grid check.",
    )
    args = parser.parse_args()

    print(f"Loading template: {args.template}")
    template_img = load_canonical_template(args.template)
    rows = [{"subject": "-", "category": "template", "file": args.template.name, **describe_nifti(template_img)}]

    for subject in args.subjects:
        print(f"\n=== {subject} ===")

        normalized_path = args.turboprep_root / subject / "normalized.nii.gz"
        if normalized_path.exists():
            norm_img = nib.load(normalized_path)
            already_identity = already_on_template_grid(norm_img, template_img)
            rows.append(
                {
                    "subject": subject,
                    "category": "normalized",
                    "file": normalized_path.name,
                    "already_on_template_grid": already_identity,
                    **describe_nifti(norm_img),
                }
            )
            print(f"  resampling {normalized_path.name} onto the template "
                  f"({'identity, already on this grid' if already_identity else 'REAL interpolation needed'})...")
            resampled = resample_saliency(normalized_path, template_img)
            rows.append(
                {
                    "subject": subject,
                    "category": "normalized (resampled)",
                    "file": normalized_path.name,
                    "already_on_template_grid": "",
                    "shape": "x".join(str(d) for d in resampled.shape),
                    "voxel_size_mm": "(forced onto template's grid by resample_to_img)",
                    "affine_diag": "",
                }
            )
        else:
            print(f"  WARNING: {normalized_path} not found -- skipping normalized T1 for {subject}.")

        subject_dir = args.saliency_dir / subject
        for model in args.models:
            f = baseline_saliency_path(subject_dir, model, args.attribution_tag, args.kind)
            if not f.exists():
                print(f"  WARNING: {f.name} not found for {subject} -- skipping {model}.")
                continue
            sal_img = nib.load(f)
            already_identity = already_on_template_grid(sal_img, template_img)
            rows.append(
                {
                    "subject": subject,
                    "category": f"saliency ({model})",
                    "file": f.name,
                    "already_on_template_grid": already_identity,
                    **describe_nifti(sal_img),
                }
            )
            print(f"  resampling {f.name} onto the template "
                  f"({'identity, already on this grid' if already_identity else 'REAL interpolation needed'})...")
            resampled = resample_saliency(f, template_img)
            rows.append(
                {
                    "subject": subject,
                    "category": f"saliency ({model}, resampled)",
                    "file": f.name,
                    "already_on_template_grid": "",
                    "shape": "x".join(str(d) for d in resampled.shape),
                    "voxel_size_mm": "(forced onto template's grid by resample_to_img)",
                    "affine_diag": "",
                }
            )

    df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print(df.to_string(index=False))

    # NOTE: every "(resampled)" row's shape always equals the template's --
    # nilearn's resample_to_img forces that by construction, so it's not a
    # meaningful check on its own (see already_on_template_grid's docstring).
    # What actually matters is already_on_template_grid on the *native*
    # rows: True means resample_to_img is a true no-op identity resample for
    # that file (matches visualize_saliency.py's module-docstring claim for
    # the 4 turboprep-based models' saliency maps -- but only when the
    # template really is turboprep's own normalization target); False means
    # a real interpolation happens every time that file gets displayed.
    native_rows = df[df["already_on_template_grid"] != ""]
    needs_real_resample = native_rows[native_rows["already_on_template_grid"] == False]  # noqa: E712
    print("\n" + "=" * 80)
    if needs_real_resample.empty:
        print("Every native file is already on the template's exact grid -- resampling is a no-op identity transform "
              "throughout.")
    else:
        print(
            f"{len(needs_real_resample)}/{len(native_rows)} native file(s) are NOT already on the template's grid -- "
            f"resampling them is a REAL interpolation, not the identity transform visualize_saliency.py's module "
            f"docstring describes for the turboprep-based models:"
        )
        print(needs_real_resample[["subject", "category", "file", "shape", "voxel_size_mm"]].to_string(index=False))


if __name__ == "__main__":
    main()
