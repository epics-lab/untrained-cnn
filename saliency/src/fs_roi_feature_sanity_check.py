"""
Independent per-feature (per-ROI) sanity check for fs_roi -- a genuinely
different question from calculate_gradcam_tabular.py's arch_/data_/input_
Adebayo et al. battery (does saliency survive randomization?): "how much
predictive signal does each SINGLE FreeSurfer ROI carry on its own?" Fits
one 1-feature RidgeCV regression per region (LogisticRegressionCV for the
classification target, sex) and reports its held-out R^2/RMSE (or
accuracy/ROC-AUC), via NESTED cross-validation -- an outer K-fold estimates
generalization, an inner K-fold (inside each outer TRAIN fold only) picks
the regularization strength, so the reported score is never optimistically
biased by having picked alpha/C on the same fold it's scored on.

This deliberately never touches the trained multi-feature checkpoint at
all -- every region is fit completely from scratch, ALONE, ignoring every
other region. A region with near-zero solo R^2 here contributes little
signal on its own, even if it happens to carry a large weight in the full
multi-feature model (large weight there can also come from collinearity
with other regions, or from noise the ridge penalty didn't fully shrink) --
this script and calculate_gradcam_tabular.py's saliency answer different
questions and are meant to be read side by side, not as duplicates.

Fitting recipe matches train_linear.py's train_ridge_head (StandardScaler
+ RidgeCV(alphas=np.logspace(-2, 5, 30)); LogisticRegressionCV for
classification) for consistency with the main model -- see that function
for the shared convention. fs_roi has no covariance-pooled features (its
cov_mask is all-False, see calculate_gradcam_tabular.py), so the
signed-sqrt step train_ridge_head applies to some other models' features
never applies here.

ALIGNMENT NOTE (read before trusting the cached .npy files directly):
train_linear.py's load_or_extract_features caches {cache_name}_
X_features_raw.npy/_subjects.npy (shared across every --target) separately
from {cache_name}_y_{target}.npy (one file per --target), with no
cross-check tying them together -- if the ROI feature table changes row
count between two different --target runs, an EARLIER target's cached y
file can silently go stale relative to the CURRENT X/_subjects.npy (this
repo's own cache currently has exactly this: X/_subjects.npy at 956 rows,
y_age.npy/y_sex.npy at 957, y_bmi.npy at 956 -- see docs/known_limitations.md
if this script's own README-adjacent docs get updated, and train_linear.py's
load_or_extract_features for the root cause). This script never reads
_y_{target}.npy at all -- it re-derives each of _subjects.npy's own
subjects' true target value fresh from the demographics file (the same
lookup_true_value calculate_gradcam_tabular.py already uses), which is
guaranteed consistent with X/_subjects.npy regardless of which target's
cache is currently stale.

Output (--output-dir, default saliency_outputs/fs_roi_feature_sanity_check/):
  fs_roi_per_feature_{target}.csv -- one row per ROI: region, r2_mean/std
    (or roc_auc_mean/std), rmse_mean/std (or accuracy_mean/std), n_subjects.
  fs_roi_per_feature_{target}.png -- two-panel horizontal bar chart (primary
    metric + secondary metric), one bar per region, sorted by the primary
    metric descending so both panels share the same top-to-bottom order.

Usage:
    python fs_roi_feature_sanity_check.py --target age
    python fs_roi_feature_sanity_check.py --target sex --top-n 15
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from calculate_gradcam_tabular import DEFAULT_CACHE_NAME, lookup_true_value
from targets import DEFAULT_PARTICIPANTS_TSV, get_target_spec
from train_linear import DEFAULT_DEMOGRAPHICS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_DIR = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "saliency_outputs" / "fs_roi_feature_sanity_check"

# Matches train_linear.py's train_ridge_head exactly (its --alt-alphas
# variant, np.logspace(0, 6, 30), isn't exposed here -- add if needed).
RIDGE_ALPHAS = np.logspace(-2, 5, 30)
LOGREG_CS = np.logspace(-5, 5, 30)


def load_features_and_labels(features_dir, cache_name, target_spec, demographics_path, participants_tsv):
    """
    X/subjects straight from the cache (always written together in one
    load_or_extract_features call, so guaranteed mutually aligned); y is
    NOT read from the cache at all -- see module docstring's ALIGNMENT NOTE
    for why -- it's re-derived per subject via the same lookup_true_value
    calculate_gradcam_tabular.py uses, so it's correct regardless of
    whether _y_{target}.npy happens to be stale relative to X/_subjects.npy.
    """
    X = np.load(features_dir / f"{cache_name}_X_features_raw.npy")
    subjects = np.load(features_dir / f"{cache_name}_subjects.npy", allow_pickle=True).astype(str).tolist()
    with open(features_dir / f"{cache_name}_feature_names.json") as f:
        feature_names = json.load(f)
    if X.shape[1] != len(feature_names):
        raise RuntimeError(f"X has {X.shape[1]} columns but feature_names.json has {len(feature_names)} entries")

    y, kept = [], []
    for i, subject_id in enumerate(subjects):
        true_value = lookup_true_value(subject_id, demographics_path, target_spec, participants_tsv)
        if true_value is None:
            continue
        y.append(true_value)
        kept.append(i)
    if len(kept) < len(subjects):
        print(
            f"Dropped {len(subjects) - len(kept)} subject(s) with no {target_spec.name} label in the "
            f"demographics file (re-derived fresh, not read from the cached _y_{target_spec.cache_suffix}.npy -- "
            "see module docstring)."
        )
    dtype = np.int64 if target_spec.task_type == "classification" else np.float64
    return X[kept], np.asarray(y, dtype=dtype), feature_names, [subjects[i] for i in kept]


def fit_one_feature(x, y, task_type, outer_splits=5, inner_splits=5, seed=0):
    """
    Nested CV for a single feature column `x` against `y`: an outer K-fold
    estimates generalization; inner RidgeCV/LogisticRegressionCV (refit
    inside each outer TRAIN fold only) selects the regularization strength.
    Never picks alpha/C on the same fold it's scored on -- that would
    optimistically bias the reported R^2/RMSE/accuracy/AUC.

    Returns {metric_mean, metric_std} across the outer folds for each
    metric ("r2"/"rmse" for regression, "accuracy"/"roc_auc" for
    classification).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    is_classification = task_type == "classification"
    outer_cv = (
        StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
        if is_classification
        else KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    )

    fold_metrics = []
    for train_idx, test_idx in outer_cv.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        if is_classification:
            n_per_class = int(np.bincount(y_train).min())
            reg = LogisticRegressionCV(Cs=LOGREG_CS, cv=min(inner_splits, n_per_class), max_iter=2000)
            reg.fit(x_train, y_train)
            accuracy = float(reg.score(x_test, y_test))
            roc_auc = float(roc_auc_score(y_test, reg.predict_proba(x_test)[:, 1])) if len(np.unique(y_test)) > 1 else float("nan")
            fold_metrics.append({"accuracy": accuracy, "roc_auc": roc_auc})
        else:
            reg = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_splits)
            reg.fit(x_train, y_train)
            pred = reg.predict(x_test)
            ss_res = float(np.sum((y_test - pred) ** 2))
            ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
            fold_metrics.append({"r2": r2, "rmse": rmse})

    keys = fold_metrics[0].keys()
    result = {}
    for k in keys:
        values = [m[k] for m in fold_metrics]
        result[f"{k}_mean"] = float(np.nanmean(values))
        result[f"{k}_std"] = float(np.nanstd(values))
    return result


def plot_per_feature_results(df, target_spec, output_path, top_n=None):
    """
    Two-panel horizontal bar chart, one row per region, sorted by the
    primary metric (R^2 for regression, ROC-AUC for classification)
    descending -- both panels share that same top-to-bottom order. Two
    separate panels, never a dual-axis single plot, since the primary/
    secondary metrics are different units/scales. Primary panel's bars are
    colored by the primary metric itself (viridis, sequential -- matches
    this repo's other magnitude figures, e.g. compute_saliency_correlations.py's
    METRIC_CMAP); the secondary panel uses one flat color since its scale
    isn't the same variable as the color mapping.
    """
    is_classification = target_spec.task_type == "classification"
    primary_col, secondary_col = ("roc_auc_mean", "accuracy_mean") if is_classification else ("r2_mean", "rmse_mean")
    primary_label, secondary_label = ("ROC-AUC", "Accuracy") if is_classification else ("R²", "RMSE")

    df = df.sort_values(primary_col, ascending=True).reset_index(drop=True)  # ascending: barh's last row plots on top
    if top_n is not None and len(df) > 2 * top_n:
        df = pd.concat([df.tail(top_n), df.head(top_n)])

    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.0, n * 0.16)), sharey=True)

    norm = plt.Normalize(vmin=df[primary_col].min(), vmax=df[primary_col].max())
    colors = plt.cm.viridis(norm(df[primary_col].to_numpy()))
    axes[0].barh(df["region"], df[primary_col], color=colors)
    axes[0].set_xlabel(primary_label)
    fit_desc = "LogisticRegressionCV" if is_classification else "RidgeCV"
    axes[0].set_title(f"{primary_label} (nested CV, single-feature {fit_desc})", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=6 if n > 40 else 8)
    if not is_classification:
        axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)

    axes[1].barh(df["region"], df[secondary_col], color="#7570b3")
    axes[1].set_xlabel(secondary_label)
    axes[1].set_title(f"{secondary_label} (same fold order)", fontsize=10)

    fig.suptitle(
        f"fs_roi per-feature sanity check ({target_spec.name}): how much does EACH single ROI predict on its own?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--cache-name", default=DEFAULT_CACHE_NAME)
    parser.add_argument("--target", choices=["age", "sex", "bmi"], default="age")
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--participants-tsv", type=Path, default=DEFAULT_PARTICIPANTS_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-splits", type=int, default=5, help="Outer CV folds (generalization estimate).")
    parser.add_argument("--inner-splits", type=int, default=5, help="Inner CV folds (alpha/C selection).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Plot only the best/worst N regions by the primary metric (default: all). The CSV always has every region.",
    )
    args = parser.parse_args()

    target_spec = get_target_spec(args.target)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, subjects = load_features_and_labels(
        args.features_dir, args.cache_name, target_spec, args.demographics, args.participants_tsv
    )
    print(f"{len(subjects)} subjects x {len(feature_names)} ROI features, target={target_spec.name}")

    primary_key = "roc_auc_mean" if target_spec.task_type == "classification" else "r2_mean"
    rows = []
    for i, region in enumerate(feature_names):
        metrics = fit_one_feature(X[:, i], y, target_spec.task_type, args.outer_splits, args.inner_splits, args.seed)
        rows.append({"region": region, **metrics, "n_subjects": len(subjects)})
        print(f"  [{i + 1}/{len(feature_names)}] {region}: {primary_key}={metrics[primary_key]:.4f}")

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / f"fs_roi_per_feature_{target_spec.name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    plot_per_feature_results(
        df, target_spec, args.output_dir / f"fs_roi_per_feature_{target_spec.name}.png", top_n=args.top_n
    )


if __name__ == "__main__":
    main()
