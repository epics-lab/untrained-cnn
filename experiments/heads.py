"""Linear head of the main benchmark, shared by the benchmark and the head timing.

Training fold StandardScaler followed by a linear model with an unpenalized bias:
  age, BMI: Ridge, closed form, minimizing sum of squared errors + alpha * ||w||^2
  sex:      LogisticRegression, L-BFGS, minimizing C * sum log loss + 0.5 * ||w||^2
alpha or C is selected by inner fivefold cross validation on the training set,
lowest mean squared error or highest accuracy, then the head is refit on the
whole training set.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5
ALPHA_GRID = np.logspace(-2, 5, 30)
C_GRID = np.logspace(-5, 5, 30)
LBFGS_MAX_ITER = 2000


def make_folds(y, task, n_splits=N_OUTER_FOLDS):
    """StratifiedKFold for sex, KFold for age and BMI, shuffled with seed 0."""
    splitter = (StratifiedKFold if task == "sex" else KFold)(n_splits=n_splits, shuffle=True, random_state=0)
    return list(splitter.split(np.zeros(len(y)), y))


def make_head(task, lam=1.0):
    if task == "sex":
        return make_pipeline(StandardScaler(), LogisticRegression(C=lam, solver="lbfgs", max_iter=LBFGS_MAX_ITER))
    return make_pipeline(StandardScaler(), Ridge(alpha=lam, solver="cholesky"))


def predict_head(model, X):
    """Logit for sex, so out >= 0 gives class 1; prediction for age and BMI."""
    X = X.astype(np.float64)
    return model.decision_function(X) if isinstance(model[-1], LogisticRegression) else model.predict(X)


def select_and_fit(X_tr, y_tr, task):
    """Inner selection over the grid, then refit at the selected value.
    Returns (model, selected alpha or C, number of fits that raised ConvergenceWarning)."""
    is_clf = task == "sex"
    key = "logisticregression__C" if is_clf else "ridge__alpha"
    gs = GridSearchCV(make_head(task), {key: C_GRID if is_clf else ALPHA_GRID},
                      cv=make_folds(y_tr, task, n_splits=N_INNER_FOLDS),
                      scoring="accuracy" if is_clf else "neg_mean_squared_error",
                      refit=True, error_score="raise")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        gs.fit(X_tr.astype(np.float64), y_tr)
    n_nonconv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return gs.best_estimator_, float(gs.best_params_[key]), n_nonconv


def fit_head(X, y, task, lam):
    """One fit at a fixed alpha or C. Returns (model, converged)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model = make_head(task, lam).fit(X.astype(np.float64), y)
    return model, not any(issubclass(x.category, ConvergenceWarning) for x in w)
