"""
Model training and evaluation module for heart disease prediction.

Provides Logistic Regression, Random Forest, SVM, and XGBoost classifiers
with optional class-weighting to address class imbalance, along with
standard evaluation metrics.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models(
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return a dictionary of configured classifiers.

    Args:
        class_weight: Pass ``"balanced"`` to enable class-weighting in
            sklearn estimators.  ``None`` uses uniform weights.
            Note: XGBoost class balancing is controlled separately via
            :func:`compute_scale_pos_weight` / ``scale_pos_weight``.
        random_state: Random seed used by each estimator.

    Returns:
        Mapping of human-readable name → unfitted sklearn-compatible estimator.
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight=class_weight,
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM": SVC(
            class_weight=class_weight,
            kernel="rbf",
            probability=True,
            random_state=random_state,
            C=1.0,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
    }
    return models


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute the ``scale_pos_weight`` parameter for XGBoost class balancing.

    The recommended value is ``(# negative samples) / (# positive samples)``.

    Args:
        y_train: Binary label array for the training set.

    Returns:
        Float ratio used as ``scale_pos_weight`` in :class:`xgboost.XGBClassifier`.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    return neg / pos if pos > 0 else 1.0


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    model_name: str = "",
) -> Dict[str, Any]:
    """Evaluate a trained classifier on a test set.

    Computes accuracy, recall (TPR), F1-score, AUC-ROC, false negative rate
    (FNR), true negative rate (TNR/specificity), and precision.

    Args:
        model: A fitted sklearn-compatible estimator.
        X_test: Test feature matrix.
        y_test: Ground-truth binary labels.
        model_name: Optional label used in log messages.

    Returns:
        Dictionary mapping metric name → numeric value, plus the raw counts
        ``n_test`` and ``n_positive``.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
    }
    return metrics


def train_and_evaluate_all(
    models: Dict[str, Any],
    X_train: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    xgb_scale_pos_weight: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate every model in *models*.

    Args:
        models: Mapping of model name → unfitted estimator (from
            :func:`get_models`).
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels (numpy array).
        y_test: Test labels (numpy array).
        xgb_scale_pos_weight: When provided, overrides the XGBoost
            ``scale_pos_weight`` parameter for class-imbalance handling.

    Returns:
        Nested dictionary: ``{model_name: {metric_name: value, "model": fitted_model}}``.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        if name == "XGBoost" and xgb_scale_pos_weight is not None:
            model.set_params(scale_pos_weight=xgb_scale_pos_weight)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        metrics["model"] = model
        results[name] = metrics

    return results
