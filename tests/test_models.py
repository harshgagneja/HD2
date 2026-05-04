"""Unit tests for the models module (src/models.py)."""

import numpy as np
import pytest

from src.models import (
    compute_scale_pos_weight,
    evaluate_model,
    get_models,
    train_and_evaluate_all,
)
from src.preprocessing import FEATURE_COLS, prepare_data
from src.utils import generate_synthetic_brfss_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prepared_data():
    """Small prepared dataset shared across all model tests."""
    df = generate_synthetic_brfss_data(n_samples=2000, random_state=42)
    return prepare_data(df, features=FEATURE_COLS, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# get_models
# ---------------------------------------------------------------------------

def test_get_models_standard_keys():
    models = get_models(class_weight=None)
    for name in ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]:
        assert name in models


def test_get_models_balanced_class_weight():
    models = get_models(class_weight="balanced")
    assert models["Logistic Regression"].class_weight == "balanced"
    assert models["Random Forest"].class_weight == "balanced"
    assert models["SVM"].class_weight == "balanced"


def test_get_models_returns_new_instances():
    models_a = get_models()
    models_b = get_models()
    assert models_a["Logistic Regression"] is not models_b["Logistic Regression"]


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------

def test_compute_scale_pos_weight_basic():
    y = np.array([0, 0, 0, 1])
    spw = compute_scale_pos_weight(y)
    assert abs(spw - 3.0) < 1e-9


def test_compute_scale_pos_weight_equal_classes():
    y = np.array([0, 1, 0, 1])
    spw = compute_scale_pos_weight(y)
    assert abs(spw - 1.0) < 1e-9


def test_compute_scale_pos_weight_no_positives():
    y = np.array([0, 0, 0])
    spw = compute_scale_pos_weight(y)
    assert spw == 1.0


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_metric_range(prepared_data):
    X_train, X_test, y_train, y_test, scaler, feature_names, age_train, age_test = prepared_data
    lr = get_models()["Logistic Regression"]
    lr.fit(X_train, y_train.values)
    metrics = evaluate_model(lr, X_test, y_test.values)

    for key in ["accuracy", "recall", "f1", "auc_roc", "false_negative_rate"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of [0, 1]"


def test_evaluate_model_counts(prepared_data):
    X_train, X_test, y_train, y_test, *_ = prepared_data
    lr = get_models()["Logistic Regression"]
    lr.fit(X_train, y_train.values)
    metrics = evaluate_model(lr, X_test, y_test.values)

    assert metrics["n_test"] == len(y_test)
    assert metrics["n_positive"] == int(y_test.sum())


# ---------------------------------------------------------------------------
# train_and_evaluate_all
# ---------------------------------------------------------------------------

def test_train_and_evaluate_all_keys(prepared_data):
    X_train, X_test, y_train, y_test, *_ = prepared_data
    fast_models = {"Logistic Regression": get_models()["Logistic Regression"]}
    results = train_and_evaluate_all(
        fast_models, X_train, X_test, y_train.values, y_test.values
    )
    assert "Logistic Regression" in results
    assert "model" in results["Logistic Regression"]


def test_train_and_evaluate_all_metric_range(prepared_data):
    X_train, X_test, y_train, y_test, *_ = prepared_data
    fast_models = {"Logistic Regression": get_models()["Logistic Regression"]}
    results = train_and_evaluate_all(
        fast_models, X_train, X_test, y_train.values, y_test.values
    )
    res = results["Logistic Regression"]
    for key in ["accuracy", "recall", "f1", "auc_roc", "false_negative_rate"]:
        assert 0.0 <= res[key] <= 1.0, f"{key} out of [0, 1]"


def test_train_and_evaluate_xgb_scale_pos_weight(prepared_data):
    """XGBoost should accept scale_pos_weight without raising."""
    X_train, X_test, y_train, y_test, *_ = prepared_data
    spw = compute_scale_pos_weight(y_train.values)
    xgb_models = {"XGBoost": get_models()["XGBoost"]}
    results = train_and_evaluate_all(
        xgb_models, X_train, X_test, y_train.values, y_test.values,
        xgb_scale_pos_weight=spw,
    )
    assert "XGBoost" in results
    assert 0.0 <= results["XGBoost"]["auc_roc"] <= 1.0
