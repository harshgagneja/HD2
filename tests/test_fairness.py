"""Unit tests for the fairness analysis module (src/fairness.py)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    compute_fairness_gap,
    compute_fairness_report,
    compute_subgroup_metrics,
    plot_subgroup_metric,
    summarize_fairness_findings,
)
from src.models import get_models, train_and_evaluate_all
from src.preprocessing import FEATURE_COLS, prepare_data
from src.utils import generate_synthetic_brfss_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analysis_data():
    """Minimal trained pipeline for fairness tests."""
    df = generate_synthetic_brfss_data(n_samples=2000, random_state=42)
    X_train, X_test, y_train, y_test, scaler, feature_names, age_train, age_test = prepare_data(
        df, features=FEATURE_COLS, test_size=0.2, random_state=42
    )
    fast_models = {"Logistic Regression": get_models()["Logistic Regression"]}
    results = train_and_evaluate_all(
        fast_models, X_train, X_test, y_train.values, y_test.values
    )
    return results, X_test, y_test, age_test


# ---------------------------------------------------------------------------
# compute_subgroup_metrics
# ---------------------------------------------------------------------------

def test_compute_subgroup_metrics_columns(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    lr = results["Logistic Regression"]["model"]
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    df_metrics = compute_subgroup_metrics(
        y_test.reset_index(drop=True),
        y_pred,
        y_proba,
        age_test.reset_index(drop=True),
    )

    for col in ["Recall (TPR)", "False Negative Rate", "F1 Score", "AUC-ROC"]:
        assert col in df_metrics.columns


def test_compute_subgroup_metrics_at_least_two_groups(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    lr = results["Logistic Regression"]["model"]
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    df_metrics = compute_subgroup_metrics(
        y_test.reset_index(drop=True),
        y_pred,
        y_proba,
        age_test.reset_index(drop=True),
    )
    assert len(df_metrics) >= 2


def test_compute_subgroup_metrics_fnr_range(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    lr = results["Logistic Regression"]["model"]
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    df_metrics = compute_subgroup_metrics(
        y_test.reset_index(drop=True),
        y_pred,
        y_proba,
        age_test.reset_index(drop=True),
    )
    assert (df_metrics["False Negative Rate"] >= 0.0).all()
    assert (df_metrics["False Negative Rate"] <= 1.0).all()


# ---------------------------------------------------------------------------
# compute_fairness_report
# ---------------------------------------------------------------------------

def test_compute_fairness_report_returns_dict(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    assert isinstance(fairness, dict)
    assert "Logistic Regression" in fairness


def test_compute_fairness_report_dataframe(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    df = fairness["Logistic Regression"]
    assert isinstance(df, pd.DataFrame)
    assert "False Negative Rate" in df.columns


# ---------------------------------------------------------------------------
# compute_fairness_gap
# ---------------------------------------------------------------------------

def test_compute_fairness_gap_non_negative(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    gaps = compute_fairness_gap(fairness)
    assert isinstance(gaps, pd.DataFrame)
    assert (gaps["Gap"] >= 0).all()


def test_compute_fairness_gap_columns(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    gaps = compute_fairness_gap(fairness)
    for col in ["Model", "Metric", "Min", "Max", "Gap"]:
        assert col in gaps.columns


# ---------------------------------------------------------------------------
# plot_subgroup_metric
# ---------------------------------------------------------------------------

def test_plot_subgroup_metric_returns_figure(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    fig = plot_subgroup_metric(fairness, metric="False Negative Rate")
    assert fig is not None


# ---------------------------------------------------------------------------
# summarize_fairness_findings
# ---------------------------------------------------------------------------

def test_summarize_fairness_findings_content(analysis_data):
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    summary = summarize_fairness_findings(results, fairness)
    assert isinstance(summary, str)
    assert "FAIRNESS ANALYSIS SUMMARY" in summary
    assert "Logistic Regression" in summary
    assert "FNR" in summary


def test_summarize_fairness_findings_fnr_gap(analysis_data):
    """Summary should include FNR Gap information."""
    results, X_test, y_test, age_test = analysis_data
    fairness = compute_fairness_report(results, X_test, y_test, age_test)
    summary = summarize_fairness_findings(results, fairness)
    assert "FNR Gap" in summary
