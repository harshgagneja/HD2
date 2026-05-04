"""
Fairness analysis module for age-group subgroup auditing.

Computes per-age-group metrics (recall, false negative rate, F1, AUC-ROC),
fairness gaps across groups, and provides publication-ready visualizations.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score


# ---------------------------------------------------------------------------
# Subgroup metric computation
# ---------------------------------------------------------------------------

def compute_subgroup_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    age_groups: pd.Series,
) -> pd.DataFrame:
    """Compute per-age-group fairness metrics.

    Args:
        y_true: Ground-truth binary labels (aligned index with *age_groups*).
        y_pred: Predicted binary labels.
        y_proba: Predicted positive-class probabilities.
        age_groups: Series of broad age-group labels aligned with *y_true*.

    Returns:
        DataFrame indexed by age-group name with columns for each metric.
        Groups with zero positive samples are omitted.
    """
    rows = []

    for group in sorted(age_groups.unique()):
        mask = age_groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_proba[mask]

        if len(yt) == 0 or int(yt.sum()) == 0:
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        try:
            auc = float(roc_auc_score(yt, yprob))
        except ValueError:
            auc = float("nan")

        rows.append(
            {
                "AgeGroup": group,
                "N": len(yt),
                "N_Positive": int(yt.sum()),
                "Prevalence": float(yt.mean()),
                "Recall (TPR)": recall,
                "False Negative Rate": fnr,
                "F1 Score": f1,
                "AUC-ROC": auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            }
        )

    return pd.DataFrame(rows).set_index("AgeGroup")


def compute_fairness_report(
    trained_models: Dict[str, Dict],
    X_test,
    y_test: pd.Series,
    age_test: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute subgroup fairness metrics for all trained models.

    Args:
        trained_models: Output of
            :func:`~src.models.train_and_evaluate_all` — a dict mapping
            model name → result dict containing a ``"model"`` key.
        X_test: Test feature matrix.
        y_test: Ground-truth binary labels for the test set.
        age_test: Broad age-group labels aligned with *y_test*.
        feature_names: Optional; not currently used but reserved for future
            feature-importance–based fairness analysis.

    Returns:
        Dict mapping model name → :class:`pandas.DataFrame` of per-age-group
        metrics (output of :func:`compute_subgroup_metrics`).
    """
    fairness_results: Dict[str, pd.DataFrame] = {}

    for name, result in trained_models.items():
        model = result.get("model") if isinstance(result, dict) else result
        if model is None:
            continue

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        metrics_df = compute_subgroup_metrics(
            y_test.reset_index(drop=True),
            y_pred,
            y_proba,
            age_test.reset_index(drop=True),
        )
        fairness_results[name] = metrics_df

    return fairness_results


def compute_fairness_gap(
    fairness_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute the fairness gap: ``max − min`` of each metric across age groups.

    A larger gap indicates greater disparity between age groups.

    Args:
        fairness_results: Output of :func:`compute_fairness_report`.

    Returns:
        DataFrame with columns ``Model``, ``Metric``, ``Min``, ``Max``, ``Gap``.
    """
    rows = []
    for model_name, df in fairness_results.items():
        for metric in ["Recall (TPR)", "False Negative Rate", "F1 Score", "AUC-ROC"]:
            if metric not in df.columns:
                continue
            vals = df[metric].dropna()
            if len(vals) < 2:
                continue
            worst_group = (
                df["False Negative Rate"].idxmax()
                if metric == "False Negative Rate"
                else ""
            )
            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "Min": float(vals.min()),
                    "Max": float(vals.max()),
                    "Gap": float(vals.max() - vals.min()),
                    "Worst Group (Max FNR)": worst_group,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_subgroup_metric(
    fairness_results: Dict[str, pd.DataFrame],
    metric: str = "False Negative Rate",
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of *metric* across age groups for every model.

    Args:
        fairness_results: Output of :func:`compute_fairness_report`.
        metric: Column name to plot (must exist in every subgroup DataFrame).
        title: Optional chart title.
        figsize: Matplotlib figure size.
        output_path: When provided, save the figure to this path.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(fairness_results.keys())
    age_groups: Optional[List[str]] = None

    for df in fairness_results.values():
        if age_groups is None:
            age_groups = list(df.index)
        break

    if age_groups is None:
        return fig

    x = np.arange(len(age_groups))
    width = 0.8 / max(len(model_names), 1)

    for i, name in enumerate(model_names):
        df = fairness_results[name]
        vals = [
            float(df.loc[g, metric]) if g in df.index else 0.0
            for g in age_groups
        ]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name, alpha=0.8)

    ax.set_xlabel("Age Group")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Age Group and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(age_groups, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_overall_vs_subgroup(
    overall_results: Dict[str, Dict],
    fairness_results: Dict[str, pd.DataFrame],
    metric: str = "False Negative Rate",
    age_group: str = "Young (18-39)",
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare overall vs. subgroup metric for each model.

    Highlights the gap between aggregate performance and performance for a
    specific age group.

    Args:
        overall_results: Output of :func:`~src.models.train_and_evaluate_all`.
        fairness_results: Output of :func:`compute_fairness_report`.
        metric: Fairness metric to compare (must map to an overall metric key).
        age_group: Age-group label to compare against overall performance.
        figsize: Matplotlib figure size.
        output_path: When provided, save the figure to this path.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    _metric_map = {
        "Recall (TPR)": "recall",
        "F1 Score": "f1",
        "AUC-ROC": "auc_roc",
        "False Negative Rate": "false_negative_rate",
    }

    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(overall_results.keys())
    overall_key = _metric_map.get(metric, metric.lower())

    overall_vals = []
    subgroup_vals = []

    for name in model_names:
        overall_vals.append(float(overall_results[name].get(overall_key, 0)))
        sub_val = 0.0
        if name in fairness_results and age_group in fairness_results[name].index:
            sub_val = float(fairness_results[name].loc[age_group, metric])
        subgroup_vals.append(sub_val)

    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width / 2, overall_vals, width, label="Overall", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, subgroup_vals, width, label=age_group, color="coral", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(f"Overall vs. {age_group}: {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Textual summary
# ---------------------------------------------------------------------------

def summarize_fairness_findings(
    overall_results: Dict[str, Dict],
    fairness_results: Dict[str, pd.DataFrame],
) -> str:
    """Generate a plain-text summary of fairness findings.

    For each model, the summary reports:
    - Overall recall, FNR, and AUC-ROC.
    - Per-subgroup FNR values.
    - The worst-performing and best-performing age groups.
    - The fairness gap (max FNR − min FNR).

    Args:
        overall_results: Output of :func:`~src.models.train_and_evaluate_all`.
        fairness_results: Output of :func:`compute_fairness_report`.

    Returns:
        Multi-line string suitable for printing to stdout or a report file.
    """
    lines = ["=" * 60, "FAIRNESS ANALYSIS SUMMARY", "=" * 60]

    for model_name in overall_results:
        if model_name not in fairness_results:
            continue

        overall = overall_results[model_name]
        subgroup = fairness_results[model_name]

        lines.append(f"\n{model_name}:")
        lines.append(f"  Overall Recall:  {overall.get('recall', 0):.3f}")
        lines.append(f"  Overall FNR:     {overall.get('false_negative_rate', 0):.3f}")
        lines.append(f"  Overall AUC-ROC: {overall.get('auc_roc', 0):.3f}")

        if "False Negative Rate" in subgroup.columns:
            fnr_series = subgroup["False Negative Rate"]
            worst_group = fnr_series.idxmax()
            best_group = fnr_series.idxmin()
            lines.append("  Subgroup FNR:")
            for group, row in subgroup.iterrows():
                fnr = row.get("False Negative Rate", 0.0)
                lines.append(f"    {group}: {fnr:.3f}  (n={row['N']})")
            lines.append(
                f"  Worst group (highest FNR): {worst_group} ({fnr_series.max():.3f})"
            )
            lines.append(
                f"  Best  group (lowest  FNR): {best_group} ({fnr_series.min():.3f})"
            )
            lines.append(f"  FNR Gap: {fnr_series.max() - fnr_series.min():.3f}")

        lines.append("\n" + "=" * 60)

    return "\n".join(lines)
