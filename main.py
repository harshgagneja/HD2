"""
main.py — Fairness-focused heart disease prediction analysis using BRFSS 2024.

Usage
-----
Run with real BRFSS data::

    python main.py --data path/to/LLCP2024.XPT

Run with synthetic data (for testing / demonstration)::

    python main.py --synthetic --n-samples 50000

See ``python main.py --help`` for all options.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from src.fairness import (
    compute_fairness_gap,
    compute_fairness_report,
    plot_overall_vs_subgroup,
    plot_subgroup_metric,
    summarize_fairness_findings,
)
from src.models import (
    compute_scale_pos_weight,
    get_models,
    train_and_evaluate_all,
)
from src.preprocessing import BROAD_AGE_GROUPS, FEATURE_COLS, load_brfss_data, prepare_data
from src.utils import (
    generate_synthetic_brfss_data,
    print_age_distribution,
    print_class_distribution,
)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_analysis(df, output_dir: str = "results", random_state: int = 42):
    """Run the complete fairness-focused classification pipeline.

    Steps:
    1. Preprocess data (clean, encode, split, scale).
    2. Train and evaluate standard models (no class weighting).
    3. Compute age-group fairness metrics for standard models.
    4. Train and evaluate class-weighted models.
    5. Compute age-group fairness metrics for weighted models.
    6. Save result tables and visualizations to *output_dir*.

    Args:
        df: Raw BRFSS-format DataFrame.
        output_dir: Directory for CSV/PNG output files.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(results_std, results_weighted, fairness_std, fairness_weighted)``.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BRFSS 2024 Heart Disease Fairness Analysis")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    print("\n[1/5] Preprocessing data …")
    (
        X_train, X_test,
        y_train, y_test,
        scaler, feature_names,
        age_train, age_test,
    ) = prepare_data(df, features=FEATURE_COLS, random_state=random_state)

    print(f"  Training samples : {len(y_train):,}")
    print(f"  Test samples     : {len(y_test):,}")
    print(f"  Features used    : {len(feature_names)}")
    print_class_distribution(y_train, "  Train")
    print_class_distribution(y_test, "  Test")
    print_age_distribution(age_test, "  Test")

    spw = compute_scale_pos_weight(y_train.values)
    print(f"\n  Class imbalance ratio (neg/pos): {spw:.2f}")

    # ------------------------------------------------------------------
    # Step 2: Standard models (no class weighting)
    # ------------------------------------------------------------------
    print("\n[2/5] Training standard models (no class weighting) …")
    models_std = get_models(class_weight=None, random_state=random_state)
    results_std = train_and_evaluate_all(
        models_std, X_train, X_test, y_train.values, y_test.values,
    )

    _print_results_table("Standard", results_std)

    # ------------------------------------------------------------------
    # Step 3: Fairness analysis — standard models
    # ------------------------------------------------------------------
    print("\n[3/5] Computing subgroup fairness metrics (standard models) …")
    fairness_std = compute_fairness_report(
        results_std, X_test, y_test, age_test,
    )
    print(summarize_fairness_findings(results_std, fairness_std))

    gaps_std = compute_fairness_gap(fairness_std)
    print("\nFairness Gaps (Standard Models):")
    print(gaps_std[["Model", "Metric", "Min", "Max", "Gap"]].to_string(index=False))
    gaps_std.to_csv(os.path.join(output_dir, "fairness_gaps_standard.csv"), index=False)

    # ------------------------------------------------------------------
    # Step 4: Class-weighted models
    # ------------------------------------------------------------------
    print("\n[4/5] Training class-weighted models (balanced class weights) …")
    models_weighted = get_models(class_weight="balanced", random_state=random_state)
    results_weighted = train_and_evaluate_all(
        models_weighted, X_train, X_test, y_train.values, y_test.values,
        xgb_scale_pos_weight=spw,
    )

    _print_results_table("Weighted", results_weighted)

    fairness_weighted = compute_fairness_report(
        results_weighted, X_test, y_test, age_test,
    )
    print(summarize_fairness_findings(results_weighted, fairness_weighted))

    gaps_weighted = compute_fairness_gap(fairness_weighted)
    gaps_weighted.to_csv(
        os.path.join(output_dir, "fairness_gaps_weighted.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Step 5: Visualizations
    # ------------------------------------------------------------------
    print("\n[5/5] Generating visualizations …")

    plot_subgroup_metric(
        fairness_std,
        metric="False Negative Rate",
        title="False Negative Rate by Age Group (Standard Models)",
        output_path=os.path.join(output_dir, "fnr_by_age_standard.png"),
    )
    plt.close("all")

    plot_subgroup_metric(
        fairness_weighted,
        metric="False Negative Rate",
        title="False Negative Rate by Age Group (Weighted Models)",
        output_path=os.path.join(output_dir, "fnr_by_age_weighted.png"),
    )
    plt.close("all")

    plot_subgroup_metric(
        fairness_std,
        metric="Recall (TPR)",
        title="Recall by Age Group (Standard Models)",
        output_path=os.path.join(output_dir, "recall_by_age_standard.png"),
    )
    plt.close("all")

    plot_overall_vs_subgroup(
        results_std,
        fairness_std,
        metric="False Negative Rate",
        age_group="Young (18-39)",
        output_path=os.path.join(output_dir, "overall_vs_young_fnr.png"),
    )
    plt.close("all")

    print(f"\n  Results saved to: {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        print(f"    {fname}")

    return results_std, results_weighted, fairness_std, fairness_weighted


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_results_table(label: str, results: dict) -> None:
    """Print a formatted performance table to stdout."""
    print(f"\nOverall Performance ({label}):")
    header = f"{'Model':<25} {'Accuracy':>9} {'Recall':>9} {'F1':>9} {'AUC-ROC':>9} {'FNR':>9}"
    print(header)
    print("-" * 65)
    for name, res in results.items():
        print(
            f"{name:<25} {res['accuracy']:>9.3f} {res['recall']:>9.3f} "
            f"{res['f1']:>9.3f} {res['auc_roc']:>9.3f} "
            f"{res['false_negative_rate']:>9.3f}"
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fairness-focused heart disease prediction analysis (BRFSS 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to BRFSS 2024 data file (.XPT or .csv)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic BRFSS-like data (no real data required)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Number of synthetic records to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output CSV and PNG files",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.synthetic:
        print(f"Generating {args.n_samples:,} synthetic BRFSS-like records …")
        df = generate_synthetic_brfss_data(
            n_samples=args.n_samples, random_state=args.random_state
        )
    elif args.data:
        print(f"Loading BRFSS data from: {args.data}")
        df = load_brfss_data(args.data)
        print(f"Loaded {len(df):,} records with {df.shape[1]} variables")
    else:
        print("No data source specified.  Running with 10,000 synthetic records.")
        df = generate_synthetic_brfss_data(n_samples=10000, random_state=args.random_state)

    run_analysis(df, output_dir=args.output_dir, random_state=args.random_state)


if __name__ == "__main__":
    main()
