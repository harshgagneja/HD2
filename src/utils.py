"""
Utility functions for the BRFSS heart disease fairness analysis.

Includes a synthetic BRFSS-like data generator used for unit testing and
offline demonstrations when the real dataset is not available.
"""

from typing import Optional

import numpy as np
import pandas as pd


def generate_synthetic_brfss_data(
    n_samples: int = 10000,
    random_state: int = 42,
    positive_rate: float = 0.12,
) -> pd.DataFrame:
    """Generate synthetic BRFSS 2024-like data for testing.

    The generator mimics the column structure and approximate variable
    distributions of the CDC BRFSS 2024 dataset, including realistic
    age-related disparities in heart disease prevalence.

    Args:
        n_samples: Number of synthetic respondents to generate.
        random_state: Random seed for reproducibility.
        positive_rate: Approximate overall heart disease prevalence rate.

    Returns:
        DataFrame with the same column names as the real BRFSS dataset
        expected by :mod:`src.preprocessing`.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Age group distribution (roughly matches national BRFSS sample)
    # _AGEG5YR codes 1-13 correspond to 18-24 … 80+
    # ------------------------------------------------------------------
    age_probs = [
        0.07, 0.07, 0.08, 0.08,   # 18-39
        0.08, 0.08, 0.09, 0.09, 0.09,  # 40-64
        0.09, 0.07, 0.05, 0.06,   # 65+
    ]
    age_groups = rng.choice(range(1, 14), size=n_samples, p=age_probs)

    # Normalized age factor used to scale age-dependent risk probabilities
    age_factor = (age_groups - 1) / 12.0  # 0.0 (youngest) → 1.0 (oldest)

    # ------------------------------------------------------------------
    # Demographic variables
    # ------------------------------------------------------------------
    sex = rng.choice([1, 2], size=n_samples, p=[0.48, 0.52])  # 1=Male, 2=Female

    # BMI stored as integer (BRFSS multiplies by 100)
    bmi = rng.normal(2750, 600, size=n_samples).clip(1500, 7000).astype(int)

    # ------------------------------------------------------------------
    # Clinical risk factors (probabilities increase with age)
    # ------------------------------------------------------------------
    bp_prob = np.clip(0.20 + 0.60 * age_factor, 0.0, 1.0)
    bphigh6 = np.where(rng.binomial(1, bp_prob), 1, 2)  # 1=Yes, 2=No

    chol_prob = np.clip(0.25 + 0.40 * age_factor, 0.0, 1.0)
    toldhi3 = np.where(rng.binomial(1, chol_prob), 1, 2)  # 1=Yes, 2=No

    diab_prob = np.clip(0.07 + 0.25 * age_factor, 0.0, 1.0)
    diabete4 = np.where(rng.binomial(1, diab_prob), 1, 3)  # 1=Yes, 3=No

    smoker3 = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.12, 0.10, 0.22, 0.56])

    exerany2 = np.where(rng.binomial(1, 0.75, size=n_samples), 1, 2)  # 1=Yes, 2=No

    # General health: 1=Excellent … 5=Poor (worse with age)
    genhlth_mean = np.clip(1.5 + 2.0 * age_factor, 1, 5)
    genhlth = rng.integers(1, 6, size=n_samples)

    menthlth = rng.integers(0, 31, size=n_samples)
    physhlth = rng.integers(0, 31, size=n_samples)

    diffwalk_prob = np.clip(0.05 + 0.40 * age_factor, 0.0, 1.0)
    diffwalk = np.where(rng.binomial(1, diffwalk_prob), 1, 2)  # 1=Yes, 2=No

    # ------------------------------------------------------------------
    # Socioeconomic variables
    # ------------------------------------------------------------------
    racegr4 = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.63, 0.12, 0.16, 0.05, 0.04])
    educa = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.01, 0.04, 0.08, 0.25, 0.30, 0.32])
    income3 = rng.choice(range(1, 12), size=n_samples)

    # ------------------------------------------------------------------
    # Heart disease target
    # Probability increases with age, hypertension, diabetes, cholesterol,
    # smoking, and male sex.
    # ------------------------------------------------------------------
    risk_score = (
        0.30 * age_factor
        + 0.15 * (bphigh6 == 1).astype(float)
        + 0.10 * (toldhi3 == 1).astype(float)
        + 0.10 * (diabete4 == 1).astype(float)
        + 0.05 * (smoker3 <= 2).astype(float)
        + 0.05 * (sex == 1).astype(float)
    )
    heart_prob = np.clip(positive_rate + risk_score * (1.0 - positive_rate), 0.01, 0.90)

    # Split probability between the two BRFSS heart disease items
    cvdinfr4 = np.where(rng.binomial(1, heart_prob * 0.60), 1, 2)  # Heart attack
    cvdcrhd4 = np.where(rng.binomial(1, heart_prob * 0.70), 1, 2)  # CHD

    return pd.DataFrame(
        {
            "_AGEG5YR": age_groups,
            "SEX": sex,
            "_BMI5": bmi,
            "BPHIGH6": bphigh6,
            "TOLDHI3": toldhi3,
            "DIABETE4": diabete4,
            "SMOKER3": smoker3,
            "EXERANY2": exerany2,
            "GENHLTH": genhlth,
            "MENTHLTH": menthlth,
            "PHYSHLTH": physhlth,
            "DIFFWALK": diffwalk,
            "_RACEGR4": racegr4,
            "EDUCA": educa,
            "INCOME3": income3,
            "CVDINFR4": cvdinfr4,
            "CVDCRHD4": cvdcrhd4,
        }
    )


def print_class_distribution(y: pd.Series, label: str = "") -> None:
    """Print class counts and proportions for a binary label series.

    Args:
        y: Binary label series.
        label: Optional prefix string for the output line.
    """
    counts = y.value_counts().sort_index()
    total = len(y)
    prefix = f"{label}: " if label else ""
    print(f"{prefix}Class distribution:")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt:,} ({cnt / total * 100:.1f}%)")


def print_age_distribution(age_groups: pd.Series, label: str = "") -> None:
    """Print age-group counts and proportions.

    Args:
        age_groups: Series of broad age-group label strings.
        label: Optional prefix string for the output line.
    """
    counts = age_groups.value_counts().sort_index()
    total = len(age_groups)
    prefix = f"{label}: " if label else ""
    print(f"{prefix}Age group distribution:")
    for group, cnt in counts.items():
        print(f"  {group}: {cnt:,} ({cnt / total * 100:.1f}%)")
