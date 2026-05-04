"""
Preprocessing module for CDC BRFSS 2024 heart disease prediction.

Handles data loading, feature selection, cleaning, encoding, and
age-group stratification for the BRFSS 2024 dataset.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Target variable columns
# ---------------------------------------------------------------------------
TARGET_COLS = ["CVDINFR4", "CVDCRHD4"]  # Heart attack or coronary heart disease

# Age group column in BRFSS 2024
AGE_COL = "_AGEG5YR"

# ---------------------------------------------------------------------------
# Key features selected for heart disease prediction
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "_AGEG5YR",   # Age group (5-year intervals, 1-13)
    "SEX",        # Sex (1=Male, 2=Female)
    "_BMI5",      # Body Mass Index (scaled by 100)
    "TOLDHI3",    # Ever told high cholesterol (1=Yes, 2=No)
    "BPHIGH6",    # Ever told high blood pressure (1=Yes, 2=No)
    "DIABETE4",   # Diabetes status (1=Yes, 2=gestational, 3=No, 4=pre-diabetes)
    "SMOKER3",    # Four-level smoker status (1-4)
    "EXERANY2",   # Exercise in past 30 days (1=Yes, 2=No)
    "GENHLTH",    # General health (1=Excellent to 5=Poor)
    "MENTHLTH",   # Mental health days not good (0-30)
    "PHYSHLTH",   # Physical health days not good (0-30)
    "DIFFWALK",   # Difficulty walking or climbing stairs (1=Yes, 2=No)
    "_RACEGR4",   # Race/ethnicity (1-5)
    "EDUCA",      # Education level (1-6)
    "INCOME3",    # Annual household income (1-11)
]

# ---------------------------------------------------------------------------
# Age group mappings
# ---------------------------------------------------------------------------
# Maps _AGEG5YR codes to 5-year-interval labels
AGE_GROUP_MAP: dict = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}

# Broad age categories used in fairness subgroup analysis
BROAD_AGE_GROUPS: dict = {
    "Young (18-39)": [1, 2, 3, 4],
    "Middle-aged (40-64)": [5, 6, 7, 8, 9],
    "Older adults (65+)": [10, 11, 12, 13],
}

# ---------------------------------------------------------------------------
# BRFSS missing / refused / unknown value codes
# ---------------------------------------------------------------------------
MISSING_CODES: dict = {
    "CVDINFR4": [7, 9],
    "CVDCRHD4": [7, 9],
    "_AGEG5YR": [14],
    "TOLDHI3": [7, 9],
    "BPHIGH6": [7, 9],
    "DIABETE4": [7, 9],
    "SMOKER3": [9],
    "EXERANY2": [7, 9],
    "GENHLTH": [7, 9],
    "DIFFWALK": [7, 9],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_brfss_data(filepath: str) -> pd.DataFrame:
    """Load BRFSS 2024 data from an XPT (SAS transport) file or CSV.

    Args:
        filepath: Path to a ``.XPT`` / ``.xpt`` SAS transport file or a
            ``.csv`` file exported from the BRFSS dataset.

    Returns:
        Raw DataFrame with all BRFSS columns preserved.
    """
    if filepath.lower().endswith(".xpt"):
        df = pd.read_sas(filepath, format="xport", encoding="latin1")
    else:
        df = pd.read_csv(filepath, low_memory=False)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary ``HeartDisease`` target variable.

    A respondent is labelled positive (1) if they were ever diagnosed with
    a heart attack (``CVDINFR4 == 1``) **or** coronary heart disease
    (``CVDCRHD4 == 1``).  Rows where *both* target columns are
    missing/refused are dropped.

    Args:
        df: Raw BRFSS DataFrame containing ``CVDINFR4`` and ``CVDCRHD4``.

    Returns:
        DataFrame with an additional ``HeartDisease`` column (0/1).
    """
    df = df.copy()

    heart_attack = df["CVDINFR4"].map({1: 1, 2: 0})  # 1=Yes, 2=No
    chd = df["CVDCRHD4"].map({1: 1, 2: 0})           # 1=Yes, 2=No

    df["HeartDisease"] = ((heart_attack == 1) | (chd == 1)).astype(int)

    # Drop rows where we cannot determine the label at all
    valid_mask = heart_attack.notna() | chd.notna()
    df = df[valid_mask].copy()

    return df


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable age group labels based on ``_AGEG5YR``.

    Adds two columns:
    - ``AgeGroup5yr``: five-year interval label (e.g. ``"18-24"``)
    - ``AgeBroadGroup``: three-way grouping used for fairness auditing

    Args:
        df: DataFrame that contains ``_AGEG5YR``.

    Returns:
        DataFrame with the two new columns appended.
    """
    df = df.copy()

    df["AgeGroup5yr"] = df[AGE_COL].map(AGE_GROUP_MAP)

    age_to_broad: dict = {}
    for broad_label, codes in BROAD_AGE_GROUPS.items():
        for code in codes:
            age_to_broad[code] = broad_label

    df["AgeBroadGroup"] = df[AGE_COL].map(age_to_broad)

    return df


def encode_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Encode BRFSS feature columns into a numeric format suitable for ML.

    BRFSS variables use numeric codes; this function standardises binary
    yes/no variables to 0/1 and maps ``DIABETE4`` to a binary indicator.
    Ordinal variables (``GENHLTH``, ``SMOKER3``, etc.) are kept as-is.

    Args:
        df: DataFrame (after :func:`create_target_variable`).
        features: List of feature column names to encode.

    Returns:
        DataFrame with encoded columns.
    """
    df = df.copy()

    # Recode Yes=1, No=2  →  1/0
    binary_cols = ["TOLDHI3", "BPHIGH6", "EXERANY2", "DIFFWALK"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})

    # DIABETE4: 1=Yes, 2=gestational-only→0, 3=No→0, 4=pre-diabetes→0
    if "DIABETE4" in df.columns:
        df["DIABETE4"] = df["DIABETE4"].map({1: 1, 2: 0, 3: 0, 4: 0})

    return df


def select_available_features(
    df: pd.DataFrame, desired_features: List[str]
) -> List[str]:
    """Return the subset of *desired_features* that are present in *df*.

    Args:
        df: The DataFrame to check against.
        desired_features: Full list of desired feature column names.

    Returns:
        Filtered list containing only columns that exist in *df*.
    """
    return [f for f in desired_features if f in df.columns]


def clean_missing_values(
    df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Replace BRFSS missing/refused codes with ``NaN`` and drop affected rows.

    Args:
        df: DataFrame (after encoding).
        features: Feature columns to check for missingness.

    Returns:
        DataFrame with rows containing any missing values in *features*,
        the target, or the age column removed.
    """
    df = df.copy()

    for col, codes in MISSING_CODES.items():
        if col in df.columns:
            df[col] = df[col].replace(codes, np.nan)

    # Drop rows with unknown age category (code 14)
    df = df[df[AGE_COL] != 14]

    available_features = [f for f in features if f in df.columns]
    df = df.dropna(subset=available_features + ["HeartDisease", AGE_COL])

    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple:
    """Run the complete preprocessing pipeline.

    Steps performed in order:

    1. Create binary target variable (``HeartDisease``).
    2. Add age-group labels.
    3. Encode features.
    4. Drop rows with missing values.
    5. Stratified train/test split.
    6. Optional StandardScaler normalization.

    Args:
        df: Raw BRFSS DataFrame.
        features: Feature columns to use; defaults to :data:`FEATURE_COLS`.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
        scale: When ``True`` apply :class:`~sklearn.preprocessing.StandardScaler`.

    Returns:
        Tuple of
        ``(X_train, X_test, y_train, y_test, scaler, feature_names,
        age_train, age_test)``
        where ``age_train`` / ``age_test`` are :class:`pandas.Series` of
        broad age-group labels aligned with the split rows.
    """
    if features is None:
        features = FEATURE_COLS

    df = create_target_variable(df)
    df = add_age_groups(df)
    df = encode_features(df, features)

    available_features = select_available_features(df, features)
    df = clean_missing_values(df, available_features)

    X = df[available_features]
    y = df["HeartDisease"]
    age_groups = df["AgeBroadGroup"]

    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X, y, age_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_names = list(X.columns)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names, age_train, age_test
