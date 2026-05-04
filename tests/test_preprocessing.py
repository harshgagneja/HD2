"""Unit tests for the preprocessing module (src/preprocessing.py)."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    AGE_GROUP_MAP,
    BROAD_AGE_GROUPS,
    FEATURE_COLS,
    add_age_groups,
    clean_missing_values,
    create_target_variable,
    encode_features,
    prepare_data,
    select_available_features,
)
from src.utils import generate_synthetic_brfss_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Small synthetic BRFSS dataframe (2,000 rows)."""
    return generate_synthetic_brfss_data(n_samples=2000, random_state=0)


# ---------------------------------------------------------------------------
# create_target_variable
# ---------------------------------------------------------------------------

def test_create_target_variable_adds_column(synthetic_df):
    df = create_target_variable(synthetic_df)
    assert "HeartDisease" in df.columns


def test_create_target_variable_binary(synthetic_df):
    df = create_target_variable(synthetic_df)
    assert set(df["HeartDisease"].unique()).issubset({0, 1})


def test_create_target_variable_nonempty(synthetic_df):
    df = create_target_variable(synthetic_df)
    assert len(df) > 0


def test_create_target_variable_or_logic():
    """A row positive on either CVDINFR4 or CVDCRHD4 should be labelled 1."""
    row = {"CVDINFR4": 1, "CVDCRHD4": 2}  # heart attack only
    df = pd.DataFrame([row])
    result = create_target_variable(df)
    assert result["HeartDisease"].iloc[0] == 1

    row2 = {"CVDINFR4": 2, "CVDCRHD4": 1}  # CHD only
    df2 = pd.DataFrame([row2])
    result2 = create_target_variable(df2)
    assert result2["HeartDisease"].iloc[0] == 1

    row3 = {"CVDINFR4": 2, "CVDCRHD4": 2}  # neither
    df3 = pd.DataFrame([row3])
    result3 = create_target_variable(df3)
    assert result3["HeartDisease"].iloc[0] == 0


# ---------------------------------------------------------------------------
# add_age_groups
# ---------------------------------------------------------------------------

def test_add_age_groups_columns(synthetic_df):
    df = create_target_variable(synthetic_df)
    df = add_age_groups(df)
    assert "AgeGroup5yr" in df.columns
    assert "AgeBroadGroup" in df.columns


def test_add_age_groups_valid_5yr(synthetic_df):
    df = create_target_variable(synthetic_df)
    df = add_age_groups(df)
    valid_labels = set(AGE_GROUP_MAP.values())
    assigned = set(df["AgeGroup5yr"].dropna())
    assert assigned.issubset(valid_labels)


def test_add_age_groups_valid_broad(synthetic_df):
    df = create_target_variable(synthetic_df)
    df = add_age_groups(df)
    valid_broad = set(BROAD_AGE_GROUPS.keys())
    assigned_broad = set(df["AgeBroadGroup"].dropna())
    assert assigned_broad.issubset(valid_broad)


# ---------------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------------

def test_encode_features_binary_range(synthetic_df):
    df = create_target_variable(synthetic_df)
    df = encode_features(df, FEATURE_COLS)
    binary_cols = ["TOLDHI3", "BPHIGH6", "EXERANY2", "DIFFWALK", "DIABETE4"]
    for col in binary_cols:
        if col in df.columns:
            vals = set(df[col].dropna().unique())
            assert vals.issubset({0, 1, 0.0, 1.0}), f"{col} contains unexpected values: {vals}"


# ---------------------------------------------------------------------------
# clean_missing_values
# ---------------------------------------------------------------------------

def test_clean_missing_values_no_nans(synthetic_df):
    df = create_target_variable(synthetic_df)
    df = add_age_groups(df)
    df = encode_features(df, FEATURE_COLS)
    available = select_available_features(df, FEATURE_COLS)
    df_clean = clean_missing_values(df, available)
    assert df_clean["HeartDisease"].isna().sum() == 0
    assert df_clean["_AGEG5YR"].isna().sum() == 0


def test_clean_missing_values_drops_rows(synthetic_df):
    df = create_target_variable(synthetic_df)
    df = add_age_groups(df)
    df = encode_features(df, FEATURE_COLS)
    available = select_available_features(df, FEATURE_COLS)
    df_clean = clean_missing_values(df, available)
    # Cleaned dataset must have fewer or equal rows
    assert len(df_clean) <= len(df)


# ---------------------------------------------------------------------------
# select_available_features
# ---------------------------------------------------------------------------

def test_select_available_features_subset():
    df = pd.DataFrame({"A": [1], "B": [2]})
    result = select_available_features(df, ["A", "C"])
    assert result == ["A"]


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

def test_prepare_data_shapes(synthetic_df):
    X_train, X_test, y_train, y_test, scaler, feature_names, age_train, age_test = prepare_data(
        synthetic_df, features=FEATURE_COLS, test_size=0.2, random_state=42
    )
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == len(feature_names)


def test_prepare_data_binary_target(synthetic_df):
    X_train, X_test, y_train, y_test, *_ = prepare_data(
        synthetic_df, features=FEATURE_COLS, random_state=42
    )
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_prepare_data_valid_age_groups(synthetic_df):
    *_, age_train, age_test = prepare_data(
        synthetic_df, features=FEATURE_COLS, random_state=42
    )
    valid_broad = set(BROAD_AGE_GROUPS.keys())
    assert set(age_test.unique()).issubset(valid_broad)


def test_prepare_data_scaler_applied(synthetic_df):
    X_train, *_ = prepare_data(synthetic_df, features=FEATURE_COLS, scale=True)
    # StandardScaler should bring feature means close to 0
    assert abs(float(X_train.mean())) < 1.0


def test_prepare_data_no_scale(synthetic_df):
    X_train, X_test, y_train, y_test, scaler, *_ = prepare_data(
        synthetic_df, features=FEATURE_COLS, scale=False
    )
    assert scaler is None
