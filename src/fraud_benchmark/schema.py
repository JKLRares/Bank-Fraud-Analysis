"""Validate the observable features shared by training and scoring."""

import numpy as np
import pandas as pd

NUMERIC = [
    "amount",
    "account_age_days",
    "hour",
    "transactions_last_24h",
    "distance_from_home_km",
    "foreign_country",
]
CATEGORICAL = ["device", "channel"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "fraud"
SCHEMA_VERSION = 1
BOUNDS = {
    "amount": (0, 1_000_000),
    "account_age_days": (0, 36_500),
    "hour": (0, 23),
    "transactions_last_24h": (0, 100_000),
    "distance_from_home_km": (0, 20_100),
    "foreign_country": (0, 1),
}
INTEGER_FEATURES = {"account_age_days", "hour", "transactions_last_24h", "foreign_country"}

# Time is an input/partitioning field, deliberately excluded from the fitted model.
CREDIT_MODEL_FEATURES = [*[f"V{i}" for i in range(1, 29)], "Amount"]
CREDIT_INPUTS = ["Time", *CREDIT_MODEL_FEATURES]


def validate_credit_features(frame: pd.DataFrame, allow_missing: bool = True) -> pd.DataFrame:
    """ULB input contract; PCA coordinates have no invented financial interpretation."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty credit-card transaction table")
    if not frame.columns.is_unique or set(frame.columns) != set(CREDIT_INPUTS):
        raise ValueError("Credit-card inputs require exactly Time, V1 through V28, and Amount")
    result = frame[CREDIT_INPUTS].copy()
    for name in CREDIT_INPUTS:
        values = result[name]
        if len(values.dropna()) and (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
        ):
            raise ValueError(f"{name}: expected real numbers or null")
        result[name] = values.astype(float)
    values = result.to_numpy()
    if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
        raise ValueError("Credit-card features must be finite; dataset cells cannot be missing")
    if (result[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative")
    return result


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Require exact columns. Missing cells are imputed later; unknown categories are allowed."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty transaction table")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Feature columns mismatch: missing={missing}, unexpected={extra}")
    result = frame.loc[:, FEATURES].copy()
    for name, (low, high) in BOUNDS.items():
        values = result[name]
        if any(isinstance(v, bool | np.bool_) for v in values.dropna()):
            raise ValueError(f"{name}: use numbers, not booleans")
        if any(not isinstance(v, int | float | np.integer | np.floating) for v in values.dropna()):
            raise ValueError(f"{name}: expected numeric values or null")
        values = values.astype(float)
        observed = values.dropna()
        if not np.isfinite(observed).all() or not observed.between(low, high).all():
            raise ValueError(f"{name}: expected finite values in [{low}, {high}]")
        if name in INTEGER_FEATURES and (observed % 1 != 0).any():
            raise ValueError(f"{name}: expected integer values")
        result[name] = values
    for name in CATEGORICAL:
        values = result[name]
        if any(not isinstance(v, str) or not v.strip() for v in values.dropna()):
            raise ValueError(f"{name}: expected nonempty strings or null")
        result[name] = values.where(values.notna(), np.nan).astype(object)
    return result


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {"transaction_id", *FEATURES, TARGET}
    if not frame.columns.is_unique or set(frame.columns) != expected:
        raise ValueError("Dataset requires transaction_id, the exact features, and fraud")
    ids = frame.transaction_id
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("transaction_id must be non-null and unique")
    y = frame[TARGET]
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("fraud must contain only binary labels 0 and 1")
    features = validate_features(frame[FEATURES])
    if features.duplicated().any():
        raise ValueError("Duplicate feature rows are not allowed across benchmark partitions")
    result = features.copy()
    result.insert(0, "transaction_id", ids)
    result[TARGET] = y.astype(int)
    return result
