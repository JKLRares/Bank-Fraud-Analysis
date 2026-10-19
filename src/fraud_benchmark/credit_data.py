"""Local ULB CSV audit and duplicate-protected temporal/group-random partitions."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_benchmark.data import Partitions
from fraud_benchmark.schema import CREDIT_INPUTS, CREDIT_MODEL_FEATURES, validate_credit_features

DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CreditData:
    frame: pd.DataFrame
    audit: dict


def load_creditcard(path: Path, fixture: bool = False) -> CreditData:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Obtain creditcard.csv from {DATASET_URL} "
            "under its current access terms; place it in data/raw/."
        )
    raw = pd.read_csv(path)
    if not raw.columns.is_unique or set(raw.columns) != {*CREDIT_INPUTS, "Class"}:
        raise ValueError("Expected the original ULB schema: Time, V1..V28, Amount, Class")
    if raw.Class.isna().any() or not raw.Class.isin([0, 1]).all():
        raise ValueError("Class must be non-null binary labels 0/1")
    features = validate_credit_features(raw[CREDIT_INPUTS], allow_missing=False)
    raw = features.assign(Class=raw.Class.astype(int))
    if fixture:
        if len(raw) > 5000:
            raise ValueError("Software fixtures are limited to 5000 rows")
    elif len(raw) != 284807 or int(raw.Class.sum()) != 492:
        raise ValueError(
            "Input does not match published ULB counts (284807 rows, 492 positives). "
            "--fixture is reserved for small software tests, never effectiveness evidence."
        )
    clean = raw.drop_duplicates(keep="first").copy()
    # Group on actual model inputs, not Time, label, or invented account identifiers.
    groups = clean.groupby(CREDIT_MODEL_FEATURES, sort=False, dropna=False)
    if groups.Class.nunique().gt(1).any():
        raise ValueError(
            "Identical model inputs have conflicting labels; resolve before evaluation"
        )
    clean["_group"] = groups.ngroup()
    audit = {
        "source": "software_fixture" if fixture else DATASET_URL,
        "evidence_kind": "software_fixture" if fixture else "historical_anonymized_benchmark",
        "raw_sha256": sha256_file(path),
        "raw_rows": len(raw),
        "raw_fraud_count": int(raw.Class.sum()),
        "raw_prevalence": float(raw.Class.mean()),
        "missing_cells": 0,
        "time_nondecreasing_in_file": bool(raw.Time.is_monotonic_increasing),
        "time_range_seconds": [float(raw.Time.min()), float(raw.Time.max())],
        "exact_duplicates_removed": len(raw) - len(clean),
        "deduplicated_rows": len(clean),
        "deduplicated_fraud_count": int(clean.Class.sum()),
        "unique_model_input_groups": int(clean._group.nunique()),
        "repeated_model_inputs_after_row_dedup": int(clean.duplicated(CREDIT_MODEL_FEATURES).sum()),
        "identity_note": "Published counts/schema checked; upstream authenticity not verified.",
    }
    return CreditData(clean, audit)


def split_creditcard(frame: pd.DataFrame, protocol: str = "chronological", seed: int = 2026):
    """No equal timestamp straddles temporal boundaries; no model-input group crosses any split."""
    if protocol == "chronological":
        low, high = frame.Time.quantile([0.6, 0.8])
        train = frame[frame.Time <= low]
        raw_validation = frame[(frame.Time > low) & (frame.Time <= high)]
        raw_test = frame[frame.Time > high]
        validation = raw_validation[~raw_validation._group.isin(train._group)]
        seen = pd.concat([train._group, raw_validation._group])
        test = raw_test[~raw_test._group.isin(seen)]
        details = {
            "time_cutoffs_seconds": [float(low), float(high)],
            "purged_validation_rows": len(raw_validation) - len(validation),
            "purged_test_rows": len(raw_test) - len(test),
            "purged_validation_frauds": int(raw_validation.Class.sum() - validation.Class.sum()),
            "purged_test_frauds": int(raw_test.Class.sum() - test.Class.sum()),
        }
    elif protocol == "group_random":
        groups = frame.groupby("_group", sort=False).Class.first()
        development, test_groups = train_test_split(
            groups, test_size=0.2, random_state=seed, stratify=groups
        )
        train_groups, validation_groups = train_test_split(
            development, test_size=0.25, random_state=seed + 1, stratify=development
        )
        train = frame[frame._group.isin(train_groups.index)]
        validation = frame[frame._group.isin(validation_groups.index)]
        test = frame[frame._group.isin(test_groups.index)]
        details = {"group_stratification": "Class of each unique model-input signature"}
    else:
        raise ValueError("protocol must be chronological or group_random")
    parts = Partitions(
        *(p.sort_values("Time", kind="stable").copy() for p in (train, validation, test))
    )
    group_sets = [set(p._group) for p in (parts.train, parts.validation, parts.test)]
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Model-input groups overlap partitions")
    for name, part in (
        ("train", parts.train),
        ("validation", parts.validation),
        ("test", parts.test),
    ):
        if part.Class.value_counts().reindex([0, 1], fill_value=0).min() < 2:
            raise ValueError(f"{name} needs at least two examples of each class; protocol unusable")
    details["partitions"] = {
        name: {
            "rows": len(part),
            "fraud_count": int(part.Class.sum()),
            "prevalence": float(part.Class.mean()),
            "groups": int(part._group.nunique()),
            "time_range_seconds": [float(part.Time.min()), float(part.Time.max())],
        }
        for name, part in (
            ("train", parts.train),
            ("validation", parts.validation),
            ("test", parts.test),
        )
    }
    return parts, details


def membership(parts: Partitions) -> dict:
    return {
        name: part.index.astype(int).tolist()
        for name, part in (
            ("train", parts.train),
            ("validation", parts.validation),
            ("test", parts.test),
        )
    }
