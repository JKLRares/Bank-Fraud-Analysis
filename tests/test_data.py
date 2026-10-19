import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.data import generate_transactions, missingness_stress, split_transactions
from fraud_benchmark.schema import FEATURES, validate_dataset, validate_features


def test_generation_is_deterministic_and_does_not_mutate_global_rng():
    np.random.seed(9)
    state = np.random.get_state()
    first = generate_transactions(2000, 2026)
    second = generate_transactions(2000, 2026)
    pd.testing.assert_frame_equal(first, second)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(state, after, strict=True))
    assert not first.equals(generate_transactions(2000, 2027))
    assert set(first.columns) == {"transaction_id", "fraud", *FEATURES}
    assert 0.01 < first.fraud.mean() < 0.15
    # Even similar amounts have both outcomes; the original deterministic amount boundary is absent.
    assert first.groupby(pd.qcut(first.amount, 4), observed=True).fraud.nunique().eq(2).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 99},
        {"rows": True},
        {"seed": -1},
        {"seed": 2**32},
        {"scenario": "bank"},
    ],
)
def test_invalid_generation_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(**kwargs)


def test_split_is_disjoint_complete_stratified_and_deterministic(dataset):
    parts = split_transactions(dataset)
    again = split_transactions(dataset)
    assert [len(parts.train), len(parts.validation), len(parts.test)] == [1200, 400, 400]
    ids = [set(part.transaction_id) for part in (parts.train, parts.validation, parts.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(dataset.transaction_id)
    for part, repeat in zip(
        (parts.train, parts.validation, parts.test),
        (again.train, again.validation, again.test),
        strict=True,
    ):
        pd.testing.assert_frame_equal(part, repeat)
        assert abs(part.fraud.mean() - dataset.fraud.mean()) < 0.005


def test_duplicates_and_missing_labels_rejected_before_split(dataset):
    duplicated = dataset.copy()
    duplicated.loc[1, FEATURES] = duplicated.loc[0, FEATURES].to_numpy()
    with pytest.raises(ValueError, match="Duplicate feature"):
        split_transactions(duplicated)
    duplicated = dataset.copy()
    duplicated.loc[1, "transaction_id"] = duplicated.loc[0, "transaction_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(duplicated)
    with pytest.raises(ValueError, match="binary"):
        validate_dataset(dataset.assign(fraud=np.nan))
    with pytest.raises(ValueError, match="at least 10"):
        split_transactions(dataset.assign(fraud=0))


@pytest.mark.parametrize(
    "column,value",
    [
        ("amount", -1),
        ("amount", np.inf),
        ("amount", "100"),
        ("amount", True),
        ("amount", 1 + 2j),
        ("hour", 24),
        ("hour", 2.5),
        ("foreign_country", 2),
        ("device", ""),
        ("channel", 5),
        ("device", ["mobile"]),
    ],
)
def test_invalid_feature_values(dataset, column, value):
    frame = dataset[FEATURES].head(2).astype(object)
    frame.at[0, column] = value
    with pytest.raises(ValueError):
        validate_features(frame)


def test_missing_values_and_unseen_categories_are_supported(dataset):
    frame = dataset[FEATURES].head(2).copy()
    frame.loc[0, :] = np.nan
    frame.loc[1, "device"] = "unknown-device"
    result = validate_features(frame[FEATURES[::-1]])
    assert result.columns.tolist() == FEATURES
    assert result.iloc[0].isna().all()
    assert result.loc[1, "device"] == "unknown-device"
    assert dataset[FEATURES].notna().all().all()


def test_schema_requires_exact_columns(dataset):
    for frame in [
        dataset[FEATURES].drop(columns="amount"),
        dataset,
        dataset[FEATURES].assign(fraud=0),
        dataset[FEATURES].iloc[:0],
        pd.concat([dataset[FEATURES], dataset[["amount"]]], axis=1),
    ]:
        with pytest.raises(ValueError):
            validate_features(frame)


def test_missingness_stress_preserves_outcomes_and_identity(dataset):
    perturbed = missingness_stress(dataset, 100)
    assert perturbed[FEATURES].isna().sum().sum() > 0
    pd.testing.assert_series_equal(dataset.fraud, perturbed.fraud)
    pd.testing.assert_series_equal(dataset.transaction_id, perturbed.transaction_id)
    pd.testing.assert_frame_equal(perturbed, missingness_stress(dataset, 100))
