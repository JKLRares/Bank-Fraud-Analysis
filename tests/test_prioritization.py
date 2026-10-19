import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud_benchmark.diagnostics import calibration_assessment, grouped_bootstrap
from fraud_benchmark.prioritization import capacity_for_rate, capacity_metrics, prioritize


def test_top_k_is_label_free_and_ties_follow_source_row_id():
    result = prioritize([0.8, 0.9, 0.9, 0.1], 2, [50, 30, 10, 20])
    assert [item["row"] for item in result] == [10, 30]
    assert [item["rank"] for item in result] == [1, 2]
    assert prioritize([0.5, 0.5], 0) == []
    assert [v["row"] for v in prioritize([0.5, 0.5], 2)] == [0, 1]
    reordered = prioritize([0.9, 0.8, 0.1, 0.9], 2, [10, 50, 20, 30])
    assert reordered == result


def test_heap_ranking_matches_full_sort():
    rng = np.random.default_rng(12)
    scores = rng.choice([0, 0.2, 0.5, 1], 101)
    ids = rng.permutation(101)
    expected = np.lexsort((ids, -scores))
    for capacity in [0, 1, 7, 100, 101]:
        assert [v["row"] for v in prioritize(scores, capacity, ids)] == ids[
            expected[:capacity]
        ].tolist()


@pytest.mark.parametrize("capacity", [-1, 5, True, 1.0, float("nan"), "1"])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        prioritize([0.1, 0.2, 0.3, 0.4], capacity)


@pytest.mark.parametrize(
    "scores,ids",
    [
        ([], []),
        ([np.nan], [0]),
        ([np.inf], [0]),
        ([1.1], [0]),
        ([True], [0]),
        ([0.1, 0.2], [0, 0]),
        ([0.1], [-1]),
        ([0.1], [0.0]),
        ([0.1], [True]),
        ([[0.1]], [0]),
        ([0.1], [0, 1]),
        (["0.1"], [0]),
    ],
)
def test_invalid_scores_and_ids(scores, ids):
    with pytest.raises(ValueError):
        prioritize(scores, 0, ids)


def test_capacity_metrics_against_hand_calculation():
    result = capacity_metrics([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], 2)
    assert result["reviewed"] == 2
    assert result["frauds_captured"] == result["false_positives"] == 1
    assert result["precision_at_k"] == result["recall_at_k"] == result["fraud_capture_rate"] == 0.5
    assert result["illustrative_cost_per_transaction"] == 25.25
    assert capacity_metrics([0, 0], [0.1, 0.2], 0)["recall_at_k"] == 0
    assert capacity_for_rate(101, 0.01) == 2


def test_calibration_counts_bins_and_proper_scores():
    result = calibration_assessment([0, 0, 1, 1], [0, 0, 1, 1])
    assert result["brier_score"] == 0
    assert result["log_loss"] < 1e-10
    assert sum(b["rows"] for b in result["reliability_bins"]) == 4
    assert sum(b["frauds"] for b in result["reliability_bins"]) == 2
    assert result["mean_score"] == result["prevalence"] == 0.5


def test_grouped_bootstrap_is_deterministic_and_respects_perfect_ranking():
    y = np.array([0, 0, 1, 1] * 20)
    first = grouped_bootstrap(y, y * 0.8 + 0.1, np.arange(80), np.arange(80), 0.25, 0.5, 3, 20)
    second = grouped_bootstrap(y, y * 0.8 + 0.1, np.arange(80), np.arange(80), 0.25, 0.5, 3, 20)
    assert first == second
    assert first["valid_replicates"] == 20
    assert first["intervals"]["average_precision"] == [1, 1]
    assert first["intervals"]["precision_at_capacity"] == [1, 1]
    assert first["intervals"]["threshold_recall"] == [1, 1]


def test_group_bootstrap_weighted_capacity_matches_expanded_cluster_samples():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.9, 0.8, 0.8, 0.5, 0.1, 0.1, 0.0])
    groups = np.array([0, 0, 1, 1, 2, 3, 4, 5])
    rng = np.random.default_rng(9)
    ap, precisions, recalls = [], [], []
    for _ in range(20):
        counts = np.bincount(rng.integers(0, 6, 6), minlength=6)[groups]
        indices = np.repeat(np.arange(8), counts)
        if len(np.unique(y[indices])) != 2:
            continue
        k = capacity_for_rate(len(indices), 0.5)
        measured = capacity_metrics(y[indices], scores[indices], k)
        ap.append(average_precision_score(y[indices], scores[indices]))
        precisions.append(measured["precision_at_k"])
        recalls.append(measured["recall_at_k"])
    actual = grouped_bootstrap(y, scores, groups, np.arange(8), 0.5, 0.5, 9, 20)
    for name, values in [
        ("average_precision", ap),
        ("precision_at_capacity", precisions),
        ("recall_at_capacity", recalls),
    ]:
        np.testing.assert_allclose(actual["intervals"][name], np.quantile(values, [0.025, 0.975]))
