import numpy as np
import pytest

from fraud_benchmark.evaluation import NO_ALERT_THRESHOLD, metrics, select_thresholds


def test_metrics_against_hand_calculation():
    result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["prevalence"] == result["alert_rate"] == 0.5
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["roc_auc"] == 0.75
    assert result["illustrative_cost_per_transaction"] == 6.5


def test_degenerate_metrics_have_explicit_null_auc_and_zero_division_policy():
    result = metrics([0, 0], [0.1, 0.2])
    assert result["roc_auc"] is None and result["average_precision"] is None
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert metrics([1, 1], [0.1, 0.2])["average_precision"] == 1


@pytest.mark.parametrize(
    "y,scores",
    [
        ([], []),
        ([0, 1], [0.2]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [np.nan, 0.2]),
        ([0, 1], [1.1, 0]),
        ([[0, 1]], [[0.1, 0.2]]),
    ],
)
def test_invalid_metric_arrays(y, scores):
    with pytest.raises(ValueError):
        metrics(y, scores)


@pytest.mark.parametrize(
    "scores",
    [
        [0.1, 0.4, 0.4, 0.8, 0.9, 1.0],
        [0.5] * 6,
        [1.0] * 6,
        [0.0] * 6,
    ],
)
def test_threshold_selection_matches_exhaustive_search_and_respects_ties(scores):
    y = [0, 1, 0, 1, 1, 0]
    choices = select_thresholds(y, scores, alert_budget=0.34, recall_target=0.8)
    boundaries = sorted({NO_ALERT_THRESHOLD, *scores}, reverse=True)
    all_metrics = [metrics(y, scores, threshold) for threshold in boundaries]
    chosen = {name: metrics(y, scores, threshold) for name, threshold in choices.items()}
    assert chosen["f1"]["f1"] == max(m["f1"] for m in all_metrics)
    assert chosen["illustrative_cost"]["illustrative_cost_per_transaction"] == min(
        m["illustrative_cost_per_transaction"] for m in all_metrics
    )
    assert chosen["alert_budget"]["alert_rate"] <= 0.34
    assert chosen["alert_budget"]["recall"] == max(
        m["recall"] for m in all_metrics if m["alert_rate"] <= 0.34
    )
    assert chosen["high_recall"]["recall"] >= 0.8
    assert chosen["high_recall"]["precision"] == max(
        m["precision"] for m in all_metrics if m["recall"] >= 0.8
    )
    assert select_thresholds(y, scores, alert_budget=0)["alert_budget"] == NO_ALERT_THRESHOLD


def test_threshold_equality_is_flagged_and_no_alert_sentinel_works():
    assert metrics([0, 1], [0.5, 1.0], 0.5)["alert_count"] == 2
    assert metrics([0, 1], [0.5, 1.0], NO_ALERT_THRESHOLD)["alert_count"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fn_cost": 0},
        {"fp_cost": np.inf},
        {"alert_budget": -0.1},
        {"recall_target": 1.1},
        {"recall_target": 0},
    ],
)
def test_invalid_policy_configuration(kwargs):
    with pytest.raises(ValueError):
        select_thresholds([0, 1], [0.1, 0.9], **kwargs)
