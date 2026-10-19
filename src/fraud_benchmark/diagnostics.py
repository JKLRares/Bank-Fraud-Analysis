"""Reliability diagnostics and conditional uncertainty for a frozen scoring model."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fraud_benchmark.evaluation import checked_arrays, validate_threshold
from fraud_benchmark.prioritization import capacity_for_rate, checked_scores

RELIABILITY_EDGES = np.array([0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])


def calibration_assessment(y, scores) -> dict:
    """Brier/log loss combine several effects; a reliability table also exposes bin support."""
    y, scores = checked_arrays(y, scores)
    bins = np.searchsorted(RELIABILITY_EDGES[1:-1], scores, side="right")
    table = []
    for index, (low, high) in enumerate(
        zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:], strict=True)
    ):
        chosen = bins == index
        if chosen.any():
            table.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "rows": int(chosen.sum()),
                    "frauds": int(y[chosen].sum()),
                    "mean_score": float(scores[chosen].mean()),
                    "observed_rate": float(y[chosen].mean()),
                }
            )
    return {
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "mean_score": float(scores.mean()),
        "prevalence": float(y.mean()),
        "reliability_bins": table,
    }


def grouped_bootstrap(
    y, scores, groups, row_ids, rate: float, threshold: float, seed: int, repeats: int = 200
) -> dict:
    """Resample identical-input groups, conditional on a frozen model and policies."""
    y, scores = checked_arrays(y, scores)
    scores, ids = checked_scores(scores, row_ids)
    validate_threshold(threshold)
    capacity_for_rate(len(y), rate)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("groups must align with observations")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 20 <= repeats <= 1000:
        raise ValueError("Bootstrap repeats must be an integer in [20, 1000]")
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max() + 1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((ids, -scores))
    samples = {
        name: []
        for name in [
            "average_precision",
            "roc_auc",
            "precision_at_capacity",
            "recall_at_capacity",
            "threshold_recall",
        ]
    }
    rejected = 0
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(0, count, count), minlength=count)
        weights = multiplicity[inverse]
        positives = float(np.dot(weights, y))
        rows = int(weights.sum())
        if positives == 0 or positives == rows:
            rejected += 1
            continue
        samples["average_precision"].append(
            average_precision_score(y, scores, sample_weight=weights)
        )
        samples["roc_auc"].append(roc_auc_score(y, scores, sample_weight=weights))
        k = capacity_for_rate(rows, rate)
        ranked_weights = weights[order]
        previous = np.cumsum(ranked_weights) - ranked_weights
        reviewed_weights = np.minimum(np.maximum(k - previous, 0), ranked_weights)
        captured = float(np.dot(reviewed_weights, y[order]))
        samples["precision_at_capacity"].append(captured / k if k else 0.0)
        samples["recall_at_capacity"].append(captured / positives)
        samples["threshold_recall"].append(
            float(np.dot(weights * (scores >= threshold), y)) / positives
        )
    return {
        "method": "95% percentile bootstrap of identical-model-input groups",
        "requested_replicates": repeats,
        "valid_replicates": repeats - rejected,
        "seed": seed,
        "intervals": {
            name: [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None
            for name, values in samples.items()
        },
        "scope": "Fixed model/policies; unknown card dependencies and future drift excluded",
    }
