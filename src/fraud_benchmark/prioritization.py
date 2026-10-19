"""Label-free capacity allocation and explicitly retrospective capture metrics."""

import heapq

import numpy as np

from fraud_benchmark.evaluation import checked_arrays


def checked_scores(scores, row_ids=None):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or scores.dtype.kind not in "fiu":
        raise ValueError("scores must be a nonempty numeric vector")
    scores = scores.astype(float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("scores must be finite and in [0, 1]")
    ids = np.arange(len(scores)) if row_ids is None else np.asarray(row_ids)
    if ids.ndim != 1 or len(ids) != len(scores) or ids.dtype.kind not in "iu":
        raise ValueError("row_ids must be an equal-length vector of unique nonnegative integers")
    if (ids < 0).any() or len(set(map(int, ids))) != len(ids):
        raise ValueError("row_ids must be unique nonnegative integers")
    return scores, ids


def prioritize(scores, capacity: int, row_ids=None) -> list[dict]:
    """Highest score first; equal scores use ascending source row ID. No labels are accepted."""
    scores, ids = checked_scores(scores, row_ids)
    if isinstance(capacity, bool) or not isinstance(capacity, int | np.integer):
        raise ValueError("capacity must be an integer")
    if not 0 <= capacity <= len(scores):
        raise ValueError("capacity must be between zero and the number of transactions")
    # Heap core: O(n log k) time/O(k) space; vector and ID validation also uses O(n) space.
    chosen = heapq.nsmallest(capacity, range(len(scores)), key=lambda i: (-scores[i], ids[i]))
    return [
        {"rank": rank, "row": int(ids[index]), "fraud_score": float(scores[index])}
        for rank, index in enumerate(chosen, start=1)
    ]


def capacity_metrics(
    y, scores, capacity: int, row_ids=None, fn_cost: float = 100, fp_cost: float = 1
) -> dict:
    y, scores = checked_arrays(y, scores)
    scores, ids = checked_scores(scores, row_ids)
    if not np.isfinite([fn_cost, fp_cost]).all() or min(fn_cost, fp_cost) <= 0:
        raise ValueError("Illustrative costs must be positive and finite")
    selected = prioritize(scores, capacity, ids)
    by_id = dict(zip(ids, y, strict=True))
    tp = sum(int(by_id[item["row"]]) for item in selected)
    fp, positives = capacity - tp, int(y.sum())
    return {
        "reviewed": int(capacity),
        "review_rate": float(capacity / len(y)),
        "frauds_captured": tp,
        "false_positives": int(fp),
        "frauds_missed": positives - tp,
        "precision_at_k": float(tp / capacity) if capacity else 0.0,
        "recall_at_k": float(tp / positives) if positives else 0.0,
        "fraud_capture_rate": float(tp / positives) if positives else 0.0,
        "fraud_count": positives,
        "prevalence": float(y.mean()),
        "illustrative_cost_per_transaction": float(
            ((positives - tp) * fn_cost + fp * fp_cost) / len(y)
        ),
    }


def capacity_for_rate(rows: int, rate: float) -> int:
    if isinstance(rows, bool) or not isinstance(rows, int | np.integer) or rows < 0:
        raise ValueError("rows must be a nonnegative integer")
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Review rate must be in [0, 1]")
    return min(rows, int(np.ceil(rows * rate)))
