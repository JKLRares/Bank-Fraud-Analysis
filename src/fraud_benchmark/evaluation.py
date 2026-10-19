"""Metrics and validation-only operating-point selection, including tied scores."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

NO_ALERT_THRESHOLD = float(np.nextafter(1.0, 2.0))


def validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0 <= threshold <= NO_ALERT_THRESHOLD:
        raise ValueError("threshold must be in [0, 1] (or the no-alert sentinel just above 1)")


def checked_arrays(y, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) == 0 or len(y) != len(scores):
        raise ValueError(
            "Labels and scores must be nonempty one-dimensional arrays of equal length"
        )
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Scores must be finite values in [0, 1]")
    return y.astype(int), scores


def metrics(y, scores, threshold: float = 0.5, fn_cost: float = 25, fp_cost: float = 1) -> dict:
    validate_threshold(threshold)
    if not np.isfinite([fn_cost, fp_cost]).all() or min(fn_cost, fp_cost) <= 0:
        raise ValueError("Illustrative costs must be positive and finite")
    y, scores = checked_arrays(y, scores)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "fraud_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
        "alert_count": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "illustrative_cost_per_transaction": float((fn * fn_cost + fp * fp_cost) / len(y)),
    }


def select_thresholds(
    y,
    scores,
    fn_cost: float = 25,
    fp_cost: float = 1,
    alert_budget: float = 0.05,
    recall_target: float = 0.8,
) -> dict[str, float]:
    """Exact observed boundaries; >= includes entire tie groups, never a partial group."""
    y, scores = checked_arrays(y, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection needs both classes")
    if not np.isfinite([fn_cost, fp_cost]).all() or min(fn_cost, fp_cost) <= 0:
        raise ValueError("Illustrative costs must be positive and finite")
    if not 0 <= alert_budget <= 1 or not 0 < recall_target <= 1:
        raise ValueError("alert_budget must be in [0, 1]; recall_target must be in (0, 1]")
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores) != 0), len(y) - 1]
    thresholds = np.r_[NO_ALERT_THRESHOLD, ranked_scores[ends]]
    alerts = np.r_[0, ends + 1]
    tp = np.r_[0, np.cumsum(ranked_y)[ends]]
    fp = alerts - tp
    fn = y.sum() - tp
    precision = np.divide(tp, alerts, out=np.zeros(len(tp), dtype=float), where=alerts > 0)
    recall = tp / y.sum()
    f1 = np.divide(2 * tp, 2 * tp + fp + fn)
    # Descending thresholds provide conservative deterministic tie breaking.
    choices = {
        "f1": np.argmax(f1),
        "illustrative_cost": np.argmin(fn * fn_cost + fp * fp_cost),
        "alert_budget": np.argmax(np.where(alerts / len(y) <= alert_budget, tp, -1)),
        "high_recall": np.argmax(np.where(recall >= recall_target, precision, -1)),
    }
    return {name: float(thresholds[index]) for name, index in choices.items()}
