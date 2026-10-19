"""Verify chronological row conservation against an existing frozen run; never train a model."""

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_benchmark.credit_data import load_creditcard, membership, split_creditcard
from fraud_benchmark.credit_experiment import verify_training


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": len(frame),
        "frauds": int(frame.Class.sum()),
        "genuine": int(frame.Class.eq(0).sum()),
    }


def verify(data_path: Path, run: Path) -> dict:
    training = verify_training(run)
    if training["config"]["protocol"] != "chronological":
        raise ValueError("This accounting check requires a chronological run")
    fixture = training["data_audit"]["evidence_kind"] == "software_fixture"
    data = load_creditcard(data_path, fixture=fixture)
    if data.audit != training["data_audit"]:
        raise ValueError("Dataset audit differs from the frozen training run")
    frame = data.frame
    raw = pd.read_csv(data_path)
    duplicate_ids = raw.index[raw.duplicated(keep="first")]
    if set(raw.index) != set(frame.index) | set(duplicate_ids):
        raise ValueError("Duplicate removal does not account for the raw source rows")
    parts, details = split_creditcard(frame, seed=training["config"]["seed"])
    recorded = json.loads((run / "split_membership.json").read_text())
    if details != training["split_audit"] or membership(parts) != recorded:
        raise ValueError("Partitions differ from the frozen run")
    low, high = details["time_cutoffs_seconds"]
    windows = {
        "train": frame[frame.Time <= low],
        "validation": frame[(frame.Time > low) & (frame.Time <= high)],
        "test": frame[frame.Time > high],
    }
    # Independent check: retain a signature only in the time window of its first occurrence.
    first_time = frame.groupby("_group").Time.transform("min")
    eligible = {
        "train": frame.Time <= low,
        "validation": (frame.Time > low) & (frame.Time <= high) & (first_time > low),
        "test": (frame.Time > high) & (first_time > high),
    }
    stages, seen = {}, set()
    for name in ("train", "validation", "test"):
        kept = getattr(parts, name)
        if set(kept.index) != set(frame.index[eligible[name]]):
            raise ValueError("Temporal purge differs from the label-independent first-time rule")
        window = windows[name]
        removed = window.loc[~window.index.isin(kept.index)]
        if not removed._group.isin(seen).all():
            raise ValueError("Discarded signature lacks an earlier-partition witness")
        stages[name] = {
            "before_purge": counts(window),
            "purged": counts(removed),
            "retained": counts(kept),
        }
        seen.update(window._group)
    frames = [parts.train, parts.validation, parts.test]
    kept = pd.concat(frames)
    removed = frame.loc[~frame.index.isin(kept.index)]
    if not kept.index.is_unique or set(kept.index) & set(removed.index):
        raise ValueError("Retained and purged rows overlap")
    if set(kept.index) | set(removed.index) != set(frame.index):
        raise ValueError("Retained/purged membership fails row conservation")
    if set(kept._group) != set(frame._group):
        raise ValueError("A signature was discarded from every partition")
    return {
        "raw_sha256": data.audit["raw_sha256"],
        "raw": counts(raw),
        "exact_duplicates_removed": counts(raw.loc[duplicate_ids]),
        "deduplicated": counts(frame),
        "time_cutoffs_seconds": [low, high],
        "chronological_windows": stages,
        "additional_purge": counts(removed),
        "final_partitions_total": counts(kept),
        "frozen_membership_matches": True,
        "all_signatures_represented": True,
        "earlier_witnesses_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--run", type=Path, default=Path("artifacts/ulb-chronological-2026"))
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.run), indent=2, allow_nan=False))
