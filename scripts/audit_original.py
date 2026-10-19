"""Reproduce original metrics without replacing any historical files."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from fraud_benchmark.evaluation import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-legacy", action="store_true", help="Also run all 3 scripts in a temp copy"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = ("all_data", "train", "test")
    frames = {name: pd.read_csv(root / f"{name}.csv") for name in names}
    train, test = frames["train"], frames["test"]
    expected_train, expected_test = train_test_split(
        frames["all_data"], test_size=0.25, random_state=12345
    )
    x_train = pd.get_dummies(train.drop(columns=["transaction_id", "fraud"]), drop_first=True)
    x_test = pd.get_dummies(test.drop(columns=["transaction_id", "fraud"]), drop_first=True)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    with threadpool_limits(limits=1):
        model = RandomForestClassifier(random_state=1234).fit(x_train, train.fraud)
        forest = metrics(test.fraud, model.predict_proba(x_test)[:, 1])
    report = {
        "source": "Original repository generate_data.py; independent synthetic fields",
        "labels": "Bernoulli(0.85 if amount > 2250 else 0.03)",
        "data": {
            name: {
                "rows": len(frame),
                "fraud_count": int(frame.fraud.sum()),
                "prevalence": float(frame.fraud.mean()),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_features": int(
                    frame.drop(columns=["transaction_id", "fraud"]).duplicated().sum()
                ),
                "sha256": hashlib.sha256((root / f"{name}.csv").read_bytes()).hexdigest(),
            }
            for name, frame in frames.items()
        },
        "split_matches_script": train.equals(expected_train.reset_index(drop=True))
        and test.equals(expected_test.reset_index(drop=True)),
        "shared_user_ids": len(set(train.user_id) & set(test.user_id)),
        "shared_transaction_ids": len(set(train.transaction_id) & set(test.transaction_id)),
        "dummy_always_legitimate": metrics(test.fraud, np.zeros(len(test))),
        "original_random_forest": forest,
        "generator_rule": metrics(test.fraud, np.where(test.amount > 2250, 0.85, 0.03)),
    }
    if args.run_legacy:
        runs = []
        with tempfile.TemporaryDirectory(prefix="fraud-original-") as temporary:
            directory = Path(temporary)
            for name in (
                "generate_data.py",
                "split_data.py",
                "EDA.py",
                "all_data.csv",
                "train.csv",
                "test.csv",
            ):
                shutil.copy2(root / name, directory / name)
            env = dict(
                os.environ,
                MPLBACKEND="Agg",
                MPLCONFIGDIR=str(directory / "mpl"),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            )
            # Evaluate saved data first; generation would replace it inside the temporary copy.
            for script in ("EDA.py", "generate_data.py", "split_data.py"):
                start = time.monotonic()
                completed = subprocess.run(
                    [sys.executable, script],
                    cwd=directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                runs.append(
                    {
                        "script": script,
                        "exit_code": completed.returncode,
                        "seconds": round(time.monotonic() - start, 2),
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
            regenerated = pd.read_csv(directory / "all_data.csv")
            report["regenerated_matches_except_dates"] = (
                frames["all_data"]
                .drop(columns="date_time")
                .equals(regenerated.drop(columns="date_time"))
            )
        report["legacy_runs"] = runs
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
