"""Offline installed-package smoke test; generated rows are software fixtures, not ML evidence."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.cli import main


def smoke(output: Path):
    output.mkdir()
    rng = np.random.default_rng(52)
    frame = pd.DataFrame(rng.normal(size=(600, 28)), columns=[f"V{i}" for i in range(1, 29)])
    frame["Time"] = np.arange(len(frame))
    frame["Amount"] = rng.uniform(0, 1000, len(frame))
    frame["Class"] = (np.arange(len(frame)) % 17 == 0).astype(int)
    data = output / "fixture.csv"
    frame.to_csv(data, index=False)
    sample = output / "request.json"
    frame.drop(columns="Class").head(5).to_json(sample, orient="records")
    commands = [
        [
            "train",
            "--data",
            str(data),
            "--fixture",
            "--output",
            str(output / "trained"),
            "--bootstrap-repeats",
            "20",
        ],
        [
            "evaluate",
            "--data",
            str(data),
            "--run",
            str(output / "trained"),
            "--output",
            str(output / "evaluated"),
        ],
        [
            "predict",
            "--model",
            str(output / "trained/model.joblib"),
            "--input",
            str(sample),
            "--output",
            str(output / "scores.json"),
        ],
        [
            "prioritize",
            "--input",
            str(output / "scores.json"),
            "--capacity",
            "2",
            "--output",
            str(output / "alerts.json"),
        ],
    ]
    for command in commands:
        if main(command) != 0:
            raise SystemExit(f"Smoke command failed: {command[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    smoke(parser.parse_args().output)
