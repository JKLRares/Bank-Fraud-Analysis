"""Trusted local model bundles and failure-safe output publication."""

import json
import os
import pickle
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from fraud_benchmark.evaluation import validate_threshold
from fraud_benchmark.schema import (
    CREDIT_INPUTS,
    FEATURES,
    SCHEMA_VERSION,
    validate_credit_features,
    validate_features,
)

DEPENDENCIES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib")


def runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in DEPENDENCIES}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    model_name: str
    threshold: float
    thresholds: dict[str, float]
    versions: dict[str, str]
    features: tuple[str, ...] = tuple(FEATURES)
    schema_version: int = SCHEMA_VERSION
    format_version: int = 1
    dataset_kind: str = "synthetic"


def load_model(path: Path) -> ModelBundle:
    """Load only locally trusted joblib files: deserialization can execute arbitrary code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(path)
    except (
        pickle.UnpicklingError,
        EOFError,
        IndexError,
        KeyError,
        TypeError,
        AttributeError,
        ImportError,
        InconsistentVersionWarning,
    ) as error:
        raise ValueError("Invalid, corrupted, or incompatible model bundle") from error
    if not isinstance(bundle, ModelBundle) or bundle.format_version != 1:
        raise ValueError("Unsupported model bundle")
    expected = {"synthetic": FEATURES, "creditcard": CREDIT_INPUTS}.get(bundle.dataset_kind)
    if (
        expected is None
        or bundle.schema_version != SCHEMA_VERSION
        or bundle.features != tuple(expected)
    ):
        raise ValueError("Model schema does not match this package")
    if bundle.versions != runtime_versions():
        raise ValueError(
            "Model dependency versions differ; recreate the pinned training environment"
        )
    validate_threshold(bundle.threshold)
    return bundle


def predict(bundle: ModelBundle, frame: pd.DataFrame, threshold: float | None = None) -> list[dict]:
    threshold = bundle.threshold if threshold is None else threshold
    validate_threshold(threshold)
    features = (
        validate_credit_features(frame)
        if bundle.dataset_kind == "creditcard"
        else validate_features(frame)
    )
    scores = bundle.pipeline.predict_proba(features)[:, 1]
    return [
        {
            "row": index,
            "fraud_score": float(score),
            "flagged": bool(score >= threshold),
            "threshold": float(threshold),
        }
        for index, score in enumerate(scores)
    ]


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextmanager
def new_artifact_directory(output: Path):
    """Reserve a new destination; publish all artifacts together or remove the empty reservation."""
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive: includes existing empty directories and symlinks.
    try:
        with tempfile.TemporaryDirectory(prefix=".fraud-run-", dir=output.parent) as temporary:
            staging = Path(temporary) / "result"
            staging.mkdir()
            yield staging
            os.replace(staging, output)
    finally:
        # Only remove our empty reservation after failure; never delete populated outputs.
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()


@contextmanager
def new_artifact_file(output: Path):
    """Publish a complete file without overwriting any existing destination."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fraud-predict-", dir=output.parent) as temporary:
        staging = Path(temporary) / "output"
        yield staging
        os.link(staging, output)  # Exclusive and atomic, including when output is a symlink.


def publish_json(output: Path, value: list[dict]) -> None:
    with new_artifact_file(output) as staging:
        write_json(staging, value)
