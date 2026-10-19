import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.artifacts import load_model, predict, publish_json
from fraud_benchmark.cli import main, read_transactions
from fraud_benchmark.data import Partitions, split_transactions
from fraud_benchmark.experiment import ExperimentConfig, run_experiment, select_model
from fraud_benchmark.models import candidates
from fraud_benchmark.schema import FEATURES, NUMERIC


def test_preprocessing_fits_only_training_rows_and_handles_empty_columns(dataset):
    parts = split_transactions(dataset)
    train = parts.train[FEATURES].copy()
    train.loc[0, "amount"] = np.nan
    model = candidates(2026)["logistic_regression"].fit(train, parts.train.fraud)
    preprocess = model.named_steps["preprocess"]
    medians = preprocess.named_transformers_["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(medians, train[NUMERIC].median().to_numpy())
    before = medians.copy()
    strange = parts.validation[FEATURES].assign(amount=np.nan, device="unseen-device")
    scores = model.predict_proba(strange)[:, 1]
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(medians, before)
    # A fully missing training column retains its position in both train and inference.
    train["amount"] = np.nan
    train["device"] = np.nan
    model.fit(train, parts.train.fraud)
    assert model.predict_proba(strange).shape == (len(strange), 2)


def test_model_selection_is_deterministic_and_has_no_test_argument(dataset):
    parts = split_transactions(dataset)
    config = ExperimentConfig(rows=len(dataset))
    first = select_model(parts.train, parts.validation, config)
    second = select_model(parts.train, parts.validation, config)
    assert first.name == second.name
    assert first.validation_models == second.validation_models
    assert first.thresholds == second.thresholds
    expected = max(
        first.validation_models, key=lambda key: first.validation_models[key]["average_precision"]
    )
    assert first.name == expected


def test_complete_run_manifests_serialization_and_predictions(completed_run):
    output, report = completed_run
    assert json.loads((output / "report.json").read_text()) == report
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest["files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    selected = json.loads((output / "selection.json").read_text())
    assert "test_operating_points" not in selected
    assert selected["selected_model"] == report["selected_model"]
    bundle = load_model(output / "model.joblib")
    frame = pd.read_csv(output / "data.csv")
    membership = json.loads((output / "split_membership.json").read_text())
    test = frame.set_index("transaction_id").loc[membership["test"]]
    scores = predict(bundle, test[FEATURES])
    saved = pd.read_csv(output / "test_predictions.csv")
    np.testing.assert_allclose([row["fraud_score"] for row in scores], saved.fraud_score)
    assert [row["flagged"] for row in scores] == saved.flagged.tolist()
    assert all(0 <= row["fraud_score"] <= 1 and type(row["flagged"]) is bool for row in scores)
    assert all(row["flagged"] for row in predict(bundle, test[FEATURES], threshold=0))
    assert (output / "precision_recall.png").stat().st_size > 1000
    assert set(report["stress_at_frozen_f1_threshold"]) == {
        "reference",
        "input_perturbation_reference",
        "covariate_shift",
        "concept_shift",
        "lower_prevalence",
        "missing_20_percent",
        "unseen_device",
    }


def test_model_schema_and_version_mismatch_are_rejected(completed_run, tmp_path):
    output, _ = completed_run
    bundle = load_model(output / "model.joblib")
    bundle.versions["numpy"] = "0.0.0"
    path = tmp_path / "bad.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="versions"):
        load_model(path)
    bundle.schema_version = -1
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="schema"):
        load_model(path)


@pytest.mark.parametrize("content", [b"", b"invalid pickle", b"\x80"])
def test_corrupted_models_produce_a_clear_cli_error(dataset, tmp_path, capsys, content):
    model = tmp_path / "broken.joblib"
    model.write_bytes(content)
    path = tmp_path / "input.json"
    path.write_text(dataset[FEATURES].head(1).to_json(orient="records"))
    assert main(["predict", "--model", str(model), "--input", str(path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_failure_cleans_staging_and_preserves_existing_outputs(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        run_experiment(output, ExperimentConfig(rows=1000))
    assert marker.read_text() == "original"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        run_experiment(empty, ExperimentConfig(rows=1000))
    assert empty.is_dir()

    def fail(*args, **kwargs):
        raise ValueError("injected training failure")

    monkeypatch.setattr("fraud_benchmark.experiment.select_model", fail)
    with pytest.raises(ValueError, match="injected"):
        run_experiment(tmp_path / "failed", ExperimentConfig(rows=1000))
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".fraud-run-*"))


def test_atomic_prediction_output_preserves_existing_file_and_symlink(tmp_path):
    output = tmp_path / "scores.json"
    publish_json(output, [{"score": 0.5}])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        publish_json(output, [{"score": 0.8}])
    assert output.read_bytes() == original
    link = tmp_path / "link.json"
    link.symlink_to(output)
    with pytest.raises(FileExistsError):
        publish_json(link, [{"score": 0.9}])
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "[]",
        "[1]",
        '[{"amount": 1, "amount": 2}]',
        '[{"amount": NaN}]',
        '[{"amount": 1}]',
    ],
)
def test_invalid_json_batches_fail_without_output(tmp_path, text, capsys):
    path = tmp_path / "bad.json"
    path.write_text(text)
    destination = tmp_path / "scores.json"
    assert (
        main(
            [
                "predict",
                "--model",
                str(tmp_path / "missing.joblib"),
                "--input",
                str(path),
                "--output",
                str(destination),
            ]
        )
        == 2
    )
    assert "error:" in capsys.readouterr().err
    assert not destination.exists()


def test_cli_prediction_and_generate(completed_run, tmp_path, dataset, capsys):
    output, _ = completed_run
    example = tmp_path / "input.json"
    example.write_text(dataset[FEATURES].head(2).to_json(orient="records"))
    destination = tmp_path / "predictions.json"
    assert (
        main(
            [
                "predict",
                "--model",
                str(output / "model.joblib"),
                "--input",
                str(example),
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    assert len(json.loads(destination.read_text())) == 2
    assert main(["generate", "--rows", "100", "--output", str(tmp_path / "data.csv")]) == 0
    original = (tmp_path / "data.csv").read_bytes()
    assert main(["generate", "--output", str(tmp_path / "data.csv")]) == 2
    assert (tmp_path / "data.csv").read_bytes() == original
    capsys.readouterr()


def test_json_row_keys_must_be_consistent(dataset, tmp_path):
    records = dataset[FEATURES].head(2).to_dict(orient="records")
    del records[1]["hour"]
    path = tmp_path / "missing-key.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Each transaction"):
        read_transactions(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 999},
        {"rows": 200001},
        {"seed": -1},
        {"fn_cost": -1},
        {"alert_budget": float("nan")},
        {"recall_target": 0},
    ],
)
def test_invalid_experiment_config_fails_before_creating_outputs(kwargs, tmp_path):
    with pytest.raises(ValueError):
        run_experiment(tmp_path / "invalid", ExperimentConfig(**kwargs))
    assert not (tmp_path / "invalid").exists()


def test_final_test_mutations_cannot_change_selection(
    completed_run, dataset, tmp_path, monkeypatch
):
    _, original_report = completed_run
    parts = split_transactions(dataset)
    altered_test = parts.test.assign(fraud=1 - parts.test.fraud, amount=999_999.0)
    monkeypatch.setattr(
        "fraud_benchmark.experiment.split_transactions",
        lambda *args: Partitions(parts.train, parts.validation, altered_test),
    )
    changed = run_experiment(tmp_path / "mutated-test", ExperimentConfig(rows=2000))
    assert changed["selected_model"] == original_report["selected_model"]
    assert changed["thresholds"] == original_report["thresholds"]
    assert changed["validation_models"] == original_report["validation_models"]
    assert changed["test_operating_points"]["f1"] != original_report["test_operating_points"]["f1"]


def test_failed_csv_write_leaves_no_partial_file(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    path = tmp_path / "data.csv"
    assert main(["generate", "--rows", "100", "--output", str(path)]) == 2
    assert not path.exists()
    assert not list(tmp_path.glob(".fraud-predict-*"))
    assert "injected disk failure" in capsys.readouterr().err
