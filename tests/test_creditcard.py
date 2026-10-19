import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.artifacts import load_model, predict
from fraud_benchmark.cli import main
from fraud_benchmark.credit_data import load_creditcard, membership, split_creditcard
from fraud_benchmark.credit_experiment import CreditConfig, evaluate_creditcard, train_creditcard
from fraud_benchmark.schema import CREDIT_INPUTS, CREDIT_MODEL_FEATURES, validate_credit_features


@pytest.fixture
def credit_csv(tmp_path):
    rng = np.random.default_rng(52)
    frame = pd.DataFrame(rng.normal(size=(600, 28)), columns=[f"V{i}" for i in range(1, 29)])
    frame["Time"] = np.repeat(np.arange(300), 2)
    frame["Amount"] = rng.uniform(0, 1000, len(frame))
    frame["Class"] = (np.arange(len(frame)) % 17 == 0).astype(int)
    # Cross-time identical inputs are conservative signature groups, not invented customer IDs.
    for index in (400, 550):
        frame.loc[index, [*CREDIT_MODEL_FEATURES, "Class"]] = frame.loc[
            0, [*CREDIT_MODEL_FEATURES, "Class"]
        ].to_numpy()
    frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    path = tmp_path / "creditcard.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def credit_run(credit_csv, tmp_path):
    output = tmp_path / "trained"
    training = train_creditcard(
        credit_csv, output, CreditConfig(bootstrap_repeats=20), fixture=True
    )
    return output, training


def test_data_audit_duplicate_removal_and_temporal_purging(credit_csv):
    data = load_creditcard(credit_csv, fixture=True)
    assert data.audit["evidence_kind"] == "software_fixture"
    assert data.audit["exact_duplicates_removed"] == 1
    assert data.audit["repeated_model_inputs_after_row_dedup"] == 2
    parts, details = split_creditcard(data.frame)
    assert parts.train.Time.max() < parts.validation.Time.min()
    assert parts.validation.Time.max() < parts.test.Time.min()
    assert details["purged_validation_rows"] == details["purged_test_rows"] == 1
    sets = [set(p._group) for p in (parts.train, parts.validation, parts.test)]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    assert 400 not in parts.validation.index and 550 not in parts.test.index


def test_group_random_split_is_stratified_disjoint_and_complete(credit_csv):
    data = load_creditcard(credit_csv, fixture=True)
    parts, _ = split_creditcard(data.frame, "group_random")
    again, _ = split_creditcard(data.frame, "group_random")
    assert membership(parts) == membership(again)
    ids = [set(p.index) for p in (parts.train, parts.validation, parts.test)]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
    assert set.union(*ids) == set(data.frame.index)
    groups = [set(p._group) for p in (parts.train, parts.validation, parts.test)]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    assert all(set(p.Class) == {0, 1} for p in (parts.train, parts.validation, parts.test))


def test_chronological_conservation_and_purging_do_not_depend_on_labels(credit_csv):
    data = load_creditcard(credit_csv, fixture=True)
    frame = data.frame
    parts, audit = split_creditcard(frame)
    kept = pd.concat([parts.train, parts.validation, parts.test])
    purged = frame.loc[~frame.index.isin(kept.index)]
    assert set(purged.index) == {400, 550}
    assert kept.index.is_unique
    assert set(kept.index).isdisjoint(purged.index)
    assert set(kept.index) | set(purged.index) == set(frame.index)
    assert len(kept) + len(purged) == len(frame)
    assert kept.Class.sum() + purged.Class.sum() == frame.Class.sum()
    assert set(kept._group) == set(frame._group)
    # These fixture purges are positive: the rule is not "remove genuine transactions".
    assert audit["purged_validation_frauds"] == audit["purged_test_frauds"] == 1
    relabeled, _ = split_creditcard(frame.assign(Class=1 - frame.Class))
    assert membership(relabeled) == membership(parts)


def test_accounting_verifier_matches_frozen_membership(credit_csv, credit_run):
    from scripts.verify_data_accounting import verify

    run, _ = credit_run
    result = verify(credit_csv, run)
    assert result["raw"]["rows"] == 601
    assert result["exact_duplicates_removed"]["rows"] == 1
    assert result["deduplicated"]["rows"] == 600
    assert result["additional_purge"] == {"rows": 2, "frauds": 2, "genuine": 0}
    assert result["final_partitions_total"]["rows"] == 598
    assert result["all_signatures_represented"]
    assert result["earlier_witnesses_verified"]


def test_loader_rejects_missing_noncanonical_and_conflicting_data(credit_csv, tmp_path):
    with pytest.raises(FileNotFoundError, match="Obtain creditcard.csv"):
        load_creditcard(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="published ULB counts"):
        load_creditcard(credit_csv)
    frame = pd.read_csv(credit_csv)
    frame.loc[400, "Class"] = 1 - frame.loc[0, "Class"]
    frame.to_csv(credit_csv, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        load_creditcard(credit_csv, fixture=True)


@pytest.mark.parametrize(
    "column,value",
    [("Time", -1), ("Amount", -1), ("V1", np.inf), ("V1", True), ("V1", "0.2"), ("V1", 1j)],
)
def test_credit_schema_rejects_invalid_values(credit_csv, column, value):
    frame = pd.read_csv(credit_csv)[CREDIT_INPUTS].head(2)
    frame[column] = pd.Series([value, frame.loc[1, column]])
    with pytest.raises(ValueError):
        validate_credit_features(frame)


def test_credit_schema_missing_values_extra_keys_and_order(credit_csv):
    frame = pd.read_csv(credit_csv)[CREDIT_INPUTS].head(2)
    frame.loc[0, "V1"] = np.nan
    normalized = validate_credit_features(frame[CREDIT_INPUTS[::-1]])
    assert normalized.columns.tolist() == CREDIT_INPUTS
    assert np.isnan(normalized.loc[0, "V1"])
    with pytest.raises(ValueError, match="finite"):
        validate_credit_features(frame, allow_missing=False)
    with pytest.raises(ValueError, match="exactly"):
        validate_credit_features(frame.assign(Class=0))


def test_training_inference_preprocessing_and_evaluation(credit_csv, credit_run, tmp_path):
    run, training = credit_run
    assert "test_thresholds" not in training
    bundle = load_model(run / "model.joblib")
    assert bundle.dataset_kind == "creditcard"
    data = load_creditcard(credit_csv, fixture=True)
    parts, _ = split_creditcard(data.frame)
    preprocessor = bundle.pipeline.named_steps["preprocess"]
    imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, parts.train[CREDIT_MODEL_FEATURES].median())
    frame = parts.test[CREDIT_INPUTS]
    assert predict(bundle, frame) == predict(bundle, frame.assign(Time=999999999))
    assert all(0 <= row["fraud_score"] <= 1 for row in predict(bundle, frame.assign(V1=np.nan)))
    report = evaluate_creditcard(credit_csv, run, tmp_path / "evaluated")
    assert report["evidence_kind"] == "software_fixture"
    assert report["training"]["selected_review_rate"] in training["config"]["review_rates"]
    assert len(report["test_capacity_curve"]) == 4
    assert report["test_uncertainty"]["valid_replicates"] > 0
    with pytest.raises(FileExistsError):
        evaluate_creditcard(credit_csv, run, tmp_path / "evaluated")


def test_evaluation_rejects_tampering_and_changed_dataset(credit_csv, credit_run, tmp_path):
    run, _ = credit_run
    original = credit_csv.read_bytes()
    credit_csv.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Dataset checksum"):
        evaluate_creditcard(credit_csv, run, tmp_path / "different-data")
    assert not (tmp_path / "different-data").exists()
    credit_csv.write_bytes(original)
    (run / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_creditcard(credit_csv, run, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_test_partition_cannot_change_training_selection(
    credit_csv, credit_run, tmp_path, monkeypatch
):
    _, original = credit_run
    data = load_creditcard(credit_csv, fixture=True)
    parts, audit = split_creditcard(data.frame)
    changed = replace(parts, test=parts.test.assign(Class=1 - parts.test.Class, V1=999))
    monkeypatch.setattr(
        "fraud_benchmark.credit_experiment.split_creditcard", lambda *args: (changed, audit)
    )
    repeated = train_creditcard(
        credit_csv, tmp_path / "changed-test", CreditConfig(bootstrap_repeats=20), fixture=True
    )
    for key in ["selected_model", "thresholds", "selected_review_rate", "validation_models"]:
        assert repeated[key] == original[key]


def test_cli_train_predict_prioritize_and_missing_data(credit_csv, tmp_path, capsys):
    run = tmp_path / "cli-trained"
    assert (
        main(
            [
                "train",
                "--fixture",
                "--data",
                str(credit_csv),
                "--output",
                str(run),
                "--bootstrap-repeats",
                "20",
            ]
        )
        == 0
    )
    sample = tmp_path / "request.json"
    sample.write_text(pd.read_csv(credit_csv)[CREDIT_INPUTS].head(5).to_json(orient="records"))
    scores = tmp_path / "scores.json"
    assert (
        main(
            [
                "predict",
                "--model",
                str(run / "model.joblib"),
                "--input",
                str(sample),
                "--output",
                str(scores),
            ]
        )
        == 0
    )
    selected = tmp_path / "selected.json"
    assert (
        main(["prioritize", "--input", str(scores), "--capacity", "2", "--output", str(selected)])
        == 0
    )
    assert len(json.loads(selected.read_text())) == 2
    assert main(["prioritize", "--input", str(scores), "--capacity", "6"]) == 2
    assert (
        main(["train", "--data", str(tmp_path / "absent.csv"), "--output", str(tmp_path / "bad")])
        == 2
    )
    assert not (tmp_path / "bad").exists()
    capsys.readouterr()


def test_control_review_rejects_a_different_frozen_seed(tmp_path, monkeypatch):
    from scripts import review_controls

    monkeypatch.setattr(
        review_controls, "load_creditcard", lambda path: SimpleNamespace(audit={"raw_sha256": "x"})
    )
    monkeypatch.setattr(
        review_controls,
        "verify_training",
        lambda path: {
            "config": {"protocol": "chronological", "seed": 7},
            "data_audit": {"raw_sha256": "x"},
        },
    )
    with pytest.raises(ValueError, match="seed 2026"):
        review_controls.review(tmp_path / "data.csv", tmp_path)
