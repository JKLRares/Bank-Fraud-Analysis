"""Train once on development data, then evaluate a frozen ULB benchmark artifact."""

import json
import logging
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from fraud_benchmark import __version__
from fraud_benchmark.artifacts import (
    ModelBundle,
    load_model,
    new_artifact_directory,
    runtime_versions,
    write_json,
)
from fraud_benchmark.credit_data import load_creditcard, membership, sha256_file, split_creditcard
from fraud_benchmark.diagnostics import calibration_assessment, grouped_bootstrap
from fraud_benchmark.evaluation import metrics, select_thresholds
from fraud_benchmark.models import candidates, fit_candidates
from fraud_benchmark.prioritization import capacity_for_rate, capacity_metrics
from fraud_benchmark.schema import CREDIT_INPUTS, CREDIT_MODEL_FEATURES

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreditConfig:
    protocol: str = "chronological"
    seed: int = 2026
    fn_cost: float = 100.0
    fp_cost: float = 1.0
    alert_budget: float = 0.005
    recall_target: float = 0.8
    review_rates: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01)
    bootstrap_repeats: int = 200

    def validate(self):
        if self.protocol not in {"chronological", "group_random"}:
            raise ValueError("Unknown credit-card protocol")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32 - 1000
        ):
            raise ValueError("seed must be an integer in [0, 2**32 - 1000)")
        if (
            not np.isfinite([self.fn_cost, self.fp_cost]).all()
            or min(self.fn_cost, self.fp_cost) <= 0
        ):
            raise ValueError("Illustrative costs must be positive and finite")
        if not 0 < self.alert_budget <= 1 or not 0 < self.recall_target <= 1:
            raise ValueError("Invalid alert budget or recall target")
        if not self.review_rates or any(
            not np.isfinite(r) or not 0 < r <= 1 for r in self.review_rates
        ):
            raise ValueError("Review rates must be finite and in (0, 1]")
        if tuple(sorted(set(self.review_rates))) != self.review_rates:
            raise ValueError("Review rates must be unique and increasing")
        if (
            isinstance(self.bootstrap_repeats, bool)
            or not isinstance(self.bootstrap_repeats, int)
            or not 20 <= self.bootstrap_repeats <= 1000
        ):
            raise ValueError("Bootstrap repeats must be an integer in [20, 1000]")


def _manifest(directory: Path):
    write_json(
        directory / "manifest.json",
        {
            "files_sha256": {
                p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()
            },
            "source_sha256": {
                p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
    )


def _capacity_curve(frame, scores, config):
    return [
        {
            "requested_review_rate": rate,
            **capacity_metrics(
                frame.Class,
                scores,
                capacity_for_rate(len(frame), rate),
                frame.index.to_numpy(),
                config.fn_cost,
                config.fp_cost,
            ),
        }
        for rate in config.review_rates
    ]


def _window_metrics(frame, scores, threshold, config):
    midpoint = float(frame.Time.median())
    return {
        name: metrics(
            frame.loc[mask, "Class"], scores[mask], threshold, config.fn_cost, config.fp_cost
        )
        for name, mask in (
            ("earlier_half", (frame.Time <= midpoint).to_numpy()),
            ("later_half", (frame.Time > midpoint).to_numpy()),
        )
        if mask.any()
    }


def train_creditcard(
    data_path: Path, output: Path, config: CreditConfig | None = None, fixture: bool = False
) -> dict:
    config = config or CreditConfig()
    config.validate()
    with new_artifact_directory(output) as directory, threadpool_limits(limits=1):
        data = load_creditcard(data_path, fixture)
        parts, split_audit = split_creditcard(data.frame, config.protocol, config.seed)
        train, validation = parts.train, parts.validation
        LOGGER.info(
            "Training %s: %d train / %d validation rows",
            config.protocol,
            len(train),
            len(validation),
        )
        models = candidates(config.seed, "creditcard")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            name, comparison = fit_candidates(
                train, validation, models, CREDIT_INPUTS, "Class", config.fn_cost, config.fp_cost
            )
        model = models[name]
        scores = model.predict_proba(validation[CREDIT_INPUTS])[:, 1]
        thresholds = select_thresholds(
            validation.Class,
            scores,
            config.fn_cost,
            config.fp_cost,
            config.alert_budget,
            config.recall_target,
        )
        capacities = _capacity_curve(validation, scores, config)
        chosen = min(
            capacities,
            key=lambda item: (item["illustrative_cost_per_transaction"], item["reviewed"]),
        )
        selected_rate = chosen["requested_review_rate"]
        bundle = ModelBundle(
            model,
            name,
            thresholds["f1"],
            thresholds,
            runtime_versions(),
            features=tuple(CREDIT_INPUTS),
            dataset_kind="creditcard",
        )
        joblib.dump(bundle, directory / "model.joblib")
        dummy = ModelBundle(
            models["dummy_prior"],
            "dummy_prior",
            0.5,
            {},
            runtime_versions(),
            features=tuple(CREDIT_INPUTS),
            dataset_kind="creditcard",
        )
        joblib.dump(dummy, directory / "dummy.joblib")
        write_json(directory / "split_membership.json", membership(parts))
        # Freeze model/policies before any test score. Diagnostics below see validation only.
        selection = {
            "package_version": __version__,
            "python": platform.python_version(),
            "dependencies": runtime_versions(),
            "config": asdict(config),
            "data_audit": data.audit,
            "split_audit": split_audit,
            "selected_model": name,
            "model_selection": "maximum validation average precision",
            "validation_models": comparison,
            "thresholds": thresholds,
            "selected_review_rate": selected_rate,
            "capacity_selection": "minimum validation cost among predefined review rates",
            "validation_capacity_curve": capacities,
            "candidate_parameters": {
                key: candidate.named_steps["classifier"].get_params()
                for key, candidate in models.items()
            },
            "model_inputs": CREDIT_MODEL_FEATURES,
            "time_role": "partition ordering only",
            "fit_warnings": [str(item.message) for item in caught],
        }
        write_json(directory / "selection.json", selection)
        selection["validation_thresholds"] = {
            key: metrics(validation.Class, scores, value, config.fn_cost, config.fp_cost)
            for key, value in thresholds.items()
        }
        selection["validation_calibration"] = {
            key: calibration_assessment(
                validation.Class, candidate.predict_proba(validation[CREDIT_INPUTS])[:, 1]
            )
            for key, candidate in models.items()
        }
        selection["validation_time_windows"] = _window_metrics(
            validation, scores, bundle.threshold, config
        )
        stability = []
        for offset in (1, 2):
            seed = config.seed + offset
            unique_groups = train._group.unique()
            selected_groups = np.random.default_rng(seed).choice(
                unique_groups, max(1, int(len(unique_groups) * 0.8)), replace=False
            )
            subset = train[train._group.isin(selected_groups)]
            probe = clone(model).set_params(classifier__random_state=seed)
            probe.fit(subset[CREDIT_INPUTS], subset.Class)
            stability.append(
                {
                    "seed": seed,
                    "training_rows": len(subset),
                    "training_frauds": int(subset.Class.sum()),
                    "validation": metrics(
                        validation.Class,
                        probe.predict_proba(validation[CREDIT_INPUTS])[:, 1],
                        bundle.threshold,
                        config.fn_cost,
                        config.fp_cost,
                    ),
                }
            )
        selection["validation_training_subset_checks"] = stability
        control = candidates(config.seed, "creditcard")["logistic_regression"]
        control.fit(
            train[CREDIT_INPUTS], np.random.default_rng(config.seed + 10).permutation(train.Class)
        )
        selection["validation_shuffled_label_control"] = metrics(
            validation.Class, control.predict_proba(validation[CREDIT_INPUTS])[:, 1]
        )
        missing = validation[CREDIT_INPUTS].copy()
        mask = (
            np.random.default_rng(config.seed + 11).random(
                (len(missing), len(CREDIT_MODEL_FEATURES))
            )
            < 0.01
        )
        missing[CREDIT_MODEL_FEATURES] = missing[CREDIT_MODEL_FEATURES].mask(mask)
        selection["validation_mcar_1pct"] = metrics(
            validation.Class,
            model.predict_proba(missing)[:, 1],
            bundle.threshold,
            config.fn_cost,
            config.fp_cost,
        )
        write_json(directory / "training.json", selection)
        _manifest(directory)
    return selection


def verify_training(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    required = {
        "model.joblib",
        "dummy.joblib",
        "selection.json",
        "training.json",
        "split_membership.json",
    }
    if not required.issubset(manifest["files_sha256"]):
        raise ValueError("Training manifest is incomplete")
    for name in required:
        if sha256_file(run / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Training artifact checksum mismatch: {name}")
    return json.loads((run / "training.json").read_text())


def evaluate_creditcard(data_path: Path, run: Path, output: Path) -> dict:
    with new_artifact_directory(output) as directory, threadpool_limits(limits=1):
        training = verify_training(run)
        config = CreditConfig(
            **{**training["config"], "review_rates": tuple(training["config"]["review_rates"])}
        )
        config.validate()
        fixture = training["data_audit"]["evidence_kind"] == "software_fixture"
        data = load_creditcard(data_path, fixture)
        if data.audit["raw_sha256"] != training["data_audit"]["raw_sha256"]:
            raise ValueError("Dataset checksum differs from the frozen training run")
        parts, _ = split_creditcard(data.frame, config.protocol, config.seed)
        if membership(parts) != json.loads((run / "split_membership.json").read_text()):
            raise ValueError("Evaluation partition membership differs from training")
        model = load_model(run / "model.joblib")
        dummy = load_model(run / "dummy.joblib")
        test = parts.test
        LOGGER.info("Evaluating frozen %s model on %d rows", config.protocol, len(test))
        scores = model.pipeline.predict_proba(test[CREDIT_INPUTS])[:, 1]
        dummy_scores = dummy.pipeline.predict_proba(test[CREDIT_INPUTS])[:, 1]
        curves = _capacity_curve(test, scores, config)
        policy = capacity_metrics(
            test.Class,
            scores,
            capacity_for_rate(len(test), training["selected_review_rate"]),
            test.index.to_numpy(),
            config.fn_cost,
            config.fp_cost,
        )
        report = {
            "evidence_kind": data.audit["evidence_kind"],
            "frozen_selection_sha256": sha256_file(run / "selection.json"),
            "training_manifest_sha256": sha256_file(run / "manifest.json"),
            "training": training,
            "test_thresholds": {
                key: metrics(test.Class, scores, threshold, config.fn_cost, config.fp_cost)
                for key, threshold in {"default_0.5": 0.5, **model.thresholds}.items()
            },
            "test_dummy": metrics(test.Class, dummy_scores, 0.5, config.fn_cost, config.fp_cost),
            "test_capacity_curve": curves,
            "test_selected_capacity": policy,
            "test_dummy_capacity_curve": _capacity_curve(test, dummy_scores, config),
            "test_calibration": calibration_assessment(test.Class, scores),
            "test_time_windows": _window_metrics(test, scores, model.threshold, config),
            "test_amount_slices": {
                str(interval): metrics(
                    group.Class, scores[positions], model.threshold, config.fn_cost, config.fp_cost
                )
                for interval, positions in test.groupby(
                    pd.cut(test.Amount, [-0.001, 1, 10, 100, 1000, np.inf]), observed=True
                ).indices.items()
                for group in [test.iloc[positions]]
            },
            "test_uncertainty": grouped_bootstrap(
                test.Class,
                scores,
                test._group,
                test.index.to_numpy(),
                training["selected_review_rate"],
                model.threshold,
                config.seed + 100,
                config.bootstrap_repeats,
            ),
            "limitations": [
                "Approximately two days of September 2013 data; no modern operational guarantee.",
                "Upstream PCA fitting and anonymization cannot be audited or refitted here.",
                "Card relationships, label delays and intervention effects are unknown.",
                "Validation selects model and operating policies; its estimates are optimistic.",
                "Risk scores are not established calibrated probabilities.",
                "Error costs are hypothetical units, not monetary losses or recoverable Amount.",
                "Capacity curves allocate a retrospective batch, not real-time daily reviews.",
                "Bootstrap uses frozen decisions and observed groups; excludes future drift.",
            ],
        }
        write_json(directory / "report.json", report)
        pd.DataFrame(
            {"row": test.index, "fraud_score": scores, "Class": test.Class.to_numpy()}
        ).to_csv(directory / "test_scores.csv", index=False)
        from fraud_benchmark.credit_reporting import figures, markdown_report

        figures(directory, report, test.Class.to_numpy(), scores)
        (directory / "report.md").write_text(markdown_report(report), encoding="utf-8")
        _manifest(directory)
    return report
