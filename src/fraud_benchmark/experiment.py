"""Freeze validation decisions before scoring the held-out test and stress scenarios."""

import hashlib
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from fraud_benchmark import __version__
from fraud_benchmark.artifacts import (
    ModelBundle,
    new_artifact_directory,
    runtime_versions,
    write_json,
)
from fraud_benchmark.data import (
    GENERATOR_VERSION,
    generate_transactions,
    missingness_stress,
    split_transactions,
)
from fraud_benchmark.evaluation import metrics, select_thresholds
from fraud_benchmark.models import candidates, fit_candidates
from fraud_benchmark.schema import FEATURES, TARGET


@dataclass(frozen=True)
class ExperimentConfig:
    rows: int = 20_000
    seed: int = 2026
    fn_cost: float = 25.0
    fp_cost: float = 1.0
    alert_budget: float = 0.05
    recall_target: float = 0.8

    def validate(self) -> None:
        if (
            isinstance(self.rows, bool)
            or not isinstance(self.rows, int)
            or not 1000 <= self.rows <= 200_000
        ):
            raise ValueError("Experiment rows must be an integer between 1000 and 200000")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32 - 1000
        ):
            raise ValueError("Experiment seed must be an integer in [0, 2**32 - 1000)")
        if (
            not np.isfinite([self.fn_cost, self.fp_cost]).all()
            or min(self.fn_cost, self.fp_cost) <= 0
        ):
            raise ValueError("Illustrative costs must be positive and finite")
        if not 0 <= self.alert_budget <= 1 or not 0 < self.recall_target <= 1:
            raise ValueError("Invalid alert budget or recall target")


@dataclass
class Selection:
    name: str
    model: Pipeline
    dummy: Pipeline
    validation_models: dict
    thresholds: dict[str, float]
    negative_control: dict


def select_model(
    train: pd.DataFrame, validation: pd.DataFrame, config: ExperimentConfig
) -> Selection:
    """No test input is accepted here. Fixed hyperparameters; validation AP ranks candidates."""
    config.validate()
    models = candidates(config.seed)
    name, comparison = fit_candidates(
        train, validation, models, FEATURES, TARGET, config.fn_cost, config.fp_cost
    )
    model = models[name]
    scores = model.predict_proba(validation[FEATURES])[:, 1]
    thresholds = select_thresholds(
        validation[TARGET],
        scores,
        config.fn_cost,
        config.fp_cost,
        config.alert_budget,
        config.recall_target,
    )
    # A shuffled training-label control tests the workflow on a deliberately destroyed association.
    control = candidates(config.seed)["logistic_regression"]
    shuffled = np.random.default_rng(config.seed + 10).permutation(train[TARGET].to_numpy())
    control.fit(train[FEATURES], shuffled)
    control_metrics = metrics(validation[TARGET], control.predict_proba(validation[FEATURES])[:, 1])
    return Selection(name, model, models["dummy_prior"], comparison, thresholds, control_metrics)


def _figures(directory: Path, y, scores, selected_metrics: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(recall, precision, where="post", label="Selected model")
    ax.axhline(np.mean(y), color="gray", linestyle="--", label="Test prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Synthetic held-out precision–recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "precision_recall.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    matrix = selected_metrics["confusion_matrix"]
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                color="white" if matrix[i][j] > np.max(matrix) / 2 else "black",
            )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        xlabel="Predicted",
        ylabel="Observed",
        title="Synthetic test · F1 policy",
    )
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def run_experiment(output: Path, config: ExperimentConfig | None = None) -> dict:
    config = config or ExperimentConfig()
    config.validate()
    with new_artifact_directory(output) as directory, threadpool_limits(limits=1):
        frame = generate_transactions(config.rows, config.seed)
        partitions = split_transactions(frame, config.seed)
        train, validation, test = partitions.train, partitions.validation, partitions.test
        data_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        frame.to_csv(directory / "data.csv", index=False, lineterminator="\n")
        membership = {
            name: part.transaction_id.tolist()
            for name, part in (("train", train), ("validation", validation), ("test", test))
        }
        write_json(directory / "split_membership.json", membership)
        selection = select_model(train, validation, config)
        validation_policies = {
            name: metrics(
                validation[TARGET],
                selection.model.predict_proba(validation[FEATURES])[:, 1],
                threshold,
                config.fn_cost,
                config.fp_cost,
            )
            for name, threshold in selection.thresholds.items()
        }
        decision = {
            "selected_model": selection.name,
            "selection_metric": "validation average precision",
            "validation_models": selection.validation_models,
            "validation_operating_points": validation_policies,
            "thresholds": selection.thresholds,
            "config": asdict(config),
            "data_sha256": data_hash,
        }
        # This record is persisted before any test score is calculated. No train+validation refit.
        write_json(directory / "selection.json", decision)
        bundle = ModelBundle(
            selection.model,
            selection.name,
            selection.thresholds["f1"],
            selection.thresholds,
            runtime_versions(),
        )
        joblib.dump(bundle, directory / "model.joblib")
        importance = permutation_importance(
            selection.model,
            validation[FEATURES],
            validation[TARGET],
            scoring="average_precision",
            n_repeats=3,
            random_state=config.seed,
            n_jobs=1,
        )
        # Post-selection diagnostics cannot change the saved model, policies or parameters.
        scores = selection.model.predict_proba(test[FEATURES])[:, 1]
        operating_points = {
            name: metrics(test[TARGET], scores, threshold, config.fn_cost, config.fp_cost)
            for name, threshold in {"default_0.5": 0.5, **selection.thresholds}.items()
        }
        dummy_metrics = metrics(
            test[TARGET],
            selection.dummy.predict_proba(test[FEATURES])[:, 1],
            fn_cost=config.fn_cost,
            fp_cost=config.fp_cost,
        )
        stress = {}
        stress_rows = len(test)
        for offset, scenario in enumerate(
            ["reference", "covariate_shift", "concept_shift", "lower_prevalence"], start=101
        ):
            fresh = generate_transactions(stress_rows, config.seed + offset, scenario)
            stress[scenario] = metrics(
                fresh[TARGET],
                selection.model.predict_proba(fresh[FEATURES])[:, 1],
                bundle.threshold,
                config.fn_cost,
                config.fp_cost,
            )
        fresh = generate_transactions(stress_rows, config.seed + 105)
        for name, perturbed in (
            ("input_perturbation_reference", fresh),
            ("missing_20_percent", missingness_stress(fresh, config.seed + 106)),
            ("unseen_device", fresh.assign(device="wearable")),
        ):
            stress[name] = metrics(
                perturbed[TARGET],
                selection.model.predict_proba(perturbed[FEATURES])[:, 1],
                bundle.threshold,
                config.fn_cost,
                config.fp_cost,
            )
        report = {
            "benchmark_version": __version__,
            "generator_version": GENERATOR_VERSION,
            "python": platform.python_version(),
            "dependencies": runtime_versions(),
            "protocol": "iid stratified 60/20/20; train-only fit; validation AP; no refit",
            **decision,
            "split_summary": {
                name: {
                    "rows": len(part),
                    "fraud_count": int(part[TARGET].sum()),
                    "prevalence": float(part[TARGET].mean()),
                }
                for name, part in (("train", train), ("validation", validation), ("test", test))
            },
            "test_dummy": dummy_metrics,
            "test_operating_points": operating_points,
            "stress_at_frozen_f1_threshold": stress,
            "validation_shuffled_label_control": selection.negative_control,
            "validation_permutation_importance": {
                feature: {"ap_drop_mean": float(mean), "ap_drop_std": float(std)}
                for feature, mean, std in zip(
                    FEATURES, importance.importances_mean, importance.importances_std, strict=True
                )
            },
            "test_errors_by_channel": {
                str(channel): metrics(
                    group[TARGET],
                    scores[test.channel.to_numpy() == channel],
                    bundle.threshold,
                    config.fn_cost,
                    config.fp_cost,
                )
                for channel, group in test.groupby("channel", sort=True)
            },
            "limitations": [
                "Designed synthetic associations; no evidence of real banking performance.",
                "Scores are not calibrated real-world fraud probabilities.",
                "One seed and one holdout; sampling uncertainty is not quantified.",
                "Validation selects model and thresholds; validation estimates are optimistic.",
                "Alert budget applies to validation only; test or deployed volume may exceed it.",
                "Costs are illustrative units; no measured operational or monetary assumptions.",
                "Stress scenarios are simulations; they cannot establish real-world robustness.",
            ],
        }
        write_json(directory / "report.json", report)
        pd.DataFrame(
            {
                "transaction_id": test.transaction_id,
                "fraud": test[TARGET],
                "fraud_score": scores,
                "flagged": scores >= bundle.threshold,
            }
        ).to_csv(directory / "test_predictions.csv", index=False)
        _figures(directory, test[TARGET], scores, operating_points["f1"])
        write_json(
            directory / "manifest.json",
            {
                "files_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(directory.iterdir())
                    if p.is_file()
                },
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob("*.py"))
                },
            },
        )
    return report
