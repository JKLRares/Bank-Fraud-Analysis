"""Validation-only review of surprising shuffled-label controls; no model/policy selection."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from fraud_benchmark.artifacts import load_model, new_artifact_file, write_json
from fraud_benchmark.credit_data import load_creditcard, split_creditcard
from fraud_benchmark.credit_experiment import verify_training
from fraud_benchmark.models import candidates
from fraud_benchmark.schema import CREDIT_INPUTS, CREDIT_MODEL_FEATURES


def review(data_path: Path, runs_root: Path) -> dict:
    data = load_creditcard(data_path)
    results = {
        "purpose": "post-hoc validation sanity check; no test scoring or selection",
        "protocols": {},
    }
    with threadpool_limits(limits=1):
        for protocol in ["chronological", "group_random"]:
            run = runs_root / f"ulb-{protocol}-2026"
            training = verify_training(run)
            if training["config"]["protocol"] != protocol or training["config"]["seed"] != 2026:
                raise ValueError("Control review requires the frozen seed 2026 reference protocols")
            if training["data_audit"]["raw_sha256"] != data.audit["raw_sha256"]:
                raise ValueError("Dataset differs from the frozen run")
            parts, _ = split_creditcard(data.frame, protocol, 2026)
            train, validation = parts.train, parts.validation
            record = {"training_label_permutations": [], "validation_label_permutation_ap": []}
            for seed in [2036, 2037, 2038, 2039]:
                shuffled = np.random.default_rng(seed).permutation(train.Class.to_numpy())
                model = candidates(2026, "creditcard")["logistic_regression"]
                model.fit(train[CREDIT_INPUTS], shuffled)
                transformed_columns = model.named_steps["preprocess"].transformers_[0][2]
                if transformed_columns != CREDIT_MODEL_FEATURES:
                    raise ValueError("Unexpected transformed feature set")
                scores = model.predict_proba(validation[CREDIT_INPUTS])[:, 1]
                record["training_label_permutations"].append(
                    {
                        "seed": seed,
                        "positive_label_overlap_with_original": int(
                            ((train.Class.to_numpy() == 1) & (shuffled == 1)).sum()
                        ),
                        "validation_ap": float(average_precision_score(validation.Class, scores)),
                    }
                )
            selected = load_model(run / "model.joblib")
            scores = selected.pipeline.predict_proba(validation[CREDIT_INPUTS])[:, 1]
            for seed in range(3000, 3020):
                shuffled = np.random.default_rng(seed).permutation(validation.Class.to_numpy())
                record["validation_label_permutation_ap"].append(
                    float(average_precision_score(shuffled, scores))
                )
            # Label-free random directions can still rank outliers in an asymmetric feature space.
            transformed = model.named_steps["preprocess"].transform(validation[CREDIT_INPUTS])
            directions = np.random.default_rng(4000).normal(size=(len(CREDIT_MODEL_FEATURES), 32))
            projections = transformed @ directions
            record["random_linear_projection_ap"] = [
                float(average_precision_score(validation.Class, projections[:, i]))
                for i in range(32)
            ]
            record["actual_validation_ap"] = float(
                average_precision_score(validation.Class, scores)
            )
            record["validation_prevalence"] = float(validation.Class.mean())
            results["protocols"][protocol] = record
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.data, args.runs_root)
    with new_artifact_file(args.output) as staging:
        write_json(staging, result)
    for protocol, values in result["protocols"].items():
        print(
            protocol,
            "projection AP range",
            min(values["random_linear_projection_ap"]),
            max(values["random_linear_projection_ap"]),
        )
