"""Train, evaluate, score and prioritize transactions; retain the synthetic benchmark."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from fraud_benchmark.artifacts import load_model, new_artifact_file, predict, publish_json
from fraud_benchmark.credit_experiment import CreditConfig, evaluate_creditcard, train_creditcard
from fraud_benchmark.data import generate_transactions
from fraud_benchmark.experiment import ExperimentConfig, run_experiment
from fraud_benchmark.prioritization import prioritize


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_transactions(path: Path, features=None) -> pd.DataFrame:
    def invalid_constant(value):
        raise ValueError(f"Nonstandard JSON value: {value}; use null for missing cells")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=invalid_constant,
    )
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ValueError("Input must be a nonempty JSON array of transaction objects")
    # A missing key in one row must not be silently converted into an imputable missing cell.
    from fraud_benchmark.schema import FEATURES

    expected = FEATURES if features is None else features
    if any(set(row) != set(expected) for row in value):
        raise ValueError("Each transaction must contain exactly the documented feature keys")
    return pd.DataFrame(value)


def read_scores(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, list) or not value:
        raise ValueError("Scores must be a nonempty JSON array")
    allowed = {"row", "fraud_score", "flagged", "threshold"}
    for row in value:
        if (
            not isinstance(row, dict)
            or not {"row", "fraud_score"}.issubset(row)
            or set(row) - allowed
        ):
            raise ValueError(
                "Each score requires row and fraud_score; target labels are not accepted"
            )
        if type(row["row"]) is not int or isinstance(row["fraud_score"], bool):
            raise ValueError("row must be an integer and fraud_score must be numeric")
    return [row["fraud_score"] for row in value], [row["row"] for row in value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the historical synthetic simulation benchmark")
    run.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    run.add_argument("--rows", type=int, default=20_000)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--fn-cost", type=float, default=25.0, help="Illustrative cost units")
    run.add_argument("--fp-cost", type=float, default=1.0, help="Illustrative cost units")
    run.add_argument("--alert-budget", type=float, default=0.05)
    run.add_argument("--recall-target", type=float, default=0.8)
    generate = commands.add_parser("generate", help="Write reference synthetic data to a new CSV")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--rows", type=int, default=20_000)
    generate.add_argument("--seed", type=int, default=2026)
    score = commands.add_parser("predict", help="Score JSON; only load trusted local model files")
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.add_argument("--threshold", type=float)
    train = commands.add_parser(
        "train", help="Freeze model and policies using ULB train/validation only"
    )
    train.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--protocol", choices=["chronological", "group_random"], default="chronological"
    )
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--fn-cost", type=float, default=100)
    train.add_argument("--fp-cost", type=float, default=1)
    train.add_argument("--alert-budget", type=float, default=0.005)
    train.add_argument("--recall-target", type=float, default=0.8)
    train.add_argument(
        "--review-rates", type=float, nargs="+", default=[0.001, 0.0025, 0.005, 0.01]
    )
    train.add_argument("--bootstrap-repeats", type=int, default=200)
    train.add_argument(
        "--fixture",
        action="store_true",
        help="Mark a <=5000-row software fixture; no real-data evidence",
    )
    evaluate = commands.add_parser(
        "evaluate", help="Score frozen held-out policies; no tuning flags"
    )
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    evaluate.add_argument("--output", type=Path, required=True)
    rank = commands.add_parser("prioritize", help="Select Top-K by score, then ascending row ID")
    rank.add_argument("--input", type=Path, required=True, help="Prediction JSON; no labels")
    rank.add_argument("--capacity", type=int, required=True)
    rank.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "run":
            config = ExperimentConfig(
                args.rows,
                args.seed,
                args.fn_cost,
                args.fp_cost,
                args.alert_budget,
                args.recall_target,
            )
            report = run_experiment(args.output, config)
            summary = {
                "output": str(args.output),
                "selected_model": report["selected_model"],
                "test_f1_policy": report["test_operating_points"]["f1"],
            }
            print(json.dumps(summary, indent=2, allow_nan=False))
        elif args.command == "generate":
            frame = generate_transactions(args.rows, args.seed)
            with new_artifact_file(args.output) as staging:
                frame.to_csv(staging, index=False, lineterminator="\n")
            print(f"Generated {len(frame)} synthetic rows in {args.output}")
        elif args.command == "predict":
            bundle = load_model(args.model)
            frame = read_transactions(args.input, bundle.features)
            with threadpool_limits(limits=1):
                predictions = predict(bundle, frame, args.threshold)
            if args.output:
                publish_json(args.output, predictions)
            else:
                print(json.dumps(predictions, indent=2, allow_nan=False))
        elif args.command == "train":
            config = CreditConfig(
                args.protocol,
                args.seed,
                args.fn_cost,
                args.fp_cost,
                args.alert_budget,
                args.recall_target,
                tuple(args.review_rates),
                args.bootstrap_repeats,
            )
            result = train_creditcard(args.data, args.output, config, args.fixture)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "evidence_kind": result["data_audit"]["evidence_kind"],
                        "selected_model": result["selected_model"],
                        "validation_ap": result["validation_models"][result["selected_model"]][
                            "average_precision"
                        ],
                        "selected_review_rate": result["selected_review_rate"],
                    },
                    indent=2,
                )
            )
        elif args.command == "evaluate":
            result = evaluate_creditcard(args.data, args.run, args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "evidence_kind": result["evidence_kind"],
                        "test_f1_policy": result["test_thresholds"]["f1"],
                        "selected_capacity": result["test_selected_capacity"],
                    },
                    indent=2,
                )
            )
        else:
            scores, ids = read_scores(args.input)
            selected = prioritize(scores, args.capacity, ids)
            if args.output:
                publish_json(args.output, selected)
            else:
                print(json.dumps(selected, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, EOFError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
