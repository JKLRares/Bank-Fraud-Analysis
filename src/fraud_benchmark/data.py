"""Versioned iid simulation and disjoint stratified partitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split

from fraud_benchmark.schema import FEATURES, TARGET, validate_dataset

GENERATOR_VERSION = "iid-context-v1"
SCENARIOS = ("reference", "covariate_shift", "concept_shift", "lower_prevalence")


def generate_transactions(
    rows: int = 20_000, seed: int = 2026, scenario: str = "reference"
) -> pd.DataFrame:
    """Sample independent contexts, then noisy labels; no persistent account or real timestamp."""
    if isinstance(rows, bool) or not isinstance(rows, int) or not 100 <= rows <= 200_000:
        raise ValueError("rows must be an integer between 100 and 200000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    shift = scenario == "covariate_shift"
    amount = np.round(np.clip(rng.lognormal(4.5 + 0.45 * shift, 1.1, rows), 1, 20_000), 2)
    age = np.clip(rng.exponential(700, rows), 0, 7_300).astype(int)
    hour = rng.integers(0, 24, rows)
    online = rng.random(rows) < (0.85 if shift else 0.65)
    foreign = rng.binomial(1, 0.25 if shift else 0.12, rows)
    count = rng.poisson(2 + 1.5 * online, rows)
    distance = np.round(
        np.clip(rng.exponential(20, rows) + foreign * rng.exponential(700, rows), 0, 20_000), 2
    )
    device = np.where(online, rng.choice(["mobile", "desktop"], rows), "terminal")
    # Designed associations, not measured banking mechanisms. Hidden risk is never exported.
    latent_risk = rng.normal(0, 1.0, rows)
    night = (hour < 6).astype(float)
    logit = (
        -5.1
        + 0.45 * np.log1p(amount / 100)
        + 0.8 * np.exp(-age / 90)
        + 0.22 * count
        + 0.55 * foreign
        + 0.35 * online
        + 0.4 * night
        + 1.0 * foreign * online
        + 0.55 * night * (count >= 5)
        + latent_risk
    )
    if scenario == "concept_shift":
        logit = logit - 1.3 * foreign * online + 1.0 * (device == "terminal")
    if scenario == "lower_prevalence":
        logit -= 1.0
    labels = rng.binomial(1, expit(logit))
    frame = pd.DataFrame(
        {
            "transaction_id": [f"{seed}-{i:07d}" for i in range(rows)],
            "amount": amount,
            "account_age_days": age,
            "hour": hour,
            "transactions_last_24h": count,
            "distance_from_home_km": distance,
            "foreign_country": foreign,
            "device": device,
            "channel": np.where(online, "online", "in_person"),
            TARGET: labels,
        }
    )
    return validate_dataset(frame)


@dataclass(frozen=True)
class Partitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_transactions(frame: pd.DataFrame, seed: int = 2026) -> Partitions:
    """60/20/20: iid rows have no shared account state or meaningful time ordering."""
    clean = validate_dataset(frame)
    if clean[TARGET].value_counts().reindex([0, 1], fill_value=0).min() < 10:
        raise ValueError("Need at least 10 examples of each class for a stratified three-way split")
    development, test = train_test_split(
        clean, test_size=0.2, stratify=clean[TARGET], random_state=seed
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=development[TARGET], random_state=(seed + 1) % 2**32
    )
    return Partitions(*(part.reset_index(drop=True) for part in (train, validation, test)))


def missingness_stress(frame: pd.DataFrame, seed: int, rate: float = 0.2) -> pd.DataFrame:
    """Mask observable cells independently of labels; preserve the existing outcomes."""
    if not 0 <= rate <= 1:
        raise ValueError("Missingness rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    result = frame.copy()
    for feature in FEATURES:
        result[feature] = result[feature].mask(rng.random(len(result)) < rate)
    return result
