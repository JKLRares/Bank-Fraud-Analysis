# Synthetic benchmark data card

Historical version 0.2 synthetic benchmark documentation. See [REAL_DATA.md](REAL_DATA.md) for the current real-data experiment.

## Scope and provenance

`iid-context-v1` is an explicitly designed simulator in `src/fraud_benchmark/data.py`.
It is separate from the original `generate_data.py` and its committed CSVs. It does not model
a bank, reproduce real accounts, use external data, or estimate actual financial fraud rates.
Its purpose is to exercise a reproducible binary-classification workflow under weak signal,
class imbalance and known changes to the generating process.

Each row is an independent transaction context. There is no persistent customer entity and no
chronological history. `hour` represents a sampled hour of day, not an ordering timestamp.
Account age and activity count are sampled context available at scoring by assumption; they
are not calculated from the other rows. `transaction_id = seed-row_number` is metadata only,
never a model input. Different scenarios with the same seed are counterfactual samples and
must not be concatenated as if their IDs were independent observations.

## Schema

All eight feature **keys** are required for every scoring row. Individual cells may be JSON
`null`. Extra keys, labels and IDs are rejected by the prediction interface. Numeric strings,
booleans, fractional integer fields, infinities and out-of-range observations are rejected.
Bounds are demonstration input contracts, not universal banking constraints.

| Field | Type and accepted range | Reference simulation |
| --- | --- | --- |
| `amount` | Number, [0, 1,000,000] | Lognormal(4.5, 1.1), clipped [1, 20,000], rounded to cents; unspecified currency units |
| `account_age_days` | Integer, [0, 36,500] | Exponential mean 700 days, clipped at 7,300, floored |
| `hour` | Integer, [0, 23] | Uniform integer hour |
| `transactions_last_24h` | Integer, [0, 100,000] | Poisson mean 3.5 online / 2 in person |
| `distance_from_home_km` | Number, [0, 20,100] | Exponential mean 20 plus foreign × exponential mean 700, clipped at 20,000 and rounded |
| `foreign_country` | Integer, 0 or 1 | Bernoulli(.12), a context flag without real geography |
| `device` | Nonempty string | Uniform mobile/desktop online; terminal in person |
| `channel` | Nonempty string | Online with probability .65; otherwise in_person |
| `fraud` | Binary target, 0 or 1 | Bernoulli of the latent logistic probability below; never a feature |

Unknown string categories are accepted for scoring and encoded with `handle_unknown="ignore"`.
That prevents a schema crash; it does not guarantee sensible scores for unknown populations.
The simulator generates complete features. Missingness is a separate perturbation experiment.

## Label mechanism

With `A = amount`, `D = account_age_days`, `N = transactions_last_24h`,
`F = foreign_country`, `O = 1(channel == online)`, `H = 1(hour < 6)` and
`U ~ Normal(0, 1)` sampled independently:

```text
z = -5.1 + .45 log(1 + A/100) + .8 exp(-D/90) + .22 N
    + .55 F + .35 O + .4 H + 1.0 F O + .55 H 1(N >= 5) + U
p = sigmoid(z)
fraud ~ Bernoulli(p)
```

Neither `U`, `z` nor `p` is exported. Observable features precede label sampling. Labels have
overlap even for similar contexts, unlike a deterministic target rule. Nevertheless, the
relationships and coefficients are invented. Learning them is still simulator recovery.
Distance is correlated with foreign status but absent from the direct label equation; device
and channel are correlated by construction. Feature importance is predictive association,
not causal evidence. Redundant variables can divide or distort permutation importance.

The committed reference run (20,000 rows, seed 2026) realizes 1,139 positives (5.695%). This
rate emerges from the chosen intercept and features; it is not fitted to banking statistics.
The seed and coefficients were fixed before the reported final test; they were not changed
to improve its result. NumPy's local `default_rng` avoids modifying global random state.

## Evaluation protocol

1. Validate schema, label domain, unique transaction IDs and absence of exact duplicate feature
   rows before splitting. Conflicting-label duplicates also fail this check.
2. Use deterministic stratified 60/20/20 partitions: independent contexts have no meaningful
   grouping or temporal ordering. Preserve membership and a source-data checksum.
3. Fit each preprocessing/model pipeline on train only: numeric median imputation, scaling,
   categorical constant imputation and dense one-hot encoding. No resampling or class weights.
   Retain all-empty training feature positions explicitly.
4. Compare prior-only dummy, logistic regression (`C=1`) and histogram gradient boosting
   (100 iterations, 15 leaves, minimum leaf size 30, learning rate .08, L2=5, no early stopping).
   Select maximum validation average precision; exact ties retain candidate insertion order.
5. Select thresholds using only validation scores. Save `selection.json` before test scoring.
   Keep the train-fitted pipeline: no train+validation refit that would change score calibration.
6. Score only the frozen selected configuration and the predefined dummy on the final test.
   Report all predefined operating points; do not choose a new winner from test results.

Average precision is sklearn's non-interpolated AP, not trapezoidal integration of the PR
curve. ROC-AUC, confusion matrix, precision, recall, F1, prevalence and alert volume are also
reported. Undefined ROC-AUC is JSON null for single-class slices; precision/recall/F1 with a
zero denominator are zero. AP is null when a slice contains no positives. Accuracy is retained
for context, never used to select a model.

The policies examine every distinct validation score with `score >= threshold`; whole score
ties stay together. A sentinel `1.0000000000000002` represents no alerts, including when a
score is exactly one. Equal objective values choose the highest threshold (fewest alerts).

| Policy | Validation objective |
| --- | --- |
| `f1` (default prediction policy) | Maximize F1 |
| `illustrative_cost` | Minimize `25 × FN + 1 × FP`, configurable positive cost units |
| `alert_budget` | Maximize recall subject to alert rate ≤ .05; ties choose fewer alerts |
| `high_recall` | Maximize precision among thresholds with recall ≥ .8 |

These constraints apply to validation; neither recall targets nor alert budgets are promises
on new data. Costs are constant hypothetical units per error, not transaction amounts, lost
money, real bank measurements or an economic business case.

## Controls and stress experiments

The model and policies stay frozen for all diagnostics. A logistic model trained on shuffled
training labels is scored against actual validation labels; its AP of .0588 versus prevalence
.057 is a useful sanity check, not a significance test or proof against all leakage.
Three-repeat validation permutation importance and held-out channel error slices are descriptive.

Every stress population contains 4,000 rows in the reference experiment. Seeds are disjoint
from the primary data seed; these populations are never used to choose a model or policy.

| Scenario | Seed offset | Change |
| --- | ---: | --- |
| Fresh reference | +101 | Same population, independent sample |
| Covariate shift | +102 | Log-amount mean 4.95, online probability .85, foreign probability .25; same conditional label mechanism |
| Concept shift | +103 | Subtract `1.3 F O`, add `1.0 × 1(device == terminal)` to the latent logit |
| Lower prevalence | +104 | Subtract 1.0 from the latent logit; this is an intercept change, not a pure prior-probability-shift experiment |
| Input perturbation reference | +105 | Matched complete-feature baseline for the next two scenarios |
| Missing cells | +105, masking +106 | Independently mask 20% of each feature's cells, retain original outcomes |
| Unseen device | +105 | Replace all device values with wearable, retain original outcomes |

Missingness and unseen-device perturbations are paired with the +105 baseline. Unknown-device
replacement is an input perturbation, not a coherent simulation of a new device's fraud process.
Covariate shift also changes prevalence naturally. AP depends on prevalence, so differences
across populations cannot all be attributed to better/worse ranking. A higher F1 under a shift
does not establish robustness. The reference uses one seed and one holdout, with no confidence
intervals or repeated-split uncertainty estimates. No result establishes real-world usefulness.

Methodology references: [sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[metrics definitions](https://scikit-learn.org/stable/modules/model_evaluation.html),
[model persistence and trusted files](https://scikit-learn.org/stable/model_persistence.html).
