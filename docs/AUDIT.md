# Original repository audit

Scope: repository at `9e18a86` on `portfolio-ml-improvements`, before the new benchmark.
All original source, datasets, PDF and text reports are preserved. The initial working tree
already had 43 deleted tracked plot images. Those deletions were left untouched.

## Inventory and behavior

| Item | Confirmed behavior |
| --- | --- |
| `generate_data.py` | Generates 1,000 synthetic transactions using NumPy and pandas; writes `all_data.csv` at import/execution |
| `split_data.py` | Seeded shuffled 75/25 split, no stratification or validation partition |
| `EDA.py` | Drops missing rows, writes descriptive reports and 43 plots, one-hot encodes train/test separately, aligns columns and trains a default random forest |
| `all_data.csv`, `train.csv`, `test.csv` | 1,000 / 750 / 250 rows; ten columns each |
| `randomforest.txt` | Accuracy, classification report and confusion matrix from the historical test |
| `*_descriptive.txt` | Numerical/categorical summaries |
| `README.pdf` | Original Romanian coursework explanation, results and plots |

No notebooks, dependency manifest, supported Python version, automated tests, CI, saved model,
reusable scoring interface or validation-based model selection existed. Imports of the
generator and EDA perform writes. Paths assume the current directory is the repository root.
The original imports require NumPy, pandas, scikit-learn and matplotlib; no versions were declared.

## Provenance and scientific validity

The source code, PDF and regeneration agree: all observations are synthetic. No bank, external
dataset or real customer source is documented. Regeneration with seed 12345 reproduces every
stored field except `date_time`, whose range depends on `datetime.now()`.

- `transaction_id` is a sequence and is excluded by the original model.
- `user_id` is uniform in 1–249, with no persistent account characteristics.
- `amount` is uniform in [1, 3000], rounded to two decimals.
- `date_time` is sampled independently from the preceding year, not sorted transaction history.
- `location`, `device` and `transaction_type` are independent choices from hand-written lists.
- `ip_address` is four independently sampled integers in 0–254. These strings can coincide
  with routable addresses, but they are randomly generated, not observed customer addresses.
- `frequency_last_24h` is Poisson(2), not computed from the rows or an account history.
- `fraud ~ Bernoulli(0.85)` when `amount > 2250`, otherwise Bernoulli(0.03).

The data have learnable signal, but it is a conspicuous single-feature shortcut. There are 195
frauds in 225 transactions above the boundary and 17 in 775 below it. The expected prevalence
is approximately 23.5%; realized prevalence is 21.2%. Neither is a measured financial fraud rate.
Other fields have no designed label relationship. A direct amount rule gives exactly the
random forest's historical classifications. This is generator-rule recovery, not evidence of
real banking generalization. The random forest's extra score ranking within the two bands
does not establish additional population signal.

| Stored partition | Rows | Fraud | Prevalence | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| All | 1000 | 212 | 21.2% | `d83cf331ba2a18cc642f14fa63de238f0f3d9fac327061eeb927f3c098fc0fe3` |
| Train | 750 | 161 | 21.47% | `ed31f25f362d62e1063f93e33deb7dbadc3a4d6a4bd7391952833b26b2f30b61` |
| Test | 250 | 51 | 20.4% | `fdfcd142605a6d1d0d0ea55391b0f240c6854c1d9b5bf86a41a5adfba759208a` |

Confirmed: no missing cells; no repeated transaction IDs, exact rows or feature rows;
train and test union equals the source; split membership matches the original script.
There are 163 shared user IDs, but the generator has no shared account state. This is not
evidence of group leakage under this generator. With meaningful repeated entities, that
conclusion would need to be revisited.

There is no confirmed direct target column in the original feature matrix, no resampling and
no numeric preprocessing fitted on the full dataset. However, independent `get_dummies` calls
with `drop_first=True` can give inconsistent reference encodings when category vocabularies
differ. Unique timestamps and IPs also create many useless columns. Dropping all incomplete
rows is untested and would silently remove predictions. EDA repeatedly examines the test set,
including target-dependent plots. History contains generator probability changes and regenerated
test results, so the old test should not be advertised as a fresh final holdout. History alone
does not establish the reason for those changes.

## Reproduced baseline

Run after installing the documented environment, **from a full repository checkout**:

```bash
python scripts/audit_original.py
python scripts/audit_original.py --run-legacy
```

The first command reports hashes, split invariants and model/rule metrics. The second also
copies the original files to a temporary directory and executes EDA, generation and splitting
there, preserving the original checkout. The wheel/source distribution's new benchmark does
not need the legacy CSVs; this historical audit does.

| Historical test, 250 rows | Accuracy | Precision | Recall | F1 | Average precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always legitimate | .7960 | .0000 | .0000 | .0000 | .2040 | .5000 |
| Original random forest | .9520 | .8545 | .9216 | .8868 | .8927 | .9463 |
| Known generator amount rule | .9520 | .8545 | .9216 | .8868 | .8035 | .9407 |

Both positive classifiers give `[[191, 8], [4, 47]]` (rows actual, columns predicted;
class order legitimate/fraud). AP and ROC-AUC were added by the audit, not recorded originally.
The known rule is a diagnostic using generator knowledge, not a trained competitor.

The original workflow ran successfully on host Python 3.10.12: EDA 100.31 s, generation 1.06 s,
split 0.90 s. EDA included a new matplotlib font cache. All 43 plots and the saved 95.2% result
were reproduced. The host warned that SciPy 1.8.0 requires NumPy <1.25 while NumPy 1.26.4
was installed. Pytest was absent. These are environment observations, not algorithm defects;
the new isolated environment fixes them. Initial sandbox DNS prevented package installation;
authorized network access then installed packages only in `.venv`.

## Git and publication inspection

The existing 13 commits and remote were inspected and preserved. Historical paths contain only
the project scripts, reports, generated CSVs and images. The largest historical blob is the
845,697-byte PDF. A targeted scan of 41 historical Python/CSV/text blobs found no private-key,
AWS access-key, GitHub-token or quoted-password patterns. This limited pattern scan is not an
exhaustive secret audit. Existing Git author contact metadata remains in history.

Generated CSVs, images and the PDF are already tracked: adding `.gitignore` does not remove
them or their history. The new benchmark writes datasets, models and run files under ignored
`artifacts/`. Small, measured reference reports and two plots in `docs/results/` are deliberate
publication artifacts. No external dataset was downloaded; no real financial information was
accessed. No files from the original project were deleted, and no commit, staging operation,
push or history rewrite was performed as part of these changes.

## Engineering decision

The original is useful coursework, but too simple as a standalone ML portfolio claim. Preserve
it as provenance and build a separate, versioned simulation benchmark. The useful additions
are experimental isolation, exact decision-policy selection, reproducibility, perturbation
analysis and reliable Python interfaces. Higher synthetic scores are not the objective.
A legitimate independently sourced dataset, after explicit approval and review of its terms
and task semantics, would be a separate future experiment; it must not be mixed into this one.
