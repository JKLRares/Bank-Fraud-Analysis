# Engineering and experimental review

Historical version 0.2 synthetic benchmark documentation. See [REAL_DATA.md](REAL_DATA.md) for the current real-data experiment.

## Assessment

The original coursework is an exploratory synthetic classifier with an amount shortcut.
The [audit](AUDIT.md) records its provenance, exact baseline and confirmed weaknesses.
The current project is a defensible simulation and Python engineering portfolio example,
with explicit limits on its scientific claims. It is not a validated financial fraud system.

The five substantive extensions are a documented noisy simulator, isolated experimental
selection, exact configurable decision policies, paired input/distribution diagnostics,
and a reusable tested CLI with reproducible artifacts. No web service, external dataset,
deep learning framework or unrelated infrastructure was added.

## Adversarial ML review

| Challenge | Evidence and remaining limitation |
| --- | --- |
| Could the original high score be trivial? | The known amount rule exactly reproduces its classification confusion matrix. Original metrics remain historical, never used as claims about the new task. |
| Can IDs or labels leak into the model? | Pipelines select exactly eight declared observable features. The strict inference schema rejects IDs, fraud and extra columns. Latent probability/noise are not exported. |
| Can related rows cross partitions? | Exact feature duplicates and repeated transaction IDs fail before splitting. Membership is disjoint and complete. The simulator has no repeated entity state; near-neighbors are iid, not shared-account records. |
| Is time used to justify a false temporal experiment? | No timestamps or sequences exist in the new benchmark. A sampled hour is an observable covariate. Stratified splitting matches the iid construction. |
| Can preprocessing see validation/test? | Each candidate's imputer, scaler and encoder are fitted on train inside its pipeline. Tests compare fitted medians to training medians and check they stay unchanged after scoring missing/unseen values. |
| Can test data choose a model or threshold? | Selection accepts only train and validation. Decisions are persisted before test scores. An integration test mutates test labels/features and confirms identical candidate comparison, model choice and thresholds. |
| Does validation tuning have a cost? | Model and thresholds share validation, so its estimates are optimistic. No refit changes the score scale. One holdout supplies the reported final result; no uncertainty intervals are claimed. |
| Was the test used to improve the generator? | The reported generator, model settings and policy objectives were fixed before its first evaluation. Subsequent runs checked reproduction and packaging; no predictive setting was changed based on those scores. |
| Are high scores suspicious? | Selected test AP is .1578, precision .1809 and recall .2237 at the F1 policy. Performance is modest. The shuffled-label logistic control has AP .0588 versus .057 prevalence; this is a sanity check, not proof of validity. |
| Could imbalance be hidden by accuracy? | Dummy accuracy is 94.3% with zero recall. Reports include AP, ROC-AUC, precision, recall, F1, confusion matrices, prevalence and alert volume. No resampling occurs. |
| Is threshold optimization correct? | Every observed boundary and a no-alert sentinel are considered. Ties are indivisible. Tests compare F1, cost, capacity and high-recall objectives to exhaustive metric calculations, including all-zero/all-one scores. |
| Is the cost model a banking claim? | FN=25 and FP=1 are hypothetical constant units. The chosen cost policy alerts 53% of test transactions. Neither operational feasibility nor monetary savings is claimed. |
| Do stress scores establish robustness? | No. Scenarios are invented, held outside selection, and affect prevalence as documented. Missing/unseen input tests use a matched unperturbed population and retain labels. |
| Do explanations establish causes? | No. Three-repeat validation permutation AP drops describe this fitted model; correlated device/channel and distance/foreign features complicate interpretation. |

The main final test contains 228 positives out of 4,000. Its F1-policy matrix is
`[[3541, 231], [177, 51]]`. The model therefore misses most positive labels. AP exceeds the
dummy's .057 baseline, but this alone establishes neither strong utility nor superiority to
all reasonable models. Logistic regression won the predefined validation comparison; the
tree model was not tuned after inspecting the test.

## Python architecture and reliability

The package separates schema validation, generation/splitting, candidate pipelines, metric and
threshold logic, orchestration, artifact handling and command parsing. Dataclasses represent
only concrete configuration, selection results, partition sets and saved models. Core logic
is callable without the CLI. The preserved legacy scripts remain outside the package.

Outputs are staged in sibling temporary paths. Complete files are published exclusively;
run directories reserve a new destination and publish the completed result together. Existing
files, empty directories and symlinks are preserved. Training/write failures clean temporary
work. This is protection against ordinary caught failures, not a durable storage transaction
against process termination or power loss. A force-killed run can leave a reserved directory
or temporary staging directory, which can be inspected and removed manually.

Model bundles preserve the fitted preprocessing and estimator, feature order, schema version,
threshold policies and core dependency versions. Corrupted, incompatible and unsupported
bundles fail clearly. Pickle/joblib is restricted to trusted files; validation cannot prevent
code execution during deserialization of malicious files. The manifest records source and
output SHA-256 values, not cryptographic authenticity.

Concrete defects found during review were fixed with corresponding verification:

- Partial CSV publication on write failure: stage first and publish exclusively; inject a
  disk error and confirm no partial file or staging directory remains.
- Corrupt model decoding produced several possible exceptions, including import failures:
  convert them to a consistent CLI error and exercise empty, malformed and truncated files.
- A default source distribution omitted the shared pytest fixture: explicitly include tests,
  examples, scripts, lock file, workflow and documentation through `MANIFEST.in`.
- Complex numbers could be converted to real numeric features with information loss: reject
  them along with booleans, malformed strings, infinities and out-of-range values.
- The confusion-matrix title was clipped: shorten the title and use contrasting count text;
  this changed only plotting, with the full metrics report remaining identical.

## Verification

The fast suite runs entirely on generated inputs and temporary paths. The verified Python
3.10.12 environment has no broken dependency requirements. The following checks are used:

```bash
python -m pip check
python -m pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
python -m build --no-isolation
```

The source-tree suite passed **65 tests without warnings** (6.76 seconds in a measured run).
Coverage includes determinism and global RNG isolation, schema validation, class/duplicate
invariants, split membership, train-only preprocessing, all-empty columns, threshold ties,
metric calculations, serialization, CLI parsing and output format, corrupted files,
failure cleanup, output preservation and test-contamination resistance. This is behavioral
coverage; no line-coverage percentage was measured.

Clean-distribution verification also passed all **65 tests without warnings** (6.45 seconds).
The source archive was extracted into a new temporary directory, and the built wheel was
installed offline into a separate target. Imports were explicitly checked to come from that
installed wheel, not the original editable source. The archive contained the shared fixture,
lock file and prediction example. Training on 1,200 generated rows and scoring the example
both succeeded outside the checkout with no legacy CSVs or pre-existing model. This reused
the isolated environment's pinned dependencies; it did not reinstall dependencies from the
network. The source archive and wheel both built successfully.

Two independent output directories reproduced the complete reference metrics JSON exactly
under the same environment. One full 20,000-row run took 3.216 seconds on the verification
host after initial plotting setup; this is an observation, not a runtime guarantee. The plot
layout fix preserved the report. Small reference artifacts are in `docs/results/`; datasets,
models and predictions remain under ignored `artifacts/`.

GitHub Actions is configured for Python 3.10 and 3.12. It installs locked dependencies, runs
lint/format/tests, builds a source archive and a wheel, installs the wheel and runs the CLI
outside the source checkout. No hosted Actions run or local Python 3.12 result is claimed.
The lock pins versions, without package hashes; OS, Python and numerical library differences
may affect exact floating-point agreement.

## Reproduction and interpretation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
fraud-benchmark run --rows 20000 --seed 2026 --output artifacts/reference-2026
fraud-benchmark predict --model artifacts/reference-2026/model.joblib \
  --input examples/transactions.json
```

Use a fresh output path when repeating a run. The exact configuration and data SHA-256 are
in `reference_report.json`; the corresponding source/output hashes are in
`reference_manifest.json`. After test inspection, future changes to the generator or
predictive settings require a separately identified evaluation rather than presenting this
known test as fresh evidence. An approved independently sourced dataset would address a
different generalization question and require its own audit, protocol and provenance.
