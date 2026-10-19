"""Small fixed candidate set with preprocessing fitted only on training rows."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_benchmark.evaluation import metrics
from fraud_benchmark.schema import CATEGORICAL, CREDIT_MODEL_FEATURES, NUMERIC


def candidates(seed: int, kind: str = "synthetic") -> dict[str, Pipeline]:
    if kind not in {"synthetic", "creditcard"}:
        raise ValueError("Unknown dataset kind")
    estimators = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            learning_rate=0.08,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=seed,
        ),
    }
    if kind == "creditcard":
        estimators = {
            "dummy_prior": estimators["dummy_prior"],
            "logistic_regression": estimators["logistic_regression"],
            "logistic_balanced": LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=seed
            ),
            "hist_gradient_boosting": estimators["hist_gradient_boosting"],
        }
    result = {}
    for name, estimator in estimators.items():
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant", fill_value="missing", keep_empty_features=True
                    ),
                ),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers = [
            ("numeric", numeric, CREDIT_MODEL_FEATURES if kind == "creditcard" else NUMERIC)
        ]
        if kind == "synthetic":
            transformers.append(("categorical", categorical, CATEGORICAL))
        result[name] = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(transformers),
                ),
                ("classifier", estimator),
            ]
        )
    return result


def fit_candidates(
    train,
    validation,
    models: dict[str, Pipeline],
    features: list[str],
    target: str,
    fn_cost: float,
    fp_cost: float,
):
    """Shared selection core. Test data cannot be passed as a third evaluation partition."""
    comparison = {}
    for name, model in models.items():
        model.fit(train[features], train[target])
        scores = model.predict_proba(validation[features])[:, 1]
        comparison[name] = metrics(validation[target], scores, fn_cost=fn_cost, fp_cost=fp_cost)
    if any(result["average_precision"] is None for result in comparison.values()):
        raise ValueError("Validation needs positive labels for model selection")
    name = max(comparison, key=lambda key: comparison[key]["average_precision"])
    return name, comparison
