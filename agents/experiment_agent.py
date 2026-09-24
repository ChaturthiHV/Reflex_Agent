"""Experiment Agent.

Runs an actual machine-learning experiment (not a simulated or LLM-guessed
result) and compares the measured metric against the target. This is the
"ground truth" step in the loop — every evaluate/replan decision downstream
is based on a real number from here, not on what a language model thinks
happened.
"""
import time

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svm_rbf": lambda **params: SVC(kernel="rbf", **params),
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def _build_model(model_name: str, params: dict):
    factory = MODEL_REGISTRY.get(model_name)
    if factory is None:
        raise ValueError(f"Unknown model '{model_name}'")
    return factory(**params)


def run(dataset_loader, approach: dict, target_metric: str, target_value: float, random_state: int = 42):
    start = time.time()
    X, y = dataset_loader()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = _build_model(approach["model"], approach["params"])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "f1_macro": round(f1_score(y_test, predictions, average="macro"), 4),
    }
    measured = metrics.get(target_metric)
    if measured is None:
        raise ValueError(f"Unsupported target_metric '{target_metric}'")

    passed = measured >= target_value
    duration = round(time.time() - start, 3)

    return {
        "model": approach["model"],
        "params": approach["params"],
        "metrics": metrics,
        "target_metric": target_metric,
        "target_value": target_value,
        "measured_value": measured,
        "passed": passed,
        "duration_seconds": duration,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
