"""Training, evaluation, and local explanations for credit-risk predictions."""

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import FEATURES, TARGET, generate_synthetic_data, load_sqlite_training_data

FACTOR_LABELS = {
    "annual_income": "Annual income",
    "loan_amount": "Loan amount",
    "credit_score": "Credit score",
    "debt_to_income": "Debt-to-income ratio",
    "employment_years": "Employment history",
    "late_payments": "Recent late payments",
    "credit_utilization": "Credit utilization",
    "savings_balance": "Savings balance",
    "loan_term_months": "Loan term",
}


@dataclass
class RiskModel:
    estimator: object
    medians: pd.Series
    leaderboard: list[dict[str, float | str]]
    training_source: str

    def predict(self, applicant: dict[str, float]) -> dict:
        row = pd.DataFrame(
            [{name: float(applicant[name]) for name in FEATURES}]
        )
        probability = float(self.estimator.predict_proba(row)[0, 1])
        category = "Low" if probability < 0.10 else "Medium" if probability < 0.30 else "High"

        factors = []
        baseline_probability = probability
        for feature in FEATURES:
            baseline = row.copy()
            baseline.loc[0, feature] = self.medians[feature]
            impact = baseline_probability - float(self.estimator.predict_proba(baseline)[0, 1])
            factors.append(
                {
                    "feature": feature,
                    "label": FACTOR_LABELS[feature],
                    "impact": round(impact, 4),
                    "direction": "increases" if impact > 0 else "reduces",
                    "value": round(float(row.loc[0, feature]), 3),
                    "reference_value": round(float(self.medians[feature]), 3),
                }
            )
        factors.sort(key=lambda factor: abs(factor["impact"]), reverse=True)

        return {
            "probability_of_default": round(probability, 4),
            "risk_category": category,
            "key_factors": factors[:3],
            "model": self.leaderboard[0]["model"],
            "training_source": self.training_source,
            "decision_support_disclaimer": (
                "This estimate is decision support only, not a lending decision. "
                "An authorized human reviewer must make any final decision."
            ),
        }


@dataclass
class CalibratedPredictor:
    estimator: object
    calibrator: LogisticRegression

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        raw_probability = np.clip(
            self.estimator.predict_proba(rows)[:, 1], 1e-6, 1 - 1e-6
        )
        log_odds = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
        positive_probability = self.calibrator.predict_proba(log_odds)[:, 1]
        return np.column_stack((1 - positive_probability, positive_probability))


def _calibrate_estimator(
    estimator: object, x_calibration: pd.DataFrame, y_calibration: pd.Series
) -> CalibratedPredictor:
    raw_probability = np.clip(
        estimator.predict_proba(x_calibration)[:, 1], 1e-6, 1 - 1e-6
    )
    log_odds = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1_000, solver="liblinear")
    calibrator.fit(log_odds, y_calibration)
    return CalibratedPredictor(estimator, calibrator)


def _fit_torch_classifier(x_train: pd.DataFrame, y_train: pd.Series):
    import torch
    from torch import nn

    torch.manual_seed(42)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_tensor = torch.tensor(x_scaled)
    y_tensor = torch.tensor(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
    network = nn.Sequential(
        nn.Linear(len(FEATURES), 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    loss_function = nn.BCEWithLogitsLoss()
    network.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_function(network(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    class TorchPredictor:
        def predict_proba(self, rows):
            network.eval()
            values = scaler.transform(rows).astype(np.float32)
            with torch.no_grad():
                positive = torch.sigmoid(network(torch.tensor(values))).numpy().ravel()
            return np.column_stack((1 - positive, positive))

    return TorchPredictor()


def _record_metrics(name: str, actual: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(actual, probabilities)), 4),
        "accuracy": round(float(accuracy_score(actual, predictions)), 4),
        "brier_score": round(float(brier_score_loss(actual, probabilities)), 4),
    }


def _log_metrics(leaderboard: list[dict]) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-risk-demo")
    with mlflow.start_run():
        selected = next(result for result in leaderboard if result["selected_for_prediction"])
        mlflow.log_param("selected_model", selected["model"])
        for result in leaderboard:
            for metric in ("roc_auc", "accuracy", "brier_score"):
                mlflow.log_metric(f"{result['model']}_test_{metric}", result[metric])
            mlflow.log_metric(
                f"{result['model']}_validation_brier_score",
                result["validation_brier_score"],
            )


def train_risk_model(data: pd.DataFrame | None = None) -> RiskModel:
    """Compare baseline classifiers and return the highest-AUC model."""
    source = "provided training data"
    if data is None:
        database_path = os.getenv("CREDIT_RISK_DB")
        if database_path:
            data = load_sqlite_training_data(database_path)
            source = "SQLite applications table"
        else:
            data = generate_synthetic_data()
            source = "synthetic demonstration data"
    missing = {*FEATURES, TARGET}.difference(data.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")
    if data[TARGET].nunique() != 2 or not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The defaulted target must contain both binary labels 0 and 1.")

    x = data.loc[:, FEATURES].astype(float)
    y = data[TARGET].astype(int)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Training features must contain only finite numeric values.")
    x_development, x_test, y_development, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_train, x_calibration_and_validation, y_train, y_calibration_and_validation = (
        train_test_split(
            x_development,
            y_development,
            test_size=0.25,
            random_state=43,
            stratify=y_development,
        )
    )
    x_calibration, x_validation, y_calibration, y_validation = train_test_split(
        x_calibration_and_validation,
        y_calibration_and_validation,
        test_size=0.5,
        random_state=44,
        stratify=y_calibration_and_validation,
    )

    candidates = [
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1_000, random_state=42, solver="liblinear"
                ),
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=120, min_samples_leaf=3, random_state=42),
        ),
    ]
    try:
        import torch

        candidates.append(
            ("pytorch_mlp", _fit_torch_classifier(x_train, y_train))
        )
    except ImportError:
        pass

    evaluated = []
    for name, estimator in candidates:
        if hasattr(estimator, "fit"):
            estimator.fit(x_train, y_train)
        calibrated = _calibrate_estimator(estimator, x_calibration, y_calibration)
        validation_probabilities = calibrated.predict_proba(x_validation)[:, 1]
        validation_metrics = _record_metrics(
            name, y_validation, validation_probabilities
        )
        evaluated.append((name, calibrated, validation_metrics))

    selected_name = min(
        evaluated, key=lambda result: result[2]["brier_score"]
    )[0]
    leaderboard = []
    selected_estimator = None
    for name, estimator, validation_metrics in evaluated:
        test_probabilities = estimator.predict_proba(x_test)[:, 1]
        metrics = _record_metrics(name, y_test, test_probabilities)
        metrics["validation_roc_auc"] = validation_metrics["roc_auc"]
        metrics["validation_brier_score"] = validation_metrics["brier_score"]
        metrics["selected_for_prediction"] = name == selected_name
        leaderboard.append(metrics)
        if name == selected_name:
            selected_estimator = estimator
    leaderboard.sort(
        key=lambda metrics: (
            not metrics["selected_for_prediction"],
            metrics["validation_brier_score"],
        )
    )
    _log_metrics(leaderboard)
    return RiskModel(
        estimator=selected_estimator,
        medians=x_development.median(),
        leaderboard=leaderboard,
        training_source=source,
    )


@lru_cache(maxsize=1)
def get_risk_model() -> RiskModel:
    return train_risk_model()
