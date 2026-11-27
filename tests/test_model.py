import sqlite3

import numpy as np
import pytest

from credit_risk.data import FEATURES, generate_synthetic_data, load_sqlite_training_data
from credit_risk.model import train_risk_model


def test_synthetic_data_is_reproducible_and_has_both_labels():
    first = generate_synthetic_data(n_samples=500, random_state=7)
    second = generate_synthetic_data(n_samples=500, random_state=7)

    assert first.equals(second)
    assert set(first.columns) == {*FEATURES, "defaulted"}
    assert set(first["defaulted"].unique()) == {0, 1}


def test_training_compares_models_and_returns_local_risk_factors():
    model = train_risk_model(generate_synthetic_data(n_samples=800))
    applicant = generate_synthetic_data(n_samples=1).iloc[0].to_dict()
    result = model.predict(applicant)

    assert len(model.leaderboard) >= 2
    assert model.leaderboard[0]["model"] in {"logistic_regression", "random_forest", "pytorch_mlp"}
    assert model.leaderboard[0]["selected_for_prediction"]
    assert all("validation_brier_score" in score for score in model.leaderboard)
    assert all("brier_score" in score for score in model.leaderboard)
    assert 0 <= result["probability_of_default"] <= 1
    assert result["risk_category"] in {"Low", "Medium", "High"}
    assert len(result["key_factors"]) == 3
    assert all(factor["direction"] in {"increases", "reduces"} for factor in result["key_factors"])
    assert "human reviewer" in result["decision_support_disclaimer"]


def test_training_rejects_non_binary_target():
    data = generate_synthetic_data(n_samples=400)
    data["defaulted"] = 0

    with pytest.raises(ValueError, match="both binary labels"):
        train_risk_model(data)


def test_prediction_rejects_non_finite_training_features():
    data = generate_synthetic_data(n_samples=400)
    data.loc[0, "annual_income"] = np.inf

    with pytest.raises(ValueError, match="finite numeric"):
        train_risk_model(data)


def test_sqlite_training_loader_reads_applications_table(tmp_path):
    expected = generate_synthetic_data(n_samples=20)
    database_path = tmp_path / "applications.sqlite"
    with sqlite3.connect(database_path) as connection:
        expected.to_sql("applications", connection, index=False)

    actual = load_sqlite_training_data(database_path)

    assert actual.equals(expected)
