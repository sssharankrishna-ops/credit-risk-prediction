from fastapi.testclient import TestClient

from credit_risk.api import app

client = TestClient(app)
VALID_APPLICANT = {
    "annual_income": 75000,
    "loan_amount": 20000,
    "credit_score": 680,
    "debt_to_income": 0.3,
    "employment_years": 5,
    "late_payments": 0,
    "credit_utilization": 0.35,
    "savings_balance": 10000,
    "loan_term_months": 36,
}


def test_health_endpoint():
    assert client.get("/health").json() == {"status": "ok"}


def test_prediction_endpoint_returns_explanation_and_disclaimer():
    response = client.post("/predict", json=VALID_APPLICANT)

    assert response.status_code == 200
    prediction = response.json()
    assert 0 <= prediction["probability_of_default"] <= 1
    assert prediction["risk_category"] in {"Low", "Medium", "High"}
    assert prediction["key_factors"]
    assert prediction["explanation"]
    assert "human reviewer" in prediction["decision_support_disclaimer"]


def test_prediction_endpoint_validates_inputs():
    invalid_applicant = {**VALID_APPLICANT, "credit_score": 200}

    assert client.post("/predict", json=invalid_applicant).status_code == 422
