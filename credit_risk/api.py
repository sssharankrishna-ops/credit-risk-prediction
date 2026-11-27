"""FastAPI service for credit-risk decision support."""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.data import FEATURES
from credit_risk.explanation import explain_prediction
from credit_risk.model import get_risk_model

app = FastAPI(
    title="Explainable Credit Risk API",
    description=(
        "Demonstration decision-support API. Its predictions are not lending decisions; "
        "final decisions require authorized human review."
    ),
)


class Applicant(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    annual_income: float = Field(ge=0, le=10_000_000)
    loan_amount: float = Field(ge=0, le=5_000_000)
    credit_score: float = Field(ge=300, le=850)
    debt_to_income: float = Field(ge=0, le=1)
    employment_years: float = Field(ge=0, le=60)
    late_payments: float = Field(ge=0, le=50)
    credit_utilization: float = Field(ge=0, le=1)
    savings_balance: float = Field(ge=0, le=10_000_000)
    loan_term_months: float = Field(ge=1, le=600)


class Prediction(BaseModel):
    probability_of_default: float
    risk_category: str
    key_factors: list[dict]
    explanation: str
    model: str
    training_source: str
    decision_support_disclaimer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/metrics")
def model_metrics() -> dict:
    model = get_risk_model()
    return {
        "selected_model": model.leaderboard[0]["model"],
        "training_source": model.training_source,
        "models": model.leaderboard,
        "features": list(FEATURES),
    }


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> dict:
    result = get_risk_model().predict(applicant.model_dump())
    result["explanation"] = explain_prediction(result)
    return result
