# Explainable Credit Risk Prediction

A small, end-to-end credit-risk decision-support prototype. It compares a logistic
regression baseline with a random forest, calibrates probability of default (PD)
estimates on a separate calibration set, assigns Low/Medium/High risk bands, and
reports local counterfactual factors for each prediction. Model selection uses a
validation set; final comparison metrics come from a separate held-out test set.
Optional extras add a PyTorch model, local Hugging Face Transformer/LangChain
explanations, and MLflow experiment tracking.

> **Important:** The default dataset is synthetic and for demonstration only. This
> prototype is not validated for lending decisions. Its predictions must not be
> used as automated decisions; final decisions require authorized human review.
> Do not use real applicant data without appropriate legal, privacy, fairness, and
> model-risk review.

## Run locally

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn credit_risk.api:app --reload
```

In another terminal, with the same environment activated:

```bash
streamlit run credit_risk/dashboard.py
```

The API is documented at <http://127.0.0.1:8000/docs>. The dashboard defaults to
that API address; set `CREDIT_RISK_API_URL` to use another address.

## API

`POST /predict` accepts these numeric inputs: `annual_income`, `loan_amount`,
`credit_score` (300–850), `debt_to_income` (0–1), `employment_years`,
`late_payments`, `credit_utilization` (0–1), `savings_balance`, and
`loan_term_months`.

The response includes PD, a risk band (Low below 10%, Medium below 30%, otherwise
High), up to three locally estimated risk factors, the selected model, and a
decision-support disclaimer. `GET /model/metrics` returns held-out test ROC AUC,
accuracy, and Brier score plus validation metrics for each candidate model.

## Data, training, and optional integrations

By default, the service trains reproducible synthetic applicant data on first use.
To train from a SQLite database instead, set `CREDIT_RISK_DB` to its path. It must
contain an `applications` table with the input columns above and a binary
`defaulted` column (`1` for default, `0` otherwise). Training-data metrics are
reported for transparency; they do not establish production suitability.

Install optional capabilities as needed:

```bash
pip install -e ".[torch]"           # include a PyTorch MLP in model comparison
pip install -e ".[explainability]"  # local Transformers + LangChain generation
pip install -e ".[tracking]"        # MLflow tracking
```

The generated explanation fallback works without optional NLP packages. To enable
Transformer generation, set `CREDIT_RISK_EXPLANATION_MODEL` to a local
text-to-text model directory already available on disk. Loading is local-only and
disables custom remote model code. To track training runs, set
`MLFLOW_TRACKING_URI` to a trusted MLflow tracking destination.

## Docker

Build and run the API image:

```bash
docker build -t credit-risk-prediction .
docker run --rm -p 8000:8000 credit-risk-prediction
```

Run the dashboard from the same image in a separate container, pointing it at the
API host reachable from that container:

```bash
docker run --rm -p 8501:8501 \
  -e CREDIT_RISK_API_URL=http://host.docker.internal:8000 \
  credit-risk-prediction \
  streamlit run credit_risk/dashboard.py --server.address=0.0.0.0
```

## Tests

```bash
pytest
```
