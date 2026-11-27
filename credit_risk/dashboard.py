"""Streamlit applicant form and model monitoring dashboard."""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("CREDIT_RISK_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Credit Risk Decision Support", page_icon="📊")
st.title("Credit Risk Decision Support")
st.caption(
    "Prototype only: estimates are not lending decisions. Final lending decisions "
    "must be made by an authorized human reviewer."
)

with st.sidebar:
    st.header("Applicant details")
    with st.form("applicant"):
        annual_income = st.number_input("Annual income ($)", min_value=0, value=75_000, step=1_000)
        loan_amount = st.number_input("Loan amount ($)", min_value=0, value=20_000, step=500)
        credit_score = st.slider("Credit score", 300, 850, 680)
        debt_to_income = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.30, 0.01)
        employment_years = st.number_input("Employment years", min_value=0.0, value=5.0, step=1.0)
        late_payments = st.number_input("Late payments (last 12 months)", min_value=0, value=0, step=1)
        credit_utilization = st.slider("Credit utilization", 0.0, 1.0, 0.35, 0.01)
        savings_balance = st.number_input("Savings balance ($)", min_value=0, value=10_000, step=500)
        loan_term_months = st.selectbox("Loan term", [12, 24, 36, 48, 60])
        submitted = st.form_submit_button("Assess risk")

if submitted:
    applicant_data = {
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "employment_years": employment_years,
        "late_payments": late_payments,
        "credit_utilization": credit_utilization,
        "savings_balance": savings_balance,
        "loan_term_months": loan_term_months,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=applicant_data, timeout=15)
        response.raise_for_status()
        prediction = response.json()
        st.metric("Probability of default", f"{prediction['probability_of_default']:.1%}")
        st.subheader(f"{prediction['risk_category']} estimated risk")
        st.write(prediction["explanation"])
        st.dataframe(
            pd.DataFrame(prediction["key_factors"])[
                ["label", "direction", "impact", "value", "reference_value"]
            ],
            hide_index=True,
        )
        st.caption(prediction["decision_support_disclaimer"])
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API at {API_URL}: {error}")

st.divider()
st.subheader("Model monitoring")
try:
    metrics = requests.get(f"{API_URL}/model/metrics", timeout=15)
    metrics.raise_for_status()
    model_data = metrics.json()
    st.caption(
        f"Selected model: {model_data['selected_model']} · "
        f"Training data: {model_data['training_source']}"
    )
    st.dataframe(pd.DataFrame(model_data["models"]), hide_index=True)
except requests.RequestException:
    st.info("Start the API to view model comparison metrics.")
