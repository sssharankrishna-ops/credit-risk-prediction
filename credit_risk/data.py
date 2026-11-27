"""Demo and SQLite-backed training data loaders."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "annual_income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "late_payments",
    "credit_utilization",
    "savings_balance",
    "loan_term_months",
)
TARGET = "defaulted"


def generate_synthetic_data(
    n_samples: int = 2_000, random_state: int = 42
) -> pd.DataFrame:
    """Create reproducible demonstration data; it is not real applicant data."""
    rng = np.random.default_rng(random_state)
    annual_income = np.clip(rng.lognormal(11.0, 0.55, n_samples), 18_000, 300_000)
    loan_amount = np.clip(rng.lognormal(10.0, 0.6, n_samples), 1_000, 100_000)
    credit_score = rng.integers(300, 851, n_samples)
    debt_to_income = rng.beta(2.2, 4.5, n_samples)
    employment_years = np.clip(rng.gamma(2.0, 3.5, n_samples), 0, 40)
    late_payments = np.clip(rng.poisson(0.45, n_samples), 0, 8)
    credit_utilization = rng.beta(2.0, 2.8, n_samples)
    savings_balance = np.clip(rng.lognormal(9.0, 1.0, n_samples), 0, 200_000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_samples)

    log_odds = (
        -1.5
        + 0.00002 * (loan_amount - 0.3 * annual_income)
        + 2.8 * debt_to_income
        - 0.012 * (credit_score - 650)
        + 0.4 * late_payments
        - 0.035 * employment_years
        + 1.5 * credit_utilization
        - 0.000015 * savings_balance
        + 0.001 * (loan_term_months - 36)
    )
    probability = 1 / (1 + np.exp(-log_odds))

    return pd.DataFrame(
        {
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "employment_years": employment_years,
            "late_payments": late_payments,
            "credit_utilization": credit_utilization,
            "savings_balance": savings_balance,
            "loan_term_months": loan_term_months,
            TARGET: rng.binomial(1, probability),
        }
    )


def load_sqlite_training_data(database_path: str | Path) -> pd.DataFrame:
    """Read an applications table from SQLite using the documented feature names."""
    with sqlite3.connect(database_path) as connection:
        data = pd.read_sql_query("SELECT * FROM applications", connection)
    required = {*FEATURES, TARGET}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")
    return data
