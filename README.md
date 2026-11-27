# AI-Powered Credit Risk Prediction System

> **Predicting Risk. Empowering Smarter Lending.**

An industry-oriented FinTech decision-support prototype that estimates a loan
applicant's **Probability of Default (PD)**, classifies risk, and explains its
predictions using generative AI.

> **Responsible AI notice:** This is a *decision-support* prototype for a
> Finance/FinTech internship. It must never be used as an autonomous loan
> approval system, and its predictions are not real customer outcomes.

---

## 1. What the system does

1. Collects borrower and loan information (validated via Pydantic).
2. Stores structured records in SQL (applicants, applications, loans, predictions, models).
3. Cleans and validates financial data (Pandas/NumPy) - never inventing missing values.
4. Performs EDA (Matplotlib/Seaborn) and engineers financial ratios:
   DTI, LTI, LTV, PTI, credit utilisation, loan exposure.
5. Trains and compares Scikit-learn models (Logistic Regression, Decision Tree,
   Random Forest, Gradient Boosting) and a PyTorch neural network.
6. Tracks every experiment in MLflow (parameters, metrics, artifacts, registry).
7. Predicts PD, applies configurable risk bands (Low < 0.20, Medium 0.20-0.50,
   High >= 0.50), and estimates Expected Credit Loss: **ECL = PD x LGD x EAD**
   (all values as decimal proportions).
8. Generates grounded natural-language explanations (Hugging Face Transformers +
   LangChain) from verified model outputs only.
9. Serves predictions through a FastAPI backend and a 6-page Streamlit dashboard.
10. Ships as Docker containers orchestrated with Docker Compose.

## 2. Technology stack (mandatory, fixed)

| Layer | Technology |
|---|---|
| Language / data | Python 3.13, SQL, NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Deep learning | PyTorch |
| MLOps | MLflow |
| Generative AI | Hugging Face Transformers, LangChain |
| Backend / Frontend | FastAPI / Streamlit |
| Version control | Git, GitHub |
| Deployment | Docker, Docker Compose |

## 3. System architecture

```
SQL Database -> validation -> Pandas/NumPy -> EDA -> feature engineering
  -> ML pipeline (Scikit-learn + PyTorch) -> evaluation -> MLflow tracking
  -> PD prediction -> risk classification -> GenAI explanation
  -> FastAPI -> Streamlit dashboard -> Docker deployment
```

## 4. Project structure

```
credit-risk-prediction/
├── data/               # raw/ and processed/ datasets (raw data is git-ignored)
├── notebooks/          # 01_data_exploration, 02_feature_engineering
├── src/
│   ├── data/           # preprocessing.py, feature_engineering.py
│   ├── models/         # train_sklearn.py, train_pytorch.py, evaluate.py, predict.py
│   ├── explainability/ # explanation.py
│   ├── database/       # db.py (schema, engine, CRUD)
│   └── utils/          # config.py, validation helpers
├── api/                # main.py (FastAPI)
├── dashboard/          # app.py (Streamlit)
├── tests/              # pytest suite
├── models/             # trained artifacts (git-ignored)
├── reports/            # EDA figures, evaluation reports
├── mlruns/             # MLflow experiment store (git-ignored)
├── verify_environment.py
├── requirements.txt
└── (Dockerfile, docker-compose.yml arrive in Phase 12)
```

## 5. Setup (Windows, Python 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python verify_environment.py    # must print RESULT: PASS
Copy-Item .env.example .env
```

## 6. Development roadmap (phase tracker)

- [x] **Phase 0** - Requirements analysis, architecture, milestones (kickoff)
- [x] **Phase 1** - Environment setup and project structure
- [ ] **Phase 2** - Dataset inspection and SQL database *(needs: chosen dataset)*
- [ ] **Phase 3** - Data preprocessing and EDA
- [ ] **Phase 4** - Financial feature engineering
- [ ] **Phase 5** - Scikit-learn model development
- [ ] **Phase 6** - PyTorch model development
- [ ] **Phase 7** - Evaluation and model comparison
- [ ] **Phase 8** - MLflow integration
- [ ] **Phase 9** - Generative AI and explanation integration
- [ ] **Phase 10** - FastAPI backend
- [ ] **Phase 11** - Streamlit dashboard
- [ ] **Phase 12** - Docker integration
- [ ] **Phase 13** - Testing and final documentation

## 7. Responsible AI and security commitments

- No protected personal characteristics are used as decision-making features.
- Model probabilities are calibrated before being interpreted as default probabilities.
- The LLM explains verified model outputs only; it never invents borrower facts
  or alters the predicted probability (deterministic fallback if unavailable).
- Applicant data is validated, access-controlled via environment configuration,
  and raw data is never committed to version control.
- All lending decisions remain subject to human review.
