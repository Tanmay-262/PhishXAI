# PhishXAI - Explainable AI for Phishing URL Detection

ML-based phishing URL detection (lexical features + Logistic Regression /
Random Forest / XGBoost) with SHAP-based local explanations translated into
plain-language reasons.

## Status

Emergency MVP build targeting the 2026-11-18 submission deadline. See
docs/PROJECT_STATUS.md for phase progress.

## Setup

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pytest tests

## Honest limitations (current state)

- Reproducibility verified on a single Windows machine (Python 3.14.0); no CI.
- FastAPI layer and the human user study are deferred (deadline scoping).
- No model has been trained yet as of B1; see docs/EXPERIMENT_LOG.md.
