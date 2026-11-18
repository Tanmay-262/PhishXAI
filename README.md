# PhishXAI - Explainable AI for Phishing URL Detection

ML-based phishing URL detection (lexical features + Logistic Regression /
Random Forest / XGBoost) with SHAP-based local explanations translated into
plain-language reasons.

## Status

Emergency MVP build targeting the 2026-11-18 submission deadline. See
docs/PROJECT_STATUS.md for phase progress. All MVP blocks (B1-B7) are
complete; submission readiness is documented in results/FINAL_RESULTS.md
and results/VIVA_PREP.md.

## Reproducing everything from this machine

    .venv\Scripts\python -m pytest tests

Regenerating data + model + explanation artifacts (needs data/raw/, which
is gitignored - re-download the 3 source files listed in
docs/DATASET_PROVENANCE.md first):

    .venv\Scripts\python scripts\build_dataset_v1.py   # dataset_v1
    .venv\Scripts\python scripts\train_models.py       # B4 bundles + metrics
    .venv\Scripts\python scripts\explain_b5.py         # B5 outputs (results/b5/)
    .venv\Scripts\python scripts\verify_artifact.py lr_v1

## Running the dashboard

    .venv\Scripts\streamlit run app\streamlit_app.py

Three tabs: Detection (verdict + top factors, defanged display, no URL
fetching), Explainability (SHAP log-odds chart, plain-language sentences,
grouped concepts), Model Performance (recorded B4 val/test metrics,
confusion matrices, limitations).

## Setup (fresh machine)

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pytest tests

Python 3.14.0 verified; stack pinned in requirements.txt.

## Honest limitations (current state)

- Reproducibility verified on a single Windows machine; no CI.
- Legitimate class is 100% synthetically constructed (disclosed in
  docs/DATASET_PROVENANCE.md and results/error_analysis_b4.md).
- FastAPI layer and the human user study are deferred (deadline scoping).

## Live deployment

The dashboard is deployed on Streamlit Community Cloud:

    https://phishxai.streamlit.app/

Verified live 2026-11-18. The Cloud project builds from this repository;
the model bundle and the recorded B4/selection artifacts are tracked in-repo
for the deploy (documented .gitignore exception), and the dashboard uses the
zero-mean SHAP background fallback when data/processed is absent (identical
for the scaled linear model). Free plan - the app sleeps when idle, so open
the URL once before a demo to warm it up. Note the deployed build may trail
later commits of this repository; check the commit the Cloud project is pinned
to before demoing.
