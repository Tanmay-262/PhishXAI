# PhishXAI - Explainable AI for Phishing URL Detection

ML-based phishing URL detection (lexical features + Logistic Regression /
Random Forest / XGBoost) with SHAP-based local explanations translated into
plain-language reasons.

## Status

Emergency MVP build targeting the 2026-11-18 submission deadline. See
docs/PROJECT_STATUS.md for phase progress.

## Reproducing artifacts

`artifacts/` (model bundles, prediction CSVs, b4_summary.json) is
gitignored. To regenerate on this machine:

    .venv\Scripts\python scripts\build_dataset_v1.py   # needs data/raw/ (see docs/DATASET_PROVENANCE.md)
    .venv\Scripts\python scripts\train_models.py       # trains + evaluates + saves bundles
    .venv\Scripts\python scripts\verify_artifact.py lr_v1

Raw data files are re-downloadable; sources + checksums are in
docs/DATASET_PROVENANCE.md. On a fresh clone without data/raw/, re-download
the three files listed there first (single-machine reproducibility; no CI).

## Setup

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pytest tests

## Honest limitations (current state)

- Reproducibility verified on a single Windows machine (Python 3.14.0); no CI.
- Legitimate class is synthetically constructed (disclosed in
  docs/DATASET_PROVENANCE.md and results/error_analysis_b4.md).
- FastAPI layer and the human user study are deferred (deadline scoping).
