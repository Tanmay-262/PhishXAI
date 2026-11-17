# PhishXAI - Reproducibility Record (B7)

Performed 2026-11-17 on the project machine (Windows, Python 3.14.0).

## What was verified in a CLEAN environment

A throwaway virtual environment (`.venv_clean`) was created from the system
interpreter and populated only from `requirements.txt` (pinned versions).
All checks below ran with that clean venv:

| Step | Result |
|---|---|
| `pip install -r requirements.txt` (12 pinned packages) | all installed; combined import smoke test passed |
| Re-download source files and verify against `data/processed/dataset_v1/manifest.json` SHA-256 | Tranco JZWZY top-10k: **MATCH** (byte-identical, pinned list ID). PSL: differs (living list, CC0; the pinned copy at `src/phishxai/features/assets/` is what the code uses - reproducible). OpenPhish feed: differs (live feed; only the preserved `data/raw/openphish_feed_2026-11-17.txt` snapshot reproduces the dataset - expected and documented) |
| Rebuild dataset: `python scripts/build_dataset_v1.py` | train/val/test CSVs **bit-identical** to the pre-B7 files (SHA-256 compared against `repro_hashes_before.txt`), registered-domain overlap still NONE |
| Retrain: `python scripts/train_models.py` | selection re-derived identically: lr_v1 wins on val PR-AUC; ranking [lr, rf, xgb] and val metrics match B4 exactly; the B4 selection timestamp was restored and the re-verification recorded as `b7_reverified_at_utc` |
| `python scripts/verify_artifact.py {lr,rf,xgb}_v1` | all three: MATCH=true (fresh-process reload, identical probabilities) |
| `python scripts/explain_b5.py` | additivity residual 1.8e-15; coverage 100% (61/61) - results/b5/ regenerated identically |
| `pytest tests` (clean venv) | 42 passed |

## What was NOT verified (honest limits)

- Not verified on another machine, OS, or Python version; no CI exists.
- PSL and OpenPhish re-downloads cannot be byte-identical by nature (living
  list / live feed). Reproducibility of the dataset therefore depends on
  the preserved `data/raw/` snapshots (SHA-256 documented in the manifest);
  re-downloading today would yield a different, newer dataset version.
- The Tranco list is reproducible because list IDs are permanent.
- `git log` timestamps and filesystem paths are machine-specific.

## How to reproduce on THIS machine

    .venv\Scripts\python -m pytest tests
    .venv\Scripts\python scripts\build_dataset_v1.py
    .venv\Scripts\python scripts\train_models.py
    .venv\Scripts\python scripts\explain_b5.py
    .venv\Scripts\streamlit run app\streamlit_app.py
