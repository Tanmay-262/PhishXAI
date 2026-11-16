# PhishXAI - Project Status

**Current phase:** B1 (Environment & repo) of the emergency MVP plan.

## Completed

- [x] B1: git repo initialized at C:\Users\Admin\Desktop\XAI
- [x] B1: project venv on Python 3.14.0; compatibility gate passed
      (numpy, pandas, scikit-learn, xgboost, shap all import on 3.14)
- [x] B1: full MVP stack installed (streamlit, plotly, matplotlib,
      seaborn, joblib, pyyaml, pytest)

## Pending

- [ ] B1: reproducibility check (throwaway venv + requirements install + imports)
- [ ] B1: commit + verification report to user
- [ ] B2: real dataset acquisition (OpenPhish + Tranco) and validation
- [ ] B3: lexical feature extraction + registry
- [ ] B4: train & evaluate baseline classifier (user approval gate after)
- [ ] B5: SHAP local explanations + template translator
- [ ] B6: minimal Streamlit dashboard
- [ ] B7: docs, tests, final commit

## Known issues

- matplotlib font-cache warning on first import (harmless).
- Sandbox blocks outbound network: pip and dataset downloads go through
  the escalated-network approval flow.

## Next actions

- Finish B1: reproducibility check, one clean commit, report to user.
