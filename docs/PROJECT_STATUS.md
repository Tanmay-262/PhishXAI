# PhishXAI - Project Status

**Current phase:** B1 complete. Awaiting user approval to start B2.

## Completed

- [x] B1: git repo initialized at C:\Users\Admin\Desktop\XAI (branch main)
- [x] B1: project venv on Python 3.14.0; compatibility gate passed
      (numpy, pandas, scikit-learn, xgboost, shap all import on 3.14)
- [x] B1: full MVP stack installed and verified (streamlit, plotly,
      matplotlib, seaborn, joblib, pyyaml, pytest)
- [x] B1: reproducibility check - fresh throwaway venv installed from
      requirements.txt, all imports verified, throwaway venv removed
- [x] B1: skeleton committed as 4bec1aa

## Pending

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
- git ownership: .git was created by the sandbox account; fixed via
  git config --global --add safe.directory C:/Users/Admin/Desktop/XAI

## Next actions

- User approval -> B2: download + inspect dataset sources, build dataset v1.
