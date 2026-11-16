# PhishXAI - Project Status

**Current phase:** B6 complete. Awaiting user approval to start B7.

## Completed

- [x] B1: repo, venv (Py 3.14), stack, repro check
- [x] B2: dataset_v1 (600 rows, domain-grouped 80/10/10, leakage-verified)
- [x] B3: registry v1 (19 lexical features), PSL extractor, 15 unit tests,
      path-artifact diagnostic on val
- [x] B4: LR/RF/XGB trained; val-F1 thresholds; lr_v1 selected pre-test;
      bundles + fresh-process verification; error analysis
- [x] B5: shap.LinearExplainer (log-odds, train background, additivity
      1.8e-15), global importance on val, 4 local cases, deterministic
      translator, 15 faithfulness tests
- [x] B6: minimal Streamlit dashboard (Detection / Explainability /
      Model Performance) as a thin wrapper over phishxai.dashboard_core;
      registry v1.2 value-aware phrases; 10 dashboard tests incl. socket-
      denial network proof; headless boot + AppTest: 0 exceptions

## Pending

- [ ] B7: final docs, full test pass, reproducibility sweep, final commit,
      submission README polish

## Known issues / limitations

- Legit side fully constructed (B2); shopee FP and uspostaldirect[.]com
  FN are documented artifacts / blind spots.
- No probability calibration, no host features, no FastAPI, no user
  study (all deferred).
- Reproducibility: single Windows machine, no CI.
- Streamlit deprecation: use_container_width replaced with width=
  (Plotly/streamlit API drift, resolved in B6).

## Next actions

- User approves B6 -> B7.
