# PhishXAI - Project Status

**Current phase:** B5 complete. Awaiting user approval to start B6.

## Completed

- [x] B1: repo, venv (Py 3.14), stack, repro check
- [x] B2: dataset_v1 (600 rows, domain-grouped 80/10/10, leakage-verified)
- [x] B3: registry v1 (19 lexical features), PSL extractor, 15 unit tests,
      path-artifact diagnostic on val
- [x] B4: LR/RF/XGB trained on train; val-F1 thresholds; lr_v1 selected
      pre-test; bundles saved + fresh-process verified; error analysis
- [x] B5: shap.LinearExplainer on lr_v1 (log-odds space, train background,
      additivity 1.8e-15), global importance on val, 4 local cases,
      registry v1.1 phrase enrichment, deterministic translator, grouped
      concepts, 15 new faithfulness tests (30 total passing)

## Pending

- [ ] B6: minimal Streamlit dashboard (user approval gate)
- [ ] B7: docs, tests, final commit

## Known issues / limitations

- Legit side fully constructed (B2); 1 FP is a constructed shopee URL.
- uspostaldirect[.]com bare-domain phishing FN shared by all B4 models.
- SHAP: correlated features (url_length/domain_length, num_dots/
  num_subdomains, num_digits/digit_in_domain_ratio,
  query_length/num_url_params) redistribute attribution; all 19 retained,
  grouped presentation concepts provided, raw values always available.
- No probability calibration; no cross-fold SHAP stability check
  (deadline scoping).
- Passing translator tests do NOT prove model accuracy.

## Next actions

- User approves B5 -> B6: minimal dashboard (Detection / Explainability /
      Model Performance pages, defanged display, no URL fetching).
