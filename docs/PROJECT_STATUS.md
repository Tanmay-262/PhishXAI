# PhishXAI - Project Status

**Current phase:** B4 complete. Awaiting user approval to start B5.

## Completed

- [x] B1: repo, venv (Py 3.14), stack, repro check
- [x] B2: dataset_v1 (600 rows, domain-grouped 80/10/10, leakage-verified)
- [x] B3: registry v1 (19 lexical features), PSL extractor, 15 unit tests,
      path-artifact diagnostic on val
- [x] B4: LR/RF/XGB trained on train; thresholds from val F1; lr_v1 selected
      (val PR-AUC 0.9991, decision recorded pre-test); test evaluated once;
      bundles saved + fresh-process verified; error analysis done

## Pending

- [ ] B5: SHAP local explanations + template translator (needs user approval)
- [ ] B6: minimal Streamlit dashboard
- [ ] B7: docs, tests, final commit

## Known issues / limitations

- Legit side fully constructed (B2); 1 FP is a constructed shopee URL -
  expected artifact, documented.
- Shared test FN uspostaldirect[.]com (bare-domain phishing) - all 3 models
  miss it; likely over-reliance on popular-domain/short-domain cues.
- val n=61 / test n=60: per-metric differences below ~5 points are not
  statistically meaningful; selection rule is a documented tiebreak, not
  a tight test.
- No probability calibration (deadline scoping); raw probabilities used.

## Next actions

- User approves B4 -> B5: TreeSHAP on lr_v1 (selected) + translator +
      faithfulness tests.
