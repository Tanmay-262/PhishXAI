# PhishXAI - Project Status

**Current phase:** B7 complete. MVP is submission-ready (subject to the
user's final review of this report). No further blocks are scheduled.

## Completed (all MVP blocks)

- [x] B1: repo, venv (Py 3.14, gate passed), full stack, single-machine repro check
- [x] B2: dataset_v1 (600 rows; 300 OpenPhish phishing + 300 CONSTRUCTED
      Tranco legits, domain-grouped 80/10/10, PSL eTLD+1, zero domain
      overlap; provenance + checksums recorded)
- [x] B3: registry v1 (19 lexical features), PSL extractor, 15 unit
      tests, path-artifact diagnostic on val (path-only PR-AUC 0.8007
      vs full 0.9963)
- [x] B4: LR/RF/XGB trained; val-F1 thresholds; **lr_v1 selected
      pre-test (timestamp 2026-11-16T21:59:26Z preserved)**; one test
      evaluation per model; bundles verified in fresh processes; error
      analysis with defanged examples
- [x] B5: shap.LinearExplainer (log-odds; additivity 1.8e-15), global
      importance on val, 4 representative local cases, registry v1.1
      phrase enrichment, deterministic translator, 15 faithfulness tests
- [x] B6: thin 3-tab Streamlit dashboard over DashboardCore; registry
      v1.2 value-aware phrases; 11 dashboard tests incl. socket-denial
      no-network proof and AppTest click-through; headless boot verified
- [x] B7: ARCHITECTURE/FINAL_RESULTS/VIVA_PREP/REPRODUCIBILITY docs;
      clean-venv reproduction (dataset bit-identical, selection
      re-derived, all bundles MATCH, 42/42 tests in both environments);
      4 rendered page screenshots (defanged URLs); PSL CC0 attribution;
      git hygiene checked; no B4/B5 result altered

## Known issues / limitations (see results/FINAL_RESULTS.md section 7)

- Legit class fully constructed; uspostaldirect[.]com FN; small splits;
  no calibration; no host features; no FastAPI; no user study;
  single-machine reproducibility, no CI.
- PSL and OpenPhish re-downloads are not byte-reproducible (living
  list / live feed) - see docs/REPRODUCIBILITY.md.

## Next actions

- User final review of this B7 report.
- GitHub push done; remote main tracks the deploy-artifact commit d740e89 and the Streamlit Cloud deployment is live at https://phishxai.streamlit.app (see README).
- Optional pre-submission: demo rehearsal (results/VIVA_PREP.md section 4), report writing from
  results/FINAL_RESULTS.md.
