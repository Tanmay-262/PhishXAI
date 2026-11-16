# PhishXAI - Project Status

**Current phase:** B3 complete. Awaiting user approval to start B4.

## Completed

- [x] B1: repo, venv (Py 3.14), stack, repro check, commits 4bec1aa + ec47978
- [x] B2: dataset_v1 (600 rows, domain-grouped 80/10/10, leakage-verified), commit 6435315
- [x] B3: feature registry v1 (19 lexical features, YAML single source of truth),
      PSL-based extractor (no network, deterministic), 15 unit tests passing,
      path-artifact diagnostic on val split (path-only PR-AUC 0.8007 vs full 0.9963)

## Pending

- [ ] B4: train & evaluate baseline classifier (LR/RF/XGBoost, val for selection;
      test set held out until final evaluation) - user approval gate
- [ ] B5: SHAP local explanations + template translator
- [ ] B6: minimal Streamlit dashboard
- [ ] B7: docs, tests, final commit

## Known issues

- High feature collinearity: url_length/domain_length (0.80),
  domain_length/num_hyphens (0.84), query_length/num_url_params (0.92) -
  will split SHAP credit; flagged for B5 global SHAP review.
- Legit side is constructed (B2 disclosure); path-only diagnostic model
  scores 0.80 PR-AUC, so the full model's 0.9963 must be defended as
  "domain-length driven, not path-presence driven".
- PSL shipped as src asset (CC0, 341 KB) so fresh clones work without data/raw.

## Next actions

- User approves B3 numbers -> B4: baseline training on train, select on val,
  ONE final test evaluation per model, honest metrics + artifact.
