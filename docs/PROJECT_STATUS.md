# PhishXAI - Project Status

**Current phase:** B2 complete. Awaiting user approval to start B3.

## Completed

- [x] B1: git repo, venv (Py 3.14, gate passed), full stack, repro check, commits 4bec1aa + ec47978
- [x] B2: downloaded OpenPhish feed (300 URLs), Tranco JZWZY top-10k, PSL;
      builder script produced dataset_v1 (600 rows, domain-grouped 80/10/10,
      zero registered-domain overlap, constructed URLs flagged); provenance
      written to docs/DATASET_PROVENANCE.md

## Pending

- [ ] B3: lexical feature extraction + registry
- [ ] B4: train & evaluate baseline classifier (user approval gate after)
- [ ] B5: SHAP local explanations + template translator
- [ ] B6: minimal Streamlit dashboard
- [ ] B7: docs, tests, final commit

## Known issues

- matplotlib font-cache warning (harmless).
- Tranco legacy top-10k URL 404s; current per-list-ID endpoint used instead.
- Legit side is fully constructed (0 observed URLs) - documented limitation.
- val/test splits are thin (61 / 60 rows) - report uncertainty honestly.

## Next actions

- User reviews B2 numbers -> approval -> B3 (registry + extractor + unit tests).
