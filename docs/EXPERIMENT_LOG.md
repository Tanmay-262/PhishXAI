# PhishXAI - Experiment Log

No experiments run yet. Entries must be added as work proceeds;
no backfilling or fabricated results.

## Format

- **Date** - **Block** - description, dataset version, seed, config, results, notes.

## Entries

(none yet)

- **2026-11-17** - **B2** - dataset_v1 built: OpenPhish feed (300 phishing, observed) + 300 constructed legit URLs (Tranco JZWZY top-1000, seeded 42, 40% bare / 60% path-constructed, 1 collision weebly.com excluded). PSL registered-domain extraction (8 exception rules ignored). Domain-grouped split 80/10/10 -> train 479 / val 61 / test 60; registered-domain overlap across splits: NONE. No model training performed. See docs/DATASET_PROVENANCE.md + data/processed/dataset_v1/manifest.json (data/ excluded from git).

- **2026-11-17** - **B3** - path-artifact diagnostic (val only, LR, seed 42, max_iter 5000): path-only [has_path, path_length] PR-AUC=0.8007, acc=0.6885, F1=0.7532, FPR@0.5=0.5357; full 19-feature LR PR-AUC=0.9963, acc=0.9672, F1=0.9697, FPR@0.5=0.0357 (val: 33 phishing / 28 legit). VAL class means: phishing avg path len 16.6, legit avg 6.6, 100% of both classes have some path. Train label-correlations: domain_length 0.843, num_dots 0.82, url_length 0.762, has_https -0.351, path_length 0.322. Collinearity >0.8: url_length<->domain_length 0.804, domain_length<->num_hyphens 0.841, num_digits<->digit_in_domain_ratio 0.819, query_length<->num_url_params 0.916. Diagnostic only; test split NOT used. No final model trained.
