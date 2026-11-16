# PhishXAI - Experiment Log

No experiments run yet. Entries must be added as work proceeds;
no backfilling or fabricated results.

## Format

- **Date** - **Block** - description, dataset version, seed, config, results, notes.

## Entries

(none yet)

- **2026-11-17** - **B2** - dataset_v1 built: OpenPhish feed (300 phishing, observed) + 300 constructed legit URLs (Tranco JZWZY top-1000, seeded 42, 40% bare / 60% path-constructed, 1 collision weebly.com excluded). PSL registered-domain extraction (8 exception rules ignored). Domain-grouped split 80/10/10 -> train 479 / val 61 / test 60; registered-domain overlap across splits: NONE. No model training performed. See docs/DATASET_PROVENANCE.md + data/processed/dataset_v1/manifest.json (data/ excluded from git).
