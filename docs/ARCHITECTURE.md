# PhishXAI - System Architecture

## 1. Problem statement and objectives

Phishing URLs are a leading initial vector for credential theft. A user
facing a suspicious link needs two answers: (1) is this URL likely
phishing, and (2) why did the classifier say so? PhishXAI is an
explainable, reproducible system that classifies URLs as phishing /
legitimate with classical ML over lexical URL features and explains every
prediction with SHAP contributions translated into plain-language,
registry-driven sentences. The contribution is the explanation layer, not
just the classifier.

## 2. Pipelines

### 2.1 Build-time (offline, run per dataset/model version)

    data sources (OpenPhish feed, Tranco list, PSL)
        -> scripts/build_dataset_v1.py  (clean, dedupe, label, PSL
           registered-domain extraction, domain-grouped 80/10/10 split)
        -> data/processed/dataset_v1/{train,val,test}.csv + manifest.json
        -> scripts/train_models.py  (extract 19 features, train LR/RF/XGB,
           val-F1 thresholds, pre-test selection, one test evaluation)
        -> artifacts/models/*.pkl + artifacts/predictions/*.csv
           + artifacts/selection_decision.json + artifacts/b4_summary.json
        -> scripts/explain_b5.py  (global SHAP on val, local cases,
           coverage) -> results/b5/*

### 2.2 Runtime (demo)

    user URL (text)
        -> DashboardCore.analyze  (feature extraction via the B3
           extractor - deterministic, no network)
        -> lr_v1 pipeline (StandardScaler -> LogisticRegression)
           -> P(phishing), verdict at the recorded threshold
        -> PhishShapExplainer (shap.LinearExplainer, log-odds space)
        -> translate() (registry-driven, value-aware phrases)
        -> Streamlit pages render verdict / SHAP chart / sentences / metrics

Streamlit is a thin presentation layer: it owns no feature, prediction,
SHAP or translation logic. That logic lives in src/phishxai so the
dashboard can be swapped or removed without touching the core.

## 3. Module responsibilities

| Module | Responsibility |
|---|---|
| src/phishxai/features/psl.py | PSL parsing; registered-domain (eTLD+1) computation; no network |
| src/phishxai/features/extractor.py | Registry-driven deterministic extraction of the 19 lexical features from a raw URL string; malformed-input safety |
| src/phishxai/features/registry.yaml | SINGLE SOURCE OF TRUTH: feature names/order/types, human names, direction-aware + value-aware phrases. Consumed by extractor, translator and dashboard |
| scripts/train_models.py | B4 training/selection/evaluation protocol (val-only selection, one test pass) |
| src/phishxai/explain/shap_explainer.py | shap.LinearExplainer wrapper for the scaled LR; log-odds space; additivity checks; LocalExplanation |
| src/phishxai/explain/translator.py | Deterministic, registry-driven translation of top-k contributions; fallback + coverage; grouped presentation concepts |
| src/phishxai/dashboard_core.py | Thin orchestrator for the dashboard: loads bundle once, wires extractor -> model -> SHAP -> translator; defanged display helpers |
| app/streamlit_app.py | 3-tab UI only: Detection, Explainability, Model Performance |
| scripts/explain_b5.py | Global/local SHAP analysis, coverage measurement |
| scripts/verify_artifact.py | Fresh-process artifact reload + prediction consistency |

## 4. Data flow and contracts

- **Feature vector:** 19 features, order = registry entry order
  (url_length, domain_length, num_dots, num_subdomains, num_hyphens,
  num_digits, digit_in_domain_ratio, num_special_chars, has_ip_host,
  has_https, has_port, has_at_symbol, has_shortener, path_length,
  has_path, query_length, num_url_params, percent_encoded_count,
  has_suspicious_keyword). Identical at training and inference time.
- **Dataset row:** url, label, source, url_kind, path_constructed,
  registered_domain, extra. Only `url` feeds the model; the provenance
  columns are documentation, never model inputs.
- **Model bundle (joblib):** estimator pipeline, feature_names (ordered),
  threshold + threshold rule, config note, seed, val/test metrics,
  timestamp. Loaded by the explainer and dashboard.
- **LocalExplanation:** url, probability, verdict, threshold, base_value
  (log-odds), decision_function (log-odds), per-feature SHAP (log-odds),
  top-k helpers. Additivity: base + sum(SHAP) = decision_function,
  verified to 1e-4 (machine precision ~1e-15 on val).
- **Translation:** ranked top-k sentences, each {feature, group,
  direction, text, shap_value, is_fallback} + coverage + fixed
  disclaimer.

## 5. Safety invariants

- No network access anywhere in the feature pipeline or dashboard;
  enforced by socket-monkeypatch tests and verified in a denial run.
- Submitted URLs are never fetched, visited, resolved or scanned.
- URLs displayed in the UI are defanged and explicitly not clickable.
- A "legitimate" verdict is rendered as a reduced-phishing-signal
  prediction, never as a safety guarantee.
- Raw datasets and model artifacts are git-ignored; only code, docs and
  small results are tracked.
