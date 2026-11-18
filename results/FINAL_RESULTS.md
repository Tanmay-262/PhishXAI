# PhishXAI - Final Results (B7)

All numbers below come from the recorded artifacts
(artifacts/selection_decision.json, artifacts/b4_summary.json,
results/b5/*). Nothing here is a new measurement; this page consolidates
them for the report and viva.

## 1. Dataset (v1)

- Phishing: 300 observed URLs from the OpenPhish feed (collected
  2026-11-17; raw file SHA-256 in docs/DATASET_PROVENANCE.md).
- Legitimate: 300 CONSTRUCTED URLs on Tranco top-1000 domains
  (40% bare-domain, 60% path-constructed from a documented 13-path pool,
  seed 42). **These are not crawled or independently verified URLs; they
  are a disclosed construction method for building a legitimate-class
  baseline.** No Tranco URL was visited.
- Split: registered-domain-grouped (PSL eTLD+1) 80/10/10 -> train 479 /
  val 61 / test 60. Zero registered-domain overlap across splits
  (verified in the manifest and re-asserted by the training script).

## 2. Features

19 lexical features (registry v1.3) extracted from the URL string only;
PSL-based registered-domain computation; no network, no dataset metadata
used as model inputs. Known collinear pairs (from B3):
url_length<->domain_length, num_dots<->num_subdomains,
num_digits<->digit_in_domain_ratio, query_length<->num_url_params.

## 3. Model selection (protocol)

- Train: train split only. Seed 42, documented minimal configs.
- Threshold: argmax F1 on val probabilities (ties: higher recall, then
  lower threshold). LR 0.2994 / RF 0.45 / XGB 0.3424.
- Selection metric: val PR-AUC, tie-break val F1. Decision: **lr_v1**,
  recorded at 2026-11-16T21:59:26Z, BEFORE any test access (file:
  artifacts/selection_decision.json - timestamp preserved, not altered).
- Caveat (stated in the artifact itself): val n=61 - a few points of
  metric difference are within uncertainty; the rule is a documented
  decision, not a statistically tight comparison.

## 4. Final metrics (unaltered, from artifacts)

### Validation (n=61; 33 phishing / 28 legitimate) - used for selection

| model | PR-AUC | F1 | accuracy | FPR@thr |
|---|---|---|---|---|
| lr_v1 (selected) | 0.9991 | 0.9851 | 0.9836 | 0.0357 |
| rf_v1 | 0.9955 | 0.9706 | 0.9672 | 0.0714 |
| xgb_v1 | 0.9937 | 0.9706 | 0.9672 | 0.0714 |

### Test (n=60; 29 phishing / 31 legitimate) - final report only

| model | PR-AUC | F1 | accuracy | FPR@thr |
|---|---|---|---|---|
| lr_v1 | 0.9937 | 0.9655 | 0.9667 | 0.0323 |
| rf_v1 | 0.9958 | 0.9825 | 0.9833 | 0.0 |
| xgb_v1 | 0.9989 | 0.9825 | 0.9833 | 0.0 |

Honest note (kept in B4 report): RF/XGB post higher test PR-AUC than LR;
LR was still the selected model because selection was val-only per
protocol. This is a documented, reproducible decision boundary.

## 5. Explainability (B5)

- Explainer: shap.LinearExplainer on the fitted LR, scaler-transformed
  inputs, LOG-ODDS space; background = full train split (n=479), seed 42;
  base value 2.620683; additivity max residual on val 1.8e-15.
- Global importance on val: domain_length (1.9148) > url_length (1.3548) >
  num_dots (1.2749) > has_https (0.8276) > num_hyphens (0.8197) >
  num_subdomains (0.6192) > num_digits (0.4885) > digit_in_domain_ratio
  (0.3982) - mean |SHAP|; all 19 retained; grouped presentation concepts
  provided alongside raw values.
- Top-5 explanation coverage on val: 100% (61/61) with registry v1.3
  value-aware phrases.
- Documented cases: the bare-domain false negative uspostaldirect[.]com
  (p=0.0732) and the constructed-URL false positive
  shopee[.]co[.]id/images/hero.png (p=0.6752) - both explained faithfully
  and disclosed as limitations, not hidden.

## 6. Dashboard (B6)

Three tabs over DashboardCore (thin wrapper, model loaded once):
Detection (defanged display, no URL fetching - enforced by socket-denial
tests), Explainability (log-odds SHAP chart + sentences + grouped
concepts + additivity line), Model Performance (val/test shown
separately with the pre-test selection decision and the limitations
block). Verified: headless boot HTTP 200; AppTest full-script run 0
exceptions; 10/10 dashboard tests at B6 (the AppTest click-through test was
added at B7, so the current suite is 11/11 dashboard tests, 43/43 overall).

## 6b. Registry v1.3 (post-B7 phrase-only fix)

The explanation registry was moved to v1.3: has_path and has_shortener
were made value-aware so their sentences always agree with the observed
flag (a sign-of-SHAP selection can otherwise emit a sentence that
contradicts the value - e.g. the old phrase 'includes a long path' was
shown for the bare-domain uspostaldirect[.]com false negative, reuters[.]com
and cdc[.]gov). This is a phrase-only change: feature names/order, the
trained lr_v1 model and every numeric result in this file are unchanged;
the B5 outputs were regenerated and diffed (only that one sentence text
changed; coverage 100% and additivity identical).

## 7. Limitations (report-ready)

1. Legitimate class is 100% constructed (Tranco domains + documented path
   pool); FPR estimates and the shopee FP are artifacts of that design.
2. Small splits (val 61 / test 60): single samples move metrics by
   2-5 points; uncertainty must be stated with every number.
3. OpenPhish labels are reported sightings, not per-URL verified.
4. Lexical-only features: bare-domain phishing URLs are the systematic
   blind spot (1 FN); host-based features (domain age, DNS, certificate)
   are the documented future work.
5. No probability calibration; thresholds absorb the need for verdicts.
6. SHAP credit is redistributed among correlated features (B3/B5); all
   19 features are retained and grouped sums are provided.
7. PSL exception (!) rules ignored (8 of ~25k rules - negligible here).
8. Reproducibility verified on one Windows machine (Python 3.14.0); no CI.
9. Deferred (deadline scoping): FastAPI layer, human user study, host
   features, extensive tuning, six-page dashboard.

## 8. How to demonstrate (demo script)

1. Launch: .venv\Scripts\streamlit run app\streamlit_app.py
2. Detection: paste a suspicious URL (defanged in the report); show the
   verdict, probability, threshold, and top factors; point out the
   "model prediction, not ground truth" caption.
3. Explainability: same URL - SHAP log-odds chart, sentences, grouped
   concepts, additivity line.
4. Model Performance: point at the val-vs-test separation and the
   pre-test selection timestamp; read the limitations block.
5. Honest beat: enter the uspostaldirect[.]com URL and show the system
   explaining its OWN miss - the strongest demo moment for explainability.
