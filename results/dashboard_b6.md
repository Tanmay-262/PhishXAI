# B6 Dashboard Report

Thin Streamlit interface over the PhishXAI package. All prediction, SHAP and
translation logic is delegated to phishxai.dashboard_core -> the B3
extractor, the B4 lr_v1 bundle, and the B5 explainer/translator. Nothing is
duplicated inside Streamlit; the app file only renders.

## Pages

Page 1 - Detection
- text input, Analyze button; empty/malformed/unsupported-scheme inputs are
  handled in-stream (no crash).
- Shows verdict, phishing probability, threshold, model name, and the URL
  in DEFANGED form ("do not click").
- Verdict is labeled a model prediction, not verified ground truth; a
  "legitimate" verdict is never described as guaranteed safe.
- Top contributing factors (top-5) with coverage %, plus an expander with
  all 19 feature values. Fixed limitation disclaimer on the page.

Page 2 - Explainability
- Per-feature SHAP bar chart (log-odds contributions; color splits
  toward/away from phishing).
- Positive and negative contribution lists.
- Registry-driven plain-language sentences (value-aware phrases, registry
  v1.2) + coverage + fixed disclaimer.
- Grouped-concepts table (signed SHAP sums; presentation only, raw values
  authoritative).
- Additivity check line: base + sum(SHAP) vs decision function, with
  residual and tolerance.

Page 3 - Model Performance
- Recorded B4 metrics (artifacts/b4_summary.json + selection_decision.json):
  validation table (used for selection, n=61) and test table (report-only,
  n=60) shown SEPARATELY with sample and class counts.
- lr_v1 selected pre-test (decision timestamp shown).
- Confusion matrices for the selected model (val + test).
- Prominent limitations block: constructed legitimate class, small splits,
  bare-domain blind spot, no calibration / host features.

## Safety

- No network access: URLs are never fetched, visited, resolved or scanned.
  Verified by tests/test_dashboard.py (socket.socket +
  create_connection monkeypatched to raise for the whole module) and by the
  7-scenario verification run under socket denial (B6 report checkpoint).
- Missing artifacts produce actionable st.error messages with the exact
  command to regenerate them - no raw stack traces or filesystem paths in
  normal operation.
- Model loaded ONCE via st.cache_resource.

## Verification

- 5 required scenarios (phishing val URL, legit val URL, bare-domain FN,
  constructed-legit FP, empty/malformed/unsupported input): predictions
  match the B4-recorded probabilities, additivity residual <= 1e-4,
  explanation sentences correspond to the top-5 |contributions|, coverage
  100% in all cases.
- Headless boot: streamlit run app\streamlit_app.py --server.headless ->
  HTTP 200 on port 8599, then AppTest (streamlit.testing.v1) executes the
  full script (all 3 tabs): 0 exceptions.
- tests/test_dashboard.py: 10/10 passing; full suite 41/41.

## How to run from a fresh environment

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    # regenerate data + model artifacts (single-machine; see
    # docs/DATASET_PROVENANCE.md for the 3 source downloads)
    .venv\Scripts\python scripts\build_dataset_v1.py
    .venv\Scripts\python scripts\train_models.py
    .venv\Scripts\streamlit run app\streamlit_app.py
