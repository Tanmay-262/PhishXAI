# PhishXAI - Viva Preparation (B7)

## 1. Project summary (say this in ~60 seconds)

We built an explainable phishing URL detection system. A user submits a
URL; the system classifies it as phishing or legitimate using a logistic
regression model over 19 lexical features extracted directly from the URL
string, and then explains the decision using SHAP contributions that a
template-based translator turns into plain-language, value-aware
sentences. The classifier was trained and selected on a domain-grouped
validation split from 300 OpenPhish phishing URLs and 300 constructed
legitimate URLs, and the SHAP layer is validated for additivity and
faithfulness. The headline contribution is the explanation layer: every
sentence maps to an actual feature with a matching contribution sign and
magnitude, and we measure coverage.

## 2. Architecture, in one sentence

An offline build pipeline (data -> features -> models -> artifacts) and a
runtime pipeline (URL -> features -> LR -> SHAP -> translator ->
Streamlit) share a YAML feature registry as the single source of truth,
so the model and the explanation can never disagree about what a feature
is.

## 3. Likely viva questions and technically accurate answers

Q: Why logistic regression if XGBoost posted higher test PR-AUC?
A: Selection was protocol-driven: highest validation PR-AUC (LR 0.9991
vs 0.9955/0.9937), with the decision recorded to a file before the test
split was opened. LR also gives exact, closed-form SHAP contributions in
log-odds space (LinearExplainer), which is cleaner for the
explainability story. The test numbers are report-only and never fed
back into selection.

Q: Why not use a deep learning model or an LLM for explanations?
A: An LLM can invent reasons the model never used, which would turn the
study into measuring persuasion rather than explanation faithfulness.
Template translation is deterministic, testable, and every sentence is
traceable to a feature + contribution sign. Deep models are out of scope
for a lexical URL-string dataset and the deadline.

Q: What do the SHAP numbers actually mean?
A: They are in log-odds (decision-function) space, not probabilities.
Base value is the model's log-odds at the background mean; the per-
feature contributions sum exactly to the difference between that base
and the URL's log-odds (additivity verified to ~1e-15). To read
probabilities you sigmoid-transform.

Q: Are the legitimate URLs real?
A: No - and we say so in every deliverable. They are constructed: Tranco
top-1000 domains with 40% bare-domain and 60% path-constructed URLs from
a documented pool. The model therefore learns "short, well-known
domain = legitimate," which does not transfer to arbitrary real
legitimate URLs. That limitation is explicit in the dataset
provenance and the report.

Q: What is the dataset's biggest artifact, and how did you handle it?
A: Legit URLs having paths while phishing URLs have paths is the risk
(the plan's "path-artifact" warning). We added a path-constructed
legit share, ran a path-only diagnostic (val PR-AUC 0.8007 vs full
0.9963 - path presence alone is not a discriminator; path LENGTH is the
signal), and documented it.

Q: How do you prevent data leakage?
A: Registered-domain-grouped splits via the Public Suffix List - no
domain appears in two splits; the builder verifies disjointness and the
training script re-asserts it. Duplicate and near-duplicate URLs are
deduped with counts in the manifest.

Q: What are the known failure modes?
A: (1) Bare-domain phishing URLs like uspostaldirect[.]com - the model
misses them because lexical features give no strong signal; host
features (domain age, DNS, certificate) are the documented fix.
(2) A high dot count on a multi-level TLD (shopee.co.id) caused the one
LR false positive - a documented artifact of the constructed class.
(3) Correlated features split SHAP credit; we keep all 19 and provide
grouped signed sums, never dropping features.

Q: How is the model reproducible?
A: Seed 42 everywhere; the pipeline, scaler, threshold, feature order,
config and seed are saved in the joblib bundle. A fresh process reloads
the bundle and reproduces identical probabilities (verify_artifact.py,
all three models, MATCH=true). Reproducibility was verified on one
Windows machine; no CI exists.

Q: What would you do with more time?
A: Host-based features (RDAP domain age, DNS, certificate age) with
caching and missing-value handling; probability calibration; a
between-subjects user study comparing verdict-only vs SHAP-plot vs
plain-language conditions; a FastAPI layer; and a larger, crawled
legitimate URL corpus to replace the constructed class.

Q: What does "faithful explanation" mean here?
A: Faithful by construction: each sentence corresponds to a top-k
feature whose SHAP sign matches the sentence's direction, and the text
is pulled from the registry (no invented content). A passing translation
test does NOT prove the model is accurate - it proves the explanation
layer is faithful to the model.

## 4. Application demonstration sequence (5 min)

1. Open the dashboard, Detection tab; paste a phishing URL; read out
   the verdict, probability, threshold and top factors; point to the
   "model prediction, not ground truth" caption.
2. Explainability tab: SHAP log-odds chart, the translated sentences,
   the grouped-concepts table, and the additivity line.
3. Model Performance tab: val vs test tables, the pre-test selection
   timestamp, and the limitations block.
4. Honest beat: enter uspostaldirect[.]com; show the system explaining
   its own miss with the same faithful pipeline.
5. If asked: run the socket-denial test and the B4 prediction-
   reproduction test live.
