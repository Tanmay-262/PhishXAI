"""B5: SHAP explainability + translator analysis for lr_v1.

- Explainer: shap.LinearExplainer on the fitted LogisticRegression,
  operating on StandardScaler-transformed inputs (log-odds space).
- Background: TRAIN split only (the data the model was fit on), seed 42.
- Global analysis: VAL split only. The test set is NOT used for any
  explanation logic; two test URLs below are reporting cases (the known
  false negative and the LR false positive).
- Outputs: results/b5/ (JSON + plot + tables).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phishxai.features.extractor import extract_features_df
from phishxai.explain import (PhishShapExplainer, group_concepts,
                              translate, load_registry)

SEED = 42
DS = ROOT / "data" / "processed" / "dataset_v1"
OUT = ROOT / "results" / "b5"
OUT.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(ROOT / "artifacts" / "models" / "lr_v1.pkl")
ex_names = bundle["feature_names"]
assert ex_names == [f["name"] for f in load_registry().values()], "registry order mismatch"

train = pd.read_csv(DS / "train.csv")
val = pd.read_csv(DS / "val.csv")
test = pd.read_csv(DS / "test.csv")
Xtr = extract_features_df(list(train["url"]))
Xva = extract_features_df(list(val["url"]))
Xte = extract_features_df(list(test["url"]))

# Background: full train split, deterministic row order (no sub-sampling
# needed at n=479; seed recorded anyway for reproducibility).
expl = PhishShapExplainer(bundle, background_X=Xtr.to_numpy(dtype=float),
                          background_size=None, seed=SEED)
print("explainer ready | base_value (log-odds) = %.6f" % expl.base_value)

# ---- 1) additivity on all val rows (tol 1e-4) ----
resid = max(expl.check_additivity(x, tol=1e-4) for x in Xva.to_numpy(dtype=float))
print("additivity max residual (val, tol=1e-4): %.3e" % resid)

# ---- 2) determinism: identical input explained twice ----
s0 = Xva.iloc[0].to_numpy(dtype=float)
l1, l2 = expl.explain_one(val["url"].iloc[0], s0), expl.explain_one(val["url"].iloc[0], s0)
det = bool(np.array_equal(l1.shap, l2.shap) and l1.probability == l2.probability)
print("determinism check (identical input twice):", "OK" if det else "FAIL")
assert det

# ---- 3) global importance on VAL ----
sv = expl.explainer.shap_values(expl.scaler.transform(Xva.to_numpy(dtype=float)))
pos = np.asarray(sv[expl.pos_class] if isinstance(sv, (list, tuple)) else sv)
mean_abs = np.abs(pos).mean(axis=0)
ranked = np.argsort(-mean_abs)
table = [{"rank": int(i + 1), "feature": ex_names[j],
          "mean_abs_shap": round(float(mean_abs[j]), 5),
          "mean_signed_shap": round(float(pos[:, j].mean()), 5)}
         for i, j in enumerate(ranked)]
(OUT / "global_importance_val.json").write_text(json.dumps(
    {"dataset": "val split (n=61)", "space": "log-odds",
     "note": "mean |SHAP| over val rows; test set not used; "
             "correlated features may share/redistribute attribution",
     "table": table}, indent=2), encoding="utf-8")
print("\nGlobal importance (val, top 8):")
for r in table[:8]:
    print("  %2d %-26s mean_abs=%.4f signed=%+.4f" %
          (r["rank"], r["feature"], r["mean_abs_shap"], r["mean_signed_shap"]))

import matplotlib
matplotlib.use("Agg")
import shap
import matplotlib.pyplot as plt
shap.summary_plot(pos, expl.scaler.transform(Xva.to_numpy(dtype=float)),
                  feature_names=ex_names, show=False,
                  title="SHAP summary - lr_v1 (val split, log-odds space)")
plt.gcf().savefig(OUT / "shap_summary_val.png", dpi=120)
plt.close("all")
print("summary plot -> results/b5/shap_summary_val.png")

# ---- 4) local explanations: 4 representative cases ----
def case(url, x_row, note):
    le = expl.explain_one(url, x_row)
    reg = load_registry()
    vals = dict(zip(le.feature_names, x_row.tolist()))
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5, registry=reg,
                   feature_values=vals)
    return {"url": url, "note": note, "verdict": le.verdict,
            "probability": round(le.probability, 6), "threshold": le.threshold,
            "base_value": round(le.base_value, 6),
            "decision_function_log_odds": round(le.decision_function, 6),
            "total_shap_sum": round(float(le.shap.sum()), 6),
            "top_positive": [[n, round(v, 4)] for n, v in le.top_positive(5)],
            "top_negative": [[n, round(v, 4)] for n, v in le.top_negative(5)],
            "sentences": [s.text + (" [fallback]" if s.is_fallback else "")
                          for s in tr.sentences],
            "coverage_top5": round(tr.coverage, 4),
            "grouped_concepts": group_concepts(le.shap.tolist(), le.feature_names),
            "disclaimer": tr.disclaimer}

cases = []
ph_probs = expl.pipeline.predict_proba(Xva.to_numpy(dtype=float))[:, 1]
val_mask_ph = (val["label"] == "phishing").to_numpy()
val_mask_leg = (val["label"] == "legitimate").to_numpy()
best_ph = int(np.argmax(np.where(val_mask_ph, ph_probs, -1.0)))
best_leg = int(np.argmin(np.where(val_mask_leg, ph_probs, 2.0)))
cases.append(case(val["url"].iloc[best_ph], Xva.iloc[best_ph].to_numpy(dtype=float),
                 "val: correctly detected phishing (highest val phishing probability)"))
cases.append(case(val["url"].iloc[best_leg], Xva.iloc[best_leg].to_numpy(dtype=float),
                 "val: correctly classified legitimate (lowest phishing probability)"))

fn = int((test["url"] == "https://uspostaldirect.com/").to_numpy().argmax())
assert test["url"].iloc[fn] == "https://uspostaldirect.com/"
cases.append(case("https://uspostaldirect.com/", Xte.iloc[fn].to_numpy(dtype=float),
                 "test: known shared FALSE NEGATIVE (all 3 B4 models missed it)"))

fp_mask = test["url"].str.contains("shopee.co.id/images/hero", regex=False).to_numpy()
fp = int(fp_mask.argmax())
assert fp_mask.any(), "LR false-positive URL not found in test set"
cases.append(case(test["url"].iloc[fp], Xte.iloc[fp].to_numpy(dtype=float),
                 "test: LR FALSE POSITIVE (constructed legitimate URL)"))

(OUT / "local_explanations.json").write_text(json.dumps(
    {"cases": cases,
     "note": "SHAP values are in log-odds space; sentences describe model "
             "behavior only, not causal proof. Raw feature-level "
             "contributions are kept alongside grouped presentation "
             "concepts."}, indent=2), encoding="utf-8")

for c in cases:
    print("\n=== " + c["note"])
    print("  verdict=%s p=%.4f (thr %.4f) base=%.3f decf=%.3f shap_sum=%.3f" %
          (c["verdict"], c["probability"], c["threshold"], c["base_value"],
           c["decision_function_log_odds"], c["total_shap_sum"]))
    print("  toward phishing :", c["top_positive"][:4])
    print("  away from phish :", c["top_negative"][:4])
    for s in c["sentences"]:
        print("  >", s)
    print("  coverage_top5:", c["coverage_top5"])

# ---- 5) coverage on all val rows (top-5) ----
covs = []
for i in range(len(Xva)):
    xrow = Xva.iloc[i].to_numpy(dtype=float)
    le = expl.explain_one(val["url"].iloc[i], xrow)
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5,
                   feature_values=dict(zip(le.feature_names, xrow.tolist())))
    covs.append(tr.coverage)
covs = np.array(covs)
(OUT / "coverage_val.json").write_text(json.dumps(
    {"split": "val (n=61)", "top_k": 5,
     "mean_coverage": round(float(covs.mean()), 4),
     "rows_at_full_coverage": int((covs >= 1.0).sum()),
     "rows_below_full": int((covs < 1.0).sum()),
     "definition": "fraction of top-5 displayed sentences backed by a "
                   "registry phrase (fallbacks tracked, not hidden)"}, indent=2),
    encoding="utf-8")
print("\ncoverage (val, top-5): mean=%.3f full=%d/%d" %
      (covs.mean(), int((covs >= 1.0).sum()), len(covs)))

print("\nB5 outputs:", sorted(p.name for p in OUT.iterdir()))
