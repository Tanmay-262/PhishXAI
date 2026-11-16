"""B3 path-artifact diagnostic (val split only; test set untouched).

Compares:
  1. path-only  : LR on [has_path, path_length]
  2. full       : LR on all 19 lexical features
Both trained on train, evaluated on val (PR-AUC, accuracy, F1, FPR at 0.5).
Also reports feature-label and feature-feature correlation artifacts.
Diagnostic only - NOT a final model.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phishxai.features.extractor import extract_features_df, FEATURE_NAMES

SEED = 42
DIR = ROOT / "data" / "processed" / "dataset_v1"

def load(split):
    df = pd.read_csv(DIR / f"{split}.csv")
    X = extract_features_df(list(df["url"]))
    y = (df["label"] == "phishing").astype(int).to_numpy()
    return X, y, df

Xtr, ytr, dtr = load("train")
Xva, yva, dva = load("val")

PATH_FEATS = ["has_path", "path_length"]

def fit_eval(cols, name):
    lr = LogisticRegression(max_iter=5000, random_state=SEED)
    lr.fit(Xtr[cols], ytr)
    p = lr.predict_proba(Xva[cols])[:, 1]
    cm = confusion_matrix(yva, (p >= 0.5).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (tn + fp) if (tn + fp) else float("nan")
    out = {"model": name, "pr_auc": round(float(average_precision_score(yva, p)), 4),
           "accuracy": round(float(accuracy_score(yva, p >= 0.5)), 4),
           "f1": round(float(f1_score(yva, (p >= 0.5).astype(int))), 4),
           "fpr_at_0.5": round(fpr, 4),
           "val_phishing": int(yva.sum()), "val_legit": int(len(yva) - yva.sum())}
    print(json.dumps(out, indent=2))
    return out

r_path = fit_eval(PATH_FEATS, "path_only (has_path, path_length)")
r_full = fit_eval(FEATURE_NAMES, "full_lexical (19 features)")

# class-level path distributions on val
print("\nVAL path feature distributions by class:")
va = dva.copy()
va["is_ph"] = va["label"] == "phishing"
va["path_length"] = Xva["path_length"].to_numpy()
print(va.groupby("is_ph")[["path_length"]].agg(
    n=("path_length", "count"),
    avg_path_len=("path_length", "mean"),
    pct_has_path=("path_length", lambda s: float((s > 0).mean())),
    pct_bare_domain=("path_length", lambda s: float((s == 0).mean())),
).round(3).to_string())

# artifact checks
ytr_f = (dtr["label"] == "phishing").astype(int).to_numpy().astype(float)
print("\nTop feature-label correlations on TRAIN (|corr| with label):")
cors = {}
for c in FEATURE_NAMES:
    x = Xtr[c].to_numpy(dtype=float)
    if np.std(x) > 0:
        cors[c] = round(float(np.corrcoef(x, ytr_f)[0, 1]), 3)
for k, v in sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:8]:
    print(f"  {k}: {v}")

print("\nFeature-feature |corr| > 0.8 on TRAIN:")
num = Xtr.to_numpy(dtype=float)
names = FEATURE_NAMES
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if num[:, i].std() > 0 and num[:, j].std() > 0:
            c = abs(float(np.corrcoef(num[:, i], num[:, j])[0, 1]))
            if c > 0.8:
                print(f"  {names[i]} <-> {names[j]}: {c:.3f}")
