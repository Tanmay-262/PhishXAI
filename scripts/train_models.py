"""B4: train, select and evaluate baseline phishing classifiers.

Order is enforced by THIS script (test set is not touched before the
selection decision is recorded):

  1. Load train/val/test, extract the 19 registered features (deterministic,
     registry-driven; no dataset metadata is used as a model input).
  2. Leakage guard: assert registered-domain sets are disjoint across splits.
  3. Train LR (StandardScaler fit on train only), RF, XGBoost on train.
  4. Val: per-model threshold = argmax F1 on val (tie -> higher recall,
     then lower threshold). Val PR-AUC + threshold metrics recorded.
  5. Selection: highest val PR-AUC, tie-break val F1. Written to
     artifacts/selection_decision.json BEFORE any test evaluation runs.
  6. Test: exactly one evaluation per model, at its val-selected threshold.
  7. Save joblib bundles (estimator + preprocessing + feature order +
     threshold + config + seed + metadata) and val/test prediction CSVs.

Run: .venv\\Scripts\\python scripts\\train_models.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                            confusion_matrix, f1_score, precision_score,
                            recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phishxai.features.extractor import FEATURE_NAMES, extract_features_df

SEED = 42
MODEL_TAG = "v1"
DS = ROOT / "data" / "processed" / "dataset_v1"
MODELS_DIR = ROOT / "artifacts" / "models"
PRED_DIR = ROOT / "artifacts" / "predictions"

# Documented, minimal configurations (no grid search; deadline scoping).
CONFIGS = {
    "lr_v1": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                       max_iter=10000, random_state=SEED)),
        ]),
        "note": "StandardScaler fit on TRAIN only (inside the pipeline fit); "
                "required for L2-regularized logistic regression.",
    },
    "rf_v1": {
        "estimator": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "note": "300 trees, default max_features='sqrt'; no scaling needed for trees.",
    },
    "xgb_v1": {
        "estimator": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                       eval_metric="logloss", random_state=SEED),
        "note": "max_depth=4 + lr=0.1 documented as minimal overfit guard on a "
                "small dataset; no tuning beyond these bounds.",
    },
}


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)), 4) if (fp + tn) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_samples": int(len(y)), "n_phishing": int(y.sum()),
        "n_legitimate": int(len(y) - y.sum()),
    }


def pick_threshold(y, p):
    """argmax F1 over all unique val probabilities; ties: higher recall,
    then lower threshold. Purely val-driven; documented in DECISIONS.md."""
    best_key, best_thr = None, None
    for t in np.unique(p):
        m = metrics_at(y, p, float(t))
        key = (m["f1"], m["recall"], -float(t))
        if best_key is None or key > best_key:
            best_key, best_thr = key, float(t)
    return best_thr


def defang(url, max_len=42):
    """Doc-safe rendering: defang dots, truncate long URLs."""
    out = url.replace(".", "[.]")
    return out if len(out) <= max_len else out[:max_len - 3] + "..."


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    splits = {}
    for s in ("train", "val", "test"):
        df = pd.read_csv(DS / f"{s}.csv")
        X = extract_features_df(list(df["url"]))
        assert list(X.columns) == FEATURE_NAMES, "feature order mismatch vs registry"
        y = (df["label"] == "phishing").astype(int).to_numpy()
        splits[s] = {"X": X, "y": y, "df": df}
        print(f"{s}: n={len(df)} phishing={int(y.sum())} legit={int(len(df) - y.sum())}")

    # Leakage guard: registered domains must stay disjoint across splits.
    rds = {s: set(pd.read_csv(DS / f"{s}.csv")["registered_domain"])
           for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (rds[a] & rds[b]), f"registered-domain overlap between {a} and {b}!"
    print("leakage guard OK: registered-domain sets disjoint across train/val/test")

    # ---- train + val (selection phase) --------------------------------
    val_results = {}
    for name, cfg in CONFIGS.items():
        est = cfg["estimator"]
        est.fit(splits["train"]["X"], splits["train"]["y"])
        p_val = est.predict_proba(splits["val"]["X"])[:, 1]
        thr = pick_threshold(splits["val"]["y"], p_val)
        m = metrics_at(splits["val"]["y"], p_val, thr)
        m["pr_auc"] = round(float(average_precision_score(splits["val"]["y"], p_val)), 4)
        m["val_pr_auc_rank"] = None  # filled after all models
        val_results[name] = {"estimator": est, "threshold": thr,
                             "metrics": m, "config_note": cfg["note"],
                             "val_probs": p_val}
        print(f"\n== {name} (val) " + json.dumps(m, indent=1))

    # ---- SELECTION (recorded BEFORE test evaluation) --------------------
    ranked = sorted(val_results, key=lambda n: (val_results[n]["metrics"]["pr_auc"],
                                                 val_results[n]["metrics"]["f1"]),
                    reverse=True)
    selected = ranked[0]
    for i, n in enumerate(ranked):
        val_results[n]["metrics"]["val_pr_auc_rank"] = i + 1
    decision = {
        "selected_model": selected,
        "rule": "highest val PR-AUC; tie-break: val F1 at its selected threshold",
        "val_results": {n: val_results[n]["metrics"] for n in ranked},
        "ranking": ranked,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Recorded before any test-set access. Test metrics below are "
                "final-report-only; they informed no decision.",
        "caveat": "val n=61 and test n=60: differences of a few metric points "
                  "are within uncertainty; the rule is a documented decision, "
                  "not a statistically tight comparison.",
    }
    (ROOT / "artifacts" / "selection_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")
    print(f"\nSELECTION (pre-test, recorded to artifacts/selection_decision.json): {selected}")

    # ---- test evaluation (once per model) ------------------------------
    test_results = {}
    for name in CONFIGS:
        est = val_results[name]["estimator"]
        p_test = est.predict_proba(splits["test"]["X"])[:, 1]
        m = metrics_at(splits["test"]["y"], p_test, val_results[name]["threshold"])
        m["pr_auc"] = round(float(average_precision_score(splits["test"]["y"], p_test)), 4)
        test_results[name] = {"metrics": m, "probs": p_test}
        print(f"\n== {name} (test) " + json.dumps(m, indent=1))

    # ---- artifact bundles ----------------------------------------------
    for name in CONFIGS:
        v = val_results[name]
        bundle = {
            "model_name": name,
            "estimator": v["estimator"],
            "feature_names": FEATURE_NAMES,
            "feature_order_note": "matches registry.yaml entry order",
            "threshold": v["threshold"],
            "threshold_rule": "argmax F1 on val (tie: higher recall, then lower threshold)",
            "config_note": v["config_note"],
            "seed": SEED,
            "dataset_version": "v1",
            "val_metrics": v["metrics"],
            "test_metrics": test_results[name]["metrics"],
            "selected": name == selected,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        joblib.dump(bundle, MODELS_DIR / f"{name}.pkl")

    # ---- prediction CSVs + error summary --------------------------------
    err_summary = {}
    for name in CONFIGS:
        v = val_results[name]
        for split in ("val", "test"):
            p = v["val_probs"] if split == "val" else test_results[name]["probs"]
            pred = (p >= v["threshold"]).astype(int)
            out = pd.DataFrame({
                "url": splits[split]["df"]["url"].to_numpy(),
                "label": splits[split]["y"],
                "predicted": pred,
                "probability": np.round(p, 6),
                "threshold": v["threshold"],
                "correct": pred == splits[split]["y"],
            })
            out.to_csv(PRED_DIR / f"{name}_{split}.csv", index=False, encoding="utf-8")
            fp = out[(out.label == 0) & (out.predicted == 1)]
            fn = out[(out.label == 1) & (out.predicted == 0)]
            err_summary[f"{name}_{split}"] = {
                "false_positives": int(len(fp)), "false_negatives": int(len(fn)),
                "fp_examples": [defang(u) for u in fp["url"].head(3)],
                "fn_examples": [defang(u) for u in fn["url"].head(3)],
            }
            print(f"\n{name} {split} errors: {json.dumps(err_summary[f'{name}_{split}'], indent=1)}")

    (ROOT / "artifacts" / "b4_summary.json").write_text(
        json.dumps({"selection": decision, "test": {n: t["metrics"] for n, t in test_results.items()},
                    "errors": err_summary}, indent=2), encoding="utf-8")
    print("\nB4 DONE. Bundles:", [p.name for p in sorted(MODELS_DIR.glob("*.pkl"))])


if __name__ == "__main__":
    main()
