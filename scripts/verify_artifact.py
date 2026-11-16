"""Reload a model bundle in a FRESH python process and check determinism.

Usage: .venv\\Scripts\\python scripts\\verify_artifact.py [model_name]
Prints the predicted probability for a fixed input URL; the expected value
is recorded in artifacts/b4_summary.json (val/test prediction CSVs).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phishxai.features.extractor import extract_features_df

name = sys.argv[1] if len(sys.argv) > 1 else "lr_v1"
bundle = joblib.load(ROOT / "artifacts" / "models" / f"{name}.pkl")
est = bundle["estimator"]

# Fixed verification input: first row of the val split (deterministic pick).
val = pd.read_csv(ROOT / "data" / "processed" / "dataset_v1" / "val.csv")
url = val["url"].iloc[0]
p = float(extract_features_df([url]).pipe(lambda X: est.predict_proba(X)[0, 1]))

# Cross-check against the CSV produced by the training run
csv_pred = pd.read_csv(ROOT / "artifacts" / "predictions" / f"{name}_val.csv")
recorded = float(csv_pred["probability"].iloc[0])

# CSV records 6-decimal probabilities; compare within that precision
match = abs(p - recorded) < 5e-6
print(json.dumps({
    "model": name, "feature_count": len(bundle["feature_names"]),
    "threshold": bundle["threshold"], "seed": bundle["seed"],
    "verification_url": url, "fresh_process_probability": p,
    "recorded_probability": recorded, "MATCH": match,
}, indent=2))
sys.exit(0 if match else 1)
