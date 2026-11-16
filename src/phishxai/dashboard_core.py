"""Dashboard core logic (B6): pure, testable, no Streamlit, no network.

Streamlit is a THIN presentation layer over this module. All feature
extraction, prediction, SHAP and translation is delegated to the B3/B4/B5
modules - nothing is duplicated here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from phishxai.features.extractor import FEATURE_NAMES, extract_features_df
from phishxai.explain import (PhishShapExplainer, group_concepts,
                              load_registry, translate)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
DATA = ROOT / "data" / "processed" / "dataset_v1"
ARTIFACTS = ROOT / "artifacts"

_DEFAKE = re.compile(r"\s+")


def defang(url: str, max_len: int = 60) -> str:
    """Defang for display: replace dots, truncate. Never clickable/fetchable."""
    out = (url or "").replace(".", "[.]")
    return out if len(out) <= max_len else out[:max_len - 3] + "..."


class MissingArtifactError(Exception):
    pass


class DashboardCore:
    """One shared instance (cached) - builds the model pipeline once."""

    def __init__(self, model_name: str = "lr_v1"):
        import joblib
        pkl = MODELS / f"{model_name}.pkl"
        if not pkl.exists():
            raise MissingArtifactError(
                f"Model artifact '{pkl.name}' not found. "
                "Run: .venv\\Scripts\\python scripts\\train_models.py  "
                "(requires data/raw/ + data/processed/ - see docs/DATASET_PROVENANCE.md).")
        bundle = joblib.load(pkl)
        self.bundle = bundle
        self.model_name = model_name
        self.threshold = float(bundle["threshold"])
        self.pipeline = bundle["estimator"]

        # SHAP background: full train split (B5-faithful). For the scaled
        # linear model, the background mean is the zero vector (scaler fit on
        # train), so a missing data/ falls back to zeros identically.
        bg = self._background()
        self.explainer = PhishShapExplainer(bundle, background_X=bg, seed=42)
        self.registry = load_registry()

    def _background(self):
        import pandas as pd
        train = DATA / "train.csv"
        if train.exists():
            urls = list(pd.read_csv(train)["url"])
            return extract_features_df(urls).to_numpy(dtype=float)
        return np.zeros((1, len(FEATURE_NAMES)), dtype=float)

    # ------------------------------------------------------------------
    def analyze(self, url: str) -> Dict:
        """Full pipeline for one URL: predict + SHAP + translation. No I/O."""
        raw = (url or "").strip()
        xrow = extract_features_df([raw]).iloc[0].to_numpy(dtype=float)
        # pipeline classes_: [0=legit, 1=phishing] for this model; use argmax
        # to be robust rather than hard-coded.
        pos = int(np.argmax(self.pipeline.classes_))
        p_phish = float(self.pipeline.predict_proba(xrow.reshape(1, -1))[0, pos])
        verdict = "phishing" if p_phish >= self.threshold else "legitimate"

        le = self.explainer.explain_one(raw, xrow)
        vals = dict(zip(FEATURE_NAMES, xrow.tolist()))
        tr = translate(le.feature_names, le.shap.tolist(), verdict,
                       p_phish, self.threshold, top_k=5,
                       registry=self.registry, feature_values=vals)
        addit = abs(le.base_value + float(le.shap.sum()) - le.decision_function)
        return {
            "url": raw, "url_defanged": defang(raw),
            "verdict": verdict, "probability": p_phish,
            "threshold": self.threshold,
            "feature_values": vals,
            "shap": le.shap.tolist(),
            "base_value": le.base_value,
            "decision_function": le.decision_function,
            "additivity_residual": addit,
            "top_positive": le.top_positive(5),
            "top_negative": le.top_negative(5),
            "sentences": [s.text + (" [fallback]" if s.is_fallback else "")
                          for s in tr.sentences],
            "sentence_details": tr.sentences,
            "grouped_concepts": group_concepts(le.shap.tolist(), le.feature_names),
            "coverage": tr.coverage,
            "disclaimer": tr.disclaimer,
        }

    # ---- model-performance data (B4 artifacts, recorded) -------------
    def performance_data(self) -> Dict:
        files = {"b4_summary": ARTIFACTS / "b4_summary.json",
                 "selection": ARTIFACTS / "selection_decision.json"}
        out = {}
        for k, p in files.items():
            if not p.exists():
                raise MissingArtifactError(
                    f"'{p.name}' not found. Run: .venv\\Scripts\\python "
                    "scripts\\train_models.py to regenerate B4 artifacts.")
            out[k] = json.loads(p.read_text(encoding="utf-8"))
        return out
