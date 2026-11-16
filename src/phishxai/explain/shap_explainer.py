"""SHAP explainability for the selected logistic-regression model (B5).

The lr_v1 bundle is a Pipeline(StandardScaler -> LogisticRegression). It is
NOT a tree model, so we use shap.LinearExplainer on the fitted Logistic
Regression (the linear head), operating on SCALED inputs.

Output space
------------
For logistic regression the LinearExplainer contributions are in
LOG-ODDS (decision-function) space, i.e.
    base + sum_j shap_j  ~=  clf.decision_function(scaled_x)
We document that space explicitly and never treat a negative log-odds score
as proof a URL is safe.

Additivity
----------
base + sum(shap) == decision_function(x) exactly for a linear model
(linear expectation = value at the mean). We still assert it with a
tolerance and surface failures as test errors.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import shap


@dataclasses.dataclass
class LocalExplanation:
    """Per-URL SHAP result for the positive (phishing) class."""
    url: str
    probability: float          # P(phishing) from the pipeline
    verdict: str                # "phishing" | "legitimate" at threshold
    threshold: float
    base_value: float           # log-odds base
    decision_function: float    # log-odds for this URL
    shap: "np.ndarray"         # shape (19,), positive-class contributions
    feature_names: List[str]

    def top_positive(self, k: int = 5) -> List[Tuple[str, float]]:
        """Top k features with the largest + (toward phishing) contributions."""
        idx = np.argsort(-self.shap)[:k]
        return [(self.feature_names[i], float(self.shap[i])) for i in idx
                if self.shap[i] > 0]

    def top_negative(self, k: int = 5) -> List[Tuple[str, float]]:
        """Top k features with the largest - (away from phishing) contributions."""
        idx = np.argsort(self.shap)[:k]
        return [(self.feature_names[i], float(self.shap[i])) for i in idx
                if self.shap[i] < 0]


class PhishShapExplainer:
    """Deterministic LinearExplainer wrapper around a lr_v1 bundle."""

    def __init__(self, bundle: dict, background_X: "np.ndarray" | None = None,
                 background_size: Optional[int] = None, seed: int = 42):
        self.pipeline = bundle["estimator"]
        self.feature_names: List[str] = list(bundle["feature_names"])
        self.threshold = float(bundle["threshold"])
        if len(self.feature_names) != 19:
            raise ValueError("registry order mismatch: expected 19 features")

        self.scaler = self.pipeline.named_steps["scaler"]
        self.clf = self.pipeline.named_steps["clf"]
        # Verify the positive class instead of assuming index 1.
        classes = [int(c) for c in self.clf.classes_]
        if 1 not in classes:
            raise ValueError(f"positive class 1 not in clf.classes_={classes}")
        self.pos_class = classes.index(1)

        if background_X is None:
            raise ValueError("SHAP background must be provided (train data only)")
        if background_size is not None and background_size < background_X.shape[0]:
            sel = np.random.RandomState(seed).choice(
                background_X.shape[0], background_size, replace=False)
            background_X = background_X[sel]
        self.bg_scaled = self.scaler.transform(np.asarray(background_X))
        # Pin the background explicitly: a full-size Independent masker means
        # no internal subsampling RNG, so base values and contributions are
        # bit-identical across processes. For a linear model SHAP is exact:
        # base = decision at background mean, phi_j = coef_j * (x_j - mean_j).
        masker = shap.maskers.Independent(self.bg_scaled,
                                          max_samples=self.bg_scaled.shape[0])
        self.explainer = shap.LinearExplainer(self.clf, masker)

        ev = getattr(self.explainer, "expected_value", None)
        ev = np.asarray(ev, dtype=float)
        if ev.ndim == 0:
            self.base_value = float(ev)
        elif ev.size == 2:
            self.base_value = float(ev[self.pos_class])
        else:
            raise ValueError(f"unexpected SHAP expected_value shape {ev.shape}")

    # -- core ----------------------------------------------------------------

    def explain_one(self, url: str, x_row: "np.ndarray") -> LocalExplanation:
        """Explain a single already-extracted feature row (no I/O here)."""
        x_row = np.asarray(x_row, dtype=float)
        if x_row.shape != (len(self.feature_names),):
            raise ValueError("feature row length != 19")
        if not np.isfinite(x_row).all():
            raise ValueError("non-finite feature value supplied to SHAP")

        x_scaled = self.scaler.transform(x_row.reshape(1, -1))
        sv = self.explainer.shap_values(x_scaled)
        # shap_values may return [neg, pos] arrays for a binary classifier.
        if isinstance(sv, (list, tuple)):
            pos_shap = np.asarray(sv[self.pos_class])[0]
        else:
            pos_shap = np.asarray(sv)[0]
        pos_shap = np.asarray(pos_shap, dtype=float).reshape(len(self.feature_names))
        if not np.isfinite(pos_shap).all():
            raise ValueError("SHAP returned non-finite contributions")

        proba = self.pipeline.predict_proba(x_row.reshape(1, -1))[0]
        p_phish = float(proba[self.pos_class])
        verdict = "phishing" if p_phish >= self.threshold else "legitimate"
        dec = float(self.clf.decision_function(x_scaled)[0])

        return LocalExplanation(url=url, probability=p_phish, verdict=verdict,
                                threshold=self.threshold, base_value=self.base_value,
                                decision_function=dec, shap=pos_shap,
                                feature_names=self.feature_names)

    def check_additivity(self, x_row: "np.ndarray", tol: float = 1e-4) -> float:
        """Return the max additivity residual; raise if it exceeds tol."""
        le = self.explain_one("", x_row)
        residual = abs(le.base_value + float(le.shap.sum()) - le.decision_function)
        if residual > tol:
            raise AssertionError(
                f"SHAP additivity violated: residual {residual:.3e} > tol {tol:.3e}")
        return residual
