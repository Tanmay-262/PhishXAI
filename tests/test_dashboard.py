# B6 integration tests: dashboard core paths + network prohibition.
# Run: .venv\Scripts\python -m pytest tests/test_dashboard.py -v
# These exercise the SAME code Streamlit renders (phishxai.dashboard_core),
# plus an import check of the app module.
import socket
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phishxai.dashboard_core import DashboardCore, defang  # noqa: E402
from phishxai.features.extractor import FEATURE_NAMES  # noqa: E402


class DenyNetwork(socket.socket):
    """Reject any real socket - proves the pipeline makes no network calls."""

    def __init__(self, *a, **k):
        raise RuntimeError("NETWORK ATTEMPTED - dashboard must be offline")

    @classmethod
    def socketpair(cls, *a, **k):
        raise RuntimeError("NETWORK ATTEMPTED")

    def connect(self, *a, **k):
        raise RuntimeError("NETWORK ATTEMPTED")

    def sendall(self, *a, **k):
        raise RuntimeError("NETWORK ATTEMPTED")


@pytest.fixture(scope="module")
def dc():
    return DashboardCore("lr_v1")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", DenyNetwork)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("NETWORK ATTEMPTED")))


def test_model_prediction_matches_b4(dc):
    # The B4 prediction CSVs recorded probabilities; the core must
    # reproduce them exactly for the same URLs.
    b4 = pd.read_csv(ROOT / "artifacts" / "predictions" / "lr_v1_val.csv")
    for _, row in b4.head(5).iterrows():
        r = dc.analyze(row["url"])
        assert abs(r["probability"] - row["probability"]) < 5e-6, \
            "dashboard prediction diverged from the B4-recorded value"
        assert r["verdict"] == ("phishing" if r["probability"] >= r["threshold"]
                                else "legitimate")


def test_known_phishing_val_url(dc):
    val = pd.read_csv(ROOT / "data" / "processed" / "dataset_v1" / "val.csv")
    ph = val[val["label"] == "phishing"].iloc[0]
    r = dc.analyze(ph["url"])
    assert r["verdict"] == "phishing"
    assert r["probability"] > 0.5
    assert r["url_defanged"].startswith("hxxp") or "[.]" in r["url_defanged"]


def test_known_legit_val_url(dc):
    val = pd.read_csv(ROOT / "data" / "processed" / "dataset_v1" / "val.csv")
    lg = val[val["label"] == "legitimate"].iloc[0]
    r = dc.analyze(lg["url"])
    assert r["verdict"] == "legitimate"
    assert r["probability"] < 0.5


def test_bare_domain_false_negative(dc):
    r = dc.analyze("https://uspostaldirect.com/")
    assert r["verdict"] == "legitimate"   # the documented B4 FN
    assert r["probability"] < dc.threshold


def test_constructed_legit_false_positive(dc):
    test = pd.read_csv(ROOT / "data" / "processed" / "dataset_v1" / "test.csv")
    mask = test["url"].str.contains("shopee.co.id/images/hero", regex=False)
    assert mask.any(), "B4 false-positive URL missing from test set"
    r = dc.analyze(test.loc[mask, "url"].iloc[0])
    assert r["verdict"] == "phishing"      # the documented B4 FP
    assert r["probability"] > dc.threshold


def test_empty_and_malformed_inputs(dc):
    for bad in ("", "   ", "not a url", "://x", "javascript:alert(1)"):
        r = dc.analyze(bad)
        assert set(r.keys()) >= {"verdict", "probability", "shap",
                                  "sentences", "coverage"}
        assert np.isfinite(r["shap"]).all(), "SHAP must stay finite"
        assert 0.0 <= r["probability"] <= 1.0


def test_additivity_within_tolerance(dc):
    r = dc.analyze("https://shopee.co.id/images/hero.png")
    assert r["additivity_residual"] <= 1e-4


def test_explanation_matches_contributions(dc):
    r = dc.analyze("https://shopee.co.id/images/hero.png")
    # top-5 sentences must correspond to the top-5 |contribution| features
    top5 = sorted(zip(FEATURE_NAMES, r["shap"]),
                   key=lambda t: -abs(t[1]))[:5]
    top5_names = [n for n, v in top5 if abs(v) >= 1e-9]
    sent_feats = [s.feature for s in r["sentence_details"]
                  if not s.is_fallback]
    for n in top5_names:
        assert n in [s.feature for s in r["sentence_details"]], \
            "a top-5 feature is missing from the explanation"


def test_no_network_used(dc):
    # autouse fixture has sockets denied for the whole module; every
    # analysis above completing without error is the proof.
    r = dc.analyze("https://example.com/")
    assert r["verdict"] in ("phishing", "legitimate")


def test_app_module_imports():
    # Import the Streamlit app module in a way that does not require a
    # running browser (Streamlit supports module import for testing).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "streamlit_app", ROOT / "app" / "streamlit_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod is not None
