# B5 unit tests: SHAP explainer + translator faithfulness.
# Run: .venv\Scripts\python -m pytest tests/test_explain.py -v
# These tests verify the EXPLANATION LAYER (faithfulness, determinism,
# coverage). Passing them proves the translator is faithful to SHAP values;
# they are NOT evidence that the underlying model is accurate.
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phishxai.features.extractor import FEATURE_NAMES, extract_features_df
from phishxai.explain import (PhishShapExplainer, group_concepts,
                              load_registry, translate)


@pytest.fixture(scope="module")
def ex():
    bundle = joblib.load(ROOT / "artifacts" / "models" / "lr_v1.pkl")
    train = pd.read_csv(ROOT / "data" / "processed" / "dataset_v1" / "train.csv")
    Xtr = extract_features_df(list(train["url"])).to_numpy(dtype=float)
    return PhishShapExplainer(bundle, background_X=Xtr, seed=42)


@pytest.fixture(scope="module")
def sample(ex):
    val = pd.read_csv(ROOT / "data" / "processed" / "dataset_v1" / "val.csv")
    url = val["url"].iloc[0]
    x = extract_features_df([url]).iloc[0].to_numpy(dtype=float)
    return ex.explain_one(url, x), url


# ---------------- SHAP correctness ----------------

def test_additivity(ex, sample):
    le, url = sample
    resid = abs(le.base_value + le.shap.sum() - le.decision_function)
    assert resid <= 1e-4, "SHAP additivity violated: %r" % resid
    assert resid < 1e-9, "linear SHAP should be machine-precision"


def test_feature_alignment(ex, sample):
    le, _ = sample
    assert list(le.feature_names) == FEATURE_NAMES
    assert len(le.shap) == 19
    assert np.isfinite(le.shap).all(), "non-finite SHAP values"


def test_positive_class_direction(ex, sample):
    le, url = sample
    p = le.probability
    if p >= 0.5:
        assert le.decision_function > le.base_value
    else:
        assert le.decision_function < le.base_value


def test_determinism(ex, sample):
    le, url = sample
    x = extract_features_df([url]).iloc[0].to_numpy(dtype=float)
    le2 = ex.explain_one(url, x)
    assert np.array_equal(le.shap, le2.shap)
    assert le.probability == le2.probability
    assert le.base_value == le2.base_value


def test_top_k_signs(sample):
    le, _ = sample
    pos = le.top_positive(5)
    neg = le.top_negative(5)
    assert all(v > 0 for _, v in pos)
    assert all(v < 0 for _, v in neg)
    mags = [abs(v) for _, v in pos]
    assert mags == sorted(mags, reverse=True)


# ---------------- translator ----------------

def test_translator_deterministic(sample):
    le, _ = sample
    t1 = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5)
    t2 = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5)
    assert [s.text for s in t1.sentences] == [s.text for s in t2.sentences]
    assert t1.coverage == t2.coverage


def test_translator_directions(sample):
    le, _ = sample
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5)
    for s in tr.sentences:
        if s.shap_value > 0:
            assert s.direction == "toward_phishing"
        else:
            assert s.direction == "toward_legitimate"


def test_translator_top_k_ranking(sample):
    le, _ = sample
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=3)
    assert len(tr.sentences) <= 3
    mags = [abs(s.shap_value) for s in tr.sentences]
    assert mags == sorted(mags, reverse=True)


def test_translator_zero_contributions_skipped():
    reg = load_registry()
    names = list(reg.keys())
    contribs = [0.0] * 19
    contribs[names.index("domain_length")] = 0.5
    tr = translate(names, contribs, "phishing", 0.9, 0.5, top_k=5, registry=reg)
    assert [s.feature for s in tr.sentences] == ["domain_length"]


def test_translator_empty_all_zero():
    reg = load_registry()
    tr = translate(list(reg.keys()), [0.0] * 19, "phishing", 0.5, 0.5,
                   top_k=5, registry=reg)
    assert tr.sentences == []
    assert tr.coverage == 1.0, "no sentences -> full coverage by definition"


def test_translator_fallback_and_coverage():
    reg = load_registry()
    names = list(reg.keys())
    nulls = [n for n in names if reg[n]["phrases"]["toward_phishing"] is None]
    target = nulls[0] if nulls else "domain_length"
    contribs = [0.0] * 19
    contribs[names.index(target)] = 1.0
    tr = translate(names, contribs, "phishing", 0.9, 0.5, top_k=5, registry=reg)
    assert len(tr.sentences) == 1
    if nulls:
        assert tr.sentences[0].is_fallback
        assert tr.coverage == 0.0
    else:
        assert tr.coverage == 1.0


def test_translator_no_fabricated_text(sample):
    le, _ = sample
    reg = load_registry()
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5, registry=reg)
    for s in tr.sentences:
        assert s.feature in reg, "sentence references an unknown feature"
        if not s.is_fallback:
            tmpl = reg[s.feature]["phrases"][s.direction]
            expected = tmpl.replace("{feature}", "the " + reg[s.feature]["human_name"])
            assert s.text == expected, "sentence deviates from registry phrase"


def test_binary_feature_phrases_match_model_directions():
    # A sentence must never claim the opposite value of a binary feature.
    # The model coefficient sign decides which VALUE pushes toward phishing;
    # the corresponding phrase direction must exist (no null gap).
    bundle = joblib.load(ROOT / "artifacts" / "models" / "lr_v1.pkl")
    coef = bundle["estimator"].named_steps["clf"].coef_[0]
    reg = load_registry()
    for i, name in enumerate(FEATURE_NAMES):
        if name in ("has_https", "has_ip_host", "has_at_symbol", "has_shortener"):
            entry = reg[name]
            if coef[i] > 0:
                assert entry["phrases"]["toward_phishing"] is not None, name
            else:
                assert entry["phrases"]["toward_legitimate"] is not None, name


def test_group_concepts_cover_all_features():
    reg = load_registry()
    names = list(reg.keys())
    groups = group_concepts([1.0 / 19] * 19, names)
    assert sum(g["n_features"] for g in groups) == 19
    assert all(g["signed_sum"] is not None for g in groups)


def test_translation_pass_is_not_model_validation():
    reg = load_registry()
    tr = translate(list(reg.keys()), [0.0] * 19, "phishing", 0.99, 0.5,
                   top_k=3, registry=reg)
    assert tr.disclaimer and "not proof" in tr.disclaimer
