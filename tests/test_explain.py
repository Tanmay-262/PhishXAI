
# B5 unit tests: SHAP explainer + translator faithfulness (updated for
# registry v1.2 value-aware phrases - B6 follow-up).
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
    vals = dict(zip(FEATURE_NAMES, x))
    return ex.explain_one(url, x), url, vals


# ---------------- SHAP correctness ----------------

def test_additivity(ex, sample):
    le, url, _ = sample
    resid = abs(le.base_value + le.shap.sum() - le.decision_function)
    assert resid <= 1e-4, "SHAP additivity violated: %r" % resid
    assert resid < 1e-9, "linear SHAP should be machine-precision"


def test_feature_alignment(ex, sample):
    le, _, _ = sample
    assert list(le.feature_names) == FEATURE_NAMES
    assert len(le.shap) == 19
    assert np.isfinite(le.shap).all(), "non-finite SHAP values"


def test_positive_class_direction(ex, sample):
    le, url, _ = sample
    p = le.probability
    if p >= 0.5:
        assert le.decision_function > le.base_value
    else:
        assert le.decision_function < le.base_value


def test_determinism(ex, sample):
    le, url, _ = sample
    x = extract_features_df([url]).iloc[0].to_numpy(dtype=float)
    le2 = ex.explain_one(url, x)
    assert np.array_equal(le.shap, le2.shap)
    assert le.probability == le2.probability
    assert le.base_value == le2.base_value


def test_top_k_signs(sample):
    le, _, _ = sample
    pos = le.top_positive(5)
    neg = le.top_negative(5)
    assert all(v > 0 for _, v in pos)
    assert all(v < 0 for _, v in neg)
    mags = [abs(v) for _, v in pos]
    assert mags == sorted(mags, reverse=True)


# ---------------- translator ----------------

def _expected_text(reg, name, direction, vals):
    # Mirror of translator substitution: {feature} -> "the <human_name>",
    # {value} -> observed value (integer when integral).
    tmpl = reg[name]["phrases"][direction]
    text = tmpl.replace("{feature}", "the " + reg[name]["human_name"])
    if "{value}" in text:
        v = vals[name]
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        text = text.replace("{value}", str(v))
    return text


def test_translator_deterministic(sample):
    le, _, vals = sample
    kw = dict(top_k=5, feature_values=vals)
    t1 = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, **kw)
    t2 = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, **kw)
    assert [s.text for s in t1.sentences] == [s.text for s in t2.sentences]
    assert t1.coverage == t2.coverage


def test_translator_directions(sample):
    le, _, vals = sample
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5,
                   feature_values=vals)
    for s in tr.sentences:
        if s.shap_value > 0:
            assert s.direction == "toward_phishing"
        else:
            assert s.direction == "toward_legitimate"


def test_translator_top_k_ranking(sample):
    le, _, vals = sample
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=3,
                   feature_values=vals)
    assert len(tr.sentences) <= 3
    mags = [abs(s.shap_value) for s in tr.sentences]
    assert mags == sorted(mags, reverse=True)


def test_translator_zero_contributions_skipped():
    reg = load_registry()
    names = list(reg.keys())
    contribs = [0.0] * 19
    contribs[names.index("domain_length")] = 0.5
    tr = translate(names, contribs, "phishing", 0.9, 0.5, top_k=5,
                   registry=reg, feature_values={"domain_length": 42})
    assert [s.feature for s in tr.sentences] == ["domain_length"]
    # v1.2 value-aware phrase must state the observed value
    assert tr.sentences[0].text == \
        "The domain name is 42 characters long, which contributed to " \
        "the phishing score."
    assert tr.coverage == 1.0


def test_translator_empty_all_zero():
    reg = load_registry()
    tr = translate(list(reg.keys()), [0.0] * 19, "phishing", 0.5, 0.5,
                   top_k=5, registry=reg)
    assert tr.sentences == []
    assert tr.coverage == 1.0, "no sentences -> full coverage by definition"


def test_translator_fallback_and_coverage():
    # Synthetic registry isolates the fallback path deterministically,
    # independent of the shipped registry's current phrase coverage.
    synth = {
        "f_known": {
            "name": "f_known", "type": "int", "computation": "test",
            "human_name": "Known Feature",
            "phrases": {"toward_phishing": "The known feature raised the score.",
                        "toward_legitimate": None},
        },
    }
    tr = translate(["f_known", "f_unknown"], [1.0, 0.5], "phishing",
                   0.9, 0.5, top_k=5, registry=synth,
                   feature_values={"f_known": 1})
    assert len(tr.sentences) == 2
    known = next(s for s in tr.sentences if s.feature == "f_known")
    unknown = next(s for s in tr.sentences if s.feature == "f_unknown")
    assert not known.is_fallback
    assert unknown.is_fallback
    assert tr.coverage == 0.5, "1 of 2 sentences from registry phrases"


def test_value_phrase_fails_closed_without_value():
    # A value-aware phrase must NOT be silently emitted without its value.
    reg = load_registry()
    names = list(reg.keys())
    contribs = [0.0] * 19
    contribs[names.index("url_length")] = 1.0
    with pytest.raises(ValueError):
        translate(names, contribs, "phishing", 0.9, 0.5, top_k=5,
                  registry=reg)  # no feature_values -> fail closed


def test_translator_no_fabricated_text(sample):
    le, _, vals = sample
    reg = load_registry()
    tr = translate(le.feature_names, le.shap.tolist(), le.verdict,
                   le.probability, le.threshold, top_k=5, registry=reg,
                   feature_values=vals)
    for s in tr.sentences:
        assert s.feature in reg, "sentence references an unknown feature"
        if not s.is_fallback:
            expected = _expected_text(reg, s.feature, s.direction, vals)
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




def test_binary_flag_phrases_agree_with_value():
    """No sentence may assert the opposite value of a binary flag.

    For a linear model the observed value of a binary feature is pinned by the
    sign of its contribution (phi = coef*(v-mean)/scale). Walk every has_*
    feature at both values and assert the emitted sentence states the observed
    value. Regression for the has_path defect: the pre-v1.3 phrase
"includes a long path" was emitted for has_path=0.
    """
    bundle = joblib.load(ROOT / "artifacts" / "models" / "lr_v1.pkl")
    clf = bundle["estimator"].named_steps["clf"]
    sc = bundle["estimator"].named_steps["scaler"]
    reg = load_registry()
    names = bundle["feature_names"]
    agree = {
        "has_path": {0: "has_path flag is 0", 1: "has_path flag is 1"},
        "has_shortener": {0: "shortening-service flag is 0", 1: "shortening-service flag is 1"},
        "has_https": {0: "no HTTPS", 1: "uses HTTPS"},
        "has_ip_host": {0: "ordinary domain", 1: "raw IP address"},
        "has_port": {0: "No unusual port", 1: "unusual port"},
        "has_at_symbol": {0: "no @ symbol", 1: "@ symbol"},
        "has_suspicious_keyword": {0: "does not use typical phishing words", 1: "mentions words like"},
    }
    checked = 0
    for name in [n for n in names if n in agree]:
        i = names.index(name)
        coef = float(clf.coef_[0][i]); mean = float(sc.mean_[i]); scale = float(sc.scale_[i])
        for v in (0, 1):
            contrib = coef * (v - mean) / scale
            if abs(contrib) < 1e-9: continue
            contribs = [0.0] * len(names); contribs[i] = contrib
            tr = translate(names, contribs, "phishing", 0.5, 0.5, top_k=5, registry=reg, feature_values={name: v})
            s = next((s for s in tr.sentences if s.feature == name), None)
            assert s is not None, name + ": no sentence for nonzero contribution"
            assert agree[name][v] in s.text, (name, v, s.text)
            checked += 1
    assert checked >= 6, "expected >= 6 flag/value checks, got %d" % checked
def test_translation_pass_is_not_model_validation():
    reg = load_registry()
    tr = translate(list(reg.keys()), [0.0] * 19, "phishing", 0.99, 0.5,
                   top_k=3, registry=reg)
    assert tr.disclaimer and "not proof" in tr.disclaimer
