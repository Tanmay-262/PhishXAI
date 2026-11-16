"""B3 unit tests: extractor + registry contract + edge cases.

Run: .venv\\Scripts\\python -m pytest tests -v
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishxai.features.extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
    extract_features,
)
from phishxai.features.psl import registered_domain

REG = Path(__file__).resolve().parents[1] / "src" / "phishxai" / "features" / "registry.yaml"


@pytest.fixture(scope="module")
def ex():
    return FeatureExtractor()


@pytest.fixture(scope="module")
def reg():
    with open(REG, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- registry schema consistency ----------------

def test_registry_schema(reg):
    feats = reg["features"]
    assert len(feats) >= 15 and len(feats) <= 25, "expected ~15-25 lexical features"
    names = [f["name"] for f in feats]
    assert len(names) == len(set(names)), "duplicate registry feature names"
    for f in feats:
        assert set(f.keys()) >= {"name", "type", "computation", "human_name", "phrases"}
        assert f["type"] in ("int", "float")
        assert set(f["phrases"].keys()) == {"toward_phishing", "toward_legitimate"}
        # direction rule: at least one direction must have a sentence
        assert (f["phrases"]["toward_phishing"] is not None
                or f["phrases"]["toward_legitimate"] is not None)


def test_registry_order_matches_extractor(ex, reg):
    assert FEATURE_NAMES == [f["name"] for f in reg["features"]]
    assert list(ex.extract("https://example.com/").keys()) == FEATURE_NAMES


# ---------------- basic values on a known URL ----------------

def test_known_url_values(ex):
    r = ex.extract("https://a.b.example.co.uk/shop/item?x=1&y=2#frag")
    assert r["has_https"] == 1
    assert r["has_path"] == 1
    assert r["path_length"] == len("/shop/item")
    assert r["query_length"] == len("x=1&y=2")
    assert r["num_url_params"] == 2
    assert r["num_dots"] >= 3
    # PSL: co.uk is a public suffix -> eTLD+1 = example.co.uk
    assert ex.registered_domain("a.b.example.co.uk") == "example.co.uk"
    assert r["num_subdomains"] == 2  # a, b under example.co.uk


def test_fragment_is_not_a_feature_input(ex):
    # fragments are client-side; ensure they are ignored consistently
    r1 = ex.extract("https://example.com/page")
    r2 = ex.extract("https://example.com/page#section")
    for k in ("path_length", "query_length", "num_url_params"):
        assert r1[k] == r2[k]


# ---------------- edge cases ----------------

def test_ip_v4_and_v6(ex):
    assert ex.extract("http://192.168.1.10/x")["has_ip_host"] == 1
    assert ex.extract("http://203.0.113.7/")["has_ip_host"] == 1
    assert ex.extract("http://[2001:db8::1]/x")["has_ip_host"] == 1


def test_at_symbol_credentials(ex):
    r = ex.extract("http://user:pass@evil.com/login")
    assert r["has_at_symbol"] == 1
    assert r["domain_length"] == len("evil.com")  # host excludes credentials


def test_port_handling(ex):
    assert ex.extract("http://example.com:8080/")["has_port"] == 1
    assert ex.extract("http://example.com:80/")["has_port"] == 0
    assert ex.extract("https://example.com:443/")["has_port"] == 0
    assert ex.extract("https://example.com/")["has_port"] == 0


def test_percent_encoding(ex):
    r = ex.extract("https://example.com/%2e%2e/admin")
    assert r["percent_encoded_count"] == 2


def test_malformed_urls_never_raise(ex):
    for bad in ("", "   ", "not a url", "://x", "http://", "example.com/no-scheme/x",
               "ht!tp://bad%%domain.com/??", None):
        r = ex.extract(bad)
        assert set(r.keys()) == set(FEATURE_NAMES)
        for v in r.values():
            assert v == v or not isinstance(v, float)  # no NaN
            assert isinstance(v, (int, float))


def test_missing_scheme_bare_url(ex):
    r = ex.extract("example.com/page")
    assert r["domain_length"] == len("example.com")
    assert r["path_length"] == len("/page")
    assert r["has_https"] == 0


def test_idn_host_counts_chars_not_bytes(ex):
    # Punycode form is what Python's urlparse sees; IDN label counts by chars
    r = ex.extract("http://xn--maana-pta.com/")
    assert r["domain_length"] == len("xn--maana-pta.com")
    assert r["has_ip_host"] == 0
    # PSL fallback: unknown TLDs return the whole host
    assert ex.registered_domain("xn--maana-pta.com") == "xn--maana-pta.com"


def test_shortener_detection(ex):
    assert ex.extract("https://bit.ly/3abc")["has_shortener"] == 1
    assert ex.extract("https://tinyurl.com/x")["has_shortener"] == 1
    assert ex.extract("https://example.com/")["has_shortener"] == 0


def test_suspicious_keyword(ex):
    assert ex.extract("https://x.com/verify-your-account")["has_suspicious_keyword"] == 1
    assert ex.extract("https://x.com/products")["has_suspicious_keyword"] == 0


# ---------------- determinism + PSL ----------------

def test_determinism(ex):
    urls = [
        "https://shopee.com.br/images/hero.png",
        "http://192.168.1.10/a/b?x=1",
        "user@weird-domain--123.co.uk:8080/deep/path?k=v#f",
        "bit.ly/abc",
    ]
    a = [ex.extract(u) for u in urls]
    b = [ex.extract(u) for u in urls]
    assert a == b, "same input must give same features (train==inference)"


def test_psl_basic_rules():
    import phishxai.features.psl as pslmod
    exact, wild, _ = pslmod.load_psl(Path(__file__).resolve().parents[1]
                                     / "src" / "phishxai" / "features" / "assets" / "public_suffix_list.dat")
    assert registered_domain("mail.google.com", exact, wild) == "google.com"
    assert registered_domain("example.co.uk", exact, wild) == "example.co.uk"
    assert registered_domain("a.b.gov.uk", exact, wild) == "b.gov.uk"
    assert registered_domain("10.0.0.5", exact, wild) == "10.0.0.5"  # IP fallback
    assert registered_domain("localhost", exact, wild) == "localhost"
