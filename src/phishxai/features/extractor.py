"""Lexical feature extractor (B3).

Design rules
------------
- The YAML registry (registry.yaml) is the single source of truth: feature
  names, order, types and human phrases all come from it.
- Deterministic: same URL + same PSL => same dict, always. Stable ordering
  => identical feature vectors during training and inference.
- NO network access. Everything comes from the URL string + a local PSL file.
- No dataset provenance fields (source, url_kind, path_constructed...) are
  ever read as model features.
- Malformed input never raises: every feature has a defined value.
"""
from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import yaml

from .psl import load_psl, registered_domain as _psl_registered_domain

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")
# PSL shipped with the package so a fresh clone can reproduce extraction
# without the gitignored data/raw files. Copied from publicsuffix.org (CC0).
DEFAULT_PSL_PATH = Path(__file__).with_name("assets") / "public_suffix_list.dat"

_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly",
    "buff.ly", "cutt.ly", "cutt.us", "rebrand.ly", "shrinkurl.com",
}
_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "security", "account", "payment", "bank", "password",
    "confirm", "support", "update", "prize", "claim", "alert",
    "recover", "unlock", "suspended", "disabled",
]
_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~/?:&=+%#@")
_PERCENT_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def _load_registry(path: Path = REGISTRY_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    feats = data["features"]
    for f_ in feats:
        if f_["type"] not in ("int", "float"):
            raise ValueError(f"registry feature {f_['name']} has illegal type {f_['type']}")
    return feats


def load_psl_bundle(path: Optional[str | Path] = None) -> Tuple[Set[str], Set[str]]:
    p = Path(path) if path else DEFAULT_PSL_PATH
    exact, wildcard, _ignored = load_psl(p)
    return exact, wildcard


def _authority_has_at(url: str) -> bool:
    """True if '@' appears in the URL's authority section (the part before
    the first path slash, after the scheme). Covers both user:pass@host
    and the trusted.com@attacker.com phishing pattern."""
    a = url
    if "://" in a:
        a = a.split("://", 1)[1]
    a = a.split("/", 1)[0]
    return "@" in a


def _extract_port(netloc: str) -> int:
    """Explicit port from netloc (0 if absent/default; -1 if unparsable).

    IPv6 netlocs ([...]:port) are not handled in the MVP - documented.
    """
    if not netloc or netloc.startswith("["):
        return 0
    tail = netloc.rsplit("@", 1)[-1]  # drop user:pass@
    if ":" not in tail:
        return 0
    raw = tail.rsplit(":", 1)[-1]
    try:
        return int(raw)
    except ValueError:
        return -1


class FeatureExtractor:
    """Registry-driven lexical extractor. Build ONE per process and reuse."""

    def __init__(self, psl_path: Optional[str | Path] = None):
        self.features = _load_registry()
        self._exact, self._wild = load_psl_bundle(psl_path)

    # -- decomposition (deterministic, no network) -----------------------

    @staticmethod
    def _parts(url: str) -> dict:
        u = (url or "").strip()
        try:
            p = urlparse(u)
        except Exception:
            p = None
        host = ""
        scheme = path = query = ""
        port = 0
        if p is not None:
            scheme = p.scheme or ""
            path = p.path or ""
            query = p.query or ""
            netloc = p.netloc or ""
            hostpart = netloc.rsplit("@", 1)[-1] if "@" in netloc else netloc
            if hostpart.startswith("["):  # IPv6 literal
                host = hostpart[1:].split("]", 1)[0].lower()
            else:
                host = hostpart.split(":", 1)[0].lower()
                port = _extract_port(netloc)
            # Bare-URL fallback: "example.com/x" without a scheme -> urlparse
            # parks everything in path. Heuristic (documented): if the first
            # slash-segment looks host-like and the input has no "://",
            # treat it as the host.
            if not host and not netloc:
                first = path.split("/", 1)[0]
                if "." in first and " " not in first and "://" not in u:
                    host = first.lower()
                    # Bare URL: move the host out of path. Path becomes the
                    # remainder after the host ("" when none).
                    path = path[len(first):]  # keeps leading slash: "example.com/page" -> "/page"

        return {"url": u, "scheme": scheme, "host": host, "path": path,
                "query": query, "port": port}

    @staticmethod
    def _is_ip(host: str) -> bool:
        if not host:
            return False
        cand = host[1:-1] if host.startswith("[") and host.endswith("]") else host
        try:
            ipaddress.ip_address(cand)
            return True
        except ValueError:
            return False

    def num_subdomains(self, host: str) -> int:
        """Host labels minus registered-domain labels, via PSL; 0 for IPs."""
        if not host or self._is_ip(host):
            return 0
        n_host = len(host.split("."))
        n_rd = len(self.registered_domain(host).split("."))
        return max(0, n_host - n_rd)

    def registered_domain(self, host: str) -> str:
        return _psl_registered_domain(host, self._exact, self._wild)

    # -- single URL -------------------------------------------------------

    def extract(self, url: str) -> Dict[str, float]:
        """Extract ALL registry features for one URL. Never raises."""
        d = self._parts(url)
        url, host, path, query = d["url"], d["host"], d["path"], d["query"]
        scheme, port = d["scheme"], d["port"]
        is_ip = self._is_ip(host)
        host_len = len(host)
        digits_in_host = sum(c.isdigit() for c in host)

        raw_vals: Dict[str, float] = {
            "url_length": len(url),
            "domain_length": host_len,
            "num_dots": url.count("."),
            "num_subdomains": self.num_subdomains(host),
            "num_hyphens": host.count("-"),
            "num_digits": sum(c.isdigit() for c in url),
            "digit_in_domain_ratio": (digits_in_host / host_len) if (host_len and not is_ip) else 0.0,
            "num_special_chars": sum(1 for c in url if c not in _ALLOWED_CHARS),
            "has_ip_host": 1 if is_ip else 0,
            "has_https": 1 if scheme == "https" else 0,
            "has_port": 1 if port > 0 and port not in (80, 443) else 0,
            "has_at_symbol": 1 if _authority_has_at(url) else 0,
            "has_shortener": 1 if self.registered_domain(host) in _SHORTENERS else 0,
            "path_length": len(path),
            "has_path": 1 if (path and path != "/") else 0,
            "query_length": len(query),
            "num_url_params": (query.count("&") + 1) if query else 0,
            "percent_encoded_count": len(_PERCENT_RE.findall(url)),
            "has_suspicious_keyword": 1 if any(k in url.lower() for k in _KEYWORDS) else 0,
        }
        # Registry order + type coercion. Every registry feature must be
        # computable; a missing one is a contract violation, not a silent 0.
        out: Dict[str, float] = {}
        for f in self.features:
            name = f["name"]
            if name not in raw_vals:
                raise KeyError(f"registry lists feature '{name}' that extractor cannot compute")
            v = raw_vals[name]
            out[name] = int(v) if f["type"] == "int" else float(v)
        return out


def make_extractor(psl_path: Optional[str | Path] = None) -> FeatureExtractor:
    return FeatureExtractor(psl_path)


FEATURE_NAMES: List[str] = [f["name"] for f in _load_registry()]


def extract_features(url: str, psl_path: Optional[str | Path] = None) -> Dict[str, float]:
    """One-shot extraction (builds an extractor per call)."""
    return make_extractor(psl_path).extract(url)


def extract_features_df(urls: List[str], psl_path: Optional[str | Path] = None):
    """Deterministic feature matrix for a list of URLs (train or infer)."""
    import pandas as pd
    ex = make_extractor(psl_path)
    rows = [ex.extract(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
