"""Registry-driven, deterministic explanation translator (B5).

Turns per-feature SHAP contributions into short, plain-language sentences.
Design rules:
- The YAML registry is the single source of truth for human names and
  direction-aware phrases. The translator never invents reasons.
- Sentences describe MODEL BEHAVIOR ("increased the model's phishing
  score"), never causal proof, and a low score is never called "safe".
- Contributions toward phishing (positive SHAP value on the positive
  class) use toward_phishing phrases; contributions away (negative) use
  toward_legitimate phrases; a null phrase for that direction produces
  a fallback sentence (tracked in coverage).
- Coverage = displayed sentences with a registry phrase / displayed.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "features" / "registry.yaml"

# Presentation-only GROUPED concepts (summing SIGNED contributions of
# member features). Raw feature-level contributions are always kept.
GROUPS: Dict[str, List[str]] = {
    "overall length": ["url_length", "path_length", "query_length"],
    "domain structure": ["num_dots", "num_subdomains", "domain_length", "num_hyphens"],
    "digits": ["num_digits", "digit_in_domain_ratio"],
    "special characters": ["num_special_chars", "percent_encoded_count"],
    "connection": ["has_https"],
    "security signals": ["has_at_symbol", "has_ip_host", "has_port", "has_shortener"],
    "keywords": ["has_suspicious_keyword"],
}


@dataclasses.dataclass
class Sentence:
    feature: str          # feature name that drove this sentence
    group: str            # presentation group it belongs to
    direction: str        # "toward_phishing" | "toward_legitimate"
    text: str
    shap_value: float
    is_fallback: bool


@dataclasses.dataclass
class Translation:
    verdict: str
    probability: float
    threshold: float
    sentences: List[Sentence]
    coverage: float            # fraction of displayed sentences from registry
    top_k: int
    disclaimer: str

    def as_text(self) -> str:
        lines = []
        for s in self.sentences:
            t = "(%+.3f) %s" % (s.shap_value, s.text)
            if s.is_fallback:
                t += " [fallback]"
            lines.append(t)
        return "\n".join(lines)


def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {f["name"]: f for f in data["features"]}


_FALLBACKS = {
    "toward_phishing":
        "This URL detail ({name}) increased the model's phishing score.",
    "toward_legitimate":
        "This URL detail ({name}) reduced the model's phishing score.",
}
_DISCLAIMER = ("These points describe why the model made this prediction; "
               "they are not proof that the URL is phishing or safe.")


def _phrase_for(reg: Dict[str, dict], feature: str, direction: str) -> Optional[str]:
    f = reg.get(feature)
    if f is None:
        return None
    tmpl = f["phrases"].get(direction)
    return (tmpl or "").strip() or None


def translate(feature_names: List[str],
              contributions: List[float],
              verdict: str, probability: float, threshold: float,
              top_k: int = 5, registry: Optional[Dict[str, dict]] = None,
              ) -> Translation:
    """Deterministic translation of per-feature SHAP contributions.

    contributions is the raw feature-level SHAP array (all 19), not the
    grouped sums - grouping is a presentation layer (group_concepts).
    """
    reg = registry if registry is not None else load_registry()
    order = [(name, float(c)) for name, c in zip(feature_names, contributions)]
    # Exactly-zero contributions mean "no effect on the score"; emitting a
    # direction sentence for them would be misleading, so they are skipped.
    order = [(n, c) for n, c in order if abs(c) >= 1e-9]
    ranked = sorted(order, key=lambda t: (-abs(t[1]), t[0]))[:top_k]

    sentences: List[Sentence] = []
    for name, c in ranked:
        direction = "toward_phishing" if c >= 0 else "toward_legitimate"
        phrase = _phrase_for(reg, name, direction)
        human = reg.get(name, {}).get("human_name", name)
        if phrase:
            text = phrase.replace("{feature}", "the " + human)
        else:
            text = _FALLBACKS[direction].format(name=human)
        group = next((g for g, mem in GROUPS.items() if name in mem), "other")
        sentences.append(Sentence(feature=name, group=group, direction=direction,
                                  text=text, shap_value=c, is_fallback=phrase is None))
    n_with = sum(1 for s in sentences if not s.is_fallback)
    coverage = (n_with / len(sentences)) if sentences else 1.0
    return Translation(verdict=verdict, probability=probability,
                       threshold=threshold, sentences=sentences,
                       coverage=coverage, top_k=top_k, disclaimer=_DISCLAIMER)


def group_concepts(contributions: List[float],
                   feature_names: List[str]) -> List[dict]:
    """Sum SIGNED contributions per non-overlapping presentation group."""
    contrib = dict(zip(feature_names, contributions))
    grouped = set()
    for members in GROUPS.values():
        grouped.update(members)
    used: Dict[str, List[str]] = {g: list(m) for g, m in GROUPS.items()}
    used["other"] = [n for n in feature_names if n not in grouped]
    out = []
    for g, members in used.items():
        total = sum(contrib[m] for m in members if m in contrib)
        out.append({"group": g, "signed_sum": round(total, 4),
                    "n_features": len(members)})
    out.sort(key=lambda d: -abs(d["signed_sum"]))
    return out
