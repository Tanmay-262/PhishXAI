"""SHAP explainability + template translation (B5)."""
from .shap_explainer import LocalExplanation, PhishShapExplainer
from .translator import GROUPS, Sentence, Translation, group_concepts, load_registry, translate

__all__ = [
    "LocalExplanation", "PhishShapExplainer", "Sentence", "Translation",
    "group_concepts", "load_registry", "translate", "GROUPS",
]
