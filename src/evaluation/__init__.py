"""
Model evaluation, PR-AUC metrics, and SHAP explainability engine.
Assigned to: Lead Engineer (You).
"""

from src.evaluation.explainability import (
    SHAPExplainerEngine,
    generate_shap_explanations,
    load_champion_model_and_features,
)

__all__ = [
    "SHAPExplainerEngine",
    "generate_shap_explanations",
    "load_champion_model_and_features",
]
