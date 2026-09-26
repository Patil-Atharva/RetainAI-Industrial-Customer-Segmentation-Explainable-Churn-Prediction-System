"""
SHAP Explainability & Feature Attribution Engine.
Computes global feature importances and per-customer local Shapley value attributions
to isolate root cause risk drivers and protective factors for churn prediction.

Assigned to: Lead Engineer (You)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Optional SHAP import with fallback
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.features.churn_features import prepare_churn_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


class SHAPExplainerEngine:
    """
    Explainable AI (XAI) engine computing TreeSHAP or feature contribution vectors.
    Explains individual churn probability predictions in human-understandable terms.
    """

    def __init__(self, model: Any, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.has_shap_explainer = False

        if HAS_SHAP:
            try:
                # Initialize TreeExplainer for tree models
                self.explainer = shap.TreeExplainer(self.model)
                self.has_shap_explainer = True
                logger.info("SHAP TreeExplainer initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP TreeExplainer: {e}. Using fallback attribution engine.")
        else:
            logger.info("SHAP package not installed. Using native Scikit-Learn feature attribution engine.")

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calculates raw SHAP values matrix (samples x features)."""
        if self.has_shap_explainer:
            try:
                shap_vals = self.explainer.shap_values(X)
                # If binary classification returns list of 2 arrays, select class 1 (churn)
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    return shap_vals[1]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    return shap_vals[:, :, 1]
                return shap_vals
            except Exception as e:
                logger.warning(f"SHAP computation failed ({e}). Falling back to feature attribution matrix.")

        # Fallback Attribution Matrix: Feature Importance * Normalized Feature Deviations
        feature_importances = getattr(self.model, "feature_importances_", None)
        if feature_importances is None:
            feature_importances = np.ones(X.shape[1]) / float(X.shape[1])

        means = np.copy(X.mean(axis=0).values)
        stds = np.copy(X.std(axis=0).values)
        stds[stds == 0] = 1.0

        # Normalized Z-scores weighted by feature importances
        z_scores = (X.values - means) / stds
        attribution_matrix = z_scores * feature_importances
        return attribution_matrix

    def get_global_feature_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Calculates mean absolute SHAP feature importances across all customers.
        """
        shap_matrix = self._compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

        cols = self.feature_names if self.feature_names else list(X.columns)
        df_importance = pd.DataFrame({
            "feature": cols,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return df_importance.head(top_n)

    def explain_customer(
        self, customer_df: pd.DataFrame, customer_id: str = "N/A", top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generates local per-customer SHAP feature attribution report.
        """
        if len(customer_df) != 1:
            customer_df = customer_df.iloc[[0]]

        # Compute churn probability
        churn_prob = float(self.model.predict_proba(customer_df)[:, 1][0])
        shap_row = self.compute_single_row_shap(customer_df)

        cols = self.feature_names if self.feature_names else list(customer_df.columns)
        vals = customer_df.values[0]

        factors = []
        for col_name, shap_val, feat_val in zip(cols, shap_row, vals):
            impact = "increases_churn_risk" if shap_val > 0 else "reduces_churn_risk"
            factors.append({
                "feature": col_name,
                "shap_value": round(float(shap_val), 4),
                "feature_value": round(float(feat_val), 4) if isinstance(feat_val, (int, float, np.number)) else str(feat_val),
                "impact": impact
            })

        # Separate risk pushers vs protective factors
        risk_factors = sorted([f for f in factors if f["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        protective_factors = sorted([f for f in factors if f["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        explanation = {
            "customer_id": str(customer_id),
            "churn_probability": round(churn_prob, 4),
            "risk_tier": "High Risk" if churn_prob >= 0.7 else ("Medium Risk" if churn_prob >= 0.3 else "Low Risk"),
            "top_risk_drivers": risk_factors,
            "top_protective_factors": protective_factors,
        }

        return explanation

    def compute_single_row_shap(self, customer_df: pd.DataFrame) -> np.ndarray:
        """Helper to compute SHAP vector for a single customer row."""
        return self._compute_shap_values(customer_df)[0]


def load_champion_model_and_features() -> Tuple[Any, pd.DataFrame, pd.Series, List[str]]:
    """Loads champion model artifact and test feature dataset."""
    model_path = MODEL_DIR / "tuned_champion_model.joblib"
    if not model_path.exists():
        model_path = MODEL_DIR / "gbdt_hist_gradient_boosting.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "baseline_random_forest.joblib"

    if not model_path.exists():
        raise FileNotFoundError("No trained churn model found in models/churn/. Please run train_gbdt.py first.")

    logger.info(f"Loading trained model artifact from {model_path}...")
    model = joblib.load(model_path)

    test_path = DATA_PROCESSED_DIR / "churn_test.csv"
    train_path = DATA_PROCESSED_DIR / "churn_train.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    _, _, X_test, y_test, engineer = prepare_churn_features(train_df, test_df, scale_features=False)
    return model, X_test, y_test, engineer.feature_names_


def generate_shap_explanations(output_dir: Path | str = MODEL_DIR) -> Dict[str, Any]:
    """
    Generates global feature importance report and sample local customer explanations.
    """
    model, X_test, y_test, feature_names = load_champion_model_and_features()
    engine = SHAPExplainerEngine(model, feature_names=feature_names)

    # 1. Global Importance
    logger.info("Computing Global SHAP Feature Importances...")
    df_global_imp = engine.get_global_feature_importance(X_test, top_n=15)

    # 2. Sample Local Explanations
    sample_customer = X_test.iloc[[0]]
    sample_explanation = engine.explain_customer(sample_customer, customer_id="SAMPLE_1001")

    report = {
        "global_top_features": df_global_imp.to_dict(orient="records"),
        "sample_customer_explanation": sample_explanation
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "shap_explainability_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved SHAP Explainability Report to {report_path}")
    return report


def main():
    """CLI runner to execute SHAP explainability analysis."""
    logger.info("=== Starting SHAP Explainability Engine ===")
    report = generate_shap_explanations()

    print("\n" + "=" * 60)
    print("GLOBAL SHAP TOP 10 RISK DRIVERS")
    print("=" * 60)
    df_imp = pd.DataFrame(report["global_top_features"])
    print(df_imp.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("SAMPLE CUSTOMER LOCAL SHAP EXPLANATION")
    print("=" * 60)
    print(json.dumps(report["sample_customer_explanation"], indent=2))
    print("=" * 60)

    logger.info("=== SHAP Explainability Engine Completed Successfully ===")


if __name__ == "__main__":
    main()
