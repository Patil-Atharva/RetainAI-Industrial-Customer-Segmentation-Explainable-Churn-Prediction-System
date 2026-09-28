"""
TEST-005 & REQ-005: SHAP Explainability Dimensions Test Suite.
Verifies SHAP feature attributions match input feature vector dimensions exactly,
and validates isolation of root cause risk drivers and protective factors.
"""

from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from src.evaluation.explainability import SHAPExplainerEngine
from src.features.churn_features import prepare_churn_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


class TestSHAPExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_path = DATA_PROCESSED_DIR / "churn_train.csv"
        test_path = DATA_PROCESSED_DIR / "churn_test.csv"
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        _, _, cls.X_test, cls.y_test, cls.engineer = prepare_churn_features(
            train_df, test_df, scale_features=False
        )

        model_path = MODEL_DIR / "tuned_champion_model.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "gbdt_hist_gradient_boosting.joblib"
        cls.model = joblib.load(model_path)
        cls.engine = SHAPExplainerEngine(cls.model, feature_names=cls.engineer.feature_names_)

    def test_shap_vector_dimension_matches_features(self):
        """Assert single customer SHAP attribution vector dimension strictly equals input feature count."""
        sample_row = self.X_test.iloc[[0]]
        shap_vec = self.engine.compute_single_row_shap(sample_row)

        self.assertEqual(
            len(shap_vec),
            sample_row.shape[1],
            f"SHAP vector dimension ({len(shap_vec)}) must match feature count ({sample_row.shape[1]})."
        )

    def test_global_importance_dimensions(self):
        """Assert global feature importance produces requested top_n ranked features."""
        df_imp = self.engine.get_global_feature_importance(self.X_test.head(100), top_n=10)
        self.assertEqual(len(df_imp), 10)
        self.assertIn("feature", df_imp.columns)
        self.assertIn("mean_abs_shap", df_imp.columns)

    def test_explain_customer_structure(self):
        """Verify top risk drivers and protective factors structure."""
        sample_row = self.X_test.iloc[[0]]
        explanation = self.engine.explain_customer(sample_row, customer_id="TEST_54007", top_k=5)

        self.assertEqual(explanation["customer_id"], "TEST_54007")
        self.assertIn(explanation["risk_tier"], ["High Risk", "Medium Risk", "Low Risk"])
        self.assertTrue(0.0 <= explanation["churn_probability"] <= 1.0)
        self.assertIsInstance(explanation["top_risk_drivers"], list)
        self.assertIsInstance(explanation["top_protective_factors"], list)


if __name__ == "__main__":
    unittest.main()
