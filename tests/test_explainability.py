"""
Unit tests for SHAP Explainability Engine (src/evaluation/explainability.py).
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.explainability import (
    SHAPExplainerEngine,
    generate_shap_explanations,
)


class TestSHAPExplainability(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_dummy = pd.DataFrame(
            np.random.randn(50, 5), columns=["Tenure", "Complain", "OrderCount", "SatisfactionScore", "CashbackAmount"]
        )
        self.y_dummy = pd.Series(np.random.choice([0, 1], size=50))
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_dummy, self.y_dummy)

    def test_global_feature_importance(self):
        """Verify calculation of global feature importance table."""
        engine = SHAPExplainerEngine(self.model, feature_names=list(self.X_dummy.columns))
        df_imp = engine.get_global_feature_importance(self.X_dummy, top_n=5)

        self.assertIsInstance(df_imp, pd.DataFrame)
        self.assertEqual(len(df_imp), 5)
        self.assertIn("feature", df_imp.columns)
        self.assertIn("mean_abs_shap", df_imp.columns)

    def test_explain_customer(self):
        """Verify individual customer risk explanation structure."""
        engine = SHAPExplainerEngine(self.model, feature_names=list(self.X_dummy.columns))
        single_row = self.X_dummy.iloc[[0]]
        explanation = engine.explain_customer(single_row, customer_id="CUST_9999", top_k=3)

        self.assertEqual(explanation["customer_id"], "CUST_9999")
        self.assertIn(explanation["risk_tier"], ["High Risk", "Medium Risk", "Low Risk"])
        self.assertTrue(0.0 <= explanation["churn_probability"] <= 1.0)
        self.assertIn("top_risk_drivers", explanation)
        self.assertIn("top_protective_factors", explanation)

    def test_generate_shap_explanations_end_to_end(self):
        """Verify end-to-end report generation."""
        report = generate_shap_explanations()
        self.assertIn("global_top_features", report)
        self.assertIn("sample_customer_explanation", report)


if __name__ == "__main__":
    unittest.main()
