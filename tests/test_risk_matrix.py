"""
Unit tests for Integrated Segment x Risk Matrix Engine (src/models/risk_matrix.py).
"""

import unittest
import numpy as np
import pandas as pd

from src.models.risk_matrix import (
    categorize_risk_tier,
    build_integrated_customer_table,
    build_segment_risk_matrix,
    identify_priority_action_cohorts,
)


class TestRiskMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create dummy RFM DataFrame
        self.rfm_df = pd.DataFrame(
            {
                "Recency": [10, 50, 100, 5, 80],
                "Frequency": [15, 5, 1, 20, 2],
                "Monetary": [1500, 500, 100, 2500, 150],
            },
            index=pd.Index([1001, 1002, 1003, 1004, 1005], name="CustomerID"),
        )
        self.labels = np.array([0, 1, 1, 0, 1])

        # Create dummy Churn DataFrame
        self.churn_df = pd.DataFrame(
            {
                "CustomerID": [1001, 1002, 1003, 1004, 1005],
                "Tenure": [12, 4, 1, 24, 2],
            }
        )
        self.churn_probs = np.array([0.10, 0.45, 0.85, 0.05, 0.90])

    def test_categorize_risk_tier(self):
        """Test categorization logic for churn probability tiers."""
        self.assertEqual(categorize_risk_tier(0.85), "High Risk")
        self.assertEqual(categorize_risk_tier(0.70), "High Risk")
        self.assertEqual(categorize_risk_tier(0.50), "Medium Risk")
        self.assertEqual(categorize_risk_tier(0.30), "Medium Risk")
        self.assertEqual(categorize_risk_tier(0.15), "Low Risk")

    def test_build_integrated_customer_table(self):
        """Test merging of RFM clusters, personas, and churn risk predictions."""
        integrated_df = build_integrated_customer_table(
            self.rfm_df, self.labels, self.churn_df, self.churn_probs
        )

        self.assertIsInstance(integrated_df, pd.DataFrame)
        self.assertEqual(len(integrated_df), 5)
        self.assertIn("Cluster_ID", integrated_df.columns)
        self.assertIn("Persona", integrated_df.columns)
        self.assertIn("Churn_Probability", integrated_df.columns)
        self.assertIn("Risk_Tier", integrated_df.columns)

    def test_build_segment_risk_matrix(self):
        """Test 2D crosstabulation matrix generation."""
        integrated_df = build_integrated_customer_table(
            self.rfm_df, self.labels, self.churn_df, self.churn_probs
        )
        matrix = build_segment_risk_matrix(integrated_df)

        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertIn("Total", matrix.columns)
        self.assertIn("Total", matrix.index)

    def test_identify_priority_action_cohorts(self):
        """Test assignment of retention action strategies and priority cohort filtering."""
        integrated_df = build_integrated_customer_table(
            self.rfm_df, self.labels, self.churn_df, self.churn_probs
        )
        priority_cohorts = identify_priority_action_cohorts(integrated_df)

        self.assertIsInstance(priority_cohorts, pd.DataFrame)
        self.assertIn("Recommended_Action", priority_cohorts.columns)
        # Should only retain High Risk and Medium Risk customers (3 out of 5)
        self.assertEqual(len(priority_cohorts), 3)


if __name__ == "__main__":
    unittest.main()
