"""
Integration test for Dashboard Data Contracts.
Asserts that all upstream artifacts and file schemas consumed by src/dashboard/app.py
exist, are non-empty, and strictly adhere to expected column types and value ranges.
"""

import json
from pathlib import Path
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"
MODELS_ROOT = PROJECT_ROOT / "models"


class TestDashboardDataContracts(unittest.TestCase):
    """Verifies all input artifacts consumed by the Streamlit dashboard."""

    def test_segment_personas_output_contract(self):
        """Verify segment_personas_output.csv schema and completeness."""
        csv_path = PROJECT_ROOT / "segment_personas_output.csv"
        self.assertTrue(csv_path.exists(), "segment_personas_output.csv must exist in project root.")

        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 0, "segment_personas_output.csv must not be empty.")

        expected_columns = {
            "Cluster_ID", "Recency_mean", "Frequency_mean", "Monetary_mean",
            "Customer_Count", "Pct_of_Base", "Persona"
        }
        missing = expected_columns - set(df.columns)
        self.assertFalse(missing, f"segment_personas_output.csv missing columns: {missing}")

        self.assertTrue((df["Customer_Count"] > 0).all(), "Customer_Count must be positive.")
        self.assertAlmostEqual(df["Pct_of_Base"].sum(), 100.0, delta=1.5, msg="Pct_of_Base must sum to ~100%.")

    def test_integrated_risk_report_contract(self):
        """Verify models/churn/integrated_risk_report.json schema."""
        json_path = MODEL_DIR / "integrated_risk_report.json"
        self.assertTrue(json_path.exists(), "models/churn/integrated_risk_report.json must exist.")

        with open(json_path, "r") as f:
            data = json.load(f)

        required_keys = {
            "total_customers", "segment_count", "high_risk_customers",
            "medium_risk_customers", "low_risk_customers", "risk_matrix",
            "top_priority_cohort_sample"
        }
        missing_keys = required_keys - set(data.keys())
        self.assertFalse(missing_keys, f"integrated_risk_report.json missing keys: {missing_keys}")

        self.assertGreater(data["total_customers"], 0)
        self.assertEqual(
            data["total_customers"],
            data["high_risk_customers"] + data["medium_risk_customers"] + data["low_risk_customers"]
        )

    def test_segment_risk_matrix_contract(self):
        """Verify models/churn/segment_risk_matrix.csv schema."""
        matrix_path = MODEL_DIR / "segment_risk_matrix.csv"
        self.assertTrue(matrix_path.exists(), "models/churn/segment_risk_matrix.csv must exist.")

        df = pd.read_csv(matrix_path, index_col=0)
        self.assertGreater(len(df), 0)

        required_columns = {"High Risk", "Medium Risk", "Low Risk", "Total"}
        missing_cols = required_columns - set(df.columns)
        self.assertFalse(missing_cols, f"segment_risk_matrix.csv missing columns: {missing_cols}")

    def test_priority_action_cohorts_contract(self):
        """Verify models/churn/priority_action_cohorts.csv schema."""
        cohorts_path = MODEL_DIR / "priority_action_cohorts.csv"
        self.assertTrue(cohorts_path.exists(), "models/churn/priority_action_cohorts.csv must exist.")

        df = pd.read_csv(cohorts_path)
        self.assertGreater(len(df), 0, "priority_action_cohorts.csv must not be empty.")

        expected_columns = {
            "CustomerID", "Persona", "Churn_Probability", "Risk_Tier", "Recommended_Action"
        }
        missing = expected_columns - set(df.columns)
        self.assertFalse(missing, f"priority_action_cohorts.csv missing columns: {missing}")

        self.assertTrue(((df["Churn_Probability"] >= 0.0) & (df["Churn_Probability"] <= 1.0)).all())
        valid_tiers = {"High Risk", "Medium Risk", "Low Risk"}
        self.assertTrue(set(df["Risk_Tier"]).issubset(valid_tiers))

    def test_customer_embeddings_contract(self):
        """Verify precomputed 2D/3D embeddings for instant scatter plot rendering."""
        embeddings_path = PROCESSED_DATA_DIR / "customer_segment_embeddings.csv"
        self.assertTrue(embeddings_path.exists(), "customer_segment_embeddings.csv must exist.")

        df = pd.read_csv(embeddings_path)
        self.assertGreater(len(df), 1000)

        expected_columns = {
            "CustomerID", "Recency", "Frequency", "Monetary", "Persona",
            "PC1", "PC2", "UMAP1", "UMAP2"
        }
        missing = expected_columns - set(df.columns)
        self.assertFalse(missing, f"customer_segment_embeddings.csv missing columns: {missing}")

    def test_silhouette_evaluation_contract(self):
        """Verify silhouette score evaluation across k=2..10."""
        sil_path = MODELS_ROOT / "silhouette_evaluation.csv"
        self.assertTrue(sil_path.exists(), "models/silhouette_evaluation.csv must exist.")

        df = pd.read_csv(sil_path)
        self.assertEqual(len(df), 9, "Must evaluate k in [2, 10] (9 points).")
        self.assertIn("silhouette", df.columns)
        self.assertIn("k", df.columns)
        self.assertTrue(((df["silhouette"] >= -1.0) & (df["silhouette"] <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
