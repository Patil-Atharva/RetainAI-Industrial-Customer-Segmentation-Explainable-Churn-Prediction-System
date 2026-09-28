"""
Contract tests for FastAPI REST API endpoints consumed by RetainAI Dashboard.
Asserts status codes, response schemas, and probability/SHAP bounds for:
  - GET  /health
  - POST /predict_churn
  - POST /segment_customer
  - GET  /explain/{customer_id}
  - GET  /segment_matrix
"""

from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from src.api.main import app, load_api_artifacts


class TestAPIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api_artifacts()
        cls.client = TestClient(app)

    def test_health_contract(self):
        """Assert /health response schema and model readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        required_keys = {"status", "model_loaded", "feature_engineer_loaded", "model_path", "version"}
        self.assertTrue(required_keys.issubset(data.keys()))
        self.assertTrue(data["model_loaded"], "Model must be loaded and ready for serving.")
        self.assertEqual(data["status"], "healthy")

    def test_predict_churn_contract(self):
        """Assert /predict_churn schema, probability bounds [0, 1], and risk tiers."""
        payload = {
            "Tenure": 12.0,
            "CityTier": 1,
            "WarehouseToHome": 10.0,
            "HourSpendOnApp": 3.0,
            "NumberOfDeviceRegistered": 3,
            "SatisfactionScore": 3,
            "NumberOfAddress": 2,
            "Complain": 0,
            "OrderAmountHikeFromlastYear": 15.0,
            "CouponUsed": 1.0,
            "OrderCount": 2.0,
            "DaySinceLastOrder": 5.0,
            "CashbackAmount": 160.0,
            "PreferredLoginDevice": "Mobile Phone",
            "PreferredPaymentMode": "Debit Card",
            "Gender": "Female",
            "PreferedOrderCat": "Laptop & Accessory",
            "MaritalStatus": "Single"
        }
        response = self.client.post("/predict_churn", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        required_keys = {
            "customer_id", "churn_probability", "churn_prediction",
            "risk_tier", "confidence_score", "model_version"
        }
        self.assertTrue(required_keys.issubset(data.keys()))

        self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
        self.assertIn(data["churn_prediction"], [0, 1])
        self.assertIn(data["risk_tier"], ["High Risk", "Medium Risk", "Low Risk"])
        self.assertTrue(0.0 <= data["confidence_score"] <= 1.0)

    def test_segment_customer_contract(self):
        """Assert /segment_customer returns valid cluster ID, persona, and strategy."""
        payload = {
            "customer_id": "CUST_TEST_001",
            "Recency": 25.0,
            "Frequency": 8.0,
            "Monetary": 2500.0
        }
        response = self.client.post("/segment_customer", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        required_keys = {"customer_id", "cluster_id", "persona", "recommended_strategy"}
        self.assertTrue(required_keys.issubset(data.keys()))

        self.assertIsInstance(data["cluster_id"], int)
        self.assertIsInstance(data["persona"], str)
        self.assertIsInstance(data["recommended_strategy"], str)
        self.assertIn(
            data["persona"],
            ["Champions", "Loyal / Steady", "At Risk", "Hibernating", "New / Low Engagement"]
        )

    def test_explain_customer_contract_valid_and_invalid(self):
        """Assert /explain/{customer_id} returns SHAP drivers, protective factors, and 404 for unknown IDs."""
        # 1. Valid customer from processed test set
        response_valid = self.client.get("/explain/54007")
        self.assertEqual(response_valid.status_code, 200)

        data = response_valid.json()
        required_keys = {
            "customer_id", "churn_probability", "risk_tier",
            "top_risk_drivers", "top_protective_factors"
        }
        self.assertTrue(required_keys.issubset(data.keys()))
        self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
        self.assertIn(data["risk_tier"], ["High Risk", "Medium Risk", "Low Risk"])

        self.assertIsInstance(data["top_risk_drivers"], list)
        self.assertIsInstance(data["top_protective_factors"], list)
        self.assertGreater(len(data["top_risk_drivers"]) + len(data["top_protective_factors"]), 0)

        # Inspect driver structure
        if data["top_risk_drivers"]:
            driver = data["top_risk_drivers"][0]
            self.assertIn("feature", driver)
            self.assertIn("shap_value", driver)
            self.assertIn("impact", driver)

        # 2. Unknown customer ID returns 404
        response_invalid = self.client.get("/explain/UNKNOWN_CUSTOMER_9999999")
        self.assertEqual(response_invalid.status_code, 404)

    def test_segment_matrix_contract(self):
        """Assert /segment_matrix returns cross-tabulated risk matrix structure."""
        response = self.client.get("/segment_matrix")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("total_customers", data)
        self.assertIn("risk_matrix", data)
        self.assertGreater(data["total_customers"], 0)


if __name__ == "__main__":
    unittest.main()
