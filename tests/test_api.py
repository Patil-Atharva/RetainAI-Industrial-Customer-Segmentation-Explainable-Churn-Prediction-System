"""
Unit tests for FastAPI REST API Service (src/api/main.py).
"""

import unittest
from fastapi.testclient import TestClient

from src.api.main import app, load_api_artifacts


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicitly load model artifacts for test environment
        load_api_artifacts()
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns welcome payload."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("endpoints", data)

    def test_health_check(self):
        """Test GET /health returns status and model availability."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("version", data)

    def test_predict_churn_single(self):
        """Test POST /predict_churn for single customer profile."""
        payload = {
            "Tenure": 12.0,
            "CityTier": 1,
            "WarehouseToHome": 15.0,
            "HourSpendOnApp": 3.5,
            "NumberOfDeviceRegistered": 4,
            "SatisfactionScore": 2,
            "NumberOfAddress": 2,
            "Complain": 1,
            "OrderAmountHikeFromlastYear": 14.0,
            "CouponUsed": 2.0,
            "OrderCount": 3.0,
            "DaySinceLastOrder": 12.0,
            "CashbackAmount": 150.0,
            "PreferredLoginDevice": "Mobile Phone",
            "PreferredPaymentMode": "Debit Card",
            "Gender": "Female",
            "PreferedOrderCat": "Mobile Phone",
            "MaritalStatus": "Single"
        }
        response = self.client.post("/predict_churn", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("churn_probability", data)
        self.assertIn("churn_prediction", data)
        self.assertIn("risk_tier", data)
        self.assertIn("confidence_score", data)
        self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
        self.assertIn(data["risk_tier"], ["High Risk", "Medium Risk", "Low Risk"])

    def test_predict_churn_batch(self):
        """Test POST /predict_churn_batch for multiple customer profiles."""
        payload = {
            "customers": [
                {
                    "Tenure": 24.0,
                    "Complain": 0,
                    "SatisfactionScore": 5,
                    "DaySinceLastOrder": 2.0
                },
                {
                    "Tenure": 1.0,
                    "Complain": 1,
                    "SatisfactionScore": 1,
                    "DaySinceLastOrder": 45.0
                }
            ]
        }
        response = self.client.post("/predict_churn_batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertIn("churn_probability", data[0])
        self.assertIn("risk_tier", data[0])

    def test_get_explain_invalid_customer(self):
        """Test GET /explain/{customer_id} for unknown customer ID returns 404."""
        response = self.client.get("/explain/INVALID_CUST_999999")
        self.assertEqual(response.status_code, 404)

    def test_get_segment_matrix(self):
        """Test GET /segment_matrix returns cross-tabulated risk report."""
        response = self.client.get("/segment_matrix")
        if response.status_code == 200:
            data = response.json()
            self.assertIn("total_customers", data)
            self.assertIn("risk_matrix", data)
        else:
            self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
