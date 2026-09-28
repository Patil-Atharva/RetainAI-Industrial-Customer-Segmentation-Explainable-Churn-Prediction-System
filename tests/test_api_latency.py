"""
TEST-006 & CON-002: REST API Latency SLA Benchmark Test.
Benchmarks inference latency under 100 requests to /predict_churn,
and asserts that p95 response time remains strictly under 150 ms.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app, load_api_artifacts


class TestAPILatency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api_artifacts()
        cls.client = TestClient(app)
        cls.sample_payload = {
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

    def _single_request(self) -> float:
        start_t = time.perf_counter()
        resp = self.client.post("/predict_churn", json=self.sample_payload)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.assertEqual(resp.status_code, 200)
        return elapsed_ms

    def test_single_request_latency_under_150ms(self):
        """Assert single request inference latency stays under 150 ms (CON-002)."""
        latencies = [self._single_request() for _ in range(10)]
        median_ms = np.median(latencies)
        self.assertLess(
            median_ms, 150.0, f"Median single request latency ({median_ms:.2f} ms) exceeded 150 ms SLA!"
        )

    def test_benchmark_100_requests_p95_under_150ms(self):
        """Benchmark 100 requests and assert p95 < 150 ms (CON-002)."""
        # Warmup
        for _ in range(5):
            self._single_request()

        latencies = [self._single_request() for _ in range(100)]

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        print(f"\n[LATENCY BENCHMARK] 100 Requests -> p50: {p50:.2f} ms | p95: {p95:.2f} ms | p99: {p99:.2f} ms")

        self.assertLess(
            p95, 150.0, f"p95 latency ({p95:.2f} ms) exceeded CON-002 boundary (< 150 ms)!"
        )


if __name__ == "__main__":
    unittest.main()
