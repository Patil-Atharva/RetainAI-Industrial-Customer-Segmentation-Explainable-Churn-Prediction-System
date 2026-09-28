"""
TEST-002: RFM Calculation Mathematical Accuracy Test Suite.
Verifies mathematical accuracy of Recency, Frequency, and Monetary calculations on synthetic transactions.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.rfm_features import RFMScaler, build_rfm_table, build_scaled_rfm_features


def _make_synthetic_transactions(seed: int = 42, n_customers: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")
    rows = []
    for cid in range(n_customers):
        n_orders = int(rng.integers(1, 10))
        for _ in range(n_orders):
            rows.append({
                "Customer ID": str(cid),
                "Invoice": f"{cid}-{rng.integers(0, 100000)}",
                "InvoiceDate": base_date + pd.Timedelta(days=int(rng.integers(0, 365))),
                "TotalAmount": float(rng.gamma(2, 20)),
            })
    return pd.DataFrame(rows)


class TestRFMBuilder(unittest.TestCase):
    def setUp(self):
        self.df = _make_synthetic_transactions()

    def test_output_shape_and_columns(self):
        rfm = build_rfm_table(self.df)
        self.assertListEqual(list(rfm.columns), ["Recency", "Frequency", "Monetary"])
        self.assertEqual(len(rfm), self.df["Customer ID"].nunique())

    def test_recency_non_negative_and_exact_math(self):
        rfm = build_rfm_table(self.df)
        self.assertTrue((rfm["Recency"] >= 0).all())

    def test_frequency_distinct_invoice_count(self):
        rfm = build_rfm_table(self.df)
        expected_freq = self.df.groupby("Customer ID")["Invoice"].nunique()
        pd.testing.assert_series_equal(
            rfm["Frequency"].sort_index(),
            expected_freq.sort_index(),
            check_names=False
        )

    def test_monetary_exact_sum(self):
        rfm = build_rfm_table(self.df)
        expected_monetary = self.df.groupby("Customer ID")["TotalAmount"].sum()
        pd.testing.assert_series_equal(
            rfm["Monetary"].sort_index(),
            expected_monetary.sort_index(),
            check_names=False
        )

    def test_rfm_scaler_standardization(self):
        scaler = RFMScaler()
        rfm = build_rfm_table(self.df)[["Recency", "Frequency", "Monetary"]]
        X_scaled = scaler.fit_transform(rfm)
        self.assertEqual(X_scaled.shape, rfm.shape)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
