"""
Unit tests for src/features/rfm_features.py.
Uses fully synthetic transaction data so these tests run without needing the
real UCI/Kaggle datasets on disk (unlike the Lead's ingestion tests).
"""
import unittest

import numpy as np
import pandas as pd

from src.features.rfm_features import RFMScaler, build_rfm_table, build_scaled_rfm_features


def _make_synthetic_transactions(seed: int = 0, n_customers: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")
    rows = []
    for cid in range(n_customers):
        n_orders = rng.integers(1, 10)
        for _ in range(n_orders):
            rows.append(
                {
                    "Customer ID": str(cid),
                    "Invoice": f"{cid}-{rng.integers(0, 100000)}",
                    "InvoiceDate": base_date + pd.Timedelta(days=int(rng.integers(0, 365))),
                    "TotalAmount": float(rng.gamma(2, 20)),
                }
            )
    return pd.DataFrame(rows)


class TestBuildRFMTable(unittest.TestCase):
    def setUp(self):
        self.df = _make_synthetic_transactions()

    def test_output_shape_and_columns(self):
        rfm = build_rfm_table(self.df)
        self.assertListEqual(list(rfm.columns), ["Recency", "Frequency", "Monetary"])
        # every synthetic customer has >0 monetary value, so none should be dropped
        self.assertEqual(len(rfm), self.df["Customer ID"].nunique())

    def test_recency_is_non_negative(self):
        rfm = build_rfm_table(self.df)
        self.assertTrue((rfm["Recency"] >= 0).all())

    def test_frequency_matches_distinct_invoice_count(self):
        rfm = build_rfm_table(self.df)
        expected_freq = self.df.groupby("Customer ID")["Invoice"].nunique()
        pd.testing.assert_series_equal(
            rfm["Frequency"].sort_index(),
            expected_freq.sort_index(),
            check_names=False,
        )

    def test_monetary_matches_total_amount_sum(self):
        rfm = build_rfm_table(self.df)
        expected_monetary = self.df.groupby("Customer ID")["TotalAmount"].sum()
        pd.testing.assert_series_equal(
            rfm["Monetary"].sort_index(),
            expected_monetary.sort_index(),
            check_names=False,
        )

    def test_custom_snapshot_date_changes_recency(self):
        rfm_default = build_rfm_table(self.df)
        later_snapshot = self.df["InvoiceDate"].max() + pd.Timedelta(days=30)
        rfm_later = build_rfm_table(self.df, snapshot_date=later_snapshot)
        # every recency value should be exactly 29 days larger
        # (default snapshot = max_date + 1 day; later snapshot = max_date + 30 days)
        diff = (rfm_later["Recency"] - rfm_default["Recency"]).unique()
        self.assertEqual(list(diff), [29])


class TestRFMScaler(unittest.TestCase):
    def setUp(self):
        self.df = _make_synthetic_transactions()
        self.rfm = build_rfm_table(self.df)[["Recency", "Frequency", "Monetary"]]

    def test_fit_transform_output_shape(self):
        scaler = RFMScaler()
        X_scaled = scaler.fit_transform(self.rfm)
        self.assertEqual(X_scaled.shape, self.rfm.shape)

    def test_scaled_features_are_approximately_standardized(self):
        scaler = RFMScaler()
        X_scaled = scaler.fit_transform(self.rfm)
        # after Yeo-Johnson + StandardScaler, each column should have
        # mean ~0 and std ~1
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-6)

    def test_fit_then_transform_matches_fit_transform(self):
        scaler_a = RFMScaler()
        out_a = scaler_a.fit_transform(self.rfm)

        scaler_b = RFMScaler()
        scaler_b.fit(self.rfm)
        out_b = scaler_b.transform(self.rfm)

        np.testing.assert_allclose(out_a, out_b, atol=1e-8)


class TestBuildScaledRFMFeatures(unittest.TestCase):
    def test_wrapper_returns_consistent_shapes(self):
        df = _make_synthetic_transactions()
        rfm_raw, X_scaled, scaler = build_scaled_rfm_features(df)
        self.assertEqual(len(rfm_raw), X_scaled.shape[0])
        self.assertEqual(X_scaled.shape[1], 3)
        self.assertIsInstance(scaler, RFMScaler)


if __name__ == "__main__":
    unittest.main()
