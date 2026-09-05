"""
Unit tests for data ingestion module (src/data/make_dataset.py).
Uses standard unittest library for zero-dependency execution.
"""

import unittest
import pandas as pd
from src.data.make_dataset import (
    load_all_datasets,
    load_churn_data,
    load_retail_data,
)


class TestMakeDataset(unittest.TestCase):
    def test_load_retail_data(self):
        """Verify that Online Retail II dataset loads and contains required columns."""
        df = load_retail_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterThan(len(df), 100000)
        expected_cols = [
            "Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID", "Country"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_load_churn_data(self):
        """Verify that E-Commerce Churn dataset loads and contains required columns."""
        df = load_churn_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterThan(len(df), 1000)
        expected_cols = [
            "CustomerID", "Churn", "Tenure", "PreferredLoginDevice",
            "CityTier", "SatisfactionScore"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertTrue(set(df["Churn"].unique()).issubset({0, 1}))

    def test_load_all_datasets(self):
        """Verify loading both datasets simultaneously."""
        df_retail, df_churn = load_all_datasets()
        self.assertGreaterThan(len(df_retail), 0)
        self.assertGreaterThan(len(df_churn), 0)

    def assertGreaterThan(self, val1, val2):
        self.assertTrue(val1 > val2, f"Expected {val1} > {val2}")


if __name__ == "__main__":
    unittest.main()
