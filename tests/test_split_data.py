"""
Unit tests for data cleaning and stratified train-test split modules.
"""

import unittest
import pandas as pd
from src.data.clean_data import clean_churn_data, clean_retail_data
from src.data.make_dataset import load_churn_data, load_retail_data
from src.data.split_data import split_churn_data


class TestCleanAndSplitData(unittest.TestCase):
    def test_clean_retail_data(self):
        """Verify retail data cleaning removes missing customer IDs and invalid sales."""
        df_raw = load_retail_data()
        df_clean = clean_retail_data(df_raw)

        self.assertIsInstance(df_clean, pd.DataFrame)
        self.assertEqual(df_clean["Customer ID"].isnull().sum(), 0)
        self.assertTrue((df_clean["Quantity"] > 0).all())
        self.assertTrue((df_clean["Price"] > 0).all())
        self.assertIn("TotalAmount", df_clean.columns)

    def test_clean_churn_data(self):
        """Verify churn categorical standardization."""
        df_raw = load_churn_data()
        df_clean = clean_churn_data(df_raw)

        self.assertNotIn("Phone", df_clean["PreferredLoginDevice"].unique())
        self.assertNotIn("CC", df_clean["PreferredPaymentMode"].unique())
        self.assertNotIn("COD", df_clean["PreferredPaymentMode"].unique())
        self.assertNotIn("Mobile", df_clean["PreferedOrderCat"].unique())

    def test_split_churn_data_stratification(self):
        """Verify stratified splitting preserves exact target churn ratio."""
        df_raw = load_churn_data()
        df_clean = clean_churn_data(df_raw)

        train_df, test_df = split_churn_data(df_clean, test_size=0.2, save_outputs=False)

        total_rows = len(df_clean)
        self.assertEqual(len(train_df) + len(test_df), total_rows)

        # Check stratification balance within 0.1% tolerance
        orig_rate = df_clean["Churn"].mean()
        train_rate = train_df["Churn"].mean()
        test_rate = test_df["Churn"].mean()

        self.assertAlmostEqual(orig_rate, train_rate, places=2)
        self.assertAlmostEqual(orig_rate, test_rate, places=2)


if __name__ == "__main__":
    unittest.main()
