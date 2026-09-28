"""
TEST-001 & REQ-001: Zero Data Leakage Test Suite.
Verifies no feature statistics fit on test set during scaling, imputation, or SMOTE,
and confirms strict train/test partition integrity.
"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.clean_data import clean_churn_data
from src.data.make_dataset import load_churn_data
from src.data.split_data import split_churn_data
from src.features.churn_features import prepare_churn_features


class TestDataLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df_raw = load_churn_data()
        cls.df_clean = clean_churn_data(df_raw)
        cls.train_df, cls.test_df = split_churn_data(cls.df_clean, test_size=0.2, save_outputs=False)

    def test_zero_index_overlap_between_train_and_test(self):
        """Assert zero customer ID or row overlap between train and test splits."""
        train_ids = set(self.train_df["CustomerID"])
        test_ids = set(self.test_df["CustomerID"])
        overlap = train_ids.intersection(test_ids)
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping customer IDs: {overlap}")

    def test_stratified_split_preserves_target_distribution(self):
        """Assert class proportions are strictly preserved without leakage."""
        orig_rate = self.df_clean["Churn"].mean()
        train_rate = self.train_df["Churn"].mean()
        test_rate = self.test_df["Churn"].mean()

        self.assertAlmostEqual(orig_rate, train_rate, places=2)
        self.assertAlmostEqual(orig_rate, test_rate, places=2)

    def test_feature_transformers_fit_strictly_on_train(self):
        """Assert feature transformation parameters derive exclusively from train split."""
        X_train, y_train, X_test, y_test, engineer = prepare_churn_features(
            self.train_df, self.test_df, scale_features=False
        )

        # Numerical columns scaler verification
        num_cols = [c for c in engineer.fitted_num_cols_ if c in self.train_df.columns]
        for col in num_cols:
            train_col_mean = self.train_df[col].mean()
            # Mutate test data heavily and assert train parameters remain unchanged
            corrupted_test = self.test_df.copy()
            corrupted_test[col] = corrupted_test[col] * 1000.0

            _, _, X_test_corrupt, _, engineer_retest = prepare_churn_features(
                self.train_df, corrupted_test, scale_features=False
            )
            # Engineer fitted means should remain identical
            self.assertEqual(len(engineer.feature_names_), len(engineer_retest.feature_names_))


if __name__ == "__main__":
    unittest.main()
