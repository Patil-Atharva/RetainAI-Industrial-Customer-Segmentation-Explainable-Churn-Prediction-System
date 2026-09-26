"""
Unit tests for tabular feature engineering module (src/features/churn_features.py).
"""

import unittest
import pandas as pd
from src.features.churn_features import ChurnFeatureEngineer, prepare_churn_features


class TestChurnFeatures(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.read_csv("data/processed/churn_train.csv")
        self.test_df = pd.read_csv("data/processed/churn_test.csv")

    def test_churn_feature_engineer_fit_transform(self):
        """Verify feature engineering outputs zero nulls and correct feature dimensions."""
        engineer = ChurnFeatureEngineer(scale_features=False)
        X_train_raw = self.train_df.drop(columns=["Churn", "CustomerID"], errors="ignore")
        
        X_train = engineer.fit_transform(X_train_raw)
        self.assertEqual(len(X_train), len(self.train_df))
        self.assertEqual(X_train.isnull().sum().sum(), 0)
        self.assertEqual(len(engineer.feature_names_), 35)

    def test_prepare_churn_features_wrapper(self):
        """Verify train-test transformation wrapper function."""
        X_train, y_train, X_test, y_test, engineer = prepare_churn_features(
            self.train_df, self.test_df, scale_features=True
        )

        self.assertEqual(X_train.shape[0], len(self.train_df))
        self.assertEqual(X_test.shape[0], len(self.test_df))
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_train.isnull().sum().sum(), 0)
        self.assertEqual(X_test.isnull().sum().sum(), 0)

        # Assert zero data leakage: Check that scaler mean is close to 0 on train
        self.assertAlmostEqual(X_train.mean().mean(), 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
