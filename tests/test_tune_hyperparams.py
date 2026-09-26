"""
Unit tests for Hyperparameter Tuning Engine (src/models/churn/tune_hyperparams.py).
"""

import unittest
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.models.churn.tune_hyperparams import (
    tune_hist_gradient_boosting,
    tune_random_forest,
)


class TestTuneHyperparams(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.read_csv("data/processed/churn_train.csv")
        # Subsample for fast unit testing execution
        self.sample_df = self.train_df.sample(n=300, random_state=42)
        
        from src.features.churn_features import prepare_churn_features
        self.X_train, self.y_train, _ = prepare_churn_features(self.sample_df, scale_features=False)

    def test_tune_hist_gradient_boosting_fast(self):
        """Verify HistGradientBoosting hyperparameter tuning optimization."""
        model, params = tune_hist_gradient_boosting(self.X_train, self.y_train, n_trials=3)
        self.assertIsInstance(model, HistGradientBoostingClassifier)
        self.assertIsInstance(params, dict)
        self.assertTrue(hasattr(model, "classes_"))

    def test_tune_random_forest_fast(self):
        """Verify Random Forest hyperparameter tuning optimization."""
        model, params = tune_random_forest(self.X_train, self.y_train, n_trials=3)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIsInstance(params, dict)
        self.assertTrue(hasattr(model, "classes_"))


if __name__ == "__main__":
    unittest.main()
