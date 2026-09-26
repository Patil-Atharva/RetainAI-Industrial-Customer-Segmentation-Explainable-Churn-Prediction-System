"""
Unit tests for baseline classification models module (src/models/churn/train_baselines.py).
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.churn_features import prepare_churn_features
from src.models.churn.train_baselines import (
    evaluate_model_performance,
    train_and_evaluate_baselines,
    train_logistic_regression,
    train_random_forest,
)


class TestTrainBaselines(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.read_csv("data/processed/churn_train.csv")
        self.test_df = pd.read_csv("data/processed/churn_test.csv")

    def test_evaluate_model_performance(self):
        """Verify metric calculation output structure and range."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.85])

        metrics = evaluate_model_performance("Test Model", y_true, y_pred, y_prob)
        self.assertEqual(metrics["model_name"], "Test Model")
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["pr_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["f1_score"] <= 1.0)

    def test_train_baselines_pipeline(self):
        """Verify end-to-end baseline training and evaluation."""
        metrics, models = train_and_evaluate_baselines(save_models=False)
        
        self.assertIn("Logistic Regression", metrics)
        self.assertIn("Random Forest", metrics)
        self.assertIn("logistic_regression", models)
        self.assertIn("random_forest", models)

        self.assertIsInstance(models["logistic_regression"], LogisticRegression)
        self.assertIsInstance(models["random_forest"], RandomForestClassifier)

        # Assert baseline performance targets
        self.assertGreater(metrics["Logistic Regression"]["roc_auc"], 0.80)
        self.assertGreater(metrics["Random Forest"]["roc_auc"], 0.90)


if __name__ == "__main__":
    unittest.main()
