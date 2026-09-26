"""
Unit tests for Gradient Boosted Decision Trees module (src/models/churn/train_gbdt.py).
"""

import unittest
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.churn.train_gbdt import (
    calculate_scale_pos_weight,
    train_and_evaluate_gbdt,
    train_hist_gradient_boosting,
)


class TestTrainGBDT(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.read_csv("data/processed/churn_train.csv")

    def test_calculate_scale_pos_weight(self):
        """Verify scale_pos_weight calculation on training targets."""
        ratio = calculate_scale_pos_weight(self.train_df["Churn"])
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 1.0)
        self.assertAlmostEqual(ratio, 4.94, places=1)

    def test_train_gbdt_pipeline(self):
        """Verify end-to-end GBDT model training and evaluation benchmarks."""
        metrics, models = train_and_evaluate_gbdt(save_models=False)
        
        self.assertIn("HistGradientBoosting", metrics)
        self.assertIn("hist_gradient_boosting", models)
        self.assertIsInstance(models["hist_gradient_boosting"], HistGradientBoostingClassifier)

        # Assert GBDT benchmark performance targets
        hgb_metrics = metrics["HistGradientBoosting"]
        self.assertGreaterEqual(hgb_metrics["roc_auc"], 0.88)
        self.assertGreaterEqual(hgb_metrics["pr_auc"], 0.82)
        self.assertGreaterEqual(hgb_metrics["f1_score"], 0.78)


if __name__ == "__main__":
    unittest.main()
