"""
TEST-004 & REQ-004: Churn Prediction Models Test Suite.
Verifies model output probabilities sum to 1.0 and pass minimum benchmark AUC thresholds (ROC-AUC >= 0.88).
"""

from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from src.features.churn_features import prepare_churn_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


class TestChurnModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_path = DATA_PROCESSED_DIR / "churn_train.csv"
        test_path = DATA_PROCESSED_DIR / "churn_test.csv"
        cls.train_df = pd.read_csv(train_path)
        cls.test_df = pd.read_csv(test_path)

        _, _, cls.X_test, cls.y_test, cls.engineer = prepare_churn_features(
            cls.train_df, cls.test_df, scale_features=False
        )

        model_path = MODEL_DIR / "tuned_champion_model.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "gbdt_hist_gradient_boosting.joblib"
        cls.model = joblib.load(model_path)

    def test_probabilities_sum_to_one(self):
        """Assert binary classification probabilities sum to 1.0 for all holdout instances."""
        probs = self.model.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_bounded_zero_one(self):
        """Assert all individual probabilities fall within [0.0, 1.0]."""
        probs = self.model.predict_proba(self.X_test)[:, 1]
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_benchmark_auc_threshold(self):
        """Verify model satisfies REQ-004 SLA: ROC-AUC >= 0.88 and PR-AUC >= 0.82."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        probs = self.model.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        pr_auc = average_precision_score(self.y_test, probs)

        self.assertGreaterEqual(
            roc_auc, 0.88, f"ROC-AUC score ({roc_auc:.4f}) failed minimum threshold 0.88 (REQ-004)"
        )
        self.assertGreaterEqual(
            pr_auc, 0.82, f"PR-AUC score ({pr_auc:.4f}) failed minimum threshold 0.82 (REQ-004)"
        )


if __name__ == "__main__":
    unittest.main()
