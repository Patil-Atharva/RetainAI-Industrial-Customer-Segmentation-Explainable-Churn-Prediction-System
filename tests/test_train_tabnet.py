"""
Unit tests for Deep Learning Tabular Model module (src/models/churn/train_tabnet.py).
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.models.churn.train_tabnet import (
    train_and_evaluate_tabnet,
    train_deep_tabular_net,
)


class TestTrainTabNet(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.read_csv("data/processed/churn_train.csv")

    def test_train_deep_tabular_net(self):
        """Verify Deep Neural Network model fitting."""
        np.random.seed(42)
        X_train_dummy = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        y_train_dummy = pd.Series(np.random.choice([0, 1], size=100))

        mlp = train_deep_tabular_net(X_train_dummy, y_train_dummy)
        self.assertIsInstance(mlp, MLPClassifier)
        self.assertTrue(hasattr(mlp, "classes_"))

    def test_train_tabnet_pipeline(self):
        """Verify end-to-end Deep Learning training and evaluation benchmark."""
        metrics, models = train_and_evaluate_tabnet(save_models=False)
        
        self.assertIn("Deep Tabular Neural Net", metrics)
        self.assertIn("deep_tabular_net", models)
        self.assertIsInstance(models["deep_tabular_net"], MLPClassifier)

        # Assert Deep Learning benchmark performance targets
        dl_metrics = metrics["Deep Tabular Neural Net"]
        self.assertGreaterEqual(dl_metrics["roc_auc"], 0.88)
        self.assertGreaterEqual(dl_metrics["pr_auc"], 0.82)
        self.assertGreaterEqual(dl_metrics["f1_score"], 0.78)


if __name__ == "__main__":
    unittest.main()
