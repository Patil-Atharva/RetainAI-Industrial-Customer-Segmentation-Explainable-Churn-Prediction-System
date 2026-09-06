"""
Unit tests for src/models/segmentation/kmeans_model.py and gmm_model.py.
Uses a synthetic, well-separated 3-blob dataset so cluster quality metrics
behave predictably (high silhouette, low davies-bouldin at k=3).
"""
import unittest

import numpy as np
from sklearn.datasets import make_blobs

from src.models.segmentation.gmm_model import (
    evaluate_gmm_grid,
    fit_gmm,
    select_best_gmm_config,
)
from src.models.segmentation.kmeans_model import evaluate_k_range, fit_kmeans, select_best_k


class TestKMeansModel(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=300, centers=3, cluster_std=0.6, random_state=42
        )

    def test_evaluate_k_range_returns_expected_columns(self):
        metrics = evaluate_k_range(self.X, k_range=range(2, 6))
        self.assertListEqual(
            list(metrics.columns), ["k", "inertia", "silhouette", "davies_bouldin"]
        )
        self.assertEqual(len(metrics), 4)  # k = 2,3,4,5

    def test_silhouette_bounded(self):
        metrics = evaluate_k_range(self.X, k_range=range(2, 6))
        self.assertTrue((metrics["silhouette"] >= -1).all())
        self.assertTrue((metrics["silhouette"] <= 1).all())

    def test_select_best_k_recovers_true_cluster_count(self):
        metrics = evaluate_k_range(self.X, k_range=range(2, 8))
        best_k = select_best_k(metrics)
        self.assertEqual(best_k, 3)

    def test_fit_kmeans_label_count_matches_k(self):
        model, labels = fit_kmeans(self.X, k=3)
        self.assertEqual(len(np.unique(labels)), 3)
        self.assertEqual(len(labels), len(self.X))


class TestGMMModel(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=300, centers=3, cluster_std=0.6, random_state=42
        )

    def test_evaluate_gmm_grid_returns_expected_columns(self):
        metrics = evaluate_gmm_grid(
            self.X, k_range=range(2, 5), covariance_types=("full", "diag")
        )
        self.assertListEqual(
            list(metrics.columns), ["k", "covariance_type", "bic", "aic", "silhouette"]
        )

    def test_select_best_gmm_config_returns_valid_types(self):
        metrics = evaluate_gmm_grid(
            self.X, k_range=range(2, 5), covariance_types=("full", "diag")
        )
        best_k, best_cov = select_best_gmm_config(metrics)
        self.assertIsInstance(best_k, int)
        self.assertIn(best_cov, ("full", "diag"))

    def test_fit_gmm_probabilities_sum_to_one(self):
        model, labels, proba = fit_gmm(self.X, k=3, covariance_type="full")
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(proba.shape, (len(self.X), 3))


if __name__ == "__main__":
    unittest.main()
