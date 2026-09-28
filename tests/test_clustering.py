"""
TEST-003 & REQ-003: Unsupervised Customer Clustering Test Suite.
Asserts Silhouette Score calculations return valid bounded values in [-1, 1],
and verifies K-Means & GMM clustering performance across k in [2, 10].
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs

from src.models.segmentation.gmm_model import evaluate_gmm_grid, fit_gmm, select_best_gmm_config
from src.models.segmentation.kmeans_model import evaluate_k_range, fit_kmeans, select_best_k


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

    def test_silhouette_score_bounded_in_neg1_to_pos1(self):
        """Assert Silhouette Score calculation returns valid bounded values in [-1, 1]."""
        metrics = evaluate_k_range(self.X, k_range=range(2, 7))
        self.assertTrue((metrics["silhouette"] >= -1.0).all())
        self.assertTrue((metrics["silhouette"] <= 1.0).all())

    def test_davies_bouldin_non_negative(self):
        """Assert Davies-Bouldin index returns non-negative scores."""
        metrics = evaluate_k_range(self.X, k_range=range(2, 7))
        self.assertTrue((metrics["davies_bouldin"] >= 0.0).all())

    def test_kmeans_cluster_reconstruction(self):
        """Verify best k selection and label assignment dimensions."""
        metrics = evaluate_k_range(self.X, k_range=range(2, 6))
        best_k = select_best_k(metrics)
        self.assertEqual(best_k, 3)

        model, labels = fit_kmeans(self.X, best_k)
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(np.unique(labels)), 3)

    def test_gmm_soft_clustering_probabilities(self):
        """Assert GMM probabilistic soft cluster assignments sum to exactly 1.0."""
        model, labels, proba = fit_gmm(self.X, k=3, covariance_type="full")
        self.assertEqual(proba.shape, (len(self.X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
