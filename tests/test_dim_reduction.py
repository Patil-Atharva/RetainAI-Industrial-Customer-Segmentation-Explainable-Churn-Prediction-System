"""
Unit tests for src/models/segmentation/dim_reduction.py.
"""
import unittest

from sklearn.datasets import make_blobs

from src.models.segmentation.dim_reduction import UMAP_AVAILABLE, pca_reduce


class TestPCAReduce(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=200, centers=3, n_features=5, random_state=42)

    def test_output_shape_2d(self):
        coords = pca_reduce(self.X, n_components=2)
        self.assertEqual(coords.shape, (200, 2))
        self.assertListEqual(list(coords.columns), ["PC1", "PC2"])

    def test_output_shape_3d(self):
        coords = pca_reduce(self.X, n_components=3)
        self.assertEqual(coords.shape, (200, 3))
        self.assertListEqual(list(coords.columns), ["PC1", "PC2", "PC3"])


@unittest.skipUnless(UMAP_AVAILABLE, "umap-learn not installed in this environment")
class TestUMAPReduce(unittest.TestCase):
    def test_output_shape_2d(self):
        from src.models.segmentation.dim_reduction import umap_reduce

        X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
        coords = umap_reduce(X, n_components=2, n_neighbors=10)
        self.assertEqual(coords.shape, (100, 2))
        self.assertListEqual(list(coords.columns), ["UMAP1", "UMAP2"])


if __name__ == "__main__":
    unittest.main()
