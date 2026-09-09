"""
Unit tests for src/models/segmentation/cluster_profiler.py.
"""
import unittest

import numpy as np
import pandas as pd

from src.models.segmentation.cluster_profiler import assign_personas, profile_clusters


class TestClusterProfiler(unittest.TestCase):
    def setUp(self):
        # 2 obviously different clusters: "big spenders who just bought" (0)
        # vs "quiet, low-value, haven't bought in ages" (1)
        self.rfm = pd.DataFrame(
            {
                "Recency": [5, 3, 4, 200, 210, 190],
                "Frequency": [15, 18, 20, 1, 2, 1],
                "Monetary": [900, 950, 1000, 20, 15, 25],
            }
        )
        self.labels = np.array([0, 0, 0, 1, 1, 1])

    def test_profile_clusters_shape_and_columns(self):
        profile = profile_clusters(self.rfm, self.labels)
        self.assertEqual(len(profile), 2)
        expected_cols = {
            "Recency_mean", "Frequency_mean", "Monetary_mean",
            "Customer_Count", "Pct_of_Base",
        }
        self.assertTrue(expected_cols.issubset(set(profile.columns)))

    def test_profile_clusters_percentages_sum_to_100(self):
        profile = profile_clusters(self.rfm, self.labels)
        self.assertAlmostEqual(profile["Pct_of_Base"].sum(), 100.0, places=4)

    def test_assign_personas_labels_champions_and_hibernating(self):
        profile = profile_clusters(self.rfm, self.labels)
        personas = assign_personas(profile)
        # cluster 0: recent + frequent + big spend -> Champions
        # cluster 1: stale + infrequent + low spend -> Hibernating
        persona_by_cluster = personas["Persona"].to_dict()
        champions_cluster = self.rfm.loc[self.labels == 0].index
        self.assertIn("Champions", personas["Persona"].values)
        self.assertIn("Hibernating", personas["Persona"].values)


if __name__ == "__main__":
    unittest.main()
