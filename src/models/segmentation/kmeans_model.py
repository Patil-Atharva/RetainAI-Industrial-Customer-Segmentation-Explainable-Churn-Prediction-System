"""
K-Means Clustering Module for Customer Segmentation.
Owned by: Junior Engineer.

Implements automated selection of the optimal number of clusters k using
three complementary diagnostics (Elbow/Inertia, Silhouette, Davies-Bouldin),
satisfying REQ-003 (evaluate k in [2, 10], target Silhouette >= 0.55) and
avoiding the "Arbitrary Cluster Selection" anti-pattern from the spec (4.3).
"""
import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def evaluate_k_range(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fits K-Means for every k in k_range and records the standard cluster
    quality diagnostics.

    Metrics:
    - inertia:        Sum of Squared Errors (SSE) -- sum of squared distances
                       of samples to their nearest centroid. Always decreases
                       as k grows, so it is used for the visual "elbow" plot,
                       never to pick k directly.
    - silhouette:      range [-1, 1], HIGHER is better. For a point i:
                       s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is
                       mean intra-cluster distance and b(i) is mean distance
                       to the nearest *other* cluster.
    - davies_bouldin:  >= 0, LOWER is better. Average similarity between each
                       cluster and its most-similar neighboring cluster.

    Returns:
        DataFrame with one row per k: ['k', 'inertia', 'silhouette', 'davies_bouldin'].
    """
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = model.fit_predict(X)

        record = {
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }
        records.append(record)
        logger.info(
            "k=%d: inertia=%.2f, silhouette=%.4f, davies_bouldin=%.4f",
            k, record["inertia"], record["silhouette"], record["davies_bouldin"],
        )

    return pd.DataFrame.from_records(records)


def select_best_k(metrics_df: pd.DataFrame) -> int:
    """
    Picks the k that maximizes the Silhouette Score -- the project's primary
    segmentation quality target (REQ-003, target >= 0.55).
    """
    best_row = metrics_df.loc[metrics_df["silhouette"].idxmax()]
    best_k = int(best_row["k"])
    logger.info("Selected k=%d (silhouette=%.4f)", best_k, best_row["silhouette"])
    return best_k


def fit_kmeans(
    X: np.ndarray, k: int, random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Fits the final K-Means model with the chosen k.

    Returns:
        (fitted_model, cluster_labels)
    """
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels
