"""
Gaussian Mixture Model (GMM) Soft-Clustering Module.
Owned by: Junior Engineer.

Provides probabilistic ("soft") clustering as an alternative to K-Means's
hard assignments: every customer gets a probability of belonging to each
segment rather than a single label. Also benchmarks the four covariance
structures required by the spec (section 6.1).
"""
import logging
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

COVARIANCE_TYPES: Sequence[str] = ("full", "tied", "diag", "spherical")


def evaluate_gmm_grid(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    covariance_types: Sequence[str] = COVARIANCE_TYPES,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Grid-searches k x covariance_type and records BIC, AIC, and Silhouette
    for each combination.

    - bic / aic: information criteria that trade off model fit against model
      complexity; LOWER is better. BIC penalizes extra free parameters more
      aggressively than AIC, which matters here because 'full' covariance has
      far more parameters than 'spherical' -- BIC discourages an overfit,
      overly-elaborate segmentation.
    - silhouette: computed on the *hard* assignment (argmax of the posterior
      responsibilities) so it is directly comparable to the K-Means results
      in kmeans_model.py.

    Returns:
        DataFrame with columns ['k', 'covariance_type', 'bic', 'aic', 'silhouette'].
    """
    records = []
    for cov_type in covariance_types:
        for k in k_range:
            model = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=random_state,
                n_init=5,
            )
            model.fit(X)
            labels = model.predict(X)

            # Silhouette is undefined with fewer than 2 populated clusters;
            # skip degenerate fits rather than letting them crash the grid.
            if len(np.unique(labels)) < 2:
                logger.warning(
                    "Skipping k=%d, covariance_type=%s: collapsed to a single cluster.",
                    k, cov_type,
                )
                continue

            records.append(
                {
                    "k": k,
                    "covariance_type": cov_type,
                    "bic": model.bic(X),
                    "aic": model.aic(X),
                    "silhouette": silhouette_score(X, labels),
                }
            )

    results = pd.DataFrame.from_records(records)
    logger.info("Evaluated %d GMM (k, covariance_type) combinations.", len(results))
    return results


def select_best_gmm_config(metrics_df: pd.DataFrame) -> Tuple[int, str]:
    """Picks the (k, covariance_type) combination with the lowest BIC."""
    best_row = metrics_df.loc[metrics_df["bic"].idxmin()]
    best_k, best_cov = int(best_row["k"]), str(best_row["covariance_type"])
    logger.info(
        "Selected GMM config: k=%d, covariance_type='%s' (BIC=%.2f)",
        best_k, best_cov, best_row["bic"],
    )
    return best_k, best_cov


def fit_gmm(
    X: np.ndarray, k: int, covariance_type: str = "full", random_state: int = 42
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    Fits the final GMM model.

    Returns:
        (fitted_model, hard_labels, soft_probabilities)
        soft_probabilities has shape (n_samples, k): each row sums to 1.0
        and gives the probability of that customer belonging to each segment
        -- useful for flagging customers who sit "on the fence" between two
        personas (e.g. 48% Champion / 45% At-Risk) rather than forcing a
        hard, over-confident label.
    """
    model = GaussianMixture(
        n_components=k, covariance_type=covariance_type, random_state=random_state, n_init=5
    )
    model.fit(X)
    labels = model.predict(X)
    probabilities = model.predict_proba(X)
    return model, labels, probabilities
