"""
Dimensionality Reduction / Manifold Visualization Module.
Owned by: Junior Engineer.

Projects the scaled RFM feature space down to 2D/3D, both for a quick visual
sanity check and to power the Streamlit UMAP Cluster Explorer (TASK-J08).

Two techniques, per ALT-001 in the architecture doc:
- PCA:  fast, linear, preserves *global* variance -- a good, cheap baseline.
- UMAP: non-linear, preserves local *and* global neighborhood structure --
  the primary visualization technique, chosen over t-SNE for better
  preservation of global structure and much faster runtime on larger data.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when umap-learn is missing
    UMAP_AVAILABLE = False
    logger.warning("umap-learn is not installed; umap_reduce() will raise if called.")


def pca_reduce(
    X: np.ndarray, n_components: int = 2, random_state: int = 42
) -> pd.DataFrame:
    """
    Linear projection via Principal Component Analysis.

    Returns:
        DataFrame with columns ['PC1', 'PC2', ...].
    """
    model = PCA(n_components=n_components, random_state=random_state)
    coords = model.fit_transform(X)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    df_coords = pd.DataFrame(coords, columns=columns)
    logger.info(
        "PCA explained variance ratio (%d components): %s",
        n_components, np.round(model.explained_variance_ratio_, 4).tolist(),
    )
    return df_coords


def umap_reduce(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Non-linear projection via UMAP.

    Args:
        n_neighbors: balances local vs. global structure preservation
            (low -> tight local clusters, high -> more global structure).
        min_dist: minimum distance allowed between points in the low-D
            embedding (lower -> tighter, more visually clumped clusters).

    Returns:
        DataFrame with columns ['UMAP1', 'UMAP2', ...].
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is required for umap_reduce(). Install it with "
            "`pip install umap-learn` (already pinned in requirements.txt)."
        )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)
    columns = [f"UMAP{i + 1}" for i in range(n_components)]
    return pd.DataFrame(coords, columns=columns)
