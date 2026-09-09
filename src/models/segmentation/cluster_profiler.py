"""
Cluster Profiling & Business Persona Assignment Module.
Owned by: Junior Engineer.

Turns raw numeric Cluster_IDs into human-readable business personas
(e.g. "Champions", "At Risk", "Hibernating") by comparing each cluster's
average Recency/Frequency/Monetary against the rest of the customer base.
This is the artifact marketing teams actually consume (spec section 3.2),
and it's the piece the Lead's risk_matrix.py will join against.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def profile_clusters(rfm_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Computes per-cluster summary statistics.

    Args:
        rfm_df: DataFrame with ['Recency', 'Frequency', 'Monetary'] columns,
            one row per customer, in the SAME row order as `labels`
            (i.e. the un-scaled rfm_raw table from build_rfm_table, not the
            scaled numpy array -- profiling should be done in real units,
            not Z-scores, or "average recency = -0.3" is meaningless to a
            marketer).
        labels: cluster assignment array from fit_kmeans() / fit_gmm().

    Returns:
        DataFrame indexed by Cluster_ID with mean R/F/M, customer count, and
        percentage share of the total customer base, sorted by spend.
    """
    df = rfm_df.copy()
    df["Cluster_ID"] = labels

    profile = df.groupby("Cluster_ID").agg(
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        Customer_Count=("Recency", "size"),
    )
    profile["Pct_of_Base"] = (profile["Customer_Count"] / len(df) * 100).round(2)
    return profile.sort_values("Monetary_mean", ascending=False)


def assign_personas(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps each cluster to a business persona using simple rules relative to
    the *median cluster* (not individual customers) on each RFM dimension:

    - Champions:            recent, frequent, and big-spending
    - At Risk:               historically frequent/high-spend, but recency
                              has drifted high (gone quiet) -- valuable
                              customers worth a retention campaign
    - New / Low Engagement:  recent but not yet frequent/high-spend
    - Hibernating:            high recency, low frequency, low spend
    - Loyal / Steady:         everything else (decent but not top-tier)

    This is intentionally a transparent rule layer rather than a second
    black-box model -- a marketer can look at the numbers and see exactly
    why a cluster got its label, mirroring the project's broader
    "explainability over black boxes" philosophy (spec 1.2).
    """
    df = profile_df.copy()

    recency_median = df["Recency_mean"].median()
    frequency_median = df["Frequency_mean"].median()
    monetary_median = df["Monetary_mean"].median()

    def label_row(row: pd.Series) -> str:
        recent = row["Recency_mean"] <= recency_median
        frequent = row["Frequency_mean"] >= frequency_median
        big_spender = row["Monetary_mean"] >= monetary_median

        if recent and frequent and big_spender:
            return "Champions"
        if not recent and (frequent or big_spender):
            return "At Risk"
        if recent and not frequent and not big_spender:
            return "New / Low Engagement"
        if not recent and not frequent and not big_spender:
            return "Hibernating"
        return "Loyal / Steady"

    df["Persona"] = df.apply(label_row, axis=1)
    logger.info("Assigned personas: %s", df["Persona"].value_counts().to_dict())
    return df
