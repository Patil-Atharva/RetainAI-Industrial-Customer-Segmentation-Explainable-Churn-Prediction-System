"""
RFM (Recency, Frequency, Monetary) Feature Engineering Module.
Owned by: Junior Engineer.

Computes per-customer RFM metrics from cleaned Online Retail II transaction
data, then applies a Yeo-Johnson power transform + Z-score scaling so the
features are ready for K-Means / GMM clustering (spec section 7.1, 8.1).
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler

logger = logging.getLogger(__name__)


def build_rfm_table(
    df: pd.DataFrame,
    customer_col: str = "Customer ID",
    invoice_col: str = "Invoice",
    date_col: str = "InvoiceDate",
    amount_col: str = "TotalAmount",
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Aggregates cleaned transaction-level data into one row per customer with
    Recency, Frequency, and Monetary columns.

    Recency  (R): days between snapshot_date and the customer's most recent
                  invoice date. Lower = purchased more recently.
    Frequency(F): count of *distinct* invoices placed by the customer.
    Monetary (M): total money spent by the customer across all transactions.

    Args:
        df: Cleaned retail transactions -- the output of
            src.data.clean_data.clean_retail_data(), which already provides
            'TotalAmount' = Quantity * Price and a parsed 'InvoiceDate'.
        customer_col, invoice_col, date_col, amount_col: source column names.
        snapshot_date: the reference "today" recency is measured against.
            Defaults to one day after the last transaction in the dataset,
            the standard convention for RFM on a fixed historical extract.

    Returns:
        DataFrame indexed by customer_col with columns
        ['Recency', 'Frequency', 'Monetary'].
    """
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    logger.info("Building RFM table using snapshot_date=%s", snapshot_date)

    rfm = df.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(invoice_col, "nunique"),
        Monetary=(amount_col, "sum"),
    )

    # Guard against non-positive Monetary values (can appear if a customer's
    # returns/discounts net out to <= 0 across the observation window).
    rfm = rfm[rfm["Monetary"] > 0]

    logger.info("RFM table built for %s unique customers.", f"{len(rfm):,}")
    return rfm


class RFMScaler(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer implementing the segmentation branch's
    pre-clustering step (PAT-001: must extend BaseEstimator/TransformerMixin
    so it can be dropped straight into an sklearn Pipeline).

    Two-stage transform:
      1. Yeo-Johnson power transform -- corrects the heavy right-skew that is
         typical of Recency/Frequency/Monetary (a handful of customers spend
         or order far more than the median customer). Unlike log transforms,
         Yeo-Johnson works even when values are zero (see spec eq. 8.1).
      2. StandardScaler (Z-score)    -- puts R, F, M on the same scale so
         Monetary (often in the thousands) doesn't dominate the Euclidean
         distance K-Means/GMM rely on.
    """

    def __init__(self) -> None:
        self.power_transformer = PowerTransformer(method="yeo-johnson")
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None) -> "RFMScaler":
        X_power = self.power_transformer.fit_transform(X)
        self.scaler.fit(X_power)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_power = self.power_transformer.transform(X)
        return self.scaler.transform(X_power)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        X_power = self.power_transformer.fit_transform(X)
        return self.scaler.fit_transform(X_power)


def build_scaled_rfm_features(
    df: pd.DataFrame, **rfm_kwargs
) -> Tuple[pd.DataFrame, np.ndarray, RFMScaler]:
    """
    Convenience wrapper used by the clustering scripts: builds the raw RFM
    table, then scales it in one call.

    Returns:
        rfm_raw:  the unscaled RFM DataFrame (kept around for human-readable
                  cluster profiling later -- you never want to profile
                  clusters using Z-scored numbers).
        X_scaled: numpy array of scaled features, ready for KMeans/GMM.
        scaler:   the *fitted* RFMScaler (reusable to transform new customers
                  the same way at inference time).
    """
    rfm_raw = build_rfm_table(df, **rfm_kwargs)
    scaler = RFMScaler()
    X_scaled = scaler.fit_transform(rfm_raw[["Recency", "Frequency", "Monetary"]])
    return rfm_raw, X_scaled, scaler
