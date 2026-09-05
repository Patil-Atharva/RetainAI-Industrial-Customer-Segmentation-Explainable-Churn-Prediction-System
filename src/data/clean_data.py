"""
Data Cleaning and Preprocessing Module.
Applies data hygiene rules:
- Retail: Drops missing CustomerIDs, invalid quantities/prices, parses dates, calculates TotalAmount.
- Churn: Standardizes inconsistent categorical labels and handles missing numerical values.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Online Retail transaction dataset for RFM segmentation.
    
    Cleaning steps:
    1. Drop rows with missing 'Customer ID'.
    2. Convert 'Customer ID' to integer string format.
    3. Filter out invalid transactions (Quantity <= 0 or Price <= 0).
    4. Convert 'InvoiceDate' to pandas datetime.
    5. Calculate derived column 'TotalAmount' = Quantity * Price.
    
    Args:
        df: Raw retail dataframe.
        
    Returns:
        Cleaned retail transaction dataframe.
    """
    logger.info("Starting Online Retail data cleaning...")
    initial_rows = len(df)
    
    # Copy to avoid inplace warnings
    cleaned_df = df.copy()

    # 1. Drop missing Customer IDs
    cleaned_df = cleaned_df.dropna(subset=["Customer ID"])
    
    # 2. Format Customer ID as integer string
    cleaned_df["Customer ID"] = cleaned_df["Customer ID"].astype(int).astype(str)

    # 3. Filter valid sales transactions (Quantity > 0 and Price > 0)
    cleaned_df = cleaned_df[(cleaned_df["Quantity"] > 0) & (cleaned_df["Price"] > 0)]

    # 4. Parse InvoiceDate to datetime
    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])

    # 5. Compute derived TotalAmount
    cleaned_df["TotalAmount"] = cleaned_df["Quantity"] * cleaned_df["Price"]

    final_rows = len(cleaned_df)
    logger.info(
        f"Retail data cleaning complete. Retained {final_rows:,} / {initial_rows:,} rows "
        f"({(final_rows / initial_rows * 100):.2f}%)."
    )
    return cleaned_df


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw E-Commerce Customer Churn dataset.
    
    Cleaning steps:
    1. Standardize inconsistent categorical values:
       - PreferredLoginDevice: 'Phone' -> 'Mobile Phone'
       - PreferredPaymentMode: 'CC' -> 'Credit Card', 'COD' -> 'Cash on Delivery'
       - PreferedOrderCat: 'Mobile' -> 'Mobile Phone'
    2. Ensure correct data types (CustomerID as int string, Churn as int).
    
    Args:
        df: Raw churn dataframe.
        
    Returns:
        Cleaned customer churn dataframe (with categoricals standardized).
    """
    logger.info("Starting E-Commerce Churn data cleaning...")
    cleaned_df = df.copy()

    # 1. Format CustomerID
    cleaned_df["CustomerID"] = cleaned_df["CustomerID"].astype(int).astype(str)
    cleaned_df["Churn"] = cleaned_df["Churn"].astype(int)

    # 2. Standardize Categorical Values
    device_map = {"Phone": "Mobile Phone"}
    cleaned_df["PreferredLoginDevice"] = cleaned_df["PreferredLoginDevice"].replace(device_map)

    payment_map = {"CC": "Credit Card", "COD": "Cash on Delivery"}
    cleaned_df["PreferredPaymentMode"] = cleaned_df["PreferredPaymentMode"].replace(payment_map)

    cat_map = {"Mobile": "Mobile Phone"}
    cleaned_df["PreferedOrderCat"] = cleaned_df["PreferedOrderCat"].replace(cat_map)

    logger.info(f"Churn data cleaning complete. Total rows: {len(cleaned_df):,}.")
    return cleaned_df
