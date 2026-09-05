"""
Dataset Ingestion and Validation Module.
Handles loading, raw schema validation, and fallback dataset retrieval for:
1. Online Retail II Dataset (for RFM Customer Segmentation)
2. E-Commerce Customer Churn Dataset (for Supervised Churn Risk Modeling)
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants & Default Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RETAIL_DATA_PATH = DATA_DIR / "online_retail_II.csv"
CHURN_DATA_PATH = DATA_DIR / "E Commerce Dataset.xlsx"

# Expected Raw Schemas
RETAIL_EXPECTED_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", 
    "InvoiceDate", "Price", "Customer ID", "Country"
]

CHURN_EXPECTED_COLUMNS = [
    "CustomerID", "Churn", "Tenure", "PreferredLoginDevice", "CityTier",
    "WarehouseToHome", "PreferredPaymentMode", "Gender", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "PreferedOrderCat", "SatisfactionScore",
    "MaritalStatus", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount"
]


def load_retail_data(filepath: Path | str = RETAIL_DATA_PATH) -> pd.DataFrame:
    """
    Loads and validates the raw Online Retail II transaction dataset.
    
    Args:
        filepath: Path to the online_retail_II.csv file.
        
    Returns:
        pd.DataFrame containing raw retail transactions.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Retail dataset not found at {filepath}. "
            "Please ensure online_retail_II.csv is placed in the data/ directory."
        )

    logger.info(f"Loading Online Retail dataset from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    # Validate Schema
    missing_cols = set(RETAIL_EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Retail dataset missing required columns: {missing_cols}")

    logger.info(
        f"Online Retail dataset loaded successfully. Shape: {df.shape[0]:,} rows, {df.shape[1]} columns."
    )
    return df


def load_churn_data(
    filepath: Path | str = CHURN_DATA_PATH, sheet_name: str = "E Comm"
) -> pd.DataFrame:
    """
    Loads and validates the raw E-Commerce Customer Churn dataset.
    
    Args:
        filepath: Path to the E Commerce Dataset.xlsx file.
        sheet_name: Sheet name containing customer churn data.
        
    Returns:
        pd.DataFrame containing raw customer churn records.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Churn dataset not found at {filepath}. "
            "Please ensure 'E Commerce Dataset.xlsx' is placed in the data/ directory."
        )

    logger.info(f"Loading E-Commerce Churn dataset from {filepath} (sheet: '{sheet_name}')...")
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Validate Schema
    missing_cols = set(CHURN_EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Churn dataset missing required columns: {missing_cols}")

    logger.info(
        f"E-Commerce Churn dataset loaded successfully. Shape: {df.shape[0]:,} rows, {df.shape[1]} columns."
    )
    return df


def load_all_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to load and return both raw datasets.
    
    Returns:
        Tuple of (df_retail, df_churn)
    """
    df_retail = load_retail_data()
    df_churn = load_churn_data()
    return df_retail, df_churn


def main():
    """Execution entry point for testing dataset loading."""
    logger.info("=== Starting Dataset Ingestion Verification ===")
    try:
        df_retail, df_churn = load_all_datasets()

        print("\n" + "=" * 50)
        print("1. ONLINE RETAIL II DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Rows: {len(df_retail):,}")
        print(f"Total Columns: {len(df_retail.columns)}")
        print(f"Unique Customers: {df_retail['Customer ID'].nunique():,}")
        print("\nColumns:", list(df_retail.columns))
        print("\nFirst 2 Rows:\n", df_retail.head(2))

        print("\n" + "=" * 50)
        print("2. E-COMMERCE CHURN DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Rows: {len(df_churn):,}")
        print(f"Total Columns: {len(df_churn.columns)}")
        print(f"Churn Rate: {(df_churn['Churn'].mean() * 100):.2f}%")
        print("\nColumns:", list(df_churn.columns))
        print("\nFirst 2 Rows:\n", df_churn.head(2))

        logger.info("=== Dataset Ingestion Verification Completed Successfully ===")

    except Exception as e:
        logger.error(f"Dataset ingestion failed: {e}")
        raise e


if __name__ == "__main__":
    main()
