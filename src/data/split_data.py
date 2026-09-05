"""
Stratified Train-Test Splitting Module.
Ensures zero data leakage by splitting cleaned datasets into training and testing holdouts
with stratified class balance before any scaling, encoding, or imputation occurs.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def split_churn_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_outputs: bool = True,
    output_dir: Path | str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train-test split on customer churn dataset.
    
    Args:
        df: Cleaned churn dataframe containing target column 'Churn'.
        test_size: Proportion of dataset to include in holdout test split (default 0.2).
        random_state: Random seed for reproducibility.
        save_outputs: If True, saves train_df and test_df to data/processed/.
        output_dir: Target directory path for saving processed CSV splits.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    if "Churn" not in df.columns:
        raise KeyError("Dataframe missing target column 'Churn'.")

    logger.info(
        f"Splitting dataset ({len(df):,} rows) with test_size={test_size}, random_state={random_state}..."
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Churn"],
    )

    # Validate Class Balance
    orig_rate = df["Churn"].mean() * 100
    train_rate = train_df["Churn"].mean() * 100
    test_rate = test_df["Churn"].mean() * 100

    logger.info(
        f"Stratification Verification -> Original Churn Rate: {orig_rate:.2f}% | "
        f"Train: {train_rate:.2f}% ({len(train_df):,} rows) | "
        f"Test: {test_rate:.2f}% ({len(test_df):,} rows)"
    )

    if save_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "churn_train.csv"
        test_path = output_dir / "churn_test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved processed train split to {train_path}")
        logger.info(f"Saved processed test split to {test_path}")

    return train_df, test_df


def main():
    """CLI runner to execute ingestion, cleaning, and stratified splitting."""
    from src.data.clean_data import clean_churn_data
    from src.data.make_dataset import load_churn_data

    logger.info("=== Executing Data Hygiene & Stratified Splitting ===")
    df_raw = load_churn_data()
    df_clean = clean_churn_data(df_raw)
    train_df, test_df = split_churn_data(df_clean, save_outputs=True)
    logger.info("=== Phase 0 Data Splitting Complete ===")


if __name__ == "__main__":
    main()
