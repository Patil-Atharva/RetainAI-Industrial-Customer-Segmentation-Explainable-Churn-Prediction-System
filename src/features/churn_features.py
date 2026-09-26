"""
Tabular Feature Engineering Module for Churn Risk Prediction.
Implements scikit-learn compatible custom transformer to engineer derived interaction features,
impute missing values using training medians, and encode categoricals without data leakage.

Assigned to: Lead Engineer (You)
"""

import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Column Definitions
NUMERICAL_COLS = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_COLS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

TARGET_COL = "Churn"
ID_COL = "CustomerID"


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn compatible transformer for engineering tabular churn features.
    Guarantees ZERO data leakage by fitting imputers, encoders, and scalers strictly on training data.
    """

    def __init__(self, scale_features: bool = False):
        self.scale_features = scale_features
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler() if scale_features else None
        
        self.feature_names_: List[str] = []
        self.fitted_num_cols_: List[str] = []
        self.is_fitted_: bool = False

    def _engineer_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes domain-specific interaction & ratio features."""
        data = df.copy()

        # Fill temporary nulls for mathematical ratios (will be formally imputed)
        tenure = data["Tenure"].fillna(data["Tenure"].median())
        order_count = data["OrderCount"].fillna(data["OrderCount"].median())
        coupon_used = data["CouponUsed"].fillna(data["CouponUsed"].median())
        satisfaction = data["SatisfactionScore"].fillna(3)
        complain = data["Complain"].fillna(0)
        days_since_last = data["DaySinceLastOrder"].fillna(data["DaySinceLastOrder"].median())

        # 1. Order Velocity: Average orders per month of tenure
        data["OrderVelocity"] = order_count / (tenure + 1.0)

        # 2. Coupon Usage Ratio: Coupon dependency per order
        data["CouponUsageRatio"] = coupon_used / (order_count + 1.0)

        # 3. Cashback Per Order: Average financial reward earned per order
        data["CashbackPerOrder"] = data["CashbackAmount"] / (order_count + 1.0)

        # 4. Complain Risk Interaction: Weighted grievance score (High complaint + Low satisfaction)
        data["ComplainRiskInteraction"] = complain * (6.0 - satisfaction)

        # 5. Inactivity Ratio: Relative days inactive vs tenure length
        data["InactivityRatio"] = days_since_last / ((tenure * 30.0) + 1.0)

        return data

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits imputers, encoders, and optional scalers strictly on training features.
        """
        logger.info("Fitting ChurnFeatureEngineer on training split...")
        X_eng = self._engineer_derived_features(X)

        # Identify all numerical columns (base + derived)
        derived_cols = [
            "OrderVelocity",
            "CouponUsageRatio",
            "CashbackPerOrder",
            "ComplainRiskInteraction",
            "InactivityRatio",
        ]
        self.fitted_num_cols_ = NUMERICAL_COLS + derived_cols

        # Fit numerical median imputer
        self.num_imputer.fit(X_eng[self.fitted_num_cols_])

        # Fit categorical OneHotEncoder
        self.cat_encoder.fit(X_eng[CATEGORICAL_COLS])

        # Construct feature names
        cat_feature_names = list(self.cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
        self.feature_names_ = self.fitted_num_cols_ + cat_feature_names

        # Fit scaler if requested
        if self.scaler is not None:
            imputed_num = self.num_imputer.transform(X_eng[self.fitted_num_cols_])
            encoded_cat = self.cat_encoder.transform(X_eng[CATEGORICAL_COLS])
            full_matrix = np.hstack([imputed_num, encoded_cat])
            self.scaler.fit(full_matrix)

        self.is_fitted_ = True
        logger.info(
            f"ChurnFeatureEngineer fitted successfully. Total features generated: {len(self.feature_names_)}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input dataframe using fitted imputers, encoders, and scalers.
        """
        if not self.is_fitted_:
            raise RuntimeError("ChurnFeatureEngineer must be fitted before calling transform().")

        X_eng = self._engineer_derived_features(X)

        # Impute numerical features
        imputed_num = self.num_imputer.transform(X_eng[self.fitted_num_cols_])

        # One-hot encode categorical features
        encoded_cat = self.cat_encoder.transform(X_eng[CATEGORICAL_COLS])

        # Combine matrices
        full_matrix = np.hstack([imputed_num, encoded_cat])

        # Scale if configured
        if self.scaler is not None:
            full_matrix = self.scaler.transform(full_matrix)

        # Return as DataFrame with preserved feature names
        return pd.DataFrame(full_matrix, columns=self.feature_names_, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def prepare_churn_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    scale_features: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.Series, ChurnFeatureEngineer],
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ChurnFeatureEngineer],
]:
    """
    Convenience wrapper to fit feature engineering on train_df and transform both train_df and test_df.
    
    Args:
        train_df: Cleaned training dataframe (from data/processed/churn_train.csv).
        test_df: Optional cleaned testing dataframe (from data/processed/churn_test.csv).
        scale_features: If True, applies StandardScaler to output features.
        
    Returns:
        If test_df is None: (X_train, y_train, engineer)
        If test_df provided: (X_train, y_train, X_test, y_test, engineer)
    """
    logger.info("Preparing tabular churn features...")
    engineer = ChurnFeatureEngineer(scale_features=scale_features)

    # Separate target and features
    X_train_raw = train_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y_train = train_df[TARGET_COL]

    X_train = engineer.fit_transform(X_train_raw)

    if test_df is not None:
        X_test_raw = test_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
        y_test = test_df[TARGET_COL]
        X_test = engineer.transform(X_test_raw)
        return X_train, y_train, X_test, y_test, engineer

    return X_train, y_train, engineer
