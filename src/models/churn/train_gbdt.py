"""
Gradient Boosted Decision Trees (GBDT) Module for Churn Risk Prediction.
Trains and evaluates HistGradientBoostingClassifier (scikit-learn native LightGBM algorithm),
XGBoost Classifier, and LightGBM Classifier with cost-sensitive class weighting (scale_pos_weight).

Assigned to: Lead Engineer (You)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Optional LightGBM import
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Optional imblearn import
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

from src.features.churn_features import prepare_churn_features
from src.models.churn.train_baselines import evaluate_model_performance

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculates negative-to-positive class ratio for cost-sensitive boosting."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    ratio = neg_count / float(pos_count)
    logger.info(f"Class Distribution -> Negatives: {neg_count:,}, Positives: {pos_count:,} | scale_pos_weight ratio: {ratio:.2f}")
    return ratio


def train_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """
    Trains HistGradientBoostingClassifier (scikit-learn native histogram-based GBDT).
    """
    logger.info("Training HistGradientBoostingClassifier (scikit-learn GBDT)...")
    hgb_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
    )
    hgb_model.fit(X_train, y_train)
    logger.info("HistGradientBoosting training complete.")
    return hgb_model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = None,
    random_state: int = 42,
):
    """
    Trains XGBoost Classifier with regularization, colsample subsampling, and cost-sensitive class weights.
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost package is not installed.")

    if scale_pos_weight is None:
        scale_pos_weight = calculate_scale_pos_weight(y_train)

    logger.info("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return xgb_model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = None,
    random_state: int = 42,
):
    """
    Trains LightGBM Classifier with leaf-wise tree growth and class weighting.
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm package is not installed.")

    if scale_pos_weight is None:
        scale_pos_weight = calculate_scale_pos_weight(y_train)

    logger.info("Training LightGBM Classifier...")
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    lgbm_model.fit(X_train, y_train)
    logger.info("LightGBM training complete.")
    return lgbm_model


def train_and_evaluate_gbdt(
    train_path: Path | str = DATA_PROCESSED_DIR / "churn_train.csv",
    test_path: Path | str = DATA_PROCESSED_DIR / "churn_test.csv",
    use_smote: bool = False,
    save_models: bool = True,
    output_dir: Path | str = MODEL_DIR,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    """
    End-to-end GBDT training pipeline supporting HistGradientBoosting, XGBoost, and LightGBM.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed splits missing at {train_path} or {test_path}.")

    logger.info(f"Loading splits for GBDT training from {train_path.parent}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Engineer Features
    X_train, y_train, X_test, y_test, engineer = prepare_churn_features(
        train_df, test_df, scale_features=False
    )

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    # Apply SMOTE if specified and available
    if use_smote and HAS_IMBLEARN:
        logger.info("Applying SMOTE resampling on training split...")
        smote = SMOTE(random_state=42)
        fit_X_train, fit_y_train = smote.fit_resample(X_train, y_train)
        weight_param = 1.0
    else:
        fit_X_train, fit_y_train = X_train, y_train
        weight_param = scale_pos_weight

    results_metrics = {}
    models_dict = {}
    output_dir = Path(output_dir)

    # 1. Train & Evaluate HistGradientBoosting (Native scikit-learn GBDT)
    hgb_model = train_hist_gradient_boosting(fit_X_train, fit_y_train)
    hgb_pred = hgb_model.predict(X_test)
    hgb_prob = hgb_model.predict_proba(X_test)[:, 1]
    hgb_metrics = evaluate_model_performance("HistGradientBoosting", y_test, hgb_pred, hgb_prob)

    results_metrics["HistGradientBoosting"] = hgb_metrics
    models_dict["hist_gradient_boosting"] = hgb_model

    if save_models:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(hgb_model, output_dir / "gbdt_hist_gradient_boosting.joblib")

    # 2. Train & Evaluate XGBoost if available
    if HAS_XGBOOST:
        xgb_model = train_xgboost(fit_X_train, fit_y_train, scale_pos_weight=weight_param)
        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_model_performance("XGBoost", y_test, xgb_pred, xgb_prob)

        results_metrics["XGBoost"] = xgb_metrics
        models_dict["xgboost"] = xgb_model

        if save_models:
            joblib.dump(xgb_model, output_dir / "gbdt_xgboost.joblib")

    # 3. Train & Evaluate LightGBM if available
    if HAS_LIGHTGBM:
        lgbm_model = train_lightgbm(fit_X_train, fit_y_train, scale_pos_weight=weight_param)
        lgbm_pred = lgbm_model.predict(X_test)
        lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_metrics = evaluate_model_performance("LightGBM", y_test, lgbm_pred, lgbm_prob)

        results_metrics["LightGBM"] = lgbm_metrics
        models_dict["lightgbm"] = lgbm_model

        if save_models:
            joblib.dump(lgbm_model, output_dir / "gbdt_lightgbm.joblib")

    return results_metrics, models_dict


def main():
    """Execution entry point for GBDT model benchmark."""
    logger.info("=== Starting GBDT Training & Benchmark Evaluation ===")
    metrics, _ = train_and_evaluate_gbdt(use_smote=False)

    print("\n" + "=" * 60)
    print("GBDT CHURN MODEL BENCHMARK SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(metrics).T.drop(columns=["model_name"], errors="ignore")
    print(summary_df.to_string())
    print("=" * 60)

    logger.info("=== GBDT Benchmark Completed Successfully ===")


if __name__ == "__main__":
    main()
