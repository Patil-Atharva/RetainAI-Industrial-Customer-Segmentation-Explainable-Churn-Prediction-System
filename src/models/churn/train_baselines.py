"""
Baseline Classification Models Module for Churn Risk Prediction.
Trains and evaluates baseline models (Logistic Regression & Random Forest Classifier)
on transformed tabular churn features, establishing performance benchmarks (ROC-AUC, PR-AUC, F1-Score).

Assigned to: Lead Engineer (You)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.churn_features import prepare_churn_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


def evaluate_model_performance(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Computes comprehensive classification metrics for churn evaluation.
    """
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, pos_label=1))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1))
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

    logger.info(f"\n=== Evaluation Metrics for {model_name} ===")
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"PR-AUC Score (Average Precision): {pr_auc:.4f}")
    logger.info(f"F1-Score (Churn Class 1): {f1:.4f}")
    logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    logger.info(f"Confusion Matrix [TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}]")

    return metrics


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> LogisticRegression:
    """
    Trains baseline Logistic Regression model with L2 regularization and balanced class weights.
    """
    logger.info("Training Logistic Regression Baseline...")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    lr_model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return lr_model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestClassifier:
    """
    Trains baseline Random Forest Classifier with balanced class weighting.
    """
    logger.info("Training Random Forest Classifier Baseline...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return rf_model


def train_and_evaluate_baselines(
    train_path: Path | str = DATA_PROCESSED_DIR / "churn_train.csv",
    test_path: Path | str = DATA_PROCESSED_DIR / "churn_test.csv",
    save_models: bool = True,
    output_dir: Path | str = MODEL_DIR,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    """
    End-to-end pipeline loading processed train/test splits, engineering features,
    training baselines, evaluating metrics, and saving joblib model artifacts.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed train/test files missing at {train_path} and {test_path}. "
            "Please run src/data/split_data.py first."
        )

    # 1. Load Data Splits
    logger.info(f"Loading processed splits from {train_path.parent}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Engineer Features
    logger.info("Preparing feature transformations for baselines...")
    X_train_scaled, y_train, X_test_scaled, y_test, engineer_scaled = prepare_churn_features(
        train_df, test_df, scale_features=True
    )
    X_train_unscaled, _, X_test_unscaled, _, engineer_unscaled = prepare_churn_features(
        train_df, test_df, scale_features=False
    )

    results_metrics = {}
    models_dict = {}

    # 3. Train & Evaluate Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model_performance("Logistic Regression", y_test, lr_pred, lr_prob)

    results_metrics["Logistic Regression"] = lr_metrics
    models_dict["logistic_regression"] = lr_model

    # 4. Train & Evaluate Random Forest
    rf_model = train_random_forest(X_train_unscaled, y_train)
    rf_pred = rf_model.predict(X_test_unscaled)
    rf_prob = rf_model.predict_proba(X_test_unscaled)[:, 1]
    rf_metrics = evaluate_model_performance("Random Forest", y_test, rf_pred, rf_prob)

    results_metrics["Random Forest"] = rf_metrics
    models_dict["random_forest"] = rf_model

    # 5. Save Artifacts if Requested
    if save_models:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lr_path = output_dir / "baseline_logistic_regression.joblib"
        rf_path = output_dir / "baseline_random_forest.joblib"
        eng_path = output_dir / "churn_feature_engineer.joblib"

        joblib.dump(lr_model, lr_path)
        joblib.dump(rf_model, rf_path)
        joblib.dump(engineer_scaled, eng_path)

        logger.info(f"Saved Logistic Regression model to {lr_path}")
        logger.info(f"Saved Random Forest model to {rf_path}")
        logger.info(f"Saved Feature Engineer artifact to {eng_path}")

    return results_metrics, models_dict


def main():
    """Execution entry point for baseline training and benchmark evaluation."""
    logger.info("=== Starting Baseline Model Training & Evaluation ===")
    metrics, _ = train_and_evaluate_baselines()

    print("\n" + "=" * 60)
    print("BASELINE CHURN MODEL BENCHMARK SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(metrics).T.drop(columns=["model_name"], errors="ignore")
    print(summary_df.to_string())
    print("=" * 60)

    logger.info("=== Baseline Model Training Completed Successfully ===")


if __name__ == "__main__":
    main()
