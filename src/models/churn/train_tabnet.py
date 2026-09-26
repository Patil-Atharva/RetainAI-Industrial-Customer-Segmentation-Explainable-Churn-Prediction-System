"""
Deep Learning Tabular Model Module (TabNet / PyTorch Deep Neural Network) for Churn Risk Prediction.
Implements Google Cloud AI's TabNet architecture and Deep Tabular Neural Networks
with sequential sparse attention and feature scaling.

Assigned to: Lead Engineer (You)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# Optional PyTorch TabNet import
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

from src.features.churn_features import prepare_churn_features
from src.models.churn.train_baselines import evaluate_model_performance

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


def train_deep_tabular_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> MLPClassifier:
    """
    Trains Deep Tabular Neural Network (Multi-Layer Perceptron with 3 hidden layers: 128 -> 64 -> 32 nodes,
    ReLU activations, Adam optimizer, and early stopping).
    """
    logger.info("Training Deep Tabular Neural Network (MLP 128-64-32)...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=random_state,
    )
    mlp_model.fit(X_train, y_train)
    logger.info("Deep Tabular Neural Network training complete.")
    return mlp_model


def train_pytorch_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray = None,
    y_valid: np.ndarray = None,
    random_state: int = 42,
):
    """
    Trains PyTorch TabNet Classifier with sequential attention mechanism.
    """
    if not HAS_TABNET:
        raise ImportError("pytorch-tabnet package is not installed.")

    logger.info("Training PyTorch TabNet Deep Learning Classifier...")
    tabnet_model = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.5),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type="sparsemax",
        seed=random_state,
        verbose=10,
    )

    eval_set = [(X_valid, y_valid)] if X_valid is not None and y_valid is not None else []
    eval_name = ["valid"] if eval_set else []

    tabnet_model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=eval_set,
        eval_name=eval_name,
        eval_metric=["auc"],
        max_epochs=100,
        patience=15,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )
    logger.info("PyTorch TabNet training complete.")
    return tabnet_model


def train_and_evaluate_tabnet(
    train_path: Path | str = DATA_PROCESSED_DIR / "churn_train.csv",
    test_path: Path | str = DATA_PROCESSED_DIR / "churn_test.csv",
    save_models: bool = True,
    output_dir: Path | str = MODEL_DIR,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    """
    End-to-end Deep Learning training pipeline evaluating Deep Tabular Neural Net & TabNet.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed splits missing at {train_path} or {test_path}.")

    logger.info(f"Loading splits for Deep Learning training from {train_path.parent}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Neural networks require standardized input feature scaling
    logger.info("Preparing Z-score scaled feature transformations for Deep Learning...")
    X_train_scaled, y_train, X_test_scaled, y_test, engineer = prepare_churn_features(
        train_df, test_df, scale_features=True
    )

    results_metrics = {}
    models_dict = {}
    output_dir = Path(output_dir)

    # 1. Train & Evaluate Deep Tabular Neural Network (MLP)
    mlp_model = train_deep_tabular_net(X_train_scaled, y_train)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_prob = mlp_model.predict_proba(X_test_scaled)[:, 1]
    mlp_metrics = evaluate_model_performance("Deep Tabular Neural Net (MLP)", y_test, mlp_pred, mlp_prob)

    results_metrics["Deep Tabular Neural Net"] = mlp_metrics
    models_dict["deep_tabular_net"] = mlp_model

    if save_models:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(mlp_model, output_dir / "deep_learning_mlp.joblib")

    # 2. Train & Evaluate PyTorch TabNet if available
    if HAS_TABNET:
        try:
            X_train_np = X_train_scaled.values.astype(np.float32)
            y_train_np = y_train.values.astype(np.int64)
            X_test_np = X_test_scaled.values.astype(np.float32)
            y_test_np = y_test.values.astype(np.int64)

            tabnet_model = train_pytorch_tabnet(X_train_np, y_train_np, X_test_np, y_test_np)
            tabnet_pred = tabnet_model.predict(X_test_np)
            tabnet_prob = tabnet_model.predict_proba(X_test_np)[:, 1]
            tabnet_metrics = evaluate_model_performance("PyTorch TabNet", y_test_np, tabnet_pred, tabnet_prob)

            results_metrics["PyTorch TabNet"] = tabnet_metrics
            models_dict["pytorch_tabnet"] = tabnet_model

            if save_models:
                tabnet_model.save_model(str(output_dir / "tabnet_model"))
        except Exception as e:
            logger.warning(f"PyTorch TabNet training failed with error: {e}. Skipping TabNet execution.")
    else:
        logger.info("pytorch-tabnet package not installed. Deep Tabular Neural Net evaluated as DL baseline.")

    return results_metrics, models_dict


def main():
    """Execution entry point for Deep Learning Tabular Model benchmark."""
    logger.info("=== Starting Deep Learning Model Training & Evaluation ===")
    metrics, _ = train_and_evaluate_tabnet()

    print("\n" + "=" * 60)
    print("DEEP LEARNING CHURN MODEL BENCHMARK SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(metrics).T.drop(columns=["model_name"], errors="ignore")
    print(summary_df.to_string())
    print("=" * 60)

    logger.info("=== Deep Learning Benchmark Completed Successfully ===")


if __name__ == "__main__":
    main()
