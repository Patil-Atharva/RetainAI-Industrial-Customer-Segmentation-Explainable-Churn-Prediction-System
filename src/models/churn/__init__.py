"""
Supervised Churn Risk models (Logistic Regression, Random Forest, XGBoost, LightGBM, PyTorch TabNet, Optuna/CV Tuning).
Assigned to: Lead Engineer (You).
"""

from src.models.churn.train_baselines import (
    evaluate_model_performance,
    train_and_evaluate_baselines,
    train_logistic_regression,
    train_random_forest,
)
from src.models.churn.train_gbdt import (
    calculate_scale_pos_weight,
    train_and_evaluate_gbdt,
    train_hist_gradient_boosting,
    train_lightgbm,
    train_xgboost,
)
from src.models.churn.train_tabnet import (
    train_and_evaluate_tabnet,
    train_deep_tabular_net,
    train_pytorch_tabnet,
)
from src.models.churn.tune_hyperparams import (
    tune_and_evaluate_all,
    tune_hist_gradient_boosting,
    tune_random_forest,
)

__all__ = [
    "train_logistic_regression",
    "train_random_forest",
    "evaluate_model_performance",
    "train_and_evaluate_baselines",
    "train_hist_gradient_boosting",
    "train_xgboost",
    "train_lightgbm",
    "calculate_scale_pos_weight",
    "train_and_evaluate_gbdt",
    "train_deep_tabular_net",
    "train_pytorch_tabnet",
    "train_and_evaluate_tabnet",
    "tune_hist_gradient_boosting",
    "tune_random_forest",
    "tune_and_evaluate_all",
]
