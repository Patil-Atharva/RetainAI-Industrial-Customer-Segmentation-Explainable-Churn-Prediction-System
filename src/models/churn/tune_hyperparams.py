"""
Hyperparameter Tuning Engine for Churn Risk Models.
Uses Optuna (Bayesian Optimization) or RandomizedSearchCV with Stratified 5-Fold Cross-Validation
to optimize PR-AUC / F1-Score across tree ensembles and deep neural networks.

Assigned to: Lead Engineer (You)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

# Optional Optuna import
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.features.churn_features import prepare_churn_features
from src.models.churn.train_baselines import evaluate_model_performance

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    random_state: int = 42,
) -> Tuple[HistGradientBoostingClassifier, Dict[str, Any]]:
    """
    Tunes HistGradientBoostingClassifier using Optuna or RandomizedSearchCV with 5-Fold Stratified CV.
    """
    logger.info(f"Starting Hyperparameter Tuning for HistGradientBoosting (trials/iter={n_trials})...")

    if HAS_OPTUNA:
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                "class_weight": "balanced",
                "random_state": random_state,
            }
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
                model = HistGradientBoostingClassifier(**params)
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_va)[:, 1]
                scores.append(average_precision_score(y_va, probs))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({"class_weight": "balanced", "random_state": random_state})
        logger.info(f"Optuna Best Params for HistGradientBoosting: {best_params} | Best PR-AUC CV: {study.best_value:.4f}")

        best_model = HistGradientBoostingClassifier(**best_params)
        best_model.fit(X_train, y_train)
        return best_model, best_params

    else:
        logger.info("Optuna not found. Tuning HistGradientBoosting via Stratified RandomizedSearchCV...")
        param_distributions = {
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "max_iter": [100, 150, 200, 250],
            "max_depth": [4, 6, 8, 10],
            "min_samples_leaf": [10, 20, 30, 40],
            "l2_regularization": [0.001, 0.01, 0.1, 1.0, 10.0],
        }
        base_model = HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state)
        pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=min(n_trials, 15),
            scoring=pr_auc_scorer,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        logger.info(f"RandomizedSearch Best Params: {search.best_params_} | Best PR-AUC CV: {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Tunes RandomForestClassifier using Optuna or RandomizedSearchCV with 5-Fold Stratified CV.
    """
    logger.info(f"Starting Hyperparameter Tuning for Random Forest (trials/iter={n_trials})...")

    if HAS_OPTUNA:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
                "max_depth": trial.suggest_int("max_depth", 6, 18),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": -1,
            }
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
                model = RandomForestClassifier(**params)
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_va)[:, 1]
                scores.append(average_precision_score(y_va, probs))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({"class_weight": "balanced", "random_state": random_state, "n_jobs": -1})
        logger.info(f"Optuna Best Params for Random Forest: {best_params} | Best PR-AUC CV: {study.best_value:.4f}")

        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X_train, y_train)
        return best_model, best_params

    else:
        logger.info("Optuna not found. Tuning Random Forest via Stratified RandomizedSearchCV...")
        param_distributions = {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 12, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        base_model = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
        pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=min(n_trials, 15),
            scoring=pr_auc_scorer,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        logger.info(f"RandomizedSearch Best Params for Random Forest: {search.best_params_} | Best PR-AUC CV: {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_


def tune_and_evaluate_all(
    train_path: Path | str = DATA_PROCESSED_DIR / "churn_train.csv",
    test_path: Path | str = DATA_PROCESSED_DIR / "churn_test.csv",
    n_trials: int = 15,
    save_models: bool = True,
    output_dir: Path | str = MODEL_DIR,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Executes full hyperparameter tuning suite across tree ensembles, retrains optimal models,
    evaluates holdout test benchmarks, and saves best tuned artifacts.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed splits missing at {train_path} or {test_path}.")

    logger.info(f"Loading data splits for hyperparameter tuning from {train_path.parent}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare features
    X_train_unscaled, y_train, X_test_unscaled, y_test, _ = prepare_churn_features(
        train_df, test_df, scale_features=False
    )

    results_metrics = {}
    best_params_all = {}
    output_dir = Path(output_dir)

    # 1. Tune & Evaluate HistGradientBoosting
    hgb_tuned_model, hgb_params = tune_hist_gradient_boosting(X_train_unscaled, y_train, n_trials=n_trials)
    hgb_pred = hgb_tuned_model.predict(X_test_unscaled)
    hgb_prob = hgb_tuned_model.predict_proba(X_test_unscaled)[:, 1]
    hgb_metrics = evaluate_model_performance("Tuned HistGradientBoosting", y_test, hgb_pred, hgb_prob)

    results_metrics["Tuned HistGradientBoosting"] = hgb_metrics
    best_params_all["HistGradientBoosting"] = {k: str(v) for k, v in hgb_params.items()}

    # 2. Tune & Evaluate Random Forest
    rf_tuned_model, rf_params = tune_random_forest(X_train_unscaled, y_train, n_trials=n_trials)
    rf_pred = rf_tuned_model.predict(X_test_unscaled)
    rf_prob = rf_tuned_model.predict_proba(X_test_unscaled)[:, 1]
    rf_metrics = evaluate_model_performance("Tuned Random Forest", y_test, rf_pred, rf_prob)

    results_metrics["Tuned Random Forest"] = rf_metrics
    best_params_all["RandomForest"] = {k: str(v) for k, v in rf_params.items()}

    # Determine Overall Champion Model based on PR-AUC
    champion_name = max(results_metrics, key=lambda k: results_metrics[k]["pr_auc"])
    champion_model = hgb_tuned_model if "HistGradientBoosting" in champion_name else rf_tuned_model

    logger.info(f"\n🏆 Champion Model Selected: {champion_name} (PR-AUC: {results_metrics[champion_name]['pr_auc']:.4f})")

    # Save artifacts
    if save_models:
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(hgb_tuned_model, output_dir / "tuned_hist_gradient_boosting.joblib")
        joblib.dump(rf_tuned_model, output_dir / "tuned_random_forest.joblib")
        joblib.dump(champion_model, output_dir / "tuned_champion_model.joblib")

        params_path = output_dir / "hyperparameter_tuning_results.json"
        with open(params_path, "w") as f:
            json.dump({"champion_model": champion_name, "best_params": best_params_all, "metrics": results_metrics}, f, indent=2)

        logger.info(f"Saved tuned model artifacts & tuning log to {output_dir}")

    return results_metrics, best_params_all


def main():
    """Execution entry point for hyperparameter tuning."""
    logger.info("=== Starting Automated Hyperparameter Tuning ===")
    metrics, best_params = tune_and_evaluate_all(n_trials=10)

    print("\n" + "=" * 65)
    print("TUNED CHURN MODEL BENCHMARK SUMMARY")
    print("=" * 65)
    summary_df = pd.DataFrame(metrics).T.drop(columns=["model_name"], errors="ignore")
    print(summary_df.to_string())
    print("=" * 65)

    logger.info("=== Hyperparameter Tuning Completed Successfully ===")


if __name__ == "__main__":
    main()
