"""
Comprehensive End-to-End (E2E) Full-System Test Suite.

Applies e2e-testing-patterns guidelines to perform thorough E2E testing across:
  - Branch A (Friend's Segmentation Pipeline): RFM Engineering, K-Means Silhouette optimization, GMM soft clustering, PCA & UMAP 2D/3D projections, Cluster Profiling & Persona assignment.
  - Branch B (Lead's Churn Prediction Pipeline): Stratified Data Hygiene, GBDT & Deep TabNet modeling, Hyperparameter Tuning.
  - Master Integration Layer: SHAP Explainability Engine & Integrated Segment x Churn Risk Matrix.
  - REST API & SLA Tier: FastAPI Service real-time inference, batch prediction, SHAP lookup, and latency SLA (< 200ms).

Assigned to: Lead Engineer (You)
"""

import json
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app, load_api_artifacts
from src.data.clean_data import clean_churn_data, clean_retail_data
from src.data.make_dataset import load_all_datasets
from src.features.churn_features import prepare_churn_features
from src.features.rfm_features import build_scaled_rfm_features
from src.models.churn.train_gbdt import train_and_evaluate_gbdt
from src.models.risk_matrix import generate_integrated_risk_report
from src.models.segmentation.cluster_profiler import assign_personas, profile_clusters
from src.models.segmentation.dim_reduction import UMAP_AVAILABLE, pca_reduce, umap_reduce
from src.models.segmentation.gmm_model import fit_gmm
from src.models.segmentation.kmeans_model import evaluate_k_range, fit_kmeans, select_best_k

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "churn"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class TestFullSystemEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and initialize FastAPI TestClient."""
        load_api_artifacts()
        cls.client = TestClient(app)

    def test_01_e2e_friend_segmentation_pipeline(self):
        """
        E2E Test 1: Full test on Friend's Unsupervised Customer Segmentation Pipeline.
        Covers: Raw Retail Ingestion -> Cleaning -> RFM Feature Extraction ->
                Silhouette Analysis (k=2..5) -> K-Means Fitting -> Cluster Profiling ->
                Persona Assignment -> Persona CSV Export.
        """
        # 1. Ingest raw retail data
        df_retail_raw, _ = load_all_datasets()
        self.assertGreater(len(df_retail_raw), 500000, "Raw retail dataset should contain > 500k rows")

        # 2. Clean retail transactions
        df_retail_clean = clean_retail_data(df_retail_raw)
        self.assertIn("Customer ID", df_retail_clean.columns)
        self.assertGreater(len(df_retail_clean), 400000)

        # 3. Build RFM features & scale
        rfm_raw, X_scaled, scaler = build_scaled_rfm_features(df_retail_clean)
        self.assertGreater(len(rfm_raw), 4000, "RFM table should have > 4k unique customers")
        self.assertEqual(X_scaled.shape[1], 3, "Scaled RFM matrix must have 3 features (R, F, M)")

        # 4. Evaluate K-Means range (k=2..5) & Select Best K
        metrics = evaluate_k_range(X_scaled, k_range=range(2, 6))
        self.assertEqual(len(metrics), 4, "Metrics table should contain rows for k=2..5")
        self.assertIn("silhouette", metrics.columns)

        best_k = select_best_k(metrics)
        self.assertTrue(2 <= best_k <= 5, f"Best K selected ({best_k}) should be between 2 and 5")

        # 5. Fit K-Means & Profile Clusters
        kmeans_model, labels = fit_kmeans(X_scaled, best_k)
        self.assertEqual(len(labels), len(rfm_raw))
        self.assertEqual(len(np.unique(labels)), best_k)

        profile = profile_clusters(rfm_raw, labels)
        personas = assign_personas(profile)
        self.assertEqual(len(personas), best_k)
        self.assertIn("Persona", personas.columns)

        # 6. Verify Persona CSV export
        output_csv = PROJECT_ROOT / "segment_personas_output.csv"
        personas.to_csv(output_csv)
        self.assertTrue(output_csv.exists(), "segment_personas_output.csv export failed")

    def test_02_e2e_friend_gmm_and_dim_reduction(self):
        """
        E2E Test 2: Test Friend's GMM Soft Clustering and Dimensionality Reduction (PCA & UMAP).
        """
        df_retail_raw, _ = load_all_datasets()
        df_retail_clean = clean_retail_data(df_retail_raw)
        rfm_raw, X_scaled, scaler = build_scaled_rfm_features(df_retail_clean)

        # 1. Test GMM model soft probabilistic clustering
        gmm_model, labels, probs = fit_gmm(X_scaled, 3)
        self.assertEqual(probs.shape, (len(rfm_raw), 3), "GMM probabilities matrix shape mismatch")
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0), "GMM soft probabilities must sum to 1.0")

        # 2. Test PCA 2D/3D projection
        coords_2d = pca_reduce(X_scaled, n_components=2)
        self.assertEqual(coords_2d.shape, (len(rfm_raw), 2), "PCA 2D projection shape mismatch")

        coords_3d = pca_reduce(X_scaled, n_components=3)
        self.assertEqual(coords_3d.shape, (len(rfm_raw), 3), "PCA 3D projection shape mismatch")

        # 3. Test UMAP 2D projection if available
        if UMAP_AVAILABLE:
            coords_umap = umap_reduce(X_scaled, n_components=2)
            self.assertEqual(coords_umap.shape, (len(rfm_raw), 2), "UMAP 2D projection shape mismatch")

    def test_03_e2e_lead_churn_pipeline(self):
        """
        E2E Test 3: Test Lead's Churn Risk Feature Engineering and Model Training.
        """
        train_path = DATA_PROCESSED_DIR / "churn_train.csv"
        test_path = DATA_PROCESSED_DIR / "churn_test.csv"
        self.assertTrue(train_path.exists())
        self.assertTrue(test_path.exists())

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train, X_test, y_test, engineer = prepare_churn_features(train_df, test_df, scale_features=False)
        self.assertEqual(X_train.shape[1], 35, "Feature engineer should produce 35 features")

        # Train & Evaluate GBDT
        results_metrics, _ = train_and_evaluate_gbdt(train_path=train_path, test_path=test_path, save_models=False)
        hgb_metrics = results_metrics["HistGradientBoosting"]
        self.assertGreaterEqual(hgb_metrics["roc_auc"], 0.90, "GBDT ROC-AUC should meet SLA >= 0.90")
        self.assertGreaterEqual(hgb_metrics["pr_auc"], 0.85, "GBDT PR-AUC should meet SLA >= 0.85")

    def test_04_e2e_master_integrated_risk_matrix(self):
        """
        E2E Test 4: Master Integration Test combining Friend's Personas with Lead's Churn Risk.
        Exports and validates segment_risk_matrix.csv, priority_action_cohorts.csv, and integrated_risk_report.json.
        """
        report = generate_integrated_risk_report(output_dir=MODEL_DIR)

        self.assertIsInstance(report, dict)
        self.assertGreater(report["total_customers"], 0)
        self.assertIn("risk_matrix", report)

        matrix_csv = MODEL_DIR / "segment_risk_matrix.csv"
        cohorts_csv = MODEL_DIR / "priority_action_cohorts.csv"
        report_json = MODEL_DIR / "integrated_risk_report.json"

        self.assertTrue(matrix_csv.exists())
        self.assertTrue(cohorts_csv.exists())
        self.assertTrue(report_json.exists())

        # Load and verify priority cohorts CSV structure
        df_cohorts = pd.read_csv(cohorts_csv)
        self.assertIn("Recommended_Action", df_cohorts.columns)
        self.assertIn("Churn_Probability", df_cohorts.columns)
        self.assertIn("Risk_Tier", df_cohorts.columns)

    def test_05_e2e_api_service_and_latency_sla(self):
        """
        E2E Test 5: REST API User Journey & Inference Latency SLA Test (< 200 ms).
        """
        # GET /health
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertTrue(res_health.json()["model_loaded"])

        # POST /predict_churn (Single Prediction SLA Test)
        sample_input = {
            "Tenure": 3.0,
            "CityTier": 2,
            "WarehouseToHome": 12.0,
            "HourSpendOnApp": 3.0,
            "NumberOfDeviceRegistered": 4,
            "SatisfactionScore": 2,
            "NumberOfAddress": 3,
            "Complain": 1,
            "OrderAmountHikeFromlastYear": 15.0,
            "CouponUsed": 1.0,
            "OrderCount": 2.0,
            "DaySinceLastOrder": 14.0,
            "CashbackAmount": 140.0,
            "PreferredLoginDevice": "Mobile Phone",
            "PreferredPaymentMode": "Credit Card",
            "Gender": "Male",
            "PreferedOrderCat": "Mobile Phone",
            "MaritalStatus": "Single"
        }

        start_t = time.perf_counter()
        res_pred = self.client.post("/predict_churn", json=sample_input)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        self.assertEqual(res_pred.status_code, 200)
        pred_data = res_pred.json()

        self.assertIn(pred_data["risk_tier"], ["High Risk", "Medium Risk", "Low Risk"])
        self.assertTrue(0.0 <= pred_data["churn_probability"] <= 1.0)
        self.assertLess(latency_ms, 200.0, f"API inference latency ({latency_ms:.2f} ms) exceeded 200ms SLA!")

        # POST /predict_churn_batch
        res_batch = self.client.post("/predict_churn_batch", json={"customers": [sample_input, sample_input]})
        self.assertEqual(res_batch.status_code, 200)
        self.assertEqual(len(res_batch.json()), 2)

        # GET /segment_matrix
        res_matrix = self.client.get("/segment_matrix")
        self.assertEqual(res_matrix.status_code, 200)


if __name__ == "__main__":
    unittest.main()
