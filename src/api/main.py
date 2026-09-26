"""
FastAPI REST API Service for Customer Churn Prediction, SHAP Explainability & Risk Matrix Serving.

Endpoints:
  GET /                    : Root welcome message & API metadata.
  GET /health              : Health check & model readiness status.
  POST /predict_churn      : Predicts churn probability, binary churn flag, and risk tier for a customer.
  POST /predict_churn_batch: Batch inference for multiple customer profiles.
  GET /explain/{customer_id}: Provides SHAP explanation (top risk drivers & protective factors).
  GET /segment_matrix      : Returns cross-tabulated Segment x Churn Risk Matrix and priority cohort summary.

Assigned to: Lead Engineer (You)
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as APIPath, status
from pydantic import BaseModel, Field

from src.evaluation.explainability import SHAPExplainerEngine
from src.features.churn_features import ChurnFeatureEngineer, prepare_churn_features

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "churn"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Global Model & Artifact Cache
model_cache: Dict[str, Any] = {}


def load_api_artifacts():
    """Loads champion model, feature engineer, and cached reports into memory."""
    global model_cache
    logger.info("Initializing FastAPI model artifacts...")

    # Load Model
    model_path = MODEL_DIR / "tuned_champion_model.joblib"
    if not model_path.exists():
        model_path = MODEL_DIR / "gbdt_hist_gradient_boosting.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "baseline_random_forest.joblib"

    if model_path.exists():
        model_cache["model"] = joblib.load(model_path)
        model_cache["model_path"] = str(model_path)
        logger.info(f"Loaded churn prediction model from {model_path}")
    else:
        model_cache["model"] = None
        model_cache["model_path"] = "None"
        logger.warning("No trained churn model found in models/churn/")

    # Load Feature Engineer
    fe_path = MODEL_DIR / "churn_feature_engineer.joblib"
    if fe_path.exists():
        model_cache["feature_engineer"] = joblib.load(fe_path)
        logger.info(f"Loaded ChurnFeatureEngineer from {fe_path}")
    else:
        # Fit on churn_train.csv if available
        train_csv = DATA_PROCESSED_DIR / "churn_train.csv"
        if train_csv.exists():
            train_df = pd.read_csv(train_csv)
            _, _, engineer = prepare_churn_features(train_df, scale_features=False)
            model_cache["feature_engineer"] = engineer
            logger.info("Fitted ChurnFeatureEngineer on data/processed/churn_train.csv")
        else:
            model_cache["feature_engineer"] = None
            logger.warning("No FeatureEngineer found and churn_train.csv unavailable.")

    # Initialize SHAP Engine if model exists
    if model_cache["model"] is not None:
        model_cache["shap_engine"] = SHAPExplainerEngine(model_cache["model"])
    else:
        model_cache["shap_engine"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager for Model Loading & Cleanup."""
    load_api_artifacts()
    yield
    model_cache.clear()
    logger.info("FastAPI service shutdown: cleared model cache.")


app = FastAPI(
    title="RetainAI - Customer Segmentation & Explainable Churn REST API",
    description="Industrial REST API exposing real-time churn prediction, SHAP risk driver explanations, and segment risk matrix analytics.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- PYDANTIC SCHEMAS ---

class CustomerFeaturesInput(BaseModel):
    Tenure: float = Field(12.0, example=12.0, description="Months customer has stayed with company")
    CityTier: int = Field(1, example=1, description="City Tier (1, 2, 3)")
    WarehouseToHome: float = Field(10.0, example=15.0, description="Distance from warehouse to home in km")
    HourSpendOnApp: float = Field(3.0, example=3.5, description="Average hours spent on app per day")
    NumberOfDeviceRegistered: int = Field(3, example=4, description="Number of registered devices")
    SatisfactionScore: int = Field(3, example=2, description="Customer satisfaction score (1 to 5)")
    NumberOfAddress: int = Field(1, example=2, description="Number of registered delivery addresses")
    Complain: int = Field(0, example=1, description="Complaint logged flag (1=Yes, 0=No)")
    OrderAmountHikeFromlastYear: float = Field(15.0, example=14.0, description="Percentage order amount hike from last year")
    CouponUsed: float = Field(1.0, example=2.0, description="Number of coupons used")
    OrderCount: float = Field(2.0, example=3.0, description="Total order count")
    DaySinceLastOrder: float = Field(5.0, example=12.0, description="Days elapsed since last order")
    CashbackAmount: float = Field(160.0, example=150.0, description="Total cashback received")
    PreferredLoginDevice: str = Field("Mobile Phone", example="Mobile Phone", description="Preferred login device")
    PreferredPaymentMode: str = Field("Debit Card", example="Debit Card", description="Preferred payment mode")
    Gender: str = Field("Female", example="Female", description="Gender")
    PreferedOrderCat: str = Field("Laptop & Accessory", example="Mobile Phone", description="Preferred order category")
    MaritalStatus: str = Field("Single", example="Single", description="Marital status")


class BatchCustomerFeaturesInput(BaseModel):
    customers: List[CustomerFeaturesInput]


class ChurnPredictionResponse(BaseModel):
    customer_id: Optional[str] = "N/A"
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    confidence_score: float
    model_version: str


class RiskDriver(BaseModel):
    feature: str
    shap_value: float
    feature_value: Any
    impact: str


class CustomerExplanationResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    top_risk_drivers: List[RiskDriver]
    top_protective_factors: List[RiskDriver]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_engineer_loaded: bool
    model_path: str
    version: str


# --- HELPER FUNCTIONS ---

def categorize_risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"


def prepare_input_dataframe(input_data: CustomerFeaturesInput) -> pd.DataFrame:
    raw_dict = input_data.model_dump()
    return pd.DataFrame([raw_dict])


# --- ENDPOINTS ---

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to RetainAI Industrial Customer Segmentation & Explainable Churn REST API",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": [
            "POST /predict_churn",
            "POST /predict_churn_batch",
            "GET /explain/{customer_id}",
            "GET /segment_matrix"
        ]
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check():
    model = model_cache.get("model")
    fe = model_cache.get("feature_engineer")
    return HealthCheckResponse(
        status="healthy" if model is not None else "degraded (no model loaded)",
        model_loaded=model is not None,
        feature_engineer_loaded=fe is not None,
        model_path=model_cache.get("model_path", "None"),
        version="1.0.0"
    )


@app.post("/predict_churn", response_model=ChurnPredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerFeaturesInput):
    model = model_cache.get("model")
    engineer = model_cache.get("feature_engineer")

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn prediction model is not loaded. Train models before requesting inference."
        )

    df_raw = prepare_input_dataframe(customer)

    if engineer is not None:
        X_trans = engineer.transform(df_raw)
    else:
        X_trans = df_raw

    prob = float(model.predict_proba(X_trans)[:, 1][0])
    pred = int(prob >= 0.5)
    tier = categorize_risk_tier(prob)
    conf = prob if pred == 1 else (1.0 - prob)

    return ChurnPredictionResponse(
        customer_id="N/A",
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        confidence_score=round(conf, 4),
        model_version=Path(model_cache.get("model_path", "tuned_champion_model")).stem
    )


@app.post("/predict_churn_batch", response_model=List[ChurnPredictionResponse], tags=["Inference"])
def predict_churn_batch(batch: BatchCustomerFeaturesInput):
    model = model_cache.get("model")
    engineer = model_cache.get("feature_engineer")

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn prediction model is not loaded."
        )

    df_raw = pd.DataFrame([c.model_dump() for c in batch.customers])

    if engineer is not None:
        X_trans = engineer.transform(df_raw)
    else:
        X_trans = df_raw

    probs = model.predict_proba(X_trans)[:, 1]
    preds = (probs >= 0.5).astype(int)

    responses = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        prob_float = float(prob)
        pred_int = int(pred)
        tier = categorize_risk_tier(prob_float)
        conf = prob_float if pred_int == 1 else (1.0 - prob_float)
        responses.append(
            ChurnPredictionResponse(
                customer_id=f"BATCH_{i+1:04d}",
                churn_probability=round(prob_float, 4),
                churn_prediction=pred_int,
                risk_tier=tier,
                confidence_score=round(conf, 4),
                model_version=Path(model_cache.get("model_path", "tuned_champion_model")).stem
            )
        )

    return responses


@app.get("/explain/{customer_id}", response_model=CustomerExplanationResponse, tags=["Explainability"])
def explain_customer(customer_id: str = APIPath(..., description="Customer ID to explain")):
    # 1. Check if precomputed explainability report contains customer
    report_json_path = MODEL_DIR / "shap_explainability_report.json"
    if report_json_path.exists():
        try:
            with open(report_json_path, "r") as f:
                report = json.load(f)
            explanations = report.get("individual_customer_explanations", {})
            if customer_id in explanations:
                data = explanations[customer_id]
                return CustomerExplanationResponse(
                    customer_id=data["customer_id"],
                    churn_probability=data["churn_probability"],
                    risk_tier=data["risk_tier"],
                    top_risk_drivers=[RiskDriver(**f) for f in data.get("top_risk_drivers", [])],
                    top_protective_factors=[RiskDriver(**f) for f in data.get("top_protective_factors", [])],
                )
        except Exception as e:
            logger.warning(f"Failed reading shap_explainability_report.json: {e}")

    # 2. Compute on the fly if test dataset customer exists
    model = model_cache.get("model")
    engineer = model_cache.get("feature_engineer")
    shap_engine = model_cache.get("shap_engine")

    if model is None or shap_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainability engine is not loaded."
        )

    # Try loading churn data
    test_csv = DATA_PROCESSED_DIR / "churn_test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        if "CustomerID" in test_df.columns and (test_df["CustomerID"].astype(str) == str(customer_id)).any():
            cust_row = test_df[test_df["CustomerID"].astype(str) == str(customer_id)].iloc[[0]]
            cust_features = cust_row.drop(columns=["Churn", "CustomerID"], errors="ignore")
            if engineer is not None:
                cust_trans = engineer.transform(cust_features)
            else:
                cust_trans = cust_features
            exp = shap_engine.explain_customer(cust_trans, customer_id=customer_id)
            return CustomerExplanationResponse(
                customer_id=exp["customer_id"],
                churn_probability=exp["churn_probability"],
                risk_tier=exp["risk_tier"],
                top_risk_drivers=[RiskDriver(**f) for f in exp.get("top_risk_drivers", [])],
                top_protective_factors=[RiskDriver(**f) for f in exp.get("top_protective_factors", [])],
            )

    # Fallback response for un-indexed synthetic / missing customer ID
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Customer ID '{customer_id}' not found in precomputed SHAP explanations or test dataset."
    )


@app.get("/segment_matrix", tags=["Analytics"])
def get_segment_matrix():
    report_json_path = MODEL_DIR / "integrated_risk_report.json"
    if not report_json_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integrated Segment x Churn Risk Matrix report not found. Run src/models/risk_matrix.py first."
        )

    with open(report_json_path, "r") as f:
        report = json.load(f)

    return report
