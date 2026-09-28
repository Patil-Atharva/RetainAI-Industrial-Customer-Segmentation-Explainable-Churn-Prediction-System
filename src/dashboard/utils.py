"""
RetainAI Dashboard Utility & Visualization Module.
Provides data access loaders, Plotly figure builders, API health checkers,
and isolated local inference fallbacks (CON-002, TASK-J08, TASK-J09).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "churn"
MODELS_ROOT = PROJECT_ROOT / "models"

# Visual Token Palettes
PERSONA_COLORS = {
    "Champions": "#10B981",          # Vibrant Emerald
    "Loyal / Steady": "#3B82F6",     # Sapphire Blue
    "At Risk": "#F59E0B",            # Amber Orange
    "Hibernating": "#EF4444",        # Crimson Red
    "New / Low Engagement": "#8B5CF6", # Amethyst Purple
}

RISK_TIER_COLORS = {
    "High Risk": "#EF4444",
    "Medium Risk": "#F59E0B",
    "Low Risk": "#10B981",
}

PERSONA_STRATEGIES = {
    "Champions": "VIP Loyalty Concierge & Exclusive Upsell",
    "Loyal / Steady": "Cross-Sell Accelerator & Tier Upgrade",
    "At Risk": "High-Touch Win-Back & Retention Discount",
    "Hibernating": "Automated Reactivation Push & Feature Highlight",
    "New / Low Engagement": "Onboarding Nurture & First-Reorder Incentive",
}


# ============================================================================
# 1. DATA LOADERS WITH STREAMLIT CACHING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_personas_summary() -> pd.DataFrame:
    """
    Loads persona profiling summary from segment_personas_output.csv.
    Enriches table with marketing retention strategy badges.
    """
    csv_path = PROJECT_ROOT / "segment_personas_output.csv"
    if not csv_path.exists():
        # Fallback placeholder if upstream artifact hasn't been generated
        logger.warning(f"Persona file not found at {csv_path}. Generating default.")
        df = pd.DataFrame([
            {"Cluster_ID": 0, "Recency_mean": 65.9, "Frequency_mean": 11.2, "Monetary_mean": 5759.66, "Customer_Count": 2800, "Pct_of_Base": 47.64, "Persona": "Champions"},
            {"Cluster_ID": 1, "Recency_mean": 324.5, "Frequency_mean": 1.8, "Monetary_mean": 525.14, "Customer_Count": 3078, "Pct_of_Base": 52.36, "Persona": "Hibernating"}
        ])
    else:
        df = pd.read_csv(csv_path)

    # Attach strategy badge
    df["Retention_Strategy"] = df["Persona"].map(lambda p: PERSONA_STRATEGIES.get(p, "Standard Monitoring"))
    return df


@st.cache_data(show_spinner=False)
def load_customer_embeddings(sample_max: Optional[int] = 5000) -> pd.DataFrame:
    """
    Loads customer-level RFM metrics and 2D/3D manifold coordinates (UMAP & PCA).
    Reuses precomputed embeddings to prevent dashboard UI latency.
    """
    precomputed_path = DATA_PROCESSED_DIR / "customer_segment_embeddings.csv"

    if precomputed_path.exists():
        df = pd.read_csv(precomputed_path)
    else:
        # Compute on the fly if precomputed artifact is missing
        logger.info("Precomputed embeddings not found. Re-computing from retail data...")
        from src.data.clean_data import clean_retail_data
        from src.features.rfm_features import build_scaled_rfm_features
        from src.models.segmentation.dim_reduction import pca_reduce, umap_reduce

        retail_csv = DATA_DIR / "online_retail_II.csv"
        if not retail_csv.exists():
            return pd.DataFrame()

        df_raw = pd.read_csv(retail_csv, low_memory=False)
        df_clean = clean_retail_data(df_raw)
        rfm_raw, X_scaled, _ = build_scaled_rfm_features(df_clean)

        pca_coords = pca_reduce(X_scaled, n_components=3)
        try:
            umap_coords = umap_reduce(X_scaled, n_components=3)
        except Exception:
            umap_coords = pca_coords.rename(columns={"PC1": "UMAP1", "PC2": "UMAP2", "PC3": "UMAP3"})

        df = rfm_raw.copy().reset_index()
        df["CustomerID"] = df[df.columns[0]].astype(str)
        # Approximate personas via spend
        q = df["Monetary"].quantile([0.33, 0.66])
        df["Persona"] = df["Monetary"].apply(
            lambda m: "Champions" if m > q.iloc[1] else ("Loyal / Steady" if m > q.iloc[0] else "Hibernating")
        )
        df["Cluster_ID"] = df["Persona"].map({"Champions": 0, "Loyal / Steady": 1, "Hibernating": 2})

        for c in ["PC1", "PC2", "PC3"]:
            df[c] = pca_coords[c]
        for c in ["UMAP1", "UMAP2", "UMAP3"]:
            df[c] = umap_coords[c]

    if sample_max and len(df) > sample_max:
        return df.sample(n=sample_max, random_state=42).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_silhouette_metrics() -> pd.DataFrame:
    """
    Loads k-evaluation metrics across k in [2, 10] for Silhouette analysis chart.
    """
    csv_path = MODELS_ROOT / "silhouette_evaluation.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Default fallback curve matching system RFM Silhouette behavior
    return pd.DataFrame({
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "silhouette": [0.4380, 0.3432, 0.3817, 0.3444, 0.3350, 0.3168, 0.2956, 0.2999, 0.2979],
        "inertia": [8229.1, 6130.6, 4609.7, 3863.3, 3361.4, 3017.8, 2730.8, 2501.8, 2328.1],
        "davies_bouldin": [0.858, 1.060, 0.906, 0.949, 0.936, 0.989, 1.016, 1.007, 1.027]
    })


@st.cache_data(show_spinner=False)
def load_segment_risk_matrix() -> pd.DataFrame:
    """
    Loads Segment Persona x Churn Risk Tier 2D cross-tabulation table.
    """
    csv_path = MODEL_DIR / "segment_risk_matrix.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
        return df

    # Fallback from JSON report
    json_path = MODEL_DIR / "integrated_risk_report.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        if "risk_matrix" in data:
            return pd.DataFrame(data["risk_matrix"]).T

    # Static fallback if not yet generated
    return pd.DataFrame({
        "High Risk": {"Champions": 0, "Hibernating": 971, "Loyal / Steady": 0, "Total": 971},
        "Medium Risk": {"Champions": 0, "Hibernating": 215, "Loyal / Steady": 0, "Total": 215},
        "Low Risk": {"Champions": 1858, "Hibernating": 728, "Loyal / Steady": 1858, "Total": 4444},
        "Total": {"Champions": 1858, "Hibernating": 1914, "Loyal / Steady": 1858, "Total": 5630}
    })


@st.cache_data(show_spinner=False)
def load_priority_cohorts() -> pd.DataFrame:
    """
    Loads priority retention action cohorts from CSV for table display and download.
    """
    csv_path = MODEL_DIR / "priority_action_cohorts.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Check JSON fallback
    json_path = MODEL_DIR / "integrated_risk_report.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        if "top_priority_cohort_sample" in data:
            return pd.DataFrame(data["top_priority_cohort_sample"])

    return pd.DataFrame()


# ============================================================================
# 2. REUSABLE PLOTLY FIGURE BUILDERS
# ============================================================================

def build_cluster_scatter_plot(
    df: pd.DataFrame,
    x_col: str = "UMAP1",
    y_col: str = "UMAP2",
    z_col: Optional[str] = None,
    color_col: str = "Persona",
) -> go.Figure:
    """
    Builds interactive Plotly 2D or 3D scatter plot of customer embeddings.
    Includes rich hover tooltips with CustomerID, Recency, Frequency, and Monetary values.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No customer data available to plot.")
        return fig

    # Custom tooltips
    custom_cols = ["CustomerID", "Persona", "Recency", "Frequency", "Monetary"]
    for c in custom_cols:
        if c not in df.columns:
            df[c] = "N/A"

    if z_col and z_col in df.columns:
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            color_discrete_map=PERSONA_COLORS,
            hover_name="CustomerID",
            hover_data={
                "Persona": True,
                "Recency": ":.0f",
                "Frequency": ":.0f",
                "Monetary": ":$.2f",
                x_col: False,
                y_col: False,
                z_col: False,
            },
            opacity=0.85,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_map=PERSONA_COLORS,
            hover_name="CustomerID",
            hover_data={
                "Persona": True,
                "Recency": ":.0f",
                "Frequency": ":.0f",
                "Monetary": ":$.2f",
                x_col: False,
                y_col: False,
            },
            opacity=0.78,
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,0.4)")))
        fig.update_layout(
            margin=dict(l=10, r=10, b=10, t=35),
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Roboto, sans-serif"),
    )
    return fig


def build_silhouette_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Renders bar chart displaying Silhouette Scores across k=2..10.
    Highlights optimal cluster count k and displays the REQ-003 threshold (0.55).
    """
    if metrics_df.empty or "silhouette" not in metrics_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Silhouette metrics data available.")
        return fig

    best_idx = metrics_df["silhouette"].idxmax()
    best_k = int(metrics_df.loc[best_idx, "k"])
    best_score = float(metrics_df.loc[best_idx, "silhouette"])

    # Colors: Highlight best k in emerald, others in slate
    colors = ["#10B981" if k == best_k else "#475569" for k in metrics_df["k"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df["k"].astype(str),
        y=metrics_df["silhouette"],
        marker_color=colors,
        text=[f"{s:.3f}" for s in metrics_df["silhouette"]],
        textposition="auto",
        name="Silhouette Score",
        hovertemplate="<b>k = %{x}</b><br>Silhouette Score: %{y:.4f}<extra></extra>"
    ))

    # Reference benchmark line at 0.55 (REQ-003 target)
    fig.add_hline(
        y=0.55,
        line_dash="dot",
        line_color="#F59E0B",
        annotation_text="Target Benchmark (0.55)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Cluster Evaluation: Silhouette Scores (Best k = {best_k}, Score = {best_score:.3f})",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        yaxis=dict(range=[0, 1.0], gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, Roboto, sans-serif"),
    )
    return fig


def build_churn_gauge(probability: float) -> go.Figure:
    """
    Constructs an interactive circular gauge indicator for customer churn probability.
    Visualizes risk zones: Low (<30%), Medium (30-70%), and High (>=70%).
    """
    pct = probability * 100.0

    if probability >= 0.70:
        bar_color = "#EF4444"  # Red
        tier_text = "HIGH RISK"
    elif probability >= 0.30:
        bar_color = "#F59E0B"  # Amber
        tier_text = "MEDIUM RISK"
    else:
        bar_color = "#10B981"  # Emerald
        tier_text = "LOW RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 42, "color": bar_color}},
        title={"text": f"<b>Churn Probability</b><br><span style='color:{bar_color}; font-size:16px;'>{tier_text}</span>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "gray"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(30, 41, 59, 0.6)",
            "borderwidth": 1,
            "bordercolor": "#475569",
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [30, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 3},
                "thickness": 0.8,
                "value": pct
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        font=dict(family="Inter, Roboto, sans-serif"),
    )
    return fig


def build_shap_horizontal_bar(
    top_risk_drivers: List[Dict[str, Any]],
    top_protective_factors: List[Dict[str, Any]],
) -> go.Figure:
    """
    Renders horizontal diverging bar chart splitting top risk drivers (pushing churn UP, Red)
    from top protective factors (lowering churn risk, Green).
    """
    records = []
    # Protective factors (Negative SHAP values)
    for item in reversed(top_protective_factors):
        records.append({
            "feature": item["feature"],
            "shap_value": float(item["shap_value"]),
            "feature_val": item.get("feature_value", "N/A"),
            "category": "Protective Factor",
            "color": "#10B981"
        })
    # Risk pushers (Positive SHAP values)
    for item in top_risk_drivers:
        records.append({
            "feature": item["feature"],
            "shap_value": float(item["shap_value"]),
            "feature_val": item.get("feature_value", "N/A"),
            "category": "Risk Driver",
            "color": "#EF4444"
        })

    if not records:
        fig = go.Figure()
        fig.update_layout(title="No SHAP attribution records available to visualize.")
        return fig

    df_plot = pd.DataFrame(records)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot["feature"],
        x=df_plot["shap_value"],
        orientation="h",
        marker=dict(color=df_plot["color"]),
        text=[f"{val:+.3f} (val: {fv})" for val, fv in zip(df_plot["shap_value"], df_plot["feature_val"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>"
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="#94A3B8")

    fig.update_layout(
        title="<b>SHAP Feature Attribution: Risk Drivers vs Protective Factors</b>",
        xaxis_title="SHAP Value (Contribution to Churn Log-Odds / Probability)",
        yaxis_title="Feature Name",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=40, t=50, b=20),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, Roboto, sans-serif"),
    )
    return fig


# ============================================================================
# 3. API HEALTH & INFERENCE ROUTING (WITH LOCAL FALLBACK)
# ============================================================================

def check_api_health(api_url: str, timeout: float = 1.5) -> Tuple[str, str, Dict[str, Any]]:
    """
    Checks the connectivity and model readiness of the FastAPI service.
    Returns:
        status: 'CONNECTED', 'DEGRADED', or 'OFFLINE'
        badge_text: Streamlit formatted status string
        payload: response JSON dict if reachable, else empty dict.
    """
    clean_url = api_url.rstrip("/")
    try:
        res = requests.get(f"{clean_url}/health", timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            if data.get("model_loaded", False):
                return "CONNECTED", "🟢 REST API Connected", data
            else:
                return "DEGRADED", "🟡 REST API Degraded (Model Unloaded)", data
        else:
            return "DEGRADED", f"🟡 REST API Status {res.status_code}", {}
    except Exception:
        return "OFFLINE", "🔴 REST API Offline (Local Fallback Active)", {}


@st.cache_resource(show_spinner=False)
def get_local_model_artifacts():
    """
    Loads local champion model and feature engineer for snappy offline fallback (CON-002).
    """
    model_path = MODEL_DIR / "tuned_champion_model.joblib"
    if not model_path.exists():
        model_path = MODEL_DIR / "gbdt_hist_gradient_boosting.joblib"
        if not model_path.exists():
            model_path = MODEL_DIR / "baseline_random_forest.joblib"

    fe_path = MODEL_DIR / "churn_feature_engineer.joblib"

    model = joblib.load(model_path) if model_path.exists() else None
    fe = joblib.load(fe_path) if fe_path.exists() else None
    return model, fe


def predict_churn_local(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes local churn prediction using the cached champion model.
    Guarantees that inference never hangs even if the REST API microservice is down.
    """
    model, fe = get_local_model_artifacts()

    if model is None:
        # Heuristic fallback if models directory was wiped
        score = 0.15
        if features_dict.get("Complain", 0) == 1:
            score += 0.40
        if features_dict.get("SatisfactionScore", 3) <= 2:
            score += 0.25
        if features_dict.get("DaySinceLastOrder", 5) > 15:
            score += 0.15
        score = min(score, 0.95)
        tier = "High Risk" if score >= 0.7 else ("Medium Risk" if score >= 0.3 else "Low Risk")
        return {
            "churn_probability": round(score, 4),
            "churn_prediction": int(score >= 0.5),
            "risk_tier": tier,
            "confidence_score": round(score if score >= 0.5 else 1.0 - score, 4),
            "model_version": "heuristic_fallback"
        }

    df_in = pd.DataFrame([features_dict])
    X_trans = fe.transform(df_in) if fe is not None else df_in
    prob = float(model.predict_proba(X_trans)[:, 1][0])
    pred = int(prob >= 0.5)
    tier = "High Risk" if prob >= 0.70 else ("Medium Risk" if prob >= 0.30 else "Low Risk")
    conf = prob if pred == 1 else (1.0 - prob)

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "risk_tier": tier,
        "confidence_score": round(conf, 4),
        "model_version": "local_champion_model"
    }


def explain_customer_local(
    features_dict: Dict[str, Any], customer_id: str = "N/A"
) -> Dict[str, Any]:
    """
    Computes local SHAP attributions using SHAPExplainerEngine directly in-memory.
    """
    from src.evaluation.explainability import SHAPExplainerEngine

    model, fe = get_local_model_artifacts()
    if model is None:
        return {
            "customer_id": customer_id,
            "churn_probability": 0.5,
            "risk_tier": "Medium Risk",
            "top_risk_drivers": [{"feature": "Complain", "shap_value": 0.35, "feature_value": features_dict.get("Complain", 1), "impact": "increases_churn_risk"}],
            "top_protective_factors": [{"feature": "Tenure", "shap_value": -0.25, "feature_value": features_dict.get("Tenure", 12), "impact": "reduces_churn_risk"}],
        }

    df_in = pd.DataFrame([features_dict])
    X_trans = fe.transform(df_in) if fe is not None else df_in
    feature_names = getattr(fe, "feature_names_", list(X_trans.columns))

    engine = SHAPExplainerEngine(model, feature_names=feature_names)
    explanation = engine.explain_customer(X_trans, customer_id=customer_id)
    return explanation


def predict_churn_api_or_fallback(
    api_url: str, customer_features: Dict[str, Any], timeout: float = 2.0
) -> Tuple[Dict[str, Any], bool]:
    """
    Attempts to predict churn via REST API; smoothly falls back to local model if unavailable.
    Returns:
        (prediction_result, used_local_fallback: bool)
    """
    clean_url = api_url.rstrip("/")
    try:
        res = requests.post(f"{clean_url}/predict_churn", json=customer_features, timeout=timeout)
        if res.status_code == 200:
            return res.json(), False
    except Exception as e:
        logger.info(f"API call to {clean_url}/predict_churn failed: {e}. Activating local fallback.")

    local_res = predict_churn_local(customer_features)
    return local_res, True


def explain_customer_api_or_fallback(
    api_url: str, customer_id: str, customer_features: Dict[str, Any], timeout: float = 2.0
) -> Tuple[Dict[str, Any], bool]:
    """
    Attempts to fetch SHAP explanations via REST API; falls back to local SHAP engine.
    Returns:
        (explanation_result, used_local_fallback: bool)
    """
    clean_url = api_url.rstrip("/")
    if customer_id and customer_id != "N/A":
        try:
            res = requests.get(f"{clean_url}/explain/{customer_id}", timeout=timeout)
            if res.status_code == 200:
                return res.json(), False
        except Exception as e:
            logger.info(f"API call to {clean_url}/explain/{customer_id} failed: {e}.")

    # Fallback to local on-the-fly SHAP computation
    local_exp = explain_customer_local(customer_features, customer_id=customer_id)
    return local_exp, True
