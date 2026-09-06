import pandas as pd
from src.data.clean_data import clean_retail_data
from src.features.rfm_features import build_scaled_rfm_features
from src.models.segmentation.kmeans_model import evaluate_k_range, select_best_k, fit_kmeans
from src.models.segmentation.cluster_profiler import profile_clusters, assign_personas

print("Loading raw data...")
df_raw = pd.read_csv("data/online_retail_II.csv", low_memory=False)
print(f"Raw rows: {len(df_raw):,}")

print("Cleaning data...")
df_clean = clean_retail_data(df_raw)
print(f"Cleaned rows: {len(df_clean):,}, unique customers: {df_clean['Customer ID'].nunique():,}")

print("Building RFM features...")
rfm_raw, X_scaled, scaler = build_scaled_rfm_features(df_clean)

print("Evaluating k=2..10 with K-Means...")
metrics = evaluate_k_range(X_scaled, k_range=range(2, 11))
print(metrics)

best_k = select_best_k(metrics)
print(f"\nBest k selected: {best_k}")

model, labels = fit_kmeans(X_scaled, best_k)
profile = profile_clusters(rfm_raw, labels)
personas = assign_personas(profile)

print("\n=== FINAL SEGMENTS ===")
print(personas)

personas.to_csv("segment_personas_output.csv")
print("\nSaved results to segment_personas_output.csv")