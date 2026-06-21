"""
Diagnostic script to analyze score distributions and fix threshold.
"""

import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from backend.services.anomaly_service import AnomalyService

service = AnomalyService()

print("=" * 60)
print("  SCORE DISTRIBUTION ANALYSIS")
print("=" * 60)

# Current threshold
print(f"\nCurrent threshold: {service.ensemble.threshold}")

# --- METRICS ---
print("\n" + "=" * 60)
print("  METRICS SCORE ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/processed/metrics_test.csv")
print(f"Records: {len(df)}, Anomalies: {df['is_anomaly'].sum()}")

if_scores = []
ae_scores = []
fused_scores = []
y_true = []

for _, row in df.iterrows():
    record = row.to_dict()
    scores = service.score_metric_record(record)
    
    fused = service.ensemble.fuse(
        if_scores=np.array([scores["if_score"]]),
        ae_scores=np.array([scores["ae_score"]])
    )[0]
    
    if_scores.append(scores["if_score"])
    ae_scores.append(scores["ae_score"])
    fused_scores.append(fused)
    y_true.append(row["is_anomaly"])

if_scores = np.array(if_scores)
ae_scores = np.array(ae_scores)
fused_scores = np.array(fused_scores)
y_true = np.array(y_true)

print(f"\nIF scores: min={if_scores.min():.4f}, max={if_scores.max():.4f}, mean={if_scores.mean():.4f}")
print(f"AE scores: min={ae_scores.min():.4f}, max={ae_scores.max():.4f}, mean={ae_scores.mean():.4f}")
print(f"Fused scores: min={fused_scores.min():.4f}, max={fused_scores.max():.4f}, mean={fused_scores.mean():.4f}")

# Percentiles for threshold selection
print("\nFused score percentiles:")
for p in [50, 70, 80, 85, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(fused_scores, p):.4f}")

# Best threshold search
print("\nSearching for optimal threshold...")
best_f1 = 0
threshold = service.ensemble.threshold
y_pred = (scores >= threshold).astype(int)



# Prediction distribution with current threshold
print(f"\nWith current threshold ({service.ensemble.threshold}):")
y_pred_current = (fused_scores >= service.ensemble.threshold).astype(int)
print(f"  Predictions: {Counter(y_pred_current)}")

# --- NETWORK ---
print("\n" + "=" * 60)
print("  NETWORK SCORE ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/processed/network_test.csv")
print(f"Records: {len(df)}, Anomalies: {df['is_anomaly'].sum()}")

net_if_scores = []
net_fused_scores = []
net_y_true = []

for _, row in df.iterrows():
    record = row.to_dict()
    df_row = pd.DataFrame([record])
    if_score = float(service.if_network.score(df_row)[0]) if service.if_network else 0.0
    
    fused = service.ensemble.fuse(
        if_scores=np.array([if_score]),
        ae_scores=None
    )[0]
    
    net_if_scores.append(if_score)
    net_fused_scores.append(fused)
    net_y_true.append(row["is_anomaly"])

net_if_scores = np.array(net_if_scores)
net_fused_scores = np.array(net_fused_scores)
net_y_true = np.array(net_y_true)

print(f"\nIF scores: min={net_if_scores.min():.4f}, max={net_if_scores.max():.4f}, mean={net_if_scores.mean():.4f}")
print(f"Fused scores: min={net_fused_scores.min():.4f}, max={net_fused_scores.max():.4f}, mean={net_fused_scores.mean():.4f}")

print("\nFused score percentiles:")
for p in [50, 70, 80, 85, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(net_fused_scores, p):.4f}")

# Best threshold search for network
print("\nSearching for optimal threshold...")
best_f1 = 0



# --- LOGS ---
print("\n" + "=" * 60)
print("  LOGS SCORE ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/processed/logs_test.csv")
print(f"Records: {len(df)}, Anomalies: {df['is_anomaly'].sum()}")

# Check if 'is_anomaly' column exists
if 'is_anomaly' in df.columns:
    print(f"  is_anomaly column exists: {df['is_anomaly'].value_counts().to_dict()}")
else:
    print("  WARNING: is_anomaly column not found!")

# Check first few rows
print(f"\n  First 3 rows:")
for _, row in df.head(3).iterrows():
    print(f"    message={str(row.get('message', ''))[:50]}..., is_anomaly={row.get('is_anomaly', 'N/A')}")