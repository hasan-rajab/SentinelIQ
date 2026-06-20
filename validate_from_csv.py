"""
Performance Validation Script - reads from pre-generated CSV files.
Faster since data is already loaded.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from backend.services.anomaly_service import AnomalyService

service = AnomalyService()

def validate_metrics_csv():
    print("\n" + "=" * 60)
    print("  METRICS VALIDATION (from CSV)")
    print("=" * 60)

    df = pd.read_csv("data/processed/metrics_test.csv")
    print(f"  Loaded {len(df)} records | anomalies = {df['is_anomaly'].sum()}")

    y_true = []
    y_pred = []
    fused_scores = []

    for _, row in df.iterrows():
        record = row.to_dict()
        scores = service.score_metric_record(record)

        fused = service.ensemble.fuse(
            if_scores=np.array([scores["if_score"]]),
            ae_scores=np.array([scores["ae_score"]])
        )[0]

        fused_scores.append(fused)
        pred = int(fused >= service.ensemble.threshold)
        y_true.append(row["is_anomaly"])
        y_pred.append(pred)

    print("\n" + "-" * 40)
    print("  Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))

    results = pd.DataFrame({"true": y_true, "pred": y_pred, "fused_score": fused_scores})
    missed = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"  Missed anomalies: {len(missed)} / {sum(y_true)}")
    if len(missed) > 0:
        print(f"  Missed anomaly avg fused score: {missed['fused_score'].mean():.4f}")


def validate_network_csv():
    print("\n" + "=" * 60)
    print("  NETWORK VALIDATION (from CSV)")
    print("=" * 60)

    df = pd.read_csv("data/processed/network_test.csv")
    print(f"  Loaded {len(df)} records | anomalies = {df['is_anomaly'].sum()}")

    y_true = []
    y_pred = []
    fused_scores = []

    for _, row in df.iterrows():
        record = row.to_dict()
        df_row = pd.DataFrame([record])
        if_score = float(service.if_network.score(df_row)[0]) if service.if_network else 0.0

        fused = service.ensemble.fuse(
            if_scores=np.array([if_score]),
            ae_scores=None
        )[0]

        fused_scores.append(fused)
        pred = int(fused >= service.ensemble.threshold)
        y_true.append(row["is_anomaly"])
        y_pred.append(pred)

    print("\n" + "-" * 40)
    print("  Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))

    results = pd.DataFrame({"true": y_true, "pred": y_pred, "fused_score": fused_scores})
    missed = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"  Missed anomalies: {len(missed)} / {sum(y_true)}")
    if len(missed) > 0:
        print(f"  Missed anomaly avg fused score: {missed['fused_score'].mean():.4f}")


def validate_logs_csv():
    print("\n" + "=" * 60)
    print("  LOGS VALIDATION (from CSV)")
    print("=" * 60)

    df = pd.read_csv("data/processed/logs_test.csv")
    print(f"  Loaded {len(df)} records | anomalies = {df['is_anomaly'].sum()}")

    y_true = []
    y_pred = []
    fused_scores = []

    for _, row in df.iterrows():
        record = row.to_dict()
        if service.bert:
            df_row = pd.DataFrame([record])
            bert_score = float(service.bert.score(df_row)[0])
        else:
            bert_score = 0.0

        fused = service.ensemble.fuse(
            if_scores=None,
            ae_scores=None,
            bert_scores=np.array([bert_score])
        )[0]

        fused_scores.append(fused)
        pred = int(fused >= service.ensemble.threshold)
        y_true.append(row["is_anomaly"])
        y_pred.append(pred)

    print("\n" + "-" * 40)
    print("  Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))

    results = pd.DataFrame({"true": y_true, "pred": y_pred, "fused_score": fused_scores})
    missed = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"  Missed anomalies: {len(missed)} / {sum(y_true)}")
    if len(missed) > 0:
        print(f"  Missed anomaly avg fused score: {missed['fused_score'].mean():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SENTINELIQ PERFORMANCE VALIDATION (from CSV)")
    print("=" * 60)
    validate_metrics_csv()
    validate_network_csv()
    validate_logs_csv()