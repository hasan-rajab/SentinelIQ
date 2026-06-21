"""
Performance Validation Script for SentinelIQ AnomalyService.

Validates the model performance on metrics, network, and log data.
Generates classification reports and analyzes missed anomalies.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(str(Path(__file__).resolve().parents[0]))

from backend.services.anomaly_service import AnomalyService
from ml.features.network_features import add_network_features


def _metric_bounds(service):
    if_bounds = ae_bounds = None
    if service.if_metrics and service.if_metrics.score_min is not None:
        if_bounds = (service.if_metrics.score_min, service.if_metrics.score_max)
    if service.ae and service.ae.score_min is not None:
        ae_bounds = (service.ae.score_min, service.ae.score_max)
    return if_bounds, ae_bounds


def _network_bounds(service):
    if_bounds = ae_bounds = None
    if service.if_network and service.if_network.score_min is not None:
        if_bounds = (service.if_network.score_min, service.if_network.score_max)
    if service.ae_network and service.ae_network.score_min is not None:
        ae_bounds = (service.ae_network.score_min, service.ae_network.score_max)
    return if_bounds, ae_bounds


def load_jsonl(path: str) -> pd.DataFrame:
    """Load JSONL file into DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} rows)")


def validate_metrics(service: AnomalyService, data_path: str):
    """Validate on metrics data."""
    print("\n" + "=" * 60)
    print("  METRICS VALIDATION")
    print("=" * 60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies = {df['is_anomaly'].sum()}")

    # Save as CSV for future use
    csv_path = "data/processed/metrics_test.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, csv_path)

    y_true = []
    y_pred = []
    fused_scores = []

    if_bounds, ae_bounds = _metric_bounds(service)

    for _, row in df.iterrows():
        record = row.to_dict()
        scores = service.score_metric_record(record)

        fused = service.ensemble.fuse(
            if_scores=np.array([scores["if_score"]]),
            ae_scores=np.array([scores["ae_score"]]),
            if_bounds=if_bounds,
            ae_bounds=ae_bounds,
        )[0]

        fused_scores.append(fused)
        pred = int(fused >= service.ensemble.threshold)
        y_true.append(row["is_anomaly"])
        y_pred.append(pred)

    print("\n" + "-" * 40)
    print("  Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))

    # Analyze missed anomalies
    results = pd.DataFrame({"true": y_true, "pred": y_pred, "fused_score": fused_scores})
    missed = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"  Missed anomalies: {len(missed)} / {sum(y_true)}")
    if len(missed) > 0:
        print(f"  Missed anomaly avg fused score: {missed['fused_score'].mean():.4f}")

    return y_true, y_pred, fused_scores


def validate_network(service: AnomalyService, data_path: str):
    """Validate on network data."""
    print("\n" + "=" * 60)
    print("  NETWORK VALIDATION")
    print("=" * 60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies = {df['is_anomaly'].sum()}")

    # Save as CSV
    csv_path = "data/processed/network_test.csv"
    save_csv(df, csv_path)

    df = add_network_features(df)
    if_bounds, ae_bounds = _network_bounds(service)

    y_true = []
    y_pred = []
    fused_scores = []

    for _, row in df.iterrows():
        record = row.to_dict()
        df_row = pd.DataFrame([record])
        df_row = add_network_features(df_row)
        if_score = float(service.if_network.score(df_row)[0]) if service.if_network else 0.0

        if service.ae_network is not None:
            ae_score = float(service.ae_network.score(df_row)[0])
            fused = float(service.ensemble.fuse_network(
                if_scores=np.array([if_score]),
                ae_scores=np.array([ae_score]),
                if_bounds=if_bounds,
                ae_bounds=ae_bounds,
            )[0])
            threshold = service.ensemble.network_threshold
        else:
            fused = if_score
            threshold = service.if_network.threshold

        fused_scores.append(fused)
        pred = int(fused >= threshold)
        y_true.append(row["is_anomaly"])
        y_pred.append(pred)

    print("\n" + "-" * 40)
    print("  Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))

    # Analyze missed anomalies
    results = pd.DataFrame({"true": y_true, "pred": y_pred, "fused_score": fused_scores})
    missed = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"  Missed anomalies: {len(missed)} / {sum(y_true)}")
    if len(missed) > 0:
        print(f"  Missed anomaly avg fused score: {missed['fused_score'].mean():.4f}")

    return y_true, y_pred, fused_scores


def validate_logs(service: AnomalyService, data_path: str):
    """Validate on log data."""
    print("\n" + "=" * 60)
    print("  LOGS VALIDATION")
    print("=" * 60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies = {df['is_anomaly'].sum()}")

    # Save as CSV
    csv_path = "data/processed/logs_test.csv"
    save_csv(df, csv_path)

    y_true = []
    y_pred = []
    fused_scores = []

    for _, row in df.iterrows():
        record = row.to_dict()

        # Logs use BERT score directly
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

    # Analyze missed anomalies
    results = pd.DataFrame({"true": y_true, "pred": y_pred, "fused_score": fused_scores})
    missed = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"  Missed anomalies: {len(missed)} / {sum(y_true)}")
    if len(missed) > 0:
        print(f"  Missed anomaly avg fused score: {missed['fused_score'].mean():.4f}")

    return y_true, y_pred, fused_scores


def main():
    print("=" * 60)
    print("  SENTINELIQ PERFORMANCE VALIDATION")
    print("=" * 60)

    # Initialize service
    print("\n  Loading models...")
    service = AnomalyService()

    # Data paths
    metrics_path = "data/simulated/metrics.jsonl"
    network_path = "data/simulated/network.jsonl"
    logs_path = "data/simulated/logs.jsonl"

    # Run validation
    metrics_results = validate_metrics(service, metrics_path)
    network_results = validate_network(service, network_path)
    logs_results = validate_logs(service, logs_path)

    # Summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Metrics:  {metrics_results[1].count()} predictions | {sum(metrics_results[1])} detected anomalies")
    print(f"  Network:  {network_results[1].count()} predictions | {sum(network_results[1])} detected anomalies")
    print(f"  Logs:     {logs_results[1].count()} predictions | {sum(logs_results[1])} detected anomalies")


if __name__ == "__main__":
    main()