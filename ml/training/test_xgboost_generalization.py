"""
SentinelIQ — XGBoost Fresh-Data Generalization Test
Validates the trained XGBoost network classifier against entirely fresh,
never-before-seen synthetic data (different random seed / new generation run)
to check whether the 1.0 F1 on the original test split was a real result
or an artifact of a small, easily-separable test set.

Run this AFTER training xgboost_network via train_xgboost_network.py.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.xgboost_network import SentinelXGBoost
from features.network_features import add_network_features


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def run_generalization_test(
    model_dir: str = "ml/saved_models",
    fresh_data_path: str = "data/simulated/network_fresh.jsonl",
):
    print("\n" + "=" * 60)
    print("  XGBoost — Fresh Data Generalization Test")
    print("=" * 60)

    model = SentinelXGBoost.load(model_dir, name="xgboost_network")

    df = load_jsonl(fresh_data_path)
    print(f"  Loaded {len(df)} FRESH records (never seen during training)")
    print(f"  Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.1%})")

    df = add_network_features(df)
    y_true = df["is_anomaly"].astype(int).values

    scores = model.score(df)
    y_pred = (scores >= model.threshold).astype(int)

    print(f"\n  Score distribution on fresh data:")
    print(f"    min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")
    print(f"    threshold={model.threshold:.4f}")

    print("\n  Classification Report (FRESH, unseen data):")
    print(classification_report(y_true, y_pred, zero_division=0))

    roc_auc = roc_auc_score(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)
    missed = int(((y_true == 1) & (y_pred == 0)).sum())
    total_anomalies = int(y_true.sum())

    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Avg Precision: {avg_precision:.4f}")
    print(f"  Missed anomalies: {missed} / {total_anomalies}")

    # Per-anomaly-type breakdown — which attack types does it actually miss?
    if "anomaly_type" in df.columns:
        print("\n  Per-attack-type recall:")
        anomaly_df = df[y_true == 1].copy()
        anomaly_df["predicted"] = y_pred[y_true == 1]
        for atype, group in anomaly_df.groupby("anomaly_type"):
            recall = group["predicted"].mean()
            print(f"    {atype:25} recall={recall:.2%}  (n={len(group)})")

    print("\n" + "=" * 60)
    if roc_auc >= 0.95 and missed / max(total_anomalies, 1) < 0.15:
        print("  VERDICT: Result holds on fresh data — not an overfitting artifact")
    elif roc_auc >= 0.85:
        print("  VERDICT: Reasonable generalization, but below original test set —")
        print("           original 1.0 F1 was likely inflated by test set size/composition")
    else:
        print("  VERDICT: Significant gap from original test set —")
        print("           original 1.0 F1 does NOT reliably hold on unseen data")
    print("=" * 60)

    return {
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_precision, 4),
        "missed": missed,
        "total_anomalies": total_anomalies,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="ml/saved_models")
    parser.add_argument("--fresh-data", default="data/simulated/network_fresh.jsonl")
    args = parser.parse_args()

    run_generalization_test(args.model_dir, args.fresh_data)