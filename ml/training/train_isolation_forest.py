"""
SentinelIQ — Train Isolation Forest
Trains on metrics.jsonl and network.jsonl, evaluates, saves weights.
Run this on Kaggle with GPU disabled (sklearn doesn't need GPU).
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.isolation_forest import SentinelIsolationForest


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def train_metrics(cfg: dict, data_path: str, save_dir: str):
    print("\n" + "="*60)
    print("  Training: Isolation Forest on METRICS")
    print("="*60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies={df['is_anomaly'].sum()}")

    feature_cols = cfg["isolation_forest"]["features"]
    y = df["is_anomaly"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    # Train only on normal samples (unsupervised setting)
    X_train_normal = X_train[y_train == 0]
    print(f"  Training on {len(X_train_normal)} normal samples (unsupervised)")

    model = SentinelIsolationForest(
        config=cfg["isolation_forest"],
        feature_cols=feature_cols,
    )
    model.fit(X_train_normal)

    print(f"\n  Evaluating on {len(X_test)} test samples...")
    metrics = model.evaluate(X_test, y_test)

    model.save(save_dir, name="isolation_forest_metrics")

    results_path = f"{save_dir}/isolation_forest_metrics_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved → {results_path}")
    return metrics


def train_network(cfg: dict, data_path: str, save_dir: str):
    print("\n" + "="*60)
    print("  Training: Isolation Forest on NETWORK FLOWS")
    print("="*60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies={df['is_anomaly'].sum()}")

    feature_cols = cfg["network"]["features"]
    y = df["is_anomaly"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    X_train_normal = X_train[y_train == 0]
    print(f"  Training on {len(X_train_normal)} normal samples (unsupervised)")

    network_cfg = {**cfg["isolation_forest"], **cfg["network"]}
    model = SentinelIsolationForest(
        config=network_cfg,
        feature_cols=feature_cols,
    )
    model.fit(X_train_normal)

    print(f"\n  Evaluating on {len(X_test)} test samples...")
    metrics = model.evaluate(X_test, y_test)

    model.save(save_dir, name="isolation_forest_network")

    results_path = f"{save_dir}/isolation_forest_network_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved → {results_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--metrics-data", default="data/simulated/metrics.jsonl")
    parser.add_argument("--network-data", default="data/simulated/network.jsonl")
    parser.add_argument("--save-dir", default="ml/saved_models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    m1 = train_metrics(cfg, args.metrics_data, args.save_dir)
    m2 = train_network(cfg, args.network_data, args.save_dir)

    print("\n" + "="*60)
    print("  Final Summary")
    print("="*60)
    print(f"  Metrics  IF — ROC-AUC={m1['roc_auc']} | F1={m1['f1']}")
    print(f"  Network  IF — ROC-AUC={m2['roc_auc']} | F1={m2['f1']}")