"""
SentinelIQ — Train Autoencoder
Trains reconstruction-based anomaly detector on metrics.jsonl.
Run on Kaggle with GPU enabled for faster training.
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.autoencoder import SentinelAutoencoder


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def train(cfg: dict, data_path: str, save_dir: str):
    print("\n" + "="*60)
    print("  Training: Autoencoder on METRICS")
    print("="*60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies={df['is_anomaly'].sum()}")

    feature_cols = cfg["isolation_forest"]["features"]   # same feature set
    ae_cfg = {**cfg["autoencoder"], "input_dim": len(feature_cols)}
    y = df["is_anomaly"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    # Split train into train/val
    X_train_normal = X_train[y_train == 0]
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test,
        test_size=0.5,
        random_state=cfg["training"]["random_state"],
        stratify=y_test,
    )

    print(f"  Train normal: {len(X_train_normal)} | Val: {len(X_val)} | Test: {len(X_test_final)}")

    model = SentinelAutoencoder(config=ae_cfg, feature_cols=feature_cols)
    X_val_normal = X_val[y_val == 0]
    model.fit(X_train_normal, val_df=X_val_normal, threshold_df=X_val)

    print(f"\n  Evaluating on {len(X_test_final)} test samples...")
    metrics = model.evaluate(X_test_final, y_test_final)

    model.save(save_dir, name="autoencoder_metrics")

    results_path = f"{save_dir}/autoencoder_metrics_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved → {results_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data", default="data/simulated/metrics.jsonl")
    parser.add_argument("--save-dir", default="ml/saved_models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    metrics = train(cfg, args.data, args.save_dir)

    print("\n" + "="*60)
    print(f"  Autoencoder — ROC-AUC={metrics['roc_auc']} | F1={metrics['f1']}")
    print("="*60)