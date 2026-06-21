"""
SentinelIQ — Train XGBoost Network Classifier
Supervised anomaly detection for network flows using labeled data.
Replaces Isolation Forest in the network ensemble.
Run locally or on Kaggle.
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.xgboost_network import SentinelXGBoost
from features.network_features import add_network_features, ENGINEERED_FEATURES


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def train(cfg: dict, data_path: str, save_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("  Training: XGBoost on NETWORK FLOWS")
    print("=" * 60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies={df['is_anomaly'].sum()}")

    df = add_network_features(df)
    print(f"  Engineered features: {len(ENGINEERED_FEATURES)} cols")

    feature_cols = cfg["network"]["features"]
    y = df["is_anomaly"].astype(int).values

    # Train / val / test split — stratified to preserve anomaly ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test,
        test_size=0.5,
        random_state=cfg["training"]["random_state"],
        stratify=y_test,
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test_final)}")
    print(f"  Train anomalies: {y_train.sum()} | Val: {y_val.sum()} | Test: {y_test_final.sum()}")

    # scale_pos_weight balances the class imbalance automatically
    # set to ratio of normal:anomaly in training set
    n_normal = int((y_train == 0).sum())
    n_anomaly = int(y_train.sum())
    scale_pos_weight = round(n_normal / max(n_anomaly, 1), 2)
    print(f"  scale_pos_weight: {scale_pos_weight} ({n_normal} normal / {n_anomaly} anomaly)")

    xgb_cfg = {
        **cfg.get("xgboost_network", {}),
        "scale_pos_weight": scale_pos_weight,
        "random_state": cfg["training"]["random_state"],
    }

    model = SentinelXGBoost(config=xgb_cfg, feature_cols=feature_cols)
    model.fit(X_train, y_train, val_df=X_val, y_val=y_val)

    print(f"\n  Evaluating on {len(X_test_final)} test samples...")
    metrics = model.evaluate(X_test_final, y_test_final)

    # Print feature importances top 10
    print("\n  Top 10 Feature Importances:")
    fi = model.feature_importance().head(10)
    for _, row in fi.iterrows():
        print(f"    {row['feature']:30} {row['importance']:.4f}")

    model.save(save_dir, name="xgboost_network")

    results_path = f"{save_dir}/xgboost_network_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data", default="data/simulated/network.jsonl")
    parser.add_argument("--save-dir", default="ml/saved_models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    metrics = train(cfg, args.data, args.save_dir)

    print("\n" + "=" * 60)
    print(f"  XGBoost Network — ROC-AUC={metrics['roc_auc']} | F1={metrics['f1']}")
    print("=" * 60)