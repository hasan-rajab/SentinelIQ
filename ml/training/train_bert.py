"""
SentinelIQ — Train BERT Log Classifier
Fine-tunes bert-base-uncased on log lines for binary anomaly classification.
Run on Kaggle with GPU enabled — BERT needs it.
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.bert_log import SentinelBertLog


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def train(cfg: dict, data_path: str, save_dir: str):
    print("\n" + "="*60)
    print("  Training: BERT on LOG SEQUENCES")
    print("="*60)

    df = load_jsonl(data_path)
    print(f"  Loaded {len(df)} records | anomalies={df['is_anomaly'].sum()}")

    y = df["is_anomaly"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=cfg["training"]["val_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y_train,
    )

    print(f"  Train: {len(X_train_split)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Train anomaly rate: {y_train_split.mean():.2%}")

    model = SentinelBertLog(config=cfg["bert_log"])
    model.fit(X_train_split, val_df=X_val)

    print(f"\n  Evaluating on {len(X_test)} test samples...")
    metrics = model.evaluate(X_test, y_test)

    model.save(save_dir, name="bert_log")

    results_path = f"{save_dir}/bert_log_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved → {results_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data", default="data/simulated/logs.jsonl")
    parser.add_argument("--save-dir", default="ml/saved_models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    metrics = train(cfg, args.data, args.save_dir)

    print("\n" + "="*60)
    print(f"  BERT Log — ROC-AUC={metrics['roc_auc']} | F1={metrics['f1']}")
    print("="*60)