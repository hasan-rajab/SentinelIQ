"""
SentinelIQ — UNSW-NB15 Real Traffic Evaluation
Tests how well the synthetic-trained XGBoost generalizes to real network data.

Phase 1: Feature mapping + inference with synthetic-trained model
Phase 2: Retrain XGBoost on UNSW-NB15, compare performance
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.models.xgboost_network import SentinelXGBoost


UNSW_PATH = Path.home() / "Downloads/UNSW-NB15_c"
MODEL_DIR  = "ml/saved_models"

# Features your model was trained on, minus the two port flags
MAPPABLE_FEATURES = [
    "bytes_out",
    "bytes_in",
    "packets",
    "duration_ms",
    "bytes_out_in_ratio",
    "bytes_per_sec",
    "bytes_per_packet",
    "byte_asymmetry",
    "packets_per_sec",
    "payload_density",
    "is_zero_bytes",
]

# Pad missing port features with zeros
ALL_FEATURES = [
    "bytes_out", "bytes_in", "packets", "duration_ms",
    "src_port", "dst_port",
    "bytes_out_in_ratio", "bytes_per_sec", "bytes_per_packet",
    "is_common_port",
    "is_suspicious_port", "byte_asymmetry", "packets_per_sec",
    "payload_density", "is_zero_bytes",
]


def map_unsw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map UNSW-NB15 columns to SentinelIQ's 15 engineered network features."""
    out = pd.DataFrame(index=df.index)

    sbytes = df["sbytes"].fillna(0).clip(lower=0)
    dbytes = df["dbytes"].fillna(0).clip(lower=0)
    spkts  = df["spkts"].fillna(0).clip(lower=0)
    dpkts  = df["dpkts"].fillna(0).clip(lower=0)
    dur    = df["dur"].fillna(0).clip(lower=0)

    total_bytes   = sbytes + dbytes
    total_packets = spkts + dpkts
    dur_sec       = dur

    out["bytes_out"]         = sbytes
    out["bytes_in"]          = dbytes
    out["packets"]           = total_packets
    out["duration_ms"]       = dur_sec * 1000

    # Port columns not in UNSW-NB15 — zero-filled
    out["src_port"]          = 0
    out["dst_port"]          = 0
    out["is_common_port"]    = 0
    out["is_suspicious_port"] = 0

    out["bytes_out_in_ratio"] = sbytes / (dbytes + 1)
    out["bytes_per_sec"]      = (total_bytes / (dur_sec + 0.01)).clip(upper=1_000_000)
    out["bytes_per_packet"]   = total_bytes / (total_packets + 1)
    out["byte_asymmetry"]     = (sbytes - dbytes).abs() / (total_bytes + 1)
    out["packets_per_sec"]    = (total_packets / (dur_sec + 0.01)).clip(upper=10_000)
    out["payload_density"]    = (df["smean"].fillna(0) / (dur_sec + 0.01)).clip(upper=100_000)
    out["is_zero_bytes"]      = (total_bytes == 0).astype(int)

    return out[ALL_FEATURES]


def evaluate(name, y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    recall    = tp / max(tp+fn, 1)
    precision = tp / max(tp+fp, 1)
    f1 = 2*precision*recall / max(precision+recall, 1e-8)
    try:
        roc = roc_auc_score(y_true, scores)
    except Exception:
        roc = 0.0
    print(f"\n  {name}")
    print(f"    Recall   : {recall:.2%}  ({tp} caught, {fn} missed)")
    print(f"    Precision: {precision:.2%}  ({fp} false positives)")
    print(f"    F1       : {f1:.4f}")
    print(f"    ROC-AUC  : {roc:.4f}")
    print(f"    Threshold: {threshold:.4f}")
    return {"recall": recall, "precision": precision, "f1": f1, "roc_auc": roc}


def phase1_generalization(test_df, y_test):
    """Run synthetic-trained model on UNSW-NB15 test set."""
    print("\n" + "="*60)
    print("  PHASE 1 — Synthetic-trained model on real traffic")
    print("="*60)

    model = SentinelXGBoost.load(MODEL_DIR, name="xgboost_network")
    X_test = map_unsw_features(test_df)

    scores = model.score(X_test)
    results = evaluate("XGBoost (synthetic-trained)", y_test, scores, model.threshold)

    print(f"\n  Per-attack-type recall:")
    for cat in test_df["attack_cat"].unique():
        if cat == "Normal":
            continue
        mask = (test_df["attack_cat"] == cat).values
        if mask.sum() == 0:
            continue
        cat_scores = scores[mask]
        cat_pred   = (cat_scores >= model.threshold).astype(int)
        recall = cat_pred.mean()
        print(f"    {cat:20} recall={recall:.2%}  (n={mask.sum()})")

    return results


def phase2_retrain(train_df, y_train, test_df, y_test):
    """Retrain XGBoost on UNSW-NB15 training set, evaluate on test set."""
    print("\n" + "="*60)
    print("  PHASE 2 — Retrained on real UNSW-NB15 traffic")
    print("="*60)

    from sklearn.model_selection import train_test_split

    X_train = map_unsw_features(train_df)
    X_test  = map_unsw_features(test_df)

    # Use a val split from training data for threshold calibration
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=42, stratify=y_train
    )

    n_normal  = int((y_tr == 0).sum())
    n_anomaly = int((y_tr == 1).sum())
    scale_pos_weight = n_normal / max(n_anomaly, 1)

    cfg = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
    }

    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Train anomalies: {y_tr.sum()} | normal: {(y_tr==0).sum()}")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    model = SentinelXGBoost(config=cfg, feature_cols=ALL_FEATURES)
    model.fit(X_tr, y_tr, val_df=X_val, y_val=y_val)

    scores  = model.score(X_test)
    results = evaluate("XGBoost (retrained on UNSW-NB15)", y_test, scores, model.threshold)

    print(f"\n  Per-attack-type recall:")
    for cat in test_df["attack_cat"].unique():
        if cat == "Normal":
            continue
        mask = (test_df["attack_cat"] == cat).values
        if mask.sum() == 0:
            continue
        cat_scores = scores[mask]
        cat_pred   = (cat_scores >= model.threshold).astype(int)
        recall = cat_pred.mean()
        print(f"    {cat:20} recall={recall:.2%}  (n={mask.sum()})")

    # Save retrained model
    model.save(MODEL_DIR, name="xgboost_unsw")
    print(f"\n  Retrained model saved as xgboost_unsw")
    return results


def main():
    print("\n" + "="*60)
    print("  SentinelIQ — UNSW-NB15 Real Traffic Evaluation")
    print("="*60)

    print("\n  Loading UNSW-NB15 datasets...")
    train_df = pd.read_csv(UNSW_PATH / "UNSW_NB15_training-set.csv")
    test_df  = pd.read_csv(UNSW_PATH / "UNSW_NB15_testing-set.csv")

    print(f"  Train: {len(train_df)} records | anomalies: {train_df['label'].sum()} ({train_df['label'].mean():.1%})")
    print(f"  Test:  {len(test_df)} records  | anomalies: {test_df['label'].sum()} ({test_df['label'].mean():.1%})")

    y_train = train_df["label"].astype(int).values
    y_test  = test_df["label"].astype(int).values

    r1 = phase1_generalization(test_df, y_test)
    r2 = phase2_retrain(train_df, y_train, test_df, y_test)

    print("\n" + "="*60)
    print("  SUMMARY — Synthetic vs Real-World Performance")
    print("="*60)
    print(f"  {'Model':<35} {'Recall':>8} {'Precision':>10} {'F1':>8} {'ROC-AUC':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*9}")
    print(f"  {'XGBoost (synthetic-trained)':<35} {r1['recall']:>8.2%} {r1['precision']:>10.2%} {r1['f1']:>8.4f} {r1['roc_auc']:>9.4f}")
    print(f"  {'XGBoost (retrained UNSW-NB15)':<35} {r2['recall']:>8.2%} {r2['precision']:>10.2%} {r2['f1']:>8.4f} {r2['roc_auc']:>9.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
