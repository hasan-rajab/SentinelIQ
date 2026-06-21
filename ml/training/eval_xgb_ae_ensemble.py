"""
SentinelIQ — XGBoost vs XGBoost+AE Ensemble Comparison
Quantifies how much recall the Autoencoder recovers on cases
that XGBoost alone misses, specifically on fresh unseen data.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.xgboost_network import SentinelXGBoost
from models.autoencoder import SentinelAutoencoder
from fusion.ensemble import SentinelEnsemble
from features.network_features import add_network_features


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def run(
    model_dir: str = "ml/saved_models",
    fresh_data: str = "data/simulated/network_fresh.jsonl",
):
    print("\n" + "=" * 60)
    print("  XGBoost vs XGBoost+AE Ensemble — Recovery Analysis")
    print("=" * 60)

    xgb = SentinelXGBoost.load(model_dir, name="xgboost_network")
    ae  = SentinelAutoencoder.load(model_dir, name="autoencoder_network")
    ensemble = SentinelEnsemble()

    df = load_jsonl(fresh_data)
    df = add_network_features(df)
    y_true = df["is_anomaly"].astype(int).values

    # --- XGBoost alone ---
    xgb_scores = xgb.score(df)
    xgb_pred   = (xgb_scores >= xgb.threshold).astype(int)

    # --- AE alone ---
    ae_scores = ae.score(df)
    ae_pred   = (ae_scores >= ae.threshold).astype(int)

    # --- Fused ---
    fused = ensemble.fuse_network_xgb(
        xgb_scores=xgb_scores,
        ae_scores=ae_scores,
        ae_threshold=ae.threshold,
    )
    # Use 0.5 as fusion threshold (XGBoost already dominant at 0.7 weight)
    fused_pred = (fused >= 0.5).astype(int)

    def summary(name, pred):
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        recall    = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        print(f"\n  {name}")
        print(f"    Recall   : {recall:.2%}  ({tp} caught, {fn} missed)")
        print(f"    Precision: {precision:.2%}  ({fp} false positives)")
        print(f"    F1       : {f1:.4f}")
        return set(np.where((y_true == 1) & (pred == 0))[0])  # missed indices

    missed_xgb    = summary("XGBoost alone", xgb_pred)
    missed_ae     = summary("AE alone", ae_pred)
    missed_fused  = summary("XGBoost + AE fused (0.7/0.3)", fused_pred)

    # Which of XGBoost's misses did the AE recover?
    recovered = missed_xgb - missed_fused
    still_missed = missed_xgb & missed_fused

    print(f"\n  --- Recovery breakdown ---")
    print(f"  XGBoost missed       : {len(missed_xgb)}")
    print(f"  AE recovered         : {len(recovered)}")
    print(f"  Still missed (both)  : {len(still_missed)}")

    if len(still_missed) > 0 and "anomaly_type" in df.columns:
        print(f"\n  Still-missed attack types:")
        for idx in still_missed:
            atype = df.iloc[idx].get("anomaly_type", "unknown")
            xgb_s = xgb_scores[idx]
            ae_s  = ae_scores[idx]
            print(f"    idx={idx}  type={atype:25}  xgb={xgb_s:.4f}  ae={ae_s:.4f}")

    if len(recovered) > 0 and "anomaly_type" in df.columns:
        print(f"\n  AE-recovered attack types:")
        for idx in recovered:
            atype = df.iloc[idx].get("anomaly_type", "unknown")
            xgb_s = xgb_scores[idx]
            ae_s  = ae_scores[idx]
            fused_s = fused[idx]
            print(f"    idx={idx}  type={atype:25}  xgb={xgb_s:.4f}  ae={ae_s:.4f}  fused={fused_s:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="ml/saved_models")
    parser.add_argument("--fresh-data", default="data/simulated/network_fresh.jsonl")
    args = parser.parse_args()
    run(args.model_dir, args.fresh_data)
