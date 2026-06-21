"""
SentinelIQ — Per-Modality Evaluation
Scores fresh data through each modality independently to identify
which one is underperforming before touching ensemble weights.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.isolation_forest import SentinelIsolationForest
from models.autoencoder import SentinelAutoencoder
from models.xgboost_network import SentinelXGBoost
from fusion.ensemble import SentinelEnsemble
from features.network_features import add_network_features


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def evaluate(name, y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    print(f"\n  {name}")
    print(f"    Recall   : {recall:.2%}  ({tp} caught, {fn} missed)")
    print(f"    Precision: {precision:.2%}  ({fp} false positives)")
    print(f"    F1       : {f1:.4f}")
    print(f"    ROC-AUC  : {roc:.4f}")
    print(f"    Threshold: {threshold:.4f}")


def main():
    model_dir  = "ml/saved_models"
    fresh_dir  = "data/simulated_fresh"
    # fall back to network_fresh if simulated_fresh doesn't exist
    net_path   = "data/simulated_fresh/network.jsonl"
    if not Path(net_path).exists():
        net_path = "data/simulated/network_fresh.jsonl"
    met_path   = "data/simulated_fresh/metrics.jsonl"
    if not Path(met_path).exists():
        met_path = "data/simulated/metrics.jsonl"

    print("\n" + "=" * 60)
    print("  SentinelIQ — Per-Modality Evaluation (fresh data)")
    print("=" * 60)

    # ── METRICS: IF ──────────────────────────────────────────────
    print("\n── METRICS ─────────────────────────────────────────────")
    df_met = load_jsonl(met_path)
    y_met  = df_met["is_anomaly"].astype(int).values

    if_met = SentinelIsolationForest.load(model_dir, name="isolation_forest_metrics")
    evaluate("Isolation Forest", y_met, if_met.score(df_met), if_met.threshold)
    del if_met

    ae_met = SentinelAutoencoder.load(model_dir, name="autoencoder_metrics")
    evaluate("Autoencoder", y_met, ae_met.score(df_met), ae_met.threshold)
    del ae_met

    # ── NETWORK: XGBoost + AE ────────────────────────────────────
    print("\n── NETWORK ─────────────────────────────────────────────")
    df_net = load_jsonl(net_path)
    df_net = add_network_features(df_net)
    y_net  = df_net["is_anomaly"].astype(int).values

    xgb = SentinelXGBoost.load(model_dir, name="xgboost_network")
    evaluate("XGBoost", y_net, xgb.score(df_net), xgb.threshold)

    ae_net = SentinelAutoencoder.load(model_dir, name="autoencoder_network")
    evaluate("Autoencoder", y_net, ae_net.score(df_net), ae_net.threshold)

    ensemble = SentinelEnsemble.load(f"{model_dir}/ensemble_config.json")
    fused_net = ensemble.fuse_network_xgb(
        xgb_scores=xgb.score(df_net),
        ae_scores=ae_net.score(df_net),
        ae_threshold=ae_net.threshold,
    )
    evaluate("XGBoost + AE fused", y_net, fused_net, ensemble.network_threshold)
    del xgb, ae_net

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
