"""
Calibrate ensemble thresholds on validation data using PR-curve F1 maximization.
Runs each modality in a separate subprocess to avoid memory pressure segfaults.
"""

import json
import sys
import tempfile
import subprocess
from pathlib import Path


def run_calibration(script: str) -> float:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    result = subprocess.run(
        [sys.executable, tmp_path],
        cwd="/Users/hassanali/Desktop/SentinelIQ",
        env={**__import__('os').environ, "PYTHONPATH": "."},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Calibration subprocess failed (exit {result.returncode})")

    # Threshold was written to a tmp json by the script
    out_path = tmp_path + ".result.json"
    with open(out_path) as f:
        return json.load(f)["threshold"]


METRICS_SCRIPT = """
import sys
sys.path.insert(0, '.')
import json, yaml, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from ml.models.isolation_forest import SentinelIsolationForest
from ml.models.autoencoder import SentinelAutoencoder
from ml.fusion.ensemble import SentinelEnsemble

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)

def pr_curve_threshold(y_true, scores, min_recall=0.0):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    valid = recall[:-1] >= min_recall
    best_idx = int(np.argmax(np.where(valid, f1, 0.0))) if valid.any() else int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx]), float(recall[best_idx])

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

if_model = SentinelIsolationForest.load('ml/saved_models', name='isolation_forest_metrics')
ae_model = SentinelAutoencoder.load('ml/saved_models', name='autoencoder_metrics')
ensemble = SentinelEnsemble()

df = load_jsonl('data/simulated/metrics.jsonl')
y  = df['is_anomaly'].astype(int).values
_, X_val, _, y_val = train_test_split(df, y,
    test_size=cfg['training']['test_size'],
    random_state=cfg['training']['random_state'],
    stratify=y)

fused = ensemble.fuse(if_scores=if_model.score(X_val), ae_scores=ae_model.score(X_val))
t, f1, recall = pr_curve_threshold(y_val, fused)
print(f'[Metrics]  threshold={t:.4f} | F1={f1:.4f} | Recall={recall:.4f}')

import __main__
out = __file__ + '.result.json'
with open(out, 'w') as f:
    json.dump({'threshold': t}, f)
"""

NETWORK_SCRIPT = """
import sys
sys.path.insert(0, '.')
import json, yaml, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from ml.models.xgboost_network import SentinelXGBoost
from ml.models.autoencoder import SentinelAutoencoder
from ml.fusion.ensemble import SentinelEnsemble
from ml.features.network_features import add_network_features

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)

def pr_curve_threshold(y_true, scores, min_recall=0.0):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    valid = recall[:-1] >= min_recall
    best_idx = int(np.argmax(np.where(valid, f1, 0.0))) if valid.any() else int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx]), float(recall[best_idx])

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

xgb_model = SentinelXGBoost.load('ml/saved_models', name='xgboost_network')
ae_model  = SentinelAutoencoder.load('ml/saved_models', name='autoencoder_network')
ensemble  = SentinelEnsemble()

df = load_jsonl('data/simulated/network.jsonl')
df = add_network_features(df)
y  = df['is_anomaly'].astype(int).values
_, X_val, _, y_val = train_test_split(df, y,
    test_size=cfg['training']['test_size'],
    random_state=cfg['training']['random_state'],
    stratify=y)

xgb_scores = xgb_model.score(X_val)
ae_scores  = ae_model.score(X_val)
fused = ensemble.fuse_network_xgb(
    xgb_scores=xgb_scores,
    ae_scores=ae_scores,
    ae_threshold=ae_model.threshold,
)
t, f1, recall = pr_curve_threshold(y_val, fused, min_recall=0.90)
print(f'[Network XGB+AE]  threshold={t:.4f} | F1={f1:.4f} | Recall={recall:.4f}')

out = __file__ + '.result.json'
with open(out, 'w') as f:
    json.dump({'threshold': t}, f)
"""

if __name__ == "__main__":
    model_dir   = "ml/saved_models"
    config_path = f"{model_dir}/ensemble_config.json"

    print("[Calibrating metrics threshold...]")
    metrics_threshold = run_calibration(METRICS_SCRIPT)

    print("\n[Calibrating network XGBoost+AE threshold...]")
    network_threshold = run_calibration(NETWORK_SCRIPT)

    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {
            "weights": {"isolation_forest": 0.30, "autoencoder": 0.30, "bert_log": 0.40},
            "strategy": "weighted_avg",
        }

    config["threshold"]         = metrics_threshold
    config["network_threshold"] = network_threshold

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved → {config_path}")
    print(f"  metrics threshold : {metrics_threshold:.4f}")
    print(f"  network threshold : {network_threshold:.4f}")
