import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent


def nb(*cells):
    n = nbf.v4.new_notebook()
    n.cells = list(cells)
    return n

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


# ── 01 Data Exploration ───────────────────────────────────────────────────────
nb01 = nb(
    md("# SentinelIQ — 01 Data Exploration\nExplore all three simulated data streams: logs, metrics, and network flows."),

    code("""\
# Setup
!git clone https://github.com/hasan-rajab/SentinelIQ.git 2>/dev/null || echo "Already cloned"
%cd /kaggle/working/SentinelIQ
import sys
sys.path.insert(0, '/kaggle/working/SentinelIQ')
"""),

    code("""\
# Generate datasets
!python data/simulated/pipeline.py --duration 120 --anomaly-rate 0.08
"""),

    code("""\
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])

logs_df    = load_jsonl('data/simulated/logs.jsonl')
metrics_df = load_jsonl('data/simulated/metrics.jsonl')
network_df = load_jsonl('data/simulated/network.jsonl')

print(f"Logs    : {len(logs_df):,} records | {logs_df['is_anomaly'].sum()} anomalies")
print(f"Metrics : {len(metrics_df):,} records | {metrics_df['is_anomaly'].sum()} anomalies")
print(f"Network : {len(network_df):,} records | {network_df['is_anomaly'].sum()} anomalies")
"""),

    md("## Logs Dataset"),

    code("""\
logs_df.head(5)
"""),

    code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Anomaly distribution
logs_df['is_anomaly'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71','#e74c3c'])
axes[0].set_title('Log Anomaly Distribution')
axes[0].set_xticklabels(['Normal', 'Anomaly'], rotation=0)

# Anomaly types
anomaly_types = logs_df[logs_df['is_anomaly']]['anomaly_type'].value_counts()
anomaly_types.plot(kind='barh', ax=axes[1], color='#e74c3c')
axes[1].set_title('Log Anomaly Types')

plt.tight_layout()
plt.show()
"""),

    code("""\
# Log type breakdown
print("Log type distribution:")
print(logs_df['log_type'].value_counts())
print()
print("Sample anomalous log messages:")
for _, row in logs_df[logs_df['is_anomaly']].head(5).iterrows():
    print(f"  [{row['anomaly_type']:25}] {row['message'][:80]}")
"""),

    md("## Metrics Dataset"),

    code("""\
metrics_df.head(5)
"""),

    code("""\
feature_cols = ['cpu_percent','mem_percent','disk_read_mbps',
                'disk_write_mbps','net_in_mbps','net_out_mbps',
                'open_connections','process_count']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    normal = metrics_df[~metrics_df['is_anomaly']][col]
    anomaly = metrics_df[metrics_df['is_anomaly']][col]
    axes[i].hist(normal, bins=30, alpha=0.6, color='#2ecc71', label='Normal')
    axes[i].hist(anomaly, bins=30, alpha=0.6, color='#e74c3c', label='Anomaly')
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

plt.suptitle('Metric Feature Distributions: Normal vs Anomaly', fontsize=14)
plt.tight_layout()
plt.show()
"""),

    code("""\
print("Metrics anomaly types:")
print(metrics_df[metrics_df['is_anomaly']]['anomaly_type'].value_counts())
print()
print("Feature statistics (normal vs anomaly):")
print(metrics_df.groupby('is_anomaly')[feature_cols].mean().T.rename(columns={False:'Normal Mean', True:'Anomaly Mean'}))
"""),

    md("## Network Dataset"),

    code("""\
network_df.head(5)
"""),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Anomaly distribution
network_df['is_anomaly'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71','#e74c3c'])
axes[0].set_title('Network Anomaly Distribution')
axes[0].set_xticklabels(['Normal','Anomaly'], rotation=0)

# Anomaly types
network_df[network_df['is_anomaly']]['anomaly_type'].value_counts().plot(
    kind='barh', ax=axes[1], color='#e74c3c')
axes[1].set_title('Network Anomaly Types')

# Protocol distribution
network_df['protocol'].value_counts().plot(kind='pie', ax=axes[2], autopct='%1.1f%%')
axes[2].set_title('Protocol Distribution')

plt.tight_layout()
plt.show()
"""),

    code("""\
net_features = ['bytes_out','bytes_in','packets','duration_ms']
print("Network feature stats (normal vs anomaly):")
print(network_df.groupby('is_anomaly')[net_features].mean().T.rename(
    columns={False:'Normal Mean', True:'Anomaly Mean'}))

print()
print("Top destination ports:")
print(network_df['dst_port'].value_counts().head(10))
"""),

    md("## Correlation Analysis (Metrics)"),

    code("""\
corr = metrics_df[feature_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Metric Feature Correlation Matrix')
plt.tight_layout()
plt.show()
"""),

    md("## Summary\n- All three data streams are generating correctly\n- Anomaly rates are ~8% as configured\n- Feature distributions show clear separation between normal and anomaly classes\n- Ready for Phase 2 model training"),
)


# ── 02 Isolation Forest ───────────────────────────────────────────────────────
nb02 = nb(
    md("# SentinelIQ — 02 Isolation Forest\nTrain and evaluate Isolation Forest on metrics and network flow data."),

    code("""\
!git clone https://github.com/hasan-rajab/SentinelIQ.git 2>/dev/null || echo "Already cloned"
%cd /kaggle/working/SentinelIQ
import sys
sys.path.insert(0, '/kaggle/working/SentinelIQ')
!pip install pyyaml joblib scikit-learn -q
"""),

    code("""\
!python data/simulated/pipeline.py --duration 120 --anomaly-rate 0.08
"""),

    code("""\
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from ml.models.isolation_forest import SentinelIsolationForest

sns.set_theme(style="darkgrid")

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])
"""),

    md("## Train on Metrics"),

    code("""\
metrics_df = load_jsonl('data/simulated/metrics.jsonl')
feature_cols = cfg['isolation_forest']['features']
y = metrics_df['is_anomaly'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    metrics_df, y, test_size=0.2, random_state=42, stratify=y)

X_train_normal = X_train[y_train == 0]
print(f"Training on {len(X_train_normal)} normal samples")
print(f"Test set: {len(X_test)} samples | {y_test.sum()} anomalies")
"""),

    code("""\
if_metrics = SentinelIsolationForest(config=cfg['isolation_forest'], feature_cols=feature_cols)
if_metrics.fit(X_train_normal)
"""),

    code("""\
metrics_results = if_metrics.evaluate(X_test, y_test)
"""),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
scores = if_metrics.score(X_test)
RocCurveDisplay.from_predictions(y_test, scores, ax=axes[0], name='Isolation Forest')
axes[0].set_title(f"ROC Curve (AUC={metrics_results['roc_auc']})")

# Precision-Recall
PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[1], name='Isolation Forest')
axes[1].set_title(f"Precision-Recall (AP={metrics_results['avg_precision']})")

# Score distributions
normal_scores = scores[y_test == 0]
anomaly_scores = scores[y_test == 1]
axes[2].hist(normal_scores, bins=30, alpha=0.6, color='#2ecc71', label='Normal')
axes[2].hist(anomaly_scores, bins=30, alpha=0.6, color='#e74c3c', label='Anomaly')
axes[2].axvline(if_metrics.threshold, color='navy', linestyle='--', label=f'Threshold={if_metrics.threshold:.3f}')
axes[2].set_title('Anomaly Score Distribution')
axes[2].legend()

plt.suptitle('Isolation Forest — Metrics', fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md("## Train on Network Flows"),

    code("""\
network_df = load_jsonl('data/simulated/network.jsonl')
net_features = cfg['network']['features']
y_net = network_df['is_anomaly'].astype(int).values

X_train_net, X_test_net, y_train_net, y_test_net = train_test_split(
    network_df, y_net, test_size=0.2, random_state=42, stratify=y_net)

X_train_net_normal = X_train_net[y_train_net == 0]

network_cfg = {**cfg['isolation_forest'], **cfg['network']}
if_network = SentinelIsolationForest(config=network_cfg, feature_cols=net_features)
if_network.fit(X_train_net_normal)
network_results = if_network.evaluate(X_test_net, y_test_net)
"""),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

net_scores = if_network.score(X_test_net)
RocCurveDisplay.from_predictions(y_test_net, net_scores, ax=axes[0], name='IF Network')
axes[0].set_title(f"ROC Curve (AUC={network_results['roc_auc']})")

PrecisionRecallDisplay.from_predictions(y_test_net, net_scores, ax=axes[1], name='IF Network')
axes[1].set_title(f"Precision-Recall (AP={network_results['avg_precision']})")

axes[2].hist(net_scores[y_test_net==0], bins=30, alpha=0.6, color='#2ecc71', label='Normal')
axes[2].hist(net_scores[y_test_net==1], bins=30, alpha=0.6, color='#e74c3c', label='Anomaly')
axes[2].axvline(if_network.threshold, color='navy', linestyle='--', label=f'Threshold')
axes[2].set_title('Score Distribution — Network')
axes[2].legend()

plt.tight_layout()
plt.show()
"""),

    md("## Save Models"),

    code("""\
import os
os.makedirs('ml/saved_models', exist_ok=True)
if_metrics.save('ml/saved_models', name='isolation_forest_metrics')
if_network.save('ml/saved_models', name='isolation_forest_network')
print("Models saved.")
"""),

    md("## Results Summary"),

    code("""\
summary = pd.DataFrame([
    {'Model': 'IF Metrics', **{k:v for k,v in metrics_results.items() if k != 'confusion_matrix'}},
    {'Model': 'IF Network', **{k:v for k,v in network_results.items() if k != 'confusion_matrix'}},
])
print(summary[['Model','roc_auc','avg_precision','precision','recall','f1','accuracy']].to_string(index=False))
"""),
)


# ── 03 Autoencoder ────────────────────────────────────────────────────────────
nb03 = nb(
    md("# SentinelIQ — 03 Autoencoder\nReconstruction-based anomaly detection on system metrics using PyTorch."),

    code("""\
!git clone https://github.com/hasan-rajab/SentinelIQ.git 2>/dev/null || echo "Already cloned"
%cd /kaggle/working/SentinelIQ
import sys
sys.path.insert(0, '/kaggle/working/SentinelIQ')
!pip install pyyaml torch scikit-learn -q
"""),

    code("""\
!python data/simulated/pipeline.py --duration 120 --anomaly-rate 0.08
"""),

    code("""\
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import torch

from ml.models.autoencoder import SentinelAutoencoder

sns.set_theme(style="darkgrid")
print(f"GPU available: {torch.cuda.is_available()}")

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])
"""),

    md("## Load & Split Data"),

    code("""\
metrics_df = load_jsonl('data/simulated/metrics.jsonl')
feature_cols = cfg['isolation_forest']['features']
ae_cfg = {**cfg['autoencoder'], 'input_dim': len(feature_cols)}

y = metrics_df['is_anomaly'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    metrics_df, y, test_size=0.2, random_state=42, stratify=y)

X_val, X_test_final, y_val, y_test_final = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

X_train_normal = X_train[y_train == 0]
print(f"Train (normal only): {len(X_train_normal)}")
print(f"Val  : {len(X_val)} | Test: {len(X_test_final)}")
"""),

    md("## Train Autoencoder"),

    code("""\
ae = SentinelAutoencoder(config=ae_cfg, feature_cols=feature_cols)
ae.fit(X_train_normal, val_df=X_val)
"""),

    md("## Evaluate"),

    code("""\
results = ae.evaluate(X_test_final, y_test_final)
"""),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scores = ae.score(X_test_final)
RocCurveDisplay.from_predictions(y_test_final, scores, ax=axes[0], name='Autoencoder')
axes[0].set_title(f"ROC Curve (AUC={results['roc_auc']})")

PrecisionRecallDisplay.from_predictions(y_test_final, scores, ax=axes[1], name='Autoencoder')
axes[1].set_title(f"Precision-Recall (AP={results['avg_precision']})")

axes[2].hist(scores[y_test_final==0], bins=30, alpha=0.6, color='#2ecc71', label='Normal')
axes[2].hist(scores[y_test_final==1], bins=30, alpha=0.6, color='#e74c3c', label='Anomaly')
axes[2].axvline(ae.threshold, color='navy', linestyle='--', label=f'Threshold={ae.threshold:.5f}')
axes[2].set_title('Reconstruction Error Distribution')
axes[2].legend()

plt.suptitle('Autoencoder — Metrics Anomaly Detection', fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md("## Latent Space Visualization"),

    code("""\
import torch
from sklearn.decomposition import PCA

ae.net.eval()
X_scaled = ae.scaler.transform(metrics_df[feature_cols].fillna(0).values)
X_tensor = torch.FloatTensor(X_scaled).to(ae.device)

with torch.no_grad():
    latent = ae.net.encode(X_tensor).cpu().numpy()

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent)

plt.figure(figsize=(8, 6))
colors = ['#e74c3c' if a else '#2ecc71' for a in metrics_df['is_anomaly']]
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, alpha=0.5, s=20)
plt.title('Autoencoder Latent Space (PCA 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color='#2ecc71', label='Normal'), Patch(color='#e74c3c', label='Anomaly')])
plt.tight_layout()
plt.show()
"""),

    md("## Save Model"),

    code("""\
import os
os.makedirs('ml/saved_models', exist_ok=True)
ae.save('ml/saved_models', name='autoencoder_metrics')
print("Autoencoder saved.")
"""),
)


# ── 04 BERT Finetuning ────────────────────────────────────────────────────────
nb04 = nb(
    md("# SentinelIQ — 04 BERT Log Classifier\nFine-tune BERT for binary log anomaly classification. Enable GPU accelerator."),

    code("""\
!git clone https://github.com/hasan-rajab/SentinelIQ.git 2>/dev/null || echo "Already cloned"
%cd /kaggle/working/SentinelIQ
import sys
sys.path.insert(0, '/kaggle/working/SentinelIQ')
!pip install transformers pyyaml torch scikit-learn -q
"""),

    code("""\
!python data/simulated/pipeline.py --duration 180 --anomaly-rate 0.08
"""),

    code("""\
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

from ml.models.bert_log import SentinelBertLog

sns.set_theme(style="darkgrid")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])
"""),

    md("## Load & Inspect Log Data"),

    code("""\
logs_df = load_jsonl('data/simulated/logs.jsonl')
print(f"Total records : {len(logs_df)}")
print(f"Anomalies     : {logs_df['is_anomaly'].sum()} ({logs_df['is_anomaly'].mean():.1%})")
print()
print("Sample messages:")
for _, row in logs_df.sample(5).iterrows():
    label = "ANOMALY" if row['is_anomaly'] else "normal "
    print(f"  [{label}] {row['message'][:90]}")
"""),

    code("""\
# Message length distribution
logs_df['msg_len'] = logs_df['message'].str.len()
plt.figure(figsize=(10, 4))
normal_len = logs_df[logs_df['is_anomaly'] == False]['msg_len']
anomaly_len = logs_df[logs_df['is_anomaly'] == True]['msg_len']
plt.hist(normal_len, bins=30, alpha=0.6, color='#2ecc71', label='Normal')
plt.hist(anomaly_len, bins=30, alpha=0.6, color='#e74c3c', label='Anomaly')
plt.title('Log Message Length Distribution')
plt.xlabel('Character length')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()
"""),

    md("## Split Data"),

    code("""\
y = logs_df['is_anomaly'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    logs_df, y, test_size=cfg['training']['test_size'],
    random_state=42, stratify=y)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=cfg['training']['val_size'],
    random_state=42, stratify=y_train)

print(f"Train : {len(X_train_final)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train anomaly rate: {y_train_final.mean():.2%}")
"""),

    md("## Fine-tune BERT\n> Make sure GPU is enabled: Settings → Accelerator → GPU T4 x2"),

    code("""\
bert = SentinelBertLog(config=cfg['bert_log'])
bert.fit(X_train_final, val_df=X_val)
"""),

    md("## Evaluate"),

    code("""\
results = bert.evaluate(X_test, y_test)
"""),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scores = bert.score(X_test)
RocCurveDisplay.from_predictions(y_test, scores, ax=axes[0], name='BERT')
axes[0].set_title(f"ROC Curve (AUC={results['roc_auc']})")

PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[1], name='BERT')
axes[1].set_title(f"Precision-Recall (AP={results['avg_precision']})")

ConfusionMatrixDisplay(confusion_matrix=np.array(results['confusion_matrix']),
    display_labels=['Normal','Anomaly']).plot(ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title('Confusion Matrix')

plt.suptitle('BERT Log Classifier', fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md("## Error Analysis"),

    code("""\
y_pred = bert.predict(X_test)
X_test_copy = X_test.copy()
X_test_copy['predicted'] = y_pred
X_test_copy['score'] = scores

# False negatives — missed anomalies
fn = X_test_copy[(X_test_copy['is_anomaly'] == True) & (X_test_copy['predicted'] == 0)]
print(f"False Negatives (missed anomalies): {len(fn)}")
for _, row in fn.head(3).iterrows():
    print(f"  score={row['score']:.3f} | {row['message'][:80]}")

print()

# False positives — wrong flags
fp = X_test_copy[(X_test_copy['is_anomaly'] == False) & (X_test_copy['predicted'] == 1)]
print(f"False Positives (wrong flags): {len(fp)}")
for _, row in fp.head(3).iterrows():
    print(f"  score={row['score']:.3f} | {row['message'][:80]}")
"""),

    md("## Save Model"),

    code("""\
import os
os.makedirs('ml/saved_models', exist_ok=True)
bert.save('ml/saved_models', name='bert_log')
print("BERT model saved.")
"""),
)


# ── 05 Fusion Ensemble ────────────────────────────────────────────────────────
nb05 = nb(
    md("# SentinelIQ — 05 Fusion Ensemble\nCombine IsolationForest, Autoencoder, and BERT scores into a unified anomaly score."),

    code("""\
import os, sys
repo = '/kaggle/working/SentinelIQ'
if os.path.exists(repo):
    !git -C /kaggle/working/SentinelIQ pull
else:
    !git clone https://github.com/hasan-rajab/SentinelIQ.git
%cd /kaggle/working/SentinelIQ
sys.path.insert(0, '/kaggle/working/SentinelIQ')
!pip install pyyaml joblib scikit-learn torch transformers shap -q
"""),

    code("""\
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from ml.models.isolation_forest import SentinelIsolationForest
from ml.models.autoencoder import SentinelAutoencoder
from ml.models.bert_log import SentinelBertLog
from ml.fusion.ensemble import SentinelEnsemble

sns.set_theme(style="darkgrid")

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])
"""),

    code("""\
# Generate fresh data
!python data/simulated/pipeline.py --duration 120 --anomaly-rate 0.08
"""),

    md("## Load Saved Models"),

    code("""\
feature_cols = cfg['isolation_forest']['features']
net_features = cfg['network']['features']

if_metrics = SentinelIsolationForest.load('ml/saved_models', name='isolation_forest_metrics')
if_network = SentinelIsolationForest.load('ml/saved_models', name='isolation_forest_network')
ae         = SentinelAutoencoder.load('ml/saved_models', name='autoencoder_metrics')
bert       = SentinelBertLog.load('ml/saved_models', name='bert_log')
print("All models loaded.")
"""),

    md("## Score Each Modality"),

    code("""\
metrics_df = load_jsonl('data/simulated/metrics.jsonl')
logs_df    = load_jsonl('data/simulated/logs.jsonl')
network_df = load_jsonl('data/simulated/network.jsonl')

# Use metrics for IF + AE, logs for BERT
# Align on shared index using metrics as base
n = min(len(metrics_df), len(logs_df))
metrics_df = metrics_df.iloc[:n].reset_index(drop=True)
logs_df    = logs_df.iloc[:n].reset_index(drop=True)

y_true = metrics_df['is_anomaly'].astype(int).values

if_scores   = if_metrics.score(metrics_df)
ae_scores   = ae.score(metrics_df)
bert_scores = bert.score(logs_df)

print(f"IF score range   : {if_scores.min():.4f} – {if_scores.max():.4f}")
print(f"AE score range   : {ae_scores.min():.6f} – {ae_scores.max():.6f}")
print(f"BERT score range : {bert_scores.min():.4f} – {bert_scores.max():.4f}")
"""),

    md("## Fuse Scores"),

    code("""\
ensemble = SentinelEnsemble(
    weights={"isolation_forest": 0.30, "autoencoder": 0.30, "bert_log": 0.40},
    strategy="weighted_avg",
    threshold=0.5,
)

fused = ensemble.fuse(if_scores=if_scores, ae_scores=ae_scores, bert_scores=bert_scores)
print(f"Fused score range: {fused.min():.4f} – {fused.max():.4f}")
"""),

    md("## Evaluate Ensemble vs Individual Models"),

    code("""\
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def quick_eval(name, scores, y_true, threshold=0.5):
    from sklearn.preprocessing import MinMaxScaler
    scores_norm = MinMaxScaler().fit_transform(scores.reshape(-1,1)).flatten()
    y_pred = (scores_norm >= threshold).astype(int)
    return {
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_true, scores_norm), 4),
        "Avg Precision": round(average_precision_score(y_true, scores_norm), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

results = pd.DataFrame([
    quick_eval("IsolationForest", if_scores, y_true),
    quick_eval("Autoencoder",     ae_scores, y_true),
    quick_eval("BERT",            bert_scores, y_true),
    {"Model": "Ensemble", "ROC-AUC": round(roc_auc_score(y_true, fused), 4),
     "Avg Precision": round(average_precision_score(y_true, fused), 4),
     "F1": round(f1_score(y_true, (fused>=0.5).astype(int), zero_division=0), 4)},
])
print(results.to_string(index=False))
"""),

    code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curves for all models
from sklearn.metrics import roc_curve
for name, scores in [("IF", if_scores), ("AE", ae_scores), ("BERT", bert_scores), ("Ensemble", fused)]:
    from sklearn.preprocessing import MinMaxScaler
    s = MinMaxScaler().fit_transform(scores.reshape(-1,1)).flatten()
    fpr, tpr, _ = roc_curve(y_true, s)
    auc = roc_auc_score(y_true, s)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0,1],[0,1],'k--')
axes[0].set_title("ROC Curves — All Models vs Ensemble")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend()

# Fused score distribution
axes[1].hist(fused[y_true==0], bins=30, alpha=0.6, color='#2ecc71', label='Normal')
axes[1].hist(fused[y_true==1], bins=30, alpha=0.6, color='#e74c3c', label='Anomaly')
axes[1].axvline(0.5, color='navy', linestyle='--', label='Threshold=0.5')
axes[1].set_title("Fused Score Distribution")
axes[1].legend()

plt.tight_layout()
plt.show()
"""),

    md("## Compare Fusion Strategies"),

    code("""\
strategies = ["weighted_avg", "max", "vote"]
strat_results = []

for strat in strategies:
    ens = SentinelEnsemble(strategy=strat, threshold=0.5)
    f = ens.fuse(if_scores=if_scores, ae_scores=ae_scores, bert_scores=bert_scores)
    from sklearn.preprocessing import MinMaxScaler
    f = MinMaxScaler().fit_transform(f.reshape(-1,1)).flatten()
    strat_results.append({
        "Strategy": strat,
        "ROC-AUC": round(roc_auc_score(y_true, f), 4),
        "F1": round(f1_score(y_true, (f>=0.5).astype(int), zero_division=0), 4),
    })

print(pd.DataFrame(strat_results).to_string(index=False))
"""),

    md("## Save Ensemble Config"),

    code("""\
os.makedirs('ml/saved_models', exist_ok=True)
ensemble.save('ml/saved_models/ensemble_config.json')
print("Ensemble config saved.")
"""),
)


# ── 06 SHAP Explainability ────────────────────────────────────────────────────
nb06 = nb(
    md("# SentinelIQ — 06 SHAP Explainability\nGenerate per-anomaly SHAP attributions and MITRE ATT&CK mappings."),

    code("""\
import os, sys
repo = '/kaggle/working/SentinelIQ'
if os.path.exists(repo):
    !git -C /kaggle/working/SentinelIQ pull
else:
    !git clone https://github.com/hasan-rajab/SentinelIQ.git
%cd /kaggle/working/SentinelIQ
sys.path.insert(0, '/kaggle/working/SentinelIQ')
!pip install pyyaml joblib scikit-learn shap matplotlib -q
"""),

    code("""\
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from ml.models.isolation_forest import SentinelIsolationForest
from ml.explainability.shap_explainer import SentinelShapExplainer
from ml.explainability.mitre_mapper import MitreMapper

with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])

feature_cols = cfg['isolation_forest']['features']
"""),

    code("""\
!python data/simulated/pipeline.py --duration 120 --anomaly-rate 0.08
"""),

    md("## Load Model & Data"),

    code("""\
metrics_df = load_jsonl('data/simulated/metrics.jsonl')
if_model = SentinelIsolationForest.load('ml/saved_models', name='isolation_forest_metrics')

X = metrics_df[feature_cols].fillna(0).values
X_scaled = if_model.scaler.transform(X)
y_true = metrics_df['is_anomaly'].astype(int).values

anomaly_idx = np.where(y_true == 1)[0]
normal_idx  = np.where(y_true == 0)[0]
print(f"Anomalies: {len(anomaly_idx)} | Normal: {len(normal_idx)}")
"""),

    md("## Fit SHAP Explainer"),

    code("""\
explainer = SentinelShapExplainer(model_type='isolation_forest', feature_cols=feature_cols)
explainer.fit(if_model.model, X_scaled[normal_idx[:200]])
shap_values = explainer.explain(X_scaled[anomaly_idx[:50]])
print(f"SHAP values shape: {shap_values.shape}")
"""),

    md("## SHAP Summary Plot"),

    code("""\
shap.summary_plot(
    shap_values,
    X_scaled[anomaly_idx[:50]],
    feature_names=feature_cols,
    show=True,
    plot_type="bar",
)
"""),

    md("## SHAP Waterfall — Single Anomaly"),

    code("""\
# Explain the highest-scoring anomaly
scores = if_model.score(metrics_df)
top_anomaly_idx = anomaly_idx[np.argmax(scores[anomaly_idx])]
top_anomaly = metrics_df.iloc[top_anomaly_idx]

print(f"Top anomaly record:")
print(f"  anomaly_type : {top_anomaly['anomaly_type']}")
print(f"  anomaly score: {scores[top_anomaly_idx]:.4f}")
for col in feature_cols:
    print(f"  {col:20}: {top_anomaly[col]}")

single_shap = explainer.explain(X_scaled[top_anomaly_idx:top_anomaly_idx+1])
attribution = explainer.explain_single(X_scaled[top_anomaly_idx])
print(f"\nTop contributing features:")
for feat, val in list(attribution.items())[:5]:
    print(f"  {feat:20}: {val:+.6f}")
"""),

    md("## MITRE ATT&CK Mapping"),

    code("""\
mapper = MitreMapper()

anomalies = metrics_df[metrics_df['is_anomaly'] == True].copy()
anomaly_types = anomalies['anomaly_type'].unique()

print("MITRE ATT&CK Mappings for detected anomalies:\n")
for atype in anomaly_types:
    m = mapper.map(atype)
    print(f"  [{atype}]")
    print(f"    Tactic    : {m.tactic} ({m.tactic_id})")
    print(f"    Technique : {m.technique} ({m.technique_id})")
    print(f"    Severity  : {m.severity.upper()}")
    print(f"    Action    : {m.recommended_action[:80]}")
    print()
"""),

    md("## Full Anomaly Explanation"),

    code("""\
from ml.explainability.shap_explainer import SentinelShapExplainer

# Build complete explanation for top anomaly
mitre = mapper.map_to_dict(top_anomaly['anomaly_type'])
fused_score = float(scores[top_anomaly_idx])
shap_attr = explainer.explain_single(X_scaled[top_anomaly_idx])

explanation = explainer.build_explanation(
    record=top_anomaly.to_dict(),
    shap_attribution=shap_attr,
    mitre_mapping=mitre,
    fused_score=fused_score,
)

print("=== ANOMALY EXPLANATION ===")
print(f"  Severity      : {explanation['severity'].upper()}")
print(f"  Fused Score   : {explanation['fused_score']}")
print(f"  MITRE Tactic  : {explanation['mitre_tactic']} ({explanation['mitre_tactic_id']})")
print(f"  Technique     : {explanation['mitre_technique']} ({explanation['mitre_technique_id']})")
print(f"  Top Features  : {explanation['top_features']}")
print(f"  Description   : {explanation['description']}")
print(f"  Action        : {explanation['recommended_action']}")
print()
print("=== NARRATIVE ===")
print(explainer.narrative(explanation))
"""),

    md("## Tactic Distribution"),

    code("""\
tactic_counts = {}
for _, row in anomalies.iterrows():
    m = mapper.map(row['anomaly_type'])
    tactic_counts[m.tactic] = tactic_counts.get(m.tactic, 0) + 1

plt.figure(figsize=(10, 4))
plt.barh(list(tactic_counts.keys()), list(tactic_counts.values()), color='#e74c3c')
plt.title('MITRE ATT&CK Tactic Distribution')
plt.xlabel('Count')
plt.tight_layout()
plt.show()
"""),
)


# ── Write all notebooks ───────────────────────────────────────────────────────
notebooks = [
    ("01_data_exploration.ipynb", nb01),
    ("02_isolation_forest.ipynb", nb02),
    ("03_autoencoder.ipynb",      nb03),
    ("04_bert_finetuning.ipynb",  nb04),
    ("05_fusion_ensemble.ipynb",  nb05),
    ("06_shap_explainability.ipynb", nb06),
]

for fname, notebook in notebooks:
    path = OUT / fname
    nbf.write(notebook, str(path))
    print(f"✅ {path}")

print("\nAll notebooks generated. Push to GitHub and open in Kaggle.")
