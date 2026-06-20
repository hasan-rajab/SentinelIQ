import sys
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from backend.services.anomaly_service import AnomalyService

service = AnomalyService()
print("autoencoder_network loaded:", service.models_loaded["autoencoder_network"])

df_net = pd.read_csv("data/processed/network_test.csv")
y_true_net = df_net["is_anomaly"].astype(int).values

fused_net = []
for _, row in df_net.iterrows():
    rec = row.to_dict()
    df1 = pd.DataFrame([rec])
    if_score = service.if_network.score(df1)[0]
    ae_score = service.ae_network.score(df1)[0]
    fused = service.ensemble.fuse_network(
    np.array([if_score]),
    np.array([ae_score]),
    if_threshold=service.if_network.threshold,
    ae_threshold=service.ae_network.threshold,
)[0]
    fused_net.append(fused)

fused_net = np.array(fused_net)
print("min:", fused_net.min(), "max:", fused_net.max(), "mean:", fused_net.mean())

from sklearn.metrics import classification_report

y_true_net = df_net["is_anomaly"].astype(int).values
y_pred_net = (fused_net >= 0.5).astype(int)  # 0.5 = "at threshold" per the new scale

print(classification_report(y_true_net, y_pred_net))
missed = ((y_true_net==1)&(y_pred_net==0)).sum()
print(f"Missed: {missed} / {y_true_net.sum()}")
from sklearn.metrics import f1_score

thresholds = np.linspace(fused_net.min(), fused_net.max(), 200)
best_t, best_f1 = 0, 0
for t in thresholds:
    pred = (fused_net >= t).astype(int)
    f1 = f1_score(y_true_net, pred, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Best threshold: {best_t:.4f} | F1: {best_f1:.4f}")
y_pred_best = (fused_net >= best_t).astype(int)
print(classification_report(y_true_net, y_pred_best))
missed_best = ((y_true_net==1)&(y_pred_best==0)).sum()
print(f"Missed: {missed_best} / {y_true_net.sum()}")

# Also check the 95%-recall threshold on this new scale
for t in sorted(thresholds):
    pred = (fused_net >= t).astype(int)
    recall = ((y_true_net==1)&(pred==1)).sum() / y_true_net.sum()
    if recall >= 0.90:
        print(f"\n90%-recall threshold: {t:.4f}")
        print(classification_report(y_true_net, pred))
        break