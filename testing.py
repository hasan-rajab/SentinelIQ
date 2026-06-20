import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from backend.services.anomaly_service import AnomalyService

service = AnomalyService()

df = pd.read_csv("data/processed/metrics_test.csv")

fused_scores = []
for _, row in df.iterrows():
    scores = service.score_metric_record(row.to_dict())
    fused = service.ensemble.fuse(
        if_scores=np.array([scores["if_score"]]),
        ae_scores=np.array([scores["ae_score"]])
    )[0]
    fused_scores.append(fused)

fused_scores = np.array(fused_scores)
print("min:", fused_scores.min(), "max:", fused_scores.max(), "mean:", fused_scores.mean())
print("threshold:", service.ensemble.threshold)
print("% above threshold:", (fused_scores >= service.ensemble.threshold).mean())

from sklearn.metrics import classification_report

y_true = df["is_anomaly"].astype(int).values
y_pred = (fused_scores >= service.ensemble.threshold).astype(int)

print(classification_report(y_true, y_pred))

missed = ((y_true == 1) & (y_pred == 0)).sum()
print(f"Missed anomalies: {missed} / {y_true.sum()}")

df_net = pd.read_csv("data/processed/network_test.csv")
net_features = service.cfg["network"]["features"]

fused_net = []
for _, row in df_net.iterrows():
    rec = row.to_dict()
    score = service.if_network.score(pd.DataFrame([rec]))[0]
    fused_net.append(score)

fused_net = np.array(fused_net)
y_true_net = df_net["is_anomaly"].astype(int).values
y_pred_net = (fused_net >= service.if_network.threshold).astype(int)

print(classification_report(y_true_net, y_pred_net))
print("Missed:", ((y_true_net==1)&(y_pred_net==0)).sum(), "/", y_true_net.sum())