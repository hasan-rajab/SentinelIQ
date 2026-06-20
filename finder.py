import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from backend.services.anomaly_service import AnomalyService

service = AnomalyService()

df_net = pd.read_csv("data/processed/network_test.csv")
y_true_net = df_net["is_anomaly"].astype(int).values

net_scores = service.if_network.score(df_net)

thresholds = np.linspace(net_scores.min(), net_scores.max(), 200)

# Option A: best F1
best_t, best_f1 = 0, 0
for t in thresholds:
    pred = (net_scores >= t).astype(int)
    f1 = f1_score(y_true_net, pred, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Best F1 threshold: {best_t:.6f} | F1: {best_f1:.4f}")
print(classification_report(y_true_net, (net_scores >= best_t).astype(int)))

# Option B: lowest threshold hitting 95% recall
for t in sorted(thresholds):
    pred = (net_scores >= t).astype(int)
    recall = ((y_true_net==1)&(pred==1)).sum() / y_true_net.sum()
    if recall >= 0.95:
        print(f"\n95%-recall threshold: {t:.6f}")
        print(classification_report(y_true_net, pred))
        break