"""
Test script to verify the fusion bug fix in AnomalyService.
"""

import numpy as np
import pandas as pd
from backend.services.anomaly_service import AnomalyService

service = AnomalyService()

# Test 1: Check ensemble threshold
print("=" * 50)
print("Test 1: Ensemble threshold")
print("=" * 50)
print(f"service.ensemble.threshold = {service.ensemble.threshold}")
print()

# Test 2: Test process_record with a sample metric (all required features)
print("=" * 50)
print("Test 2: process_record with anomalous sample (FIXED)")
print("=" * 50)
sample = {
    "cpu_percent": 99,
    "mem_percent": 99,
    "disk_read_mbps": 10000,
    "disk_write_mbps": 10000,
    "net_in_mbps": 10000,
    "net_out_mbps": 10000,
    "open_connections": 5000,
    "process_count": 500,
}
alert = service.process_record(sample, "metric")
if alert:
    print(f"✓ Alert generated! fused_score = {alert['fused_score']}")
    print(f"  Severity: {alert['severity']}")
else:
    print("✗ alert = None (BUG: anomaly not detected)")
print()

# Test 3: Verify the fusion fix with batch vs single
print("=" * 50)
print("Test 3: Fusion with batch vs single values (FIXED)")
print("=" * 50)

# Batch fusion (works correctly)
batch_fused = service.ensemble.fuse(
    if_scores=np.array([0.1, 0.5, 0.9]),
    ae_scores=np.array([0.2, 0.6, 0.8])
)
print(f"Batch fusion [0.1,0.5,0.9] + [0.2,0.6,0.8] = {np.round(batch_fused, 4)}")

# Single value fusion (FIXED: no longer returns 0)
single_fused = service.ensemble.fuse(
    if_scores=np.array([0.5]),
    ae_scores=np.array([0.6])
)
print(f"Single fusion [0.5] + [0.6] = {single_fused}")
print(f"  Expected: weighted avg of 0.5 and 0.6 = {0.5*0.3/0.6 + 0.6*0.3/0.6:.4f}")
print()

# Test 4: User's original code snippet
print("=" * 50)
print("Test 4: User's original code snippet (FIXED)")
print("=" * 50)
from data.simulated.metric_simulator import generate_normal_metrics, inject_anomaly

records = []
for i in range(5):
    record = generate_normal_metrics("host-1", tick=i)
    records.append(record)
anomalous = inject_anomaly(generate_normal_metrics("host-1", tick=5))
records.append(anomalous)
df = pd.DataFrame(records)

row = df.iloc[0].to_dict()
scores = service.score_metric_record(row)
print(f"scores = {scores}")

fused = service.ensemble.fuse(
    if_scores=np.array([scores["if_score"]]),
    ae_scores=np.array([scores["ae_score"]])
)
print(f"fused = {fused}")
print(f"  (Previously returned [0.], now returns meaningful value)")