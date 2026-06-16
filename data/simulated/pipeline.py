"""
SentinelIQ — Data Pipeline
Runs all three simulators concurrently and saves labeled datasets
ready for ML training in Phase 2.
"""

import threading
import json
import os
import argparse
import pandas as pd
from pathlib import Path

from log_simulator import stream_logs
from metric_simulator import stream_metrics
from network_simulator import stream_network


def run_simulator(generator, output_path: str, label: str):
    records = list(generator)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(records)
    anomaly_count = df["is_anomaly"].sum()
    print(f"[{label}] ✅ {len(df)} records | {anomaly_count} anomalies ({anomaly_count/len(df)*100:.1f}%) → {output_path}")
    return df


def generate_datasets(
    duration: int = 60,
    anomaly_rate: float = 0.08,
    output_dir: str = "data/simulated",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SentinelIQ — Generating Training Data")
    print(f"  Duration: {duration}s | Anomaly rate: {anomaly_rate*100:.0f}%")
    print(f"{'='*60}\n")

    results = {}
    threads = []

    configs = [
        (stream_logs,    {"rate_per_second": 10, "anomaly_rate": anomaly_rate, "duration_seconds": duration},
         f"{output_dir}/logs.jsonl",    "LOGS   "),
        (stream_metrics, {"rate_per_second": 2,  "anomaly_rate": anomaly_rate, "duration_seconds": duration},
         f"{output_dir}/metrics.jsonl", "METRICS"),
        (stream_network, {"rate_per_second": 20, "anomaly_rate": anomaly_rate, "duration_seconds": duration},
         f"{output_dir}/network.jsonl", "NETWORK"),
    ]

    def worker(fn, kwargs, path, label):
        gen = fn(**kwargs)
        results[label] = run_simulator(gen, path, label)

    for fn, kwargs, path, label in configs:
        t = threading.Thread(target=worker, args=(fn, kwargs, path, label))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # ── Save combined summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Dataset Summary")
    print(f"{'='*60}")
    total = 0
    total_anomalies = 0
    for label, df in results.items():
        total += len(df)
        total_anomalies += df["is_anomaly"].sum()
        anomaly_types = df[df["is_anomaly"]]["anomaly_type"].value_counts().to_dict()
        print(f"\n  {label.strip()}")
        print(f"    Total records : {len(df):,}")
        print(f"    Anomalies     : {df['is_anomaly'].sum():,}")
        print(f"    Anomaly types : {anomaly_types}")

    print(f"\n  TOTAL: {total:,} records | {total_anomalies:,} anomalies")
    print(f"\n  Files saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelIQ Data Pipeline")
    parser.add_argument("--duration", type=int, default=60, help="Seconds per simulator")
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--output-dir", type=str, default="data/simulated")
    args = parser.parse_args()

    generate_datasets(
        duration=args.duration,
        anomaly_rate=args.anomaly_rate,
        output_dir=args.output_dir,
    )