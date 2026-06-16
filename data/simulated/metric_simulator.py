"""
SentinelIQ — Metric Stream Simulator
Generates realistic system metrics (CPU, memory, disk, network)
with synthetic anomalies injected at configurable rates.
"""

import random
import time
import json
import datetime
import math
from typing import Generator


# ── Baseline profiles per host ────────────────────────────────────────────────
HOST_PROFILES = {
    "host-1": {"cpu_base": 25, "mem_base": 60, "net_base": 10},
    "host-2": {"cpu_base": 40, "mem_base": 70, "net_base": 20},
    "host-3": {"cpu_base": 15, "mem_base": 45, "net_base": 5},
    "host-4": {"cpu_base": 55, "mem_base": 80, "net_base": 30},
    "host-5": {"cpu_base": 30, "mem_base": 55, "net_base": 15},
}


def _diurnal_factor(hour: int) -> float:
    """Simulate load variation through the day (peaks at 10am and 3pm)."""
    return 0.7 + 0.3 * (math.sin((hour - 6) * math.pi / 12) ** 2)


def generate_normal_metrics(host: str, tick: int) -> dict:
    profile = HOST_PROFILES[host]
    hour = datetime.datetime.utcnow().hour
    factor = _diurnal_factor(hour)

    cpu = min(95, max(2, profile["cpu_base"] * factor + random.gauss(0, 3)))
    mem = min(95, max(10, profile["mem_base"] + random.gauss(0, 2)))
    disk_read = max(0, profile["net_base"] * 0.5 + random.gauss(0, 1))   # MB/s
    disk_write = max(0, profile["net_base"] * 0.3 + random.gauss(0, 0.5))
    net_in = max(0, profile["net_base"] * factor + random.gauss(0, 2))   # Mbps
    net_out = max(0, profile["net_base"] * 0.6 * factor + random.gauss(0, 1))
    open_conns = max(0, int(50 * factor + random.gauss(0, 5)))
    proc_count = max(50, int(120 + random.gauss(0, 10)))

    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "host": host,
        "tick": tick,
        "cpu_percent": round(cpu, 2),
        "mem_percent": round(mem, 2),
        "disk_read_mbps": round(disk_read, 3),
        "disk_write_mbps": round(disk_write, 3),
        "net_in_mbps": round(net_in, 3),
        "net_out_mbps": round(net_out, 3),
        "open_connections": open_conns,
        "process_count": proc_count,
        "is_anomaly": False,
        "anomaly_type": None,
    }


def inject_anomaly(record: dict) -> dict:
    """Inject one of several realistic anomaly patterns."""
    anomaly_type = random.choice([
        "cpu_spike",
        "memory_leak",
        "network_exfiltration",
        "disk_flood",
        "connection_storm",
        "process_bomb",
    ])

    record = record.copy()
    record["is_anomaly"] = True
    record["anomaly_type"] = anomaly_type

    if anomaly_type == "cpu_spike":
        record["cpu_percent"] = round(random.uniform(92, 100), 2)
        record["process_count"] = random.randint(300, 500)

    elif anomaly_type == "memory_leak":
        record["mem_percent"] = round(random.uniform(93, 99), 2)
        record["cpu_percent"] = record["cpu_percent"] * 1.4

    elif anomaly_type == "network_exfiltration":
        record["net_out_mbps"] = round(random.uniform(200, 500), 3)  # massive outbound spike
        record["open_connections"] = random.randint(500, 2000)

    elif anomaly_type == "disk_flood":
        record["disk_write_mbps"] = round(random.uniform(300, 800), 3)
        record["disk_read_mbps"] = round(random.uniform(100, 300), 3)

    elif anomaly_type == "connection_storm":
        record["open_connections"] = random.randint(5000, 20000)
        record["net_in_mbps"] = round(random.uniform(100, 300), 3)

    elif anomaly_type == "process_bomb":
        record["process_count"] = random.randint(800, 2000)
        record["cpu_percent"] = round(random.uniform(88, 100), 2)
        record["mem_percent"] = round(random.uniform(85, 99), 2)

    return record


def stream_metrics(
    rate_per_second: float = 2.0,
    anomaly_rate: float = 0.05,
    duration_seconds: int = None,
    output_file: str = None,
) -> Generator[dict, None, None]:
    """
    Yields metric records for all hosts at the given rate.
    """
    interval = 1.0 / rate_per_second
    hosts = list(HOST_PROFILES.keys())
    tick = 0
    start = time.time()

    f = open(output_file, "w") if output_file else None

    try:
        while True:
            for host in hosts:
                record = generate_normal_metrics(host, tick)
                if random.random() < anomaly_rate:
                    record = inject_anomaly(record)
                if f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                yield record
            tick += 1
            if duration_seconds and (time.time() - start) >= duration_seconds:
                break
            time.sleep(interval)
    finally:
        if f:
            f.close()
        print(f"[metric_simulator] Generated {tick * len(hosts)} records.")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelIQ Metric Simulator")
    parser.add_argument("--rate", type=float, default=2.0, help="Ticks per second")
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/simulated/metrics.jsonl")
    args = parser.parse_args()

    print(f"[SentinelIQ] Streaming metrics → {args.output}\n")

    for record in stream_metrics(
        rate_per_second=args.rate,
        anomaly_rate=args.anomaly_rate,
        duration_seconds=args.duration if args.duration > 0 else None,
        output_file=args.output,
    ):
        label = "🔴 ANOMALY" if record["is_anomaly"] else "🟢 normal "
        print(
            f"[{label}] [{record['host']}] "
            f"CPU={record['cpu_percent']:5.1f}%  "
            f"MEM={record['mem_percent']:5.1f}%  "
            f"NET_OUT={record['net_out_mbps']:7.2f}Mbps  "
            f"CONNS={record['open_connections']:5d}"
            + (f"  [{record['anomaly_type']}]" if record["is_anomaly"] else "")
        )