"""Synthetic telemetry producer for SentinelIQ's Kafka demo pipeline.

The producer intentionally strips simulator ground-truth labels before events
leave the generator process. This keeps the serving path representative of a
real telemetry stream and makes label leakage structurally impossible.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from confluent_kafka import Producer

from data.simulated.log_simulator import generate_apache, generate_auth, generate_syslog
from data.simulated.metric_simulator import HOST_PROFILES, generate_normal_metrics, inject_anomaly
from data.simulated.network_simulator import ANOMALY_GENERATORS, NORMAL_GENERATORS

TOPICS = {
    "log": "sentineliq.logs",
    "metric": "sentineliq.metrics",
    "network": "sentineliq.network",
}


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove evaluation-only labels from the live telemetry envelope."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"is_anomaly", "anomaly_type"}
    }


def generate_event(tick: int) -> tuple[str, dict[str, Any]]:
    modality = random.choice(["log", "log", "metric", "metric", "network"])

    if modality == "log":
        generator = random.choice([generate_syslog, generate_auth, generate_apache])
        record = generator(anomaly_rate=0.15)
    elif modality == "metric":
        host = random.choice(list(HOST_PROFILES.keys()))
        record = generate_normal_metrics(host, tick)
        if random.random() < 0.15:
            record = inject_anomaly(record)
    else:
        generator = (
            random.choice(ANOMALY_GENERATORS)
            if random.random() < 0.15
            else random.choice(NORMAL_GENERATORS)
        )
        record = generator()

    return modality, sanitize_record(record)


def main() -> None:
    producer = Producer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
            "client.id": "sentineliq-demo-producer",
            "enable.idempotence": True,
            "acks": "all",
        }
    )
    interval = float(os.getenv("SENTINELIQ_DEMO_INTERVAL_SECONDS", "0.5"))
    tick = 0

    while True:
        modality, record = generate_event(tick)
        envelope = {"modality": modality, "record": record}
        producer.produce(
            TOPICS[modality],
            key=str(record.get("host") or record.get("source") or record.get("src_ip") or tick),
            value=json.dumps(envelope, separators=(",", ":"), default=str),
        )
        producer.poll(0)
        tick += 1
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
