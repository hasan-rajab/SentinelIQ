"""Kafka-to-inference bridge for SentinelIQ.

Consumes telemetry envelopes, validates their shape, and forwards them to the
backend's authenticated /ingest endpoint. Offsets are committed only after the
backend accepts the record, providing at-least-once delivery semantics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from urllib import error, request

from confluent_kafka import Consumer, KafkaError

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("sentineliq.kafka-consumer")

TOPICS = ["sentineliq.logs", "sentineliq.metrics", "sentineliq.network"]


def _forward(envelope: dict) -> bool:
    endpoint = os.getenv("SENTINELIQ_INGEST_URL", "http://localhost:8000/ingest")
    api_key = os.getenv("SENTINELIQ_INGEST_API_KEY", "")
    body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=10) as response:
            return 200 <= response.status < 300
    except (error.URLError, TimeoutError):
        logger.exception("Failed to forward telemetry to inference API")
        return False


def main() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
            "group.id": os.getenv("KAFKA_CONSUMER_GROUP", "sentineliq-inference-v1"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe(TOPICS)

    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Kafka error: %s", message.error())
                continue

            try:
                envelope = json.loads(message.value().decode("utf-8"))
                if envelope.get("modality") not in {"log", "metric", "network"}:
                    raise ValueError("invalid modality")
                if not isinstance(envelope.get("record"), dict):
                    raise ValueError("record must be an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                logger.exception("Dropping malformed Kafka event")
                consumer.commit(message=message, asynchronous=False)
                continue

            if _forward(envelope):
                consumer.commit(message=message, asynchronous=False)
            else:
                # Avoid a tight retry loop during transient backend failures.
                time.sleep(2)
    finally:
        consumer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
