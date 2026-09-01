"""Durable alert persistence for SentinelIQ.

Uses SQLAlchemy Core so the same repository works with local SQLite and the
PostgreSQL service used by Docker Compose. The full alert payload is retained
as JSON while indexed columns support common SOC filters.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, JSON, MetaData, String, Table, Column, create_engine, delete, select, update
from sqlalchemy.engine import Engine

metadata = MetaData()

alerts_table = Table(
    "alerts",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("source", String(255), nullable=False, index=True),
    Column("modality", String(32), nullable=False, index=True),
    Column("anomaly_type", String(128), nullable=False, index=True),
    Column("fused_score", Float, nullable=False),
    Column("severity", String(32), nullable=False, index=True),
    Column("is_acknowledged", Boolean, nullable=False, default=False, index=True),
    Column("payload", JSON, nullable=False),
)


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _plain_json(payload: dict) -> dict:
    """Normalize NumPy/datetime values before handing payload to DB drivers."""
    return json.loads(json.dumps(payload, default=_json_default))


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


class AlertRepository:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        metadata.create_all(self.engine)

    def save(self, alert: dict) -> None:
        payload = _plain_json(alert)
        row = {
            "id": alert["id"],
            "timestamp": _parse_timestamp(alert["timestamp"]),
            "source": str(alert["source"]),
            "modality": str(alert["modality"]),
            "anomaly_type": str(alert["anomaly_type"]),
            "fused_score": float(alert["fused_score"]),
            "severity": str(alert["severity"]),
            "is_acknowledged": bool(alert.get("is_acknowledged", False)),
            "payload": payload,
        }
        with self.engine.begin() as connection:
            connection.execute(alerts_table.insert().values(**row))

    def list(self, limit: int = 100, severity: Optional[str] = None) -> list[dict]:
        query = select(alerts_table.c.payload).order_by(alerts_table.c.timestamp.desc()).limit(limit)
        if severity:
            query = query.where(alerts_table.c.severity == severity)
        with self.engine.connect() as connection:
            return [dict(row.payload) for row in connection.execute(query)]

    def get(self, alert_id: str) -> Optional[dict]:
        query = select(alerts_table.c.payload).where(alerts_table.c.id == alert_id)
        with self.engine.connect() as connection:
            row = connection.execute(query).first()
            return dict(row.payload) if row else None

    def acknowledge(self, alert_id: str) -> bool:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(alerts_table.c.payload).where(alerts_table.c.id == alert_id)
            ).first()
            if row is None:
                return False

            payload = dict(row.payload)
            payload["is_acknowledged"] = True
            result = connection.execute(
                update(alerts_table)
                .where(alerts_table.c.id == alert_id)
                .values(is_acknowledged=True, payload=payload)
            )
            return result.rowcount > 0

    def delete_all(self) -> None:
        """Testing/development helper; not exposed through the API."""
        with self.engine.begin() as connection:
            connection.execute(delete(alerts_table))

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(1))
            return True
        except Exception:
            return False
