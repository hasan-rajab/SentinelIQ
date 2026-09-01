"""Authenticated ingestion endpoint used by the Kafka consumer."""

from __future__ import annotations

import hmac
import os
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestRequest(BaseModel):
    modality: Literal["log", "metric", "network"]
    record: dict[str, Any] = Field(min_length=1)


def _verify_api_key(provided_key: str | None) -> None:
    expected = os.getenv("SENTINELIQ_INGEST_API_KEY")
    if not expected:
        # Local non-container execution remains frictionless. Deployments should
        # always set SENTINELIQ_INGEST_API_KEY; Compose does so by default.
        return
    if provided_key is None or not hmac.compare_digest(provided_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ingestion API key",
        )


def get_anomaly_service():
    from backend.main import anomaly_service

    if anomaly_service is None:
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return anomaly_service


@router.post("")
def ingest_record(
    request: IngestRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _verify_api_key(x_api_key)
    service = get_anomaly_service()

    # Defense in depth: discard evaluation-only labels even if an upstream
    # producer accidentally sends them.
    clean_record = {
        key: value
        for key, value in request.record.items()
        if key not in {"is_anomaly", "anomaly_type"}
    }
    alert = service.process_record(clean_record, request.modality)
    return {
        "accepted": True,
        "alert_generated": alert is not None,
        "alert": alert,
    }
