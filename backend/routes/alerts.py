"""
SentinelIQ — Alerts Routes
REST endpoints for listing, retrieving, and acknowledging anomaly alerts.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.schemas.models import (
    AlertListResponse, AnomalyAlert,
    AcknowledgeRequest, AcknowledgeResponse,
)

router = APIRouter(prefix="/alerts", tags=["alerts"])


def get_alert_service():
    from backend.main import alert_service
    return alert_service


@router.get("", response_model=AlertListResponse)
def list_alerts(
    limit: int = Query(100, le=1000),
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
):
    service = get_alert_service()
    alerts = service.list_alerts(limit=limit, severity=severity)
    return AlertListResponse(total=len(alerts), alerts=alerts)


@router.get("/{alert_id}", response_model=AnomalyAlert)
def get_alert(alert_id: str):
    service = get_alert_service()
    alert = service.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/acknowledge", response_model=AcknowledgeResponse)
def acknowledge_alert(request: AcknowledgeRequest):
    service = get_alert_service()
    success = service.acknowledge(request.alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AcknowledgeResponse(
        alert_id=request.alert_id,
        is_acknowledged=True,
        message="Alert acknowledged successfully",
    )


@router.get("/stats/by-severity")
def stats_by_severity():
    service = get_alert_service()
    return service.count_by_severity()


@router.get("/stats/by-tactic")
def stats_by_tactic():
    service = get_alert_service()
    return service.count_by_tactic()