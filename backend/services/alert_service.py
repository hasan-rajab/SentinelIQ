"""
SentinelIQ — Alert Service
Thin wrapper around AnomalyService's in-memory alert store.
Swap this for a real database (Postgres/SQLite) in production.
"""

from typing import Optional, List
from backend.services.anomaly_service import AnomalyService


class AlertService:
    def __init__(self, anomaly_service: AnomalyService):
        self.anomaly_service = anomaly_service

    def list_alerts(self, limit: int = 100, severity: Optional[str] = None) -> List[dict]:
        return self.anomaly_service.get_alerts(limit=limit, severity=severity)

    def get_alert(self, alert_id: str) -> Optional[dict]:
        return self.anomaly_service.get_alert(alert_id)

    def acknowledge(self, alert_id: str) -> bool:
        return self.anomaly_service.acknowledge_alert(alert_id)

    def count_by_severity(self) -> dict:
        alerts = self.anomaly_service.get_alerts(limit=10_000)
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for a in alerts:
            sev = a.get("severity", "low")
            if sev in counts:
                counts[sev] += 1
        return counts

    def count_by_tactic(self) -> dict:
        alerts = self.anomaly_service.get_alerts(limit=10_000)
        counts = {}
        for a in alerts:
            tactic = a.get("mitre_tactic", "Unknown")
            counts[tactic] = counts.get(tactic, 0) + 1
        return counts