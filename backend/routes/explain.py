"""
SentinelIQ — Explain Routes
Returns full SHAP + MITRE explanation for a given alert.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas.models import ExplainResponse

router = APIRouter(prefix="/explain", tags=["explainability"])


def get_alert_service():
    from backend.main import alert_service
    return alert_service


@router.get("/{alert_id}", response_model=ExplainResponse)
def explain_alert(alert_id: str):
    service = get_alert_service()
    alert = service.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    return ExplainResponse(
        alert_id=alert["id"],
        fused_score=alert["fused_score"],
        severity=alert["severity"],
        mitre_tactic=alert["mitre_tactic"],
        mitre_tactic_id=alert["mitre_tactic_id"],
        mitre_technique=alert["mitre_technique"],
        mitre_technique_id=alert["mitre_technique_id"],
        description=alert["description"],
        recommended_action=alert["recommended_action"],
        top_features=alert["top_features"],
        feature_values=alert["feature_values"],
        shap_attribution=alert["shap_attribution"],
        narrative=alert["narrative"],
    )