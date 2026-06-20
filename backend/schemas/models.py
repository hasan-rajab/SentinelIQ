"""
SentinelIQ — API Schemas
Pydantic models defining request/response shapes for the backend.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AnomalyAlert(BaseModel):
    id: str
    timestamp: datetime
    source: str
    modality: str               # "log" | "metric" | "network"
    anomaly_type: str
    fused_score: float
    severity: str                # low | medium | high | critical
    is_acknowledged: bool = False

    # Explainability
    mitre_tactic: str
    mitre_tactic_id: str
    mitre_technique: str
    mitre_technique_id: str
    description: str
    recommended_action: str
    top_features: List[str]
    feature_values: Dict[str, Any]
    shap_attribution: Dict[str, float]
    narrative: Optional[str] = None

    # Raw record snapshot
    raw_record: Dict[str, Any]


class AlertListResponse(BaseModel):
    total: int
    alerts: List[AnomalyAlert]


class AcknowledgeRequest(BaseModel):
    alert_id: str
    acknowledged_by: Optional[str] = None
    note: Optional[str] = None


class AcknowledgeResponse(BaseModel):
    alert_id: str
    is_acknowledged: bool
    message: str


class ExplainResponse(BaseModel):
    alert_id: str
    fused_score: float
    severity: str
    mitre_tactic: str
    mitre_tactic_id: str
    mitre_technique: str
    mitre_technique_id: str
    description: str
    recommended_action: str
    top_features: List[str]
    feature_values: Dict[str, Any]
    shap_attribution: Dict[str, float]
    narrative: str


class FederatedNodeStatus(BaseModel):
    node_id: int
    status: str                  # "idle" | "training" | "evaluating" | "offline"
    last_round: int
    last_loss: float
    n_samples: int
    n_anomalies: int


class FederatedStatusResponse(BaseModel):
    is_training: bool
    current_round: int
    total_rounds: int
    nodes: List[FederatedNodeStatus]
    round_history: List[Dict[str, Any]]


class StreamStats(BaseModel):
    total_records_processed: int
    total_anomalies_detected: int
    anomaly_rate: float
    records_per_second: float
    active_since: datetime


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"