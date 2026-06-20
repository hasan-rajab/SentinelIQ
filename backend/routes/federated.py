"""
SentinelIQ — Federated Routes
Exposes federated training status and round history to the frontend.
"""

import json
from pathlib import Path
from fastapi import APIRouter

from backend.schemas.models import FederatedStatusResponse, FederatedNodeStatus

router = APIRouter(prefix="/federated", tags=["federated"])

ROUND_HISTORY_PATH = "ml/saved_models/federated/round_history.json"


@router.get("/status", response_model=FederatedStatusResponse)
def federated_status():
    history = []
    if Path(ROUND_HISTORY_PATH).exists():
        with open(ROUND_HISTORY_PATH) as f:
            history = json.load(f)

    last_round = history[-1] if history else {"round": 0, "n_clients": 0, "avg_loss": 0.0}

    # Mock node statuses derived from last known round
    nodes = [
        FederatedNodeStatus(
            node_id=i,
            status="idle",
            last_round=last_round.get("round", 0),
            last_loss=last_round.get("avg_loss", 0.0),
            n_samples=last_round.get("total_examples", 0) // max(last_round.get("n_clients", 1), 1),
            n_anomalies=0,
        )
        for i in range(last_round.get("n_clients", 3))
    ]

    return FederatedStatusResponse(
        is_training=False,
        current_round=last_round.get("round", 0),
        total_rounds=5,
        nodes=nodes,
        round_history=history,
    )


@router.get("/history")
def federated_history():
    if Path(ROUND_HISTORY_PATH).exists():
        with open(ROUND_HISTORY_PATH) as f:
            return json.load(f)
    return []