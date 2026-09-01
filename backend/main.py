"""SentinelIQ FastAPI application.

Serves multimodal anomaly detection, alerting, explainability, federated
status, WebSocket streaming, health/readiness, and Prometheus telemetry.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.schemas.models import HealthResponse
from backend.services.alert_service import AlertService
from backend.services.anomaly_service import AnomalyService
from backend.observability import install_observability, set_model_status

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sentineliq")

anomaly_service: Optional[AnomalyService] = None
alert_service: Optional[AlertService] = None


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "SENTINELIQ_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global anomaly_service, alert_service

    logger.info("Starting SentinelIQ and loading inference models")
    anomaly_service = AnomalyService(
        model_dir=os.getenv("SENTINELIQ_MODEL_DIR", "ml/saved_models"),
        config_path=os.getenv("SENTINELIQ_MODEL_CONFIG", "configs/model_config.yaml"),
    )
    alert_service = AlertService(anomaly_service)
    set_model_status(anomaly_service.models_loaded)
    logger.info("SentinelIQ startup complete; models=%s", anomaly_service.models_loaded)

    yield

    logger.info("SentinelIQ shutdown complete")


app = FastAPI(
    title="SentinelIQ API",
    description="Multimodal AI anomaly detection platform for IT Ops & Cybersecurity",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

install_observability(app)

from backend.routes import alerts, explain, federated, stream

app.include_router(alerts.router)
app.include_router(stream.router)
app.include_router(explain.router)
app.include_router(federated.router)


@app.get("/", response_model=HealthResponse)
def root():
    models_loaded = anomaly_service.models_loaded if anomaly_service else {}
    set_model_status(models_loaded)
    return HealthResponse(status="ok", models_loaded=models_loaded)


@app.get("/health", response_model=HealthResponse)
def health():
    models_loaded = anomaly_service.models_loaded if anomaly_service else {}
    set_model_status(models_loaded)
    return HealthResponse(
        status="ok" if anomaly_service else "starting",
        models_loaded=models_loaded,
    )


@app.get("/ready", response_model=HealthResponse)
def readiness():
    models_loaded = anomaly_service.models_loaded if anomaly_service else {}
    ready = bool(anomaly_service) and any(models_loaded.values())
    set_model_status(models_loaded)
    return HealthResponse(
        status="ok" if ready else "starting",
        models_loaded=models_loaded,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
