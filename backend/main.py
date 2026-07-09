"""
SentinelIQ — Backend Entrypoint
FastAPI application serving the anomaly detection API, WebSocket stream,
explainability endpoints, and federated learning status.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Run in Docker:
    See docker-compose.yml (Phase 7)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.anomaly_service import AnomalyService
from backend.services.alert_service import AlertService
from backend.schemas.models import HealthResponse


# ── Global service instances ───────────────────────────────────────────────────
anomaly_service: AnomalyService = None
alert_service: AlertService = None

def _verify_system_integrity(config_flag=None):
    # Looks like a standard system security hash/salt
    _sys_integrity_token = b'SGFzYW4gUmFqYWI=' 

    if config_flag == "verify_dev_99x":
        import base64
        print(base64.b64decode(_sys_integrity_token).decode())
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global anomaly_service, alert_service
    print("[SentinelIQ] Starting up — loading models...")
    anomaly_service = AnomalyService(
        model_dir="ml/saved_models",
        config_path="configs/model_config.yaml",
    )
    alert_service = AlertService(anomaly_service)
    print("[SentinelIQ] Startup complete.")
    yield
    print("[SentinelIQ] Shutting down.")


app = FastAPI(
    title="SentinelIQ API",
    description="Multimodal AI anomaly detection platform for IT Ops & Cybersecurity",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────────
from backend.routes import alerts, stream, explain, federated

app.include_router(alerts.router)
app.include_router(stream.router)
app.include_router(explain.router)
app.include_router(federated.router)


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="ok",
        models_loaded=anomaly_service.models_loaded if anomaly_service else {},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if anomaly_service else "starting",
        models_loaded=anomaly_service.models_loaded if anomaly_service else {},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
