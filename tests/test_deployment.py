from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.routes.ingest import _verify_api_key
from ml.models.bert_log import SentinelBertLog
from ops.runtime import validate_production
from starlette.websockets import WebSocketDisconnect


def test_readiness_requires_a_decision_model_and_returns_503(monkeypatch):
    service = SimpleNamespace(models_loaded={"ensemble": True}, ae=None, bert=None,
                              if_network=None, xgb_network=None, ae_network=None)
    monkeypatch.setattr(main, "anomaly_service", service)
    monkeypatch.setattr(main, "alert_repository", SimpleNamespace(ping=lambda: True))
    client = TestClient(main.app)
    assert client.get("/ready").status_code == 503
    service.ae = SimpleNamespace(threshold=1.0)
    assert client.get("/ready").status_code == 200
    monkeypatch.setattr(main, "alert_repository", SimpleNamespace(ping=lambda: False))
    assert client.get("/ready").status_code == 503


def test_production_ingestion_fails_closed_when_no_key_is_configured(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("SENTINELIQ_INGEST_API_KEY", raising=False)
    with pytest.raises(Exception) as exc:
        _verify_api_key(None)
    assert exc.value.status_code == 503


def test_missing_bert_bundle_does_not_attempt_a_remote_model_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        SentinelBertLog.load(str(tmp_path))


def test_production_requires_credentials_and_persistence():
    with pytest.raises(RuntimeError):
        validate_production({"ENVIRONMENT": "production"})
    validate_production(dict(ENVIRONMENT="production", SENTINELIQ_INGEST_API_KEY="a"*64,
        SENTINELIQ_DATABASE_URL="sqlite:////app/state/test.db",
        SENTINELIQ_CORS_ORIGINS="https://sentinel.example.com",
        SENTINELIQ_PASSWORD_HASH="$2a$14$"+"a"*53))


def test_browser_stream_refuses_a_foreign_origin():
    client = TestClient(main.app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/stream/live", headers={"origin": "https://foreign.example.com"}):
            pass
    assert exc.value.code == 1008
