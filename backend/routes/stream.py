"""
SentinelIQ — Stream Routes
WebSocket endpoint that streams simulated data through the anomaly
detection pipeline in real-time and pushes alerts to connected clients.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data.simulated.log_simulator import generate_syslog, generate_auth, generate_apache
from data.simulated.metric_simulator import generate_normal_metrics, inject_anomaly, HOST_PROFILES
from data.simulated.network_simulator import (
    ANOMALY_GENERATORS, NORMAL_GENERATORS,
)

router = APIRouter(prefix="/stream", tags=["streaming"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()


def get_anomaly_service():
    from backend.main import anomaly_service
    return anomaly_service


@router.websocket("/live")
async def stream_live(websocket: WebSocket):
    """
    Streams simulated multimodal data through the detection pipeline.
    Sends every record processed, with alert details attached if anomalous.
    """
    await manager.connect(websocket)
    service = get_anomaly_service()
    tick = 0

    try:
        while True:
            import random

            modality_choice = random.choice(["log", "log", "metric", "metric", "network"])

            if modality_choice == "log":
                gen = random.choice([generate_syslog, generate_auth, generate_apache])
                record = gen(anomaly_rate=0.15)
                alert = service.process_record(record, modality="log")

            elif modality_choice == "metric":
                host = random.choice(list(HOST_PROFILES.keys()))
                record = generate_normal_metrics(host, tick)
                if random.random() < 0.15:
                    record = inject_anomaly(record)
                alert = service.process_record(record, modality="metric")

            else:  # network
                if random.random() < 0.15:
                    record = random.choice(ANOMALY_GENERATORS)()
                else:
                    record = random.choice(NORMAL_GENERATORS)()
                alert = service.process_record(record, modality="network")

            payload = {
                "type": "alert" if alert else "record",
                "modality": modality_choice,
                "record": record,
                "alert": alert,
            }

            await websocket.send_json(payload)
            tick += 1
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[stream] Error: {e}")
        manager.disconnect(websocket)


@router.get("/stats")
def get_stream_stats():
    service = get_anomaly_service()
    return service.get_stats()