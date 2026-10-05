"""Exercise the built production image locally in CI with test-only credentials."""
from __future__ import annotations

import base64
import json
import secrets
import subprocess
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    image = "sentineliq-production:ci"
    password = secrets.token_urlsafe(32)
    ingest_key = secrets.token_urlsafe(48)
    password_hash = subprocess.check_output([
        "docker", "run", "--rm", "--entrypoint", "caddy", image,
        "hash-password", "--plaintext", password], text=True).strip()
    volume = 'sentineliq-ci-state-' + secrets.token_hex(6)
    subprocess.run(['docker', 'volume', 'create', volume], check=True, stdout=subprocess.DEVNULL)
    container = subprocess.check_output([
        "docker", "run", "-d", "--name", "sentineliq-production-ci",
        "--user", "0", "--mount", f"type=volume,source={volume},target=/app/state",
        "-p", "127.0.0.1:18080:8080",
        "-e", "PORT=8080", "-e", "SENTINELIQ_USERNAME=ci",
        "-e", "SENTINELIQ_CORS_ORIGINS=https://sentinel.example.test",
        "-e", "SENTINELIQ_PASSWORD_HASH=" + password_hash,
        "-e", "SENTINELIQ_INGEST_API_KEY=" + ingest_key, image], text=True).strip()
    origin = "http://127.0.0.1:18080"
    auth = "Basic " + base64.b64encode(f"ci:{password}".encode()).decode()
    def request(path, *, data=None, authenticated=True, api_key=None):
        headers = {"Authorization": auth} if authenticated else {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if api_key:
            headers["X-API-Key"] = api_key
        req = Request(origin + path, data=None if data is None else json.dumps(data).encode(), headers=headers)
        with urlopen(req, timeout=5) as response:
            return response.status, response.read()
    def wait_ready():
        deadline = time.monotonic() + 120
        while True:
            try:
                status, body = request("/ready", authenticated=False)
                if status == 200:
                    break
            except (HTTPError, URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Production container did not become ready within 120 seconds")
            time.sleep(1)
    try:
        wait_ready()
        for path in ("/", "/api/alerts"):
            try:
                request(path, authenticated=False)
                raise AssertionError("Protected route accepted an unauthenticated request")
            except HTTPError as exc:
                assert exc.code == 401
        assert request("/")[0] == 200
        assert request('/demo',authenticated=False)[0]==200
        assert json.loads(request('/api/demo/report',authenticated=False)[1])['test']['records']==121
        alert_count=json.loads(request('/api/alerts')[1])['total']
        public_sample=dict(cpu_percent=32,mem_percent=54,disk_read_mbps=12,disk_write_mbps=8,
            net_in_mbps=24,net_out_mbps=18,open_connections=110,process_count=85)
        public_result=json.loads(request('/api/demo/score',data=public_sample,authenticated=False)[1])
        assert isinstance(public_result['flagged'],bool)
        assert json.loads(request('/api/alerts')[1])['total']==alert_count
        model_state = json.loads(request("/api/health")[1])["models_loaded"]
        assert model_state["autoencoder"] and model_state["isolation_forest_network"]
        assert not model_state["bert_log"]
        payload = {"modality": "metric", "record": {
            "cpu_percent": 99, "mem_percent": 95, "disk_read_mbps": 900,
            "disk_write_mbps": 300, "net_in_mbps": 100, "net_out_mbps": 250,
            "open_connections": 6000, "process_count": 900}}
        try:
            request("/api/ingest", data=payload)
            raise AssertionError("Ingestion accepted a missing ingestion key")
        except HTTPError as exc:
            assert exc.code == 401
        result = json.loads(request("/api/ingest", data=payload, api_key=ingest_key)[1])
        assert result["alert_generated"]
        assert json.loads(request("/api/alerts")[1])["total"] > 0
        with socket.create_connection(("127.0.0.1", 18080), timeout=10) as connection:
            websocket_key = base64.b64encode(secrets.token_bytes(16)).decode()
            handshake = ("GET /api/stream/live HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                f"Sec-WebSocket-Key: {websocket_key}\r\nAuthorization: {auth}\r\n"
                "Origin: https://sentinel.example.test\r\n\r\n")
            connection.sendall(handshake.encode())
            response = b""
            while b"\r\n\r\n" not in response:
                response += connection.recv(4096)
            assert response.split(b"\r\n")[0].endswith(b"101 Switching Protocols")
            remainder = response.split(b"\r\n\r\n", 1)[1]
            assert remainder or connection.recv(4096)
        subprocess.run(['docker', 'restart', container], check=True, stdout=subprocess.DEVNULL)
        wait_ready()
        assert json.loads(request('/api/alerts')[1])['total'] > 0
        print("Production image smoke passed: mounted-volume readiness, login, models, ingestion, restart persistence and authenticated WebSocket.")
    except BaseException:
        subprocess.run(["docker", "logs", container], check=False)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(['docker', 'volume', 'rm', volume], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
