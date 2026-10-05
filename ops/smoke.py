"""Exercise the built production image locally in CI with test-only credentials."""
from __future__ import annotations

import base64
import json
import secrets
import subprocess
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
    container = subprocess.check_output([
        "docker", "run", "-d", "--name", "sentineliq-production-ci",
        "-p", "127.0.0.1:18080:8080",
        "-e", "PORT=8080", "-e", "SENTINELIQ_USERNAME=ci",
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
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                status, body = request("/ready", authenticated=False)
                if status == 200:
                    break
            except (HTTPError, URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Production container did not become ready within 120 seconds")
            time.sleep(1)
        for path in ("/", "/api/alerts"):
            try:
                request(path, authenticated=False)
                raise AssertionError("Protected route accepted an unauthenticated request")
            except HTTPError as exc:
                assert exc.code == 401
        assert request("/")[0] == 200
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
        print("Production image smoke passed: readiness, login, models, ingestion and persisted alert.")
    except BaseException:
        subprocess.run(["docker", "logs", container], check=False)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
