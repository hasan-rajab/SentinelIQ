"""Run FastAPI, Next.js and their authenticated WebSocket-capable ingress."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.runtime import validate_production


def main():
    validate_production()
    port = int(os.environ.get("PORT", "8080"))
    if not 1024 <= port <= 65535:
        sys.exit("PORT must be between 1024 and 65535")
    data = Path("/app/state")
    data.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        for base, dirs, files in os.walk(data):
            for path in [Path(base)] + [Path(base) / name for name in dirs + files]:
                if not path.is_symlink():
                    os.chown(path, 10001, 10001)
        os.setgroups([])
        os.setgid(10001)
        os.setuid(10001)
    frontend_env = dict(os.environ, PORT="3000", HOSTNAME="127.0.0.1")
    commands = [
        ([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"], "/app", dict(os.environ)),
        (["node", "node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "--port", "3000"], "/app/frontend", frontend_env),
        (["caddy", "run", "--config", "ops/Caddyfile", "--adapter", "caddyfile"], "/app", dict(os.environ, PORT=str(port))),
    ]
    children = []
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for command, cwd, env in commands:
            children.append(subprocess.Popen(command, cwd=cwd, env=env))
        while True:
            for child in children:
                if child.poll() is not None:
                    raise SystemExit(child.returncode or 1)
            time.sleep(0.25)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
