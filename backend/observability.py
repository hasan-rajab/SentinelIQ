"""Operational telemetry for the SentinelIQ API.

Provides low-cardinality Prometheus metrics, request IDs, and structured
request logs without coupling the application to a specific APM vendor.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Mapping

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger("sentineliq.http")

HTTP_REQUESTS = Counter(
    "sentineliq_http_requests_total",
    "Total HTTP requests processed by SentinelIQ.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "sentineliq_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "route"],
)
MODEL_LOADED = Gauge(
    "sentineliq_model_loaded",
    "Whether an inference model is currently loaded (1=yes, 0=no).",
    ["model"],
)


def set_model_status(models_loaded: Mapping[str, bool]) -> None:
    """Publish model readiness as Prometheus gauges."""
    for model, loaded in models_loaded.items():
        MODEL_LOADED.labels(model=model).set(1 if loaded else 0)


def install_observability(app: FastAPI) -> None:
    """Attach request telemetry and the Prometheus scrape endpoint."""

    @app.middleware("http")
    async def telemetry_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            route_obj = request.scope.get("route")
            route = getattr(route_obj, "path", request.url.path)
            method = request.method

            HTTP_REQUESTS.labels(
                method=method,
                route=route,
                status=str(status_code),
            ).inc()
            HTTP_LATENCY.labels(method=method, route=route).observe(elapsed)

            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": method,
                        "route": route,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1000, 2),
                    }
                )
            )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
