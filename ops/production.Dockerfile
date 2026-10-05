FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
RUN npm run build

FROM caddy:2 AS ingress

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 \
    SENTINELIQ_DATABASE_URL=sqlite:////app/state/sentineliq.db \
    SENTINELIQ_MODEL_DIR=/app/ml/saved_models
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sentineliq
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.0 \
    && pip install --no-cache-dir -r backend/requirements.txt
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=ingress /usr/bin/caddy /usr/local/bin/caddy
COPY --chown=sentineliq:sentineliq . .
COPY --from=frontend --chown=sentineliq:sentineliq /build/.next frontend/.next
COPY --from=frontend --chown=sentineliq:sentineliq /build/node_modules frontend/node_modules
RUN mkdir -p /app/state && chown -R sentineliq:sentineliq /app/state
USER sentineliq
EXPOSE 8080
CMD ["python", "ops/serve.py"]
