# SentinelIQ deployment

Status: deployment configuration prepared; no live deployment has been verified.

Deploy the `codex/data-analytics-evidence-2026-10-05` branch. The root
`railway.json` selects `ops/production.Dockerfile`, which builds the Next.js
frontend and original Python model service. The production entrypoint runs:

- FastAPI on loopback port 8000, with one worker;
- Next.js on loopback port 3000;
- Caddy on the host's `PORT`, defaulting to 8080.

Caddy authenticates the UI and `/api/*` with HTTP Basic authentication, proxies
API requests to FastAPI and preserves WebSocket upgrades for `/api/stream/live`.
Only `/health` and `/ready` bypass the browser login. Hosting terminates public
HTTPS. Neither internal service is exposed directly.

## Required settings and persistence

| Variable | Value or purpose |
| --- | --- |
| `ENVIRONMENT` | `production` |
| `SENTINELIQ_USERNAME` | Browser login name, default `owner` |
| `SENTINELIQ_PASSWORD_HASH` | bcrypt output of `caddy hash-password` |
| `SENTINELIQ_INGEST_API_KEY` | Separate generated random secret, at least 32 characters |
| `SENTINELIQ_DATABASE_URL` | `sqlite:////app/state/sentineliq.db` for this single-replica profile |
| `SENTINELIQ_CORS_ORIGINS` | Exact deployed HTTPS origin; also checks browser WebSocket origin |
| `RAILWAY_RUN_UID` | `0` for volume initialization; the entrypoint then drops to UID 10001 |

Store the password hash and ingestion key in host secret settings. Attach a
volume at `/app/state`; alerts survive restarts and redeployments. `/app/data`
contains application simulation modules and must remain part of the image.
Use one
replica with SQLite. The included optional Compose stack remains available
for PostgreSQL and Kafka; this web deployment does not silently provision them.

## Models and data scope

The repository includes metric-autoencoder weights and Isolation Forest
artifacts for metrics and network records. The runtime pins scikit-learn 1.9.0,
matching the version embedded in the saved estimators. The production image copies those
committed files. BERT, network-autoencoder weights and XGBoost artifacts are
optional and currently absent: `/health` reports them as unavailable. Missing
BERT files are detected locally without attempting a remote model download.

The live WebSocket feeds simulated telemetry through the loaded models. It
is a live application demonstration, not a monitored enterprise SOC. Historical
federated-round records describe saved simulations. The separate held-out
Isolation Forest benchmark does not measure the complete deployed ensemble.

## Verification and rollback

`/ready` returns HTTP 503 until the alert database responds and a model used
for actual decision-making is loaded; an ensemble configuration alone is
insufficient. Startup refuses missing login credentials, ingestion key or
database configuration. Ingestion also refuses unauthenticated requests.

Verify HTTP 401 for the protected UI, log in, inspect model availability,
connect the WebSocket and confirm generated alerts and persistence after
restart. Verify the ingestion key independently of browser authentication.
The supervisor stops the whole container if any of its three processes exits.
Roll back to a prior verified image while retaining the alert volume.
