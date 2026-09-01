# SentinelIQ

**Production-style multimodal anomaly intelligence for IT operations and cybersecurity.**

SentinelIQ is an end-to-end AI engineering project that ingests logs, system metrics, and network flows, scores them with modality-specific ML models, generates model-aligned explanations, maps inferred incidents to MITRE ATT&CK, persists alerts, and exposes the system through a FastAPI service and Next.js SOC dashboard.

The repository is intentionally explicit about one distinction: **synthetic evaluation labels are never allowed into the serving decision path.**

## What this project demonstrates

- multimodal ML serving: metrics, logs, and network telemetry
- supervised + unsupervised anomaly detection
- label-leakage-safe inference
- streaming ingestion with Kafka
- durable PostgreSQL alert persistence
- model-aligned feature attribution
- MITRE ATT&CK incident mapping
- FastAPI + WebSocket serving
- Prometheus observability + Grafana
- Dockerized local infrastructure
- CI regression gates for ML integrity and application builds
- federated-learning experiments with Flower

## Production architecture

```text
                         ┌──────────────────────────────┐
                         │ Synthetic demo generators    │
                         │ (labels stripped at source)  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Kafka                                                                      │
│ sentineliq.logs  |  sentineliq.metrics  |  sentineliq.network             │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ manual offset commit
                               ▼
                     ┌──────────────────────┐
                     │ Kafka consumer       │
                     └──────────┬───────────┘
                                │ authenticated POST /ingest
                                ▼
                     ┌──────────────────────┐
                     │ FastAPI serving      │
                     └──────────┬───────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
┌───────────────┐      ┌────────────────┐      ┌──────────────────┐
│ Metrics       │      │ Logs           │      │ Network          │
│ Autoencoder   │      │ BERT           │      │ XGBoost + AE     │
└───────┬───────┘      └───────┬────────┘      └────────┬─────────┘
        └───────────────────────┼─────────────────────────┘
                                ▼
                    model score + calibrated threshold
                                │
                                ▼
                  model-aligned feature attribution
                                │
                                ▼
                  telemetry-derived incident category
                                │
                                ▼
                       MITRE ATT&CK mapping
                                │
                      ┌─────────┴──────────┐
                      ▼                    ▼
                PostgreSQL          Prometheus metrics
                      │                    │
                      ▼                    ▼
               Next.js dashboard       Grafana
```

A separate WebSocket demo path can feed generated records directly through the same anomaly service for interactive visualization.

## ML integrity contract

The simulator includes `is_anomaly` and `anomaly_type` fields so models can be trained and evaluated. Those fields are **ground truth, not features**.

SentinelIQ enforces this contract at multiple boundaries:

1. The Kafka producer removes both fields before publishing telemetry.
2. `/ingest` strips them again as defense in depth.
3. `AnomalyService` makes the alert decision only from the deployed model score and its calibrated threshold.
4. The incident category is inferred from observed telemetry after the model decision; it is not copied from the simulator label.
5. Raw alert evidence excludes the simulator labels.
6. CI contains regression tests proving a positive simulator label cannot force an alert below threshold and a model can alert when a simulator label says normal.

## Decision-model-aligned explanations

Explanations describe the model that actually produced the serving decision:

- **Metrics:** per-feature autoencoder reconstruction error.
- **Network, XGBoost path:** booster-native per-feature contribution values.
- **Network, AE fallback:** per-feature autoencoder reconstruction error.
- **Logs:** BERT score + telemetry-derived incident classification; no fabricated numeric feature attribution is emitted.

The API retains the legacy `shap_attribution` response field for frontend compatibility, but its values are now generated from the deployed decision model rather than an unrelated explainer.

## Stack

| Layer | Technology |
| --- | --- |
| ML | XGBoost, PyTorch Autoencoder, Hugging Face BERT, Isolation Forest |
| Feature engineering | Pandas, NumPy, scikit-learn |
| Streaming | Apache Kafka / Confluent Python client |
| Serving | FastAPI, WebSockets, Pydantic |
| Persistence | PostgreSQL via SQLAlchemy; SQLite fallback for local execution |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Observability | Prometheus metrics, structured request logs, request IDs, Grafana |
| Containers | Docker, Docker Compose |
| CI | GitHub Actions |
| Federated experiments | Flower |

## Quick start

### Prerequisites

- Docker with Compose
- Git

### Run the full demo stack

```bash
cp .env.example .env
```

Change the placeholder PostgreSQL, ingestion, and Grafana credentials in `.env`, then run:

```bash
docker compose --profile demo up --build
```

The `demo` profile starts a synthetic telemetry producer. For infrastructure without generated traffic, omit `--profile demo`.

### Local services

| Service | Address |
| --- | --- |
| SOC dashboard | http://localhost:3000 |
| FastAPI | http://localhost:8000 |
| OpenAPI docs | http://localhost:8000/docs |
| Readiness | http://localhost:8000/ready |
| Prometheus metrics | http://localhost:8000/metrics |
| Prometheus UI | http://localhost:9090 |
| Grafana | http://localhost:3001 |
| Kafka host listener | localhost:29092 |

Grafana automatically provisions Prometheus as its default datasource.

## Run without Docker

The backend defaults to SQLite when `SENTINELIQ_DATABASE_URL` is not set.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Then:

```bash
cd frontend
npm ci
npm run dev
```

## Model artifacts

Large artifacts, particularly BERT weights, are intentionally excluded from Git. Place trained/downloaded artifacts under `ml/saved_models/`.

Expected artifacts can include:

```text
ml/saved_models/
├── isolation_forest_metrics_*
├── isolation_forest_network_*
├── autoencoder_metrics_*
├── autoencoder_network_*
├── xgboost_network_*
├── bert_log/
├── bert_log_meta.json
└── ensemble_config.json
```

Missing optional artifacts degrade the relevant modality and are reported through model readiness state instead of crashing the complete API.

## Testing and CI

The CI pipeline validates:

```text
Python source compilation
        ↓
Inference-integrity regression tests
        ↓
FastAPI import contract

Next.js dependency install
        ↓
Production frontend build

Docker Compose configuration validation
```

Run the backend integrity tests locally with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Synthetic benchmark results

The following results came from fresh simulator-generated data that was separate from the model training samples used in the original experiments:

| Modality | Model | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: |
| Metrics | Autoencoder | 100.00% | 97.62% | 0.988 |
| Network | XGBoost | 85.85% | 100.00% | 0.924 |
| Network | XGBoost + Autoencoder | 99.06% | 100.00% | 0.995 |
| Logs | BERT | 100.00% | 100.00% | 1.000 |

These numbers are **synthetic upper-bound research benchmarks, not production performance claims**. The generators have cleaner class separation than real enterprise telemetry. Real evaluation should use external datasets and organization-specific traffic before any operational deployment.

## Known limitations

- The checked-in benchmark data is synthetic.
- Large BERT/model artifacts must be supplied separately.
- MITRE mapping is heuristic after anomaly detection; it is not a substitute for analyst investigation.
- The federated-learning module is an experiment and is not part of the primary serving data plane.
- Local Compose credentials are development defaults and must be changed before exposure outside a trusted machine.
- A production internet-facing deployment should place the dashboard and analyst APIs behind enterprise identity/IAM rather than relying on local Compose defaults.

## Repository map

```text
SentinelIQ/
├── backend/
│   ├── routes/              # alerts, explainability, ingest, stream, federated status
│   ├── services/            # inference, alert access, anomaly rules
│   ├── observability.py     # request metrics/logging
│   └── storage.py           # PostgreSQL/SQLite alert repository
├── data/simulated/          # demo telemetry generators
├── ingestion/               # Kafka producer and consumer
├── ml/
│   ├── models/
│   ├── features/
│   ├── fusion/
│   ├── explainability/
│   ├── training/
│   └── saved_models/
├── federated/
├── frontend/
├── ops/
│   ├── prometheus/
│   └── grafana/
├── tests/
├── configs/
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Design decisions worth discussing in an interview

**Why not let the synthetic label determine alerts?**  
Because that converts evaluation ground truth into a serving feature and produces invalid performance evidence. The serving path is now isolated from those labels.

**Why combine supervised and unsupervised network models?**  
XGBoost is useful for known attack patterns while the autoencoder contributes a structurally different novelty signal. The two scores are calibrated onto compatible ranges before fusion.

**Why manual Kafka commits?**  
The consumer commits an offset only after `/ingest` accepts the event, giving the demo at-least-once processing behavior instead of silently losing records when the backend is unavailable.

**Why PostgreSQL plus SQLite?**  
PostgreSQL models the deployed architecture; SQLite keeps local development and tests frictionless without creating a second persistence abstraction.

**Why separate `/health` and `/ready`?**  
Liveness answers whether the process is running. Readiness also checks that the database is reachable and at least one inference artifact is available.

## Roadmap

The highest-value next experiments are external benchmark evaluation, online drift monitoring, production IAM, model registry/release automation, and cloud infrastructure-as-code.

## License

MIT License.
