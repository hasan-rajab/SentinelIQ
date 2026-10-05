# SentinelIQ

**Multimodal anomaly intelligence for security operations — designed to help analysts turn noisy telemetry into prioritized, explainable investigations.**

SentinelIQ is a production-style AI engineering project for logs, system metrics and network flows. It combines modality-specific ML models, streaming ingestion, explainability, MITRE ATT&CK context, durable alert storage and a SOC-facing application.

The business problem is straightforward:

> **Security teams do not need another model score. They need a defensible path from raw telemetry → anomaly evidence → incident context → analyst action.**

SentinelIQ is designed around that path.

> **Scope:** benchmark data in this repository is synthetic. Results are useful for regression and engineering validation, not claims of real-enterprise detection performance.

---

## Reproducible metric benchmark

Run `python -m ml.training.benchmark_metrics` for a fresh chronological train/validation/test evaluation on the committed **synthetic** metric stream. The current held-out result is **0.941 record F1 on 121 test records**, including **9 positive records: 8 TP / 0 FP / 1 FN**. Per-host/type breakdowns, the source hash, threshold-selection policy and environment versions are retained in the [benchmark report](docs/benchmarks/metric_report.json).

See [scope, reproduction and limitations](docs/METRIC_BENCHMARK.md). This small simulated benchmark does not measure real security-incident accuracy or the complete BERT/autoencoder/ensemble stack.

## Executive view

| Security-operations need | SentinelIQ approach |
|---|---|
| Prioritize suspicious telemetry | Modality-specific anomaly scoring across metrics, logs and network flows |
| Explain why something was flagged | Model-aligned feature attribution instead of generic explanations |
| Add security context | Telemetry-derived incident classification mapped to MITRE ATT&CK |
| Avoid invalid ML evidence | Synthetic ground-truth labels are excluded from the serving decision path |
| Support streaming operations | Kafka ingestion with manual commits and persistent alerts |
| Operate the service | FastAPI, PostgreSQL, Prometheus, Grafana, Docker and CI |

---

## Business value

SentinelIQ is designed to improve the **quality of analyst attention**, not to claim autonomous threat response.

The value hypothesis is:

1. identify unusual telemetry earlier;
2. preserve the evidence behind the alert;
3. explain which observed features drove the decision;
4. map the event into recognizable security context;
5. let analysts investigate through a consistent application surface.

In a real deployment, success should be measured with operational KPIs such as alert precision, analyst time-to-triage, escalation quality, false-positive burden and detection coverage — not model accuracy alone.

---

## Architecture

```text
Logs / metrics / network telemetry
              ↓
            Kafka
              ↓
       ingestion consumer
              ↓
           FastAPI
        ┌─────┼─────┐
        ↓     ↓     ↓
     Metrics  Logs  Network
       AE     BERT  XGBoost + AE
        └─────┼─────┘
              ↓
 model score + calibrated threshold
              ↓
 model-aligned explanation
              ↓
 telemetry-derived incident category
              ↓
       MITRE ATT&CK context
        ┌─────┴─────┐
        ↓           ↓
   PostgreSQL   Prometheus
        ↓           ↓
 Next.js SOC     Grafana
 dashboard
```

### ML integrity boundary

The synthetic simulator contains `is_anomaly` and `anomaly_type` for training/evaluation, but those fields are **not allowed to determine production-style serving decisions**.

SentinelIQ enforces that at multiple boundaries:

- producer strips synthetic labels before publishing;
- `/ingest` strips them again;
- alert decisions come from deployed-model scores and calibrated thresholds;
- incident categories are inferred from observed telemetry;
- raw alert evidence excludes simulator labels;
- regression tests prove the label cannot force or suppress the serving decision.

---

## Verified engineering evidence

**Reference CI:** SentinelIQ CI run #18  
**Date:** 10 September 2026  
**Conclusion:** success

The backend-integrity job completed:

- Python source compilation
- **8/8 inference-integrity tests**
- FastAPI import validation

The same workflow also completed:

- frontend production build
- Docker Compose configuration validation

### Synthetic benchmark

| Modality | Model | Recall | Precision | F1 |
|---|---|---:|---:|---:|
| Metrics | Autoencoder | 100.00% | 97.62% | 0.988 |
| Network | XGBoost | 85.85% | 100.00% | 0.924 |
| Network | XGBoost + Autoencoder | 99.06% | 100.00% | 0.995 |
| Logs | BERT | 100.00% | 100.00% | 1.000 |

These are **synthetic upper-bound research benchmarks**. The generated data has cleaner separation than real security telemetry, so external datasets and organization-specific traffic would be required before operational use.

---

## Why the architecture matters

### Supervised + unsupervised signals
Known attack patterns and novel behavior require different detection assumptions. The network path therefore combines XGBoost and autoencoder signals instead of relying on a single model family.

### Explanation follows the decision model
- metrics: per-feature reconstruction error;
- network / XGBoost: booster-native feature contributions;
- network / AE fallback: reconstruction error;
- logs: BERT score plus telemetry-derived incident classification.

The system does not fabricate numeric attribution where the deployed model does not support it.

### Streaming reliability
Kafka offsets are committed only after the ingestion API accepts the record, supporting at-least-once processing behavior in the demo architecture.

---

## Technology

**ML:** XGBoost · PyTorch Autoencoder · Hugging Face BERT · Isolation Forest  
**Streaming:** Apache Kafka  
**Serving:** FastAPI · WebSockets  
**Persistence:** PostgreSQL · SQLite fallback  
**Frontend:** Next.js · TypeScript  
**Observability:** Prometheus · Grafana · structured logs  
**Delivery:** Docker · Docker Compose · GitHub Actions  
**Security context:** MITRE ATT&CK  
**Experiments:** Flower federated-learning module

---

## Run the demo

```bash
cp .env.example .env
docker compose --profile demo up --build
```

The demo profile generates synthetic telemetry. Remove the profile for infrastructure without generated traffic.

Key surfaces:

- SOC dashboard: `http://localhost:3000`
- FastAPI/OpenAPI: `http://localhost:8000/docs`
- readiness: `http://localhost:8000/ready`
- metrics: `http://localhost:8000/metrics`
- Grafana: `http://localhost:3001`

---

## What I would change for a real SOC

A production deployment would require:

- enterprise identity and analyst RBAC;
- calibrated evaluation on external and organization-specific data;
- model registry and controlled release process;
- drift/performance monitoring;
- hardened secrets/networking;
- SIEM/SOAR integration;
- incident-response operating model and rollback procedures.

SentinelIQ is therefore best read as evidence of **AI security architecture, ML-integrity thinking and operational system design**, not autonomous cyber defense.

