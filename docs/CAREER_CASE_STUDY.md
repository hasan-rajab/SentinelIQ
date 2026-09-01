# SentinelIQ — Career Case Study

## One-line explanation

SentinelIQ is a production-style multimodal AI observability platform that detects anomalous behavior across infrastructure metrics, network telemetry, and logs, then turns model output into persisted, explainable, observable incidents.

## 30-second recruiter version

I built SentinelIQ to move anomaly detection beyond a notebook. The system serves modality-specific models behind FastAPI, ingests telemetry through Kafka, persists alerts in PostgreSQL, exposes Prometheus/Grafana observability, and runs a Next.js SOC dashboard. During hardening I found and removed inference label leakage, aligned explanations with the models that actually made the decisions, and added CI tests specifically around ML integrity.

## 90-second interview version

The original challenge was detecting unusual infrastructure, network, and log behavior with different model families. Metrics use autoencoder-based anomaly scoring, network telemetry uses XGBoost with an autoencoder fallback, and logs use transformer-based classification. The difficult part was not training the models; it was making the serving path trustworthy.

I found that synthetic ground-truth fields such as `is_anomaly` and `anomaly_type` could contaminate inference if they crossed the wrong boundary, so I redesigned the ingestion path to strip labels at the producer, API, and service levels and added regression tests proving labels cannot force a prediction. I also replaced explanations that were disconnected from the actual decision model with model-aligned attribution: reconstruction error for autoencoders and booster-native contributions for XGBoost.

I then added Kafka ingestion, PostgreSQL alert persistence, Prometheus/Grafana observability, Dockerized services, authenticated ingestion, readiness checks, structured logging, and CI gates covering backend integrity, frontend builds, and Compose configuration. The biggest lesson was that production ML quality is a systems property: model behavior, data contracts, persistence, observability, security, and regression tests all have to agree.

## Architecture story

```text
telemetry
  -> Kafka
  -> authenticated ingestion
  -> modality-specific inference
  -> calibrated decision threshold
  -> model-aligned attribution
  -> incident categorization / MITRE mapping
  -> PostgreSQL alert record
  -> dashboard + Prometheus/Grafana
```

## Engineering problems I can defend in an interview

### 1. Label leakage
**Problem:** simulator labels existed in generated records and could create invalid inference behavior if treated as features or decision inputs.

**Fix:** strip ground-truth fields at multiple serving boundaries and test that model scores, not labels, determine alerts.

**Lesson:** offline labels and online features must have an explicit contract.

### 2. Explanation mismatch
**Problem:** an explanation path can be technically valid while still misleading if it explains a different model from the one that generated the decision.

**Fix:** use autoencoder reconstruction error for AE decisions and XGBoost-native contributions for booster decisions; do not fabricate numeric attribution for logs.

**Lesson:** explainability is part of model integrity, not UI decoration.

### 3. Demo architecture vs production-style architecture
**Problem:** a model + API demo does not prove operational readiness.

**Fix:** add streaming ingestion, persistence, readiness, observability, containers, authentication, and CI release gates.

**Lesson:** production AI engineering starts after model training.

## Evidence

- multimodal model serving
- Kafka streaming ingestion
- PostgreSQL / SQLite persistence abstraction
- FastAPI + WebSockets
- Prometheus + Grafana
- Docker / Compose
- Next.js production build
- ML-integrity regression tests
- model-aligned feature attribution
- authenticated ingestion

## Strong interview questions this project answers

**Why use different models by modality?**  
Because metrics, network flows, and free-form logs have different statistical structures and representation needs. A single model would simplify deployment but weaken modality-specific inductive bias.

**Why not let a model automatically block incidents?**  
The project is an anomaly-intelligence and prioritization layer. Operational blocking requires organization-specific risk tolerances, false-positive evidence, identity controls, rollback, and human governance.

**What would you change at 100x traffic?**  
Partition Kafka by tenant/source, run stateless inference replicas, separate heavy model serving from API orchestration, use managed PostgreSQL/object storage, introduce a registry and deployment controller, add autoscaling, distributed tracing, and tenant-aware rate limits.

**What is the biggest limitation?**  
The current benchmark is synthetic. It demonstrates system integrity and model mechanics, not production detection quality on a real enterprise environment.

## CV-ready bullets

- Engineered a multimodal anomaly-intelligence platform combining autoencoder, XGBoost, transformer, and unsupervised models across infrastructure metrics, network telemetry, and logs.
- Built a production-style serving stack with FastAPI, Kafka streaming, PostgreSQL persistence, Dockerized services, and Prometheus/Grafana observability.
- Identified and eliminated inference label leakage and redesigned feature attribution to align explanations with the models responsible for production decisions.
- Implemented CI regression gates covering ML integrity, backend compilation, frontend production builds, and deployment configuration.

## Claims boundary

SentinelIQ is a portfolio-grade production-style system. Synthetic benchmark results are not production security guarantees, and the project should not be represented as a deployed commercial SOC platform.
