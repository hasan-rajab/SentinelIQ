# Palantir FDSE Interview Drill — SentinelIQ

This drill is intentionally adversarial. It is designed to make the project defensible in an FDSE interview without overstating portfolio-scale evidence as production deployment.

## 30-second pitch

SentinelIQ is a production-style multimodal anomaly-intelligence system for IT operations and security. It consumes metrics, logs and network telemetry through Kafka, scores each modality with the model actually deployed for that path, persists alerts, exposes them through FastAPI and a Next.js dashboard, and instruments the service with Prometheus/Grafana. The central integrity decision is that synthetic ground-truth labels are stripped before serving and cannot influence the alert decision.

## Hostile questions and defensible answers

### Why Kafka rather than direct HTTP ingestion?
Kafka decouples telemetry producers from inference, buffers bursts, and gives the consumer an explicit offset/replay model. The current implementation is intentionally small, but the architecture separates ingestion from serving so either side can scale independently.

### Is your Kafka delivery really at-least-once?
The intended contract is: commit only after `/ingest` accepts the record. A failed forward must not allow a later offset from the same partition to be committed past it. The consumer therefore rewinds the failed partition to the failed offset before retrying. Malformed events are a different class: they are logged and committed as poison records because retrying invalid JSON forever would block the partition. A production system would normally send those to a DLQ.

### Why not exactly-once?
Exactly-once across Kafka, HTTP inference and PostgreSQL would require a stronger end-to-end transaction/idempotency design. The portfolio system chooses at-least-once ingestion and should make alert persistence idempotent before claiming duplicate-free end-to-end semantics.

### Why combine supervised and unsupervised models?
Known attack patterns and novelty are different signals. XGBoost can capture supervised structure while the autoencoder provides a structurally different anomaly signal. Their scores are calibrated before fusion; the purpose is complementary evidence, not model count for its own sake.

### Why did you remove simulator labels from serving?
Because ground truth is evaluation data. If `is_anomaly` or `anomaly_type` participates in the live decision, the reported performance is leakage and the serving system is invalid. SentinelIQ strips those labels before Kafka publication and again at the ingestion boundary, then makes the decision from model score and calibrated threshold.

### Why is attribution model-aligned?
An explanation should describe the model that made the decision. Metrics use autoencoder reconstruction error; XGBoost network decisions use booster-native contributions; the AE fallback uses reconstruction attribution. Producing SHAP values from an unrelated model would look sophisticated but be false evidence.

### Why PostgreSQL plus SQLite?
PostgreSQL represents the deployed persistence architecture; SQLite is a frictionless local/test fallback behind the same alert-repository boundary. At larger scale I would use PostgreSQL with migrations, connection pooling, indexes/partitioning by time, retention policies and tenant/organization boundaries.

### What breaks first at 100x traffic?
Likely bottlenecks are synchronous inference latency, model memory per worker, HTTP forwarding from the Kafka bridge, database write throughput and the single consumer group topology. I would measure per-stage latency first, then scale consumers by partition, batch where model semantics allow, move inference behind a horizontally scalable service, and partition/retention-tune storage.

### What happens if a model artifact is missing?
The service records model readiness and degrades that modality rather than crashing the whole API. That is useful for local/recovery behavior, but production routing should alert on degraded readiness and enforce an SLO for required modalities.

### Why not use an LLM to classify every incident?
The alert decision and high-confidence mapping should remain deterministic/model-grounded. An LLM can help summarize evidence for an analyst, but it should not become an uncalibrated source of truth for whether an incident occurred.

### What is the weakest evidence in this project?
The benchmark data is synthetic and cleaner than enterprise telemetry. The correct claim is that the repository demonstrates engineering and ML-integrity behavior; it does not prove production detection quality. The next evidence would be external benchmark data, drift evaluation and organization-specific calibration.

### If a customer says false positives cost more than misses, what changes?
I would make the objective explicit, inspect precision/recall and operational review capacity, recalibrate thresholds per modality, potentially route borderline cases to lower-severity review, and validate against a customer-agreed cost function instead of optimizing F1 blindly.

## Code anchors to be ready to open

- `backend/services/anomaly_service.py` — serving decisions, fallback behavior and model-aligned attribution
- `ingestion/kafka_consumer.py` — offset handling and delivery semantics
- `ml/fusion/ensemble.py` — score fusion and thresholds
- `backend/storage.py` — durable alert repository
- `tests/` — leakage/integrity regression evidence
- `.github/workflows/ci.yml` — release gates

## Claims boundary

Do not claim production SOC deployment, real-enterprise detection accuracy, exactly-once delivery, or proven Internet-scale throughput. Claim production-style architecture, reproducible tests, explicit integrity controls, synthetic benchmark evidence and a concrete scale path.
