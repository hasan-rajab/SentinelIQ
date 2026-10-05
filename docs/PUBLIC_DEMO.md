# Public anomaly lab

Share https://sentineliq-production.up.railway.app/demo with visitors.
The exact public Caddy allowlist contains `/demo`, `/api/demo/report` and `/api/demo/score`.
Every other UI/API path retains owner Basic authentication; private ingestion also retains its API key.

The numeric sandbox accepts eight bounded telemetry fields, rejects unknown keys and foreign browser
origins, limits request frequency and simultaneous inference, and uses the saved deployed metrics
Isolation Forest. It never invokes alert persistence, private ingestion or model training.
No logs, host identifiers or arbitrary model paths are accepted. Scores are not probabilities.

The separate measured benchmark contains 605 synthetic records with chronological 363/121/121
train/validation/test splits. Test positives: nine. TP8, FP0, FN1, TN112; F1 .941, recall .889.
Threshold selection uses validation only. The missed memory-leak record is shown, rather than
hidden by aggregate performance. This benchmark does not validate the deployed model, BERT,
autoencoder, multimodal ensemble or real incident outcomes. See `docs/benchmarks/metric_report.json`.

Production-image smoke checks public scoring without credentials, unchanged alert counts after
inference, and continued 401 responses on private UI/alert routes.
