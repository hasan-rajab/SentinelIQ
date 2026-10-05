# Held-out metric benchmark with sample counts

```bash
python -m ml.training.benchmark_metrics
```

This standalone evaluation trains a fresh Isolation Forest on the committed **synthetic** metric stream. It does not overwrite the saved production artifacts or evaluate BERT, the autoencoder or the ensemble.

The 605 rows are ordered by UTC timestamp and original source-row ID, with 363 training, 121 validation and 121 test rows. Only the eight observed metric columns enter the model/scaler. Simulator `is_anomaly` and `anomaly_type` fields are forbidden features. Validation labels select the threshold by F1, then precision, then higher threshold; test labels only score the frozen result.

The current run contains **9 positive / 112 negative test records**. It reports **8 TP / 0 FP / 1 FN / 112 TN**, giving **0.941 record-level F1** and **0.889 recall**. The always-normal baseline has 0 recall/F1 despite high accuracy. Per-host and per-anomaly-type breakdowns show where records are missed. These are record metrics, not event/incident metrics.

See [`benchmarks/metric_report.json`](benchmarks/metric_report.json) for counts, thresholds, split boundaries, data SHA-256 and environment versions, and [`benchmarks/metric_test_predictions.csv`](benchmarks/metric_test_predictions.csv) for held-out labels/scores/predictions. Re-run to verify results under your environment; small version differences can change fitted trees.

The source covers roughly one minute of simulated activity across five hosts. Nine positive test records are a small sample. These results demonstrate a reproducible evaluation and error-analysis workflow; they do not establish real threat-detection accuracy, wide generalization or production security assurance.

Tests cover exact confusion counts, positive support, single-class AUC handling, threshold selection, disjoint partitions, finite scores and binary labels. CI additionally generates and retains the labelled benchmark artifacts. Existing inference-integrity checks continue to test that simulator labels cannot force a live alert.

Defensible concise CV wording: “Added a reproducible held-out Isolation Forest benchmark and host/type error analysis; achieved 0.941 record F1 on 121 simulated test records (9 positives; 8 TP, 0 FP, 1 FN).” Keep the simulated scope in the same bullet.
