"""
SentinelIQ — Anomaly Detection Service
Orchestrates the full pipeline: scoring → fusion → SHAP → MITRE mapping
→ narrative generation, for a single incoming record or batch.
"""

import sys
import json
import uuid
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.models.isolation_forest import SentinelIsolationForest
from ml.models.autoencoder import SentinelAutoencoder
from ml.models.bert_log import SentinelBertLog
from ml.fusion.ensemble import SentinelEnsemble
from ml.explainability.shap_explainer import SentinelShapExplainer
from ml.explainability.mitre_mapper import MitreMapper


class AnomalyService:
    """
    Singleton-style service that loads all trained models once
    and exposes scoring + explanation methods for the API layer.
    """

    def __init__(
        self,
        model_dir: str = "ml/saved_models",
        config_path: str = "configs/model_config.yaml",
    ):
        self.model_dir = model_dir
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.feature_cols = self.cfg["isolation_forest"]["features"]
        self.net_features = self.cfg["network"]["features"]

        self.models_loaded = {
            "isolation_forest_metrics": False,
            "isolation_forest_network": False,
            "autoencoder": False,
            "autoencoder_network": False,
            "bert_log": False,
            "ensemble": False,
        }

        self.if_metrics: Optional[SentinelIsolationForest] = None
        self.if_network: Optional[SentinelIsolationForest] = None
        self.ae: Optional[SentinelAutoencoder] = None
        self.ae_network: Optional[SentinelAutoencoder] = None
        self.bert: Optional[SentinelBertLog] = None
        self.ensemble: Optional[SentinelEnsemble] = None
        self.shap_explainer: Optional[SentinelShapExplainer] = None
        self.mitre_mapper = MitreMapper()

        self._load_models()

        # In-memory alert store (replace with DB in production)
        self.alerts: dict = {}

        # Stream stats
        self.stats = {
            "total_records_processed": 0,
            "total_anomalies_detected": 0,
            "active_since": datetime.utcnow(),
        }

    def _load_models(self):
        try:
            self.if_metrics = SentinelIsolationForest.load(self.model_dir, name="isolation_forest_metrics")
            self.models_loaded["isolation_forest_metrics"] = True
        except FileNotFoundError:
            print("[AnomalyService] WARNING: isolation_forest_metrics not found")

        try:
            self.if_network = SentinelIsolationForest.load(self.model_dir, name="isolation_forest_network")
            self.models_loaded["isolation_forest_network"] = True
        except FileNotFoundError:
            print("[AnomalyService] WARNING: isolation_forest_network not found")

        try:
            self.ae = SentinelAutoencoder.load(self.model_dir, name="autoencoder_metrics")
            self.models_loaded["autoencoder"] = True
        except FileNotFoundError:
            print("[AnomalyService] WARNING: autoencoder_metrics not found")

        try:
            self.ae_network = SentinelAutoencoder.load(self.model_dir, name="autoencoder_network")
            self.models_loaded["autoencoder_network"] = True
        except FileNotFoundError:
            print("[AnomalyService] WARNING: autoencoder_network not found "
                  "(network detection will fall back to Isolation Forest only)")

        try:
            self.bert = SentinelBertLog.load(self.model_dir, name="bert_log")
            self.models_loaded["bert_log"] = True
        except FileNotFoundError:
            print("[AnomalyService] WARNING: bert_log not found")

        try:
            self.ensemble = SentinelEnsemble.load(f"{self.model_dir}/ensemble_config.json")
            self.models_loaded["ensemble"] = True
        except FileNotFoundError:
            print("[AnomalyService] WARNING: ensemble_config not found, using defaults")
            self.ensemble = SentinelEnsemble()

        # Fit SHAP explainer on IF metrics model using real background data
        if self.if_metrics is not None:
            self.shap_explainer = SentinelShapExplainer(
                model_type="isolation_forest",
                feature_cols=self.feature_cols,
            )
            self._fit_shap_background()

        print(f"[AnomalyService] Models loaded: {self.models_loaded}")

    def _fit_shap_background(self):
        """
        Fit the SHAP explainer using a sample of normal records from the
        simulated metrics dataset as background. Falls back silently if
        no data is available yet (e.g. before pipeline.py has been run).
        """
        try:
            metrics_path = "data/simulated/metrics.jsonl"
            if not Path(metrics_path).exists():
                print("[AnomalyService] WARNING: no metrics data found for SHAP background, "
                      "explanations will use top-feature heuristic until data exists")
                return

            df = pd.DataFrame([json.loads(l) for l in open(metrics_path)])
            normal_df = df[df["is_anomaly"] == False]
            if len(normal_df) < 10:
                print("[AnomalyService] WARNING: not enough normal samples for SHAP background")
                return

            X = normal_df[self.feature_cols].fillna(0).values
            X_scaled = self.if_metrics.scaler.transform(X)
            background = X_scaled[: min(200, len(X_scaled))]

            self.shap_explainer.fit(self.if_metrics.model, background)
            print(f"[AnomalyService] SHAP explainer fitted on {len(background)} background samples")
        except Exception as e:
            print(f"[AnomalyService] WARNING: SHAP background fit failed: {e}")

    def score_metric_record(self, record: dict) -> dict:
        """Score a single metric record through IF + Autoencoder."""
        df = pd.DataFrame([record])

        if_score = float(self.if_metrics.score(df)[0]) if self.if_metrics else None
        ae_score = float(self.ae.score(df)[0]) if self.ae else None

        return {"if_score": if_score, "ae_score": ae_score}

    def score_log_record(self, record: dict) -> dict:
        """Score a single log record through BERT."""
        df = pd.DataFrame([record])
        bert_score = float(self.bert.score(df)[0]) if self.bert else None
        return {"bert_score": bert_score}

    def process_record(self, record: dict, modality: str) -> Optional[dict]:
        """
        Process an incoming record (metric/log/network), score it,
        fuse, explain, and store as an alert if anomalous.

        Returns the alert dict if anomalous, else None.
        """
        self.stats["total_records_processed"] += 1

        if modality == "metric":
            scores = self.score_metric_record(record)
            fused = self.ensemble.fuse(
                if_scores=np.array([scores["if_score"]]) if scores["if_score"] is not None else None,
                ae_scores=np.array([scores["ae_score"]]) if scores["ae_score"] is not None else None,
            )[0]
        elif modality == "log":
            scores = self.score_log_record(record)
            fused = scores["bert_score"] if scores["bert_score"] is not None else 0.0
        elif modality == "network":
            df = pd.DataFrame([record])
            if_score = float(self.if_network.score(df)[0]) if self.if_network else 0.0
            if self.ae_network is not None:
                ae_score = float(self.ae_network.score(df)[0])
                fused = float(self.ensemble.fuse_network(
                    if_scores=np.array([if_score]),
                    ae_scores=np.array([ae_score]),
                    if_threshold=self.if_network.threshold,
                    ae_threshold=self.ae_network.threshold,
                )[0])
            else:
                # Fallback: IF-only scoring if network autoencoder isn't trained yet
                fused = if_score
        else:
            raise ValueError(f"Unknown modality: {modality}")

        is_anomaly = record.get("is_anomaly", fused >= self.ensemble.threshold)

        if not is_anomaly:
            return None

        self.stats["total_anomalies_detected"] += 1

        alert_id = str(uuid.uuid4())
        anomaly_type = record.get("anomaly_type", "unknown")
        mitre = self.mitre_mapper.map_to_dict(anomaly_type)

        # SHAP attribution — real per-record explanation for metric modality
        shap_attr = {}
        top_features = list(record.keys())[:3]

        if modality == "metric" and self.shap_explainer is not None and self.shap_explainer.explainer is not None:
            try:
                df_single = pd.DataFrame([record])
                X = df_single[self.feature_cols].fillna(0).values
                X_scaled = self.if_metrics.scaler.transform(X)
                shap_attr = self.shap_explainer.explain_single(X_scaled[0])
                top_features = list(shap_attr.keys())[:3]
            except Exception as e:
                print(f"[AnomalyService] SHAP explain failed for alert: {e}")
                shap_attr = {}
                top_features = sorted(
                    record.keys() & set(self.feature_cols),
                    key=lambda k: abs(record.get(k, 0)), reverse=True
                )[:3]
        elif modality == "metric":
            # Fallback heuristic when SHAP isn't fitted yet
            top_features = sorted(
                record.keys() & set(self.feature_cols),
                key=lambda k: abs(record.get(k, 0)), reverse=True
            )[:3]

        narrative = (
            f"A {mitre['severity'].upper()} severity anomaly ({anomaly_type}) was detected "
            f"via {modality} analysis with a fused score of {fused:.3f}. "
            f"This corresponds to MITRE ATT&CK technique {mitre['technique']} "
            f"({mitre['technique_id']}) under tactic {mitre['tactic']}. "
            f"{mitre['description']} Recommended action: {mitre['recommended_action']}"
        )

        alert = {
            "id": alert_id,
            "timestamp": record.get("timestamp", datetime.utcnow().isoformat()),
            "source": record.get("source", record.get("host", record.get("src_ip", "unknown"))),
            "modality": modality,
            "anomaly_type": anomaly_type,
            "fused_score": round(float(fused), 4),
            "severity": mitre["severity"],
            "is_acknowledged": False,
            "mitre_tactic": mitre["tactic"],
            "mitre_tactic_id": mitre["tactic_id"],
            "mitre_technique": mitre["technique"],
            "mitre_technique_id": mitre["technique_id"],
            "description": mitre["description"],
            "recommended_action": mitre["recommended_action"],
            "top_features": top_features,
            "feature_values": {k: record.get(k) for k in top_features},
            "shap_attribution": shap_attr,
            "narrative": narrative,
            "raw_record": record,
        }

        self.alerts[alert_id] = alert
        return alert

    def get_alerts(self, limit: int = 100, severity: Optional[str] = None) -> list:
        alerts = list(self.alerts.values())
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        alerts.sort(key=lambda a: a["timestamp"], reverse=True)
        return alerts[:limit]

    def get_alert(self, alert_id: str) -> Optional[dict]:
        return self.alerts.get(alert_id)

    def acknowledge_alert(self, alert_id: str) -> bool:
        if alert_id in self.alerts:
            self.alerts[alert_id]["is_acknowledged"] = True
            return True
        return False

    def get_stats(self) -> dict:
        elapsed = (datetime.utcnow() - self.stats["active_since"]).total_seconds()
        rps = self.stats["total_records_processed"] / max(elapsed, 1)
        rate = (
            self.stats["total_anomalies_detected"] / max(self.stats["total_records_processed"], 1)
        )
        return {
            "total_records_processed": self.stats["total_records_processed"],
            "total_anomalies_detected": self.stats["total_anomalies_detected"],
            "anomaly_rate": round(rate, 4),
            "records_per_second": round(rps, 2),
            "active_since": self.stats["active_since"],
        }
