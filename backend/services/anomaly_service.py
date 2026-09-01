"""SentinelIQ anomaly-detection service.

Orchestrates model loading, modality-specific scoring, decision-model-aligned
feature attribution, MITRE ATT&CK mapping, durable alert creation, and stream
metrics. Simulator ground-truth labels never participate in live inference.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.services.anomaly_rules import infer_anomaly_type
from ml.explainability.mitre_mapper import MitreMapper
from ml.explainability.model_attribution import (
    autoencoder_reconstruction_attribution,
    top_features as attribution_top_features,
    xgboost_contribution_attribution,
)
from ml.features.network_features import add_network_features
from ml.fusion.ensemble import SentinelEnsemble
from ml.models.autoencoder import SentinelAutoencoder
from ml.models.bert_log import SentinelBertLog
from ml.models.isolation_forest import SentinelIsolationForest
from ml.models.xgboost_network import SentinelXGBoost

logger = logging.getLogger("sentineliq.anomaly")


class AnomalyService:
    """Load trained artifacts once and expose safe live inference operations."""

    def __init__(
        self,
        model_dir: str = "ml/saved_models",
        config_path: str = "configs/model_config.yaml",
        alert_repository=None,
    ):
        self.model_dir = model_dir
        self.alert_repository = alert_repository
        with open(config_path, encoding="utf-8") as config_file:
            self.cfg = yaml.safe_load(config_file)

        self.feature_cols = self.cfg["isolation_forest"]["features"]
        self.net_features = self.cfg["network"]["features"]

        self.models_loaded = {
            "isolation_forest_metrics": False,
            "isolation_forest_network": False,
            "autoencoder": False,
            "autoencoder_network": False,
            "xgboost_network": False,
            "bert_log": False,
            "ensemble": False,
        }

        self.if_metrics: Optional[SentinelIsolationForest] = None
        self.if_network: Optional[SentinelIsolationForest] = None
        self.ae: Optional[SentinelAutoencoder] = None
        self.ae_network: Optional[SentinelAutoencoder] = None
        self.xgb_network: Optional[SentinelXGBoost] = None
        self.bert: Optional[SentinelBertLog] = None
        self.ensemble: Optional[SentinelEnsemble] = None
        self.mitre_mapper = MitreMapper()

        self._load_models()

        self.alerts: dict[str, dict] = {}
        self.stats = {
            "total_records_processed": 0,
            "total_anomalies_detected": 0,
            "active_since": datetime.now(timezone.utc),
        }

    def _load_models(self) -> None:
        self.if_metrics = self._load_optional(
            "isolation_forest_metrics",
            SentinelIsolationForest.load,
            self.model_dir,
            name="isolation_forest_metrics",
        )
        self.if_network = self._load_optional(
            "isolation_forest_network",
            SentinelIsolationForest.load,
            self.model_dir,
            name="isolation_forest_network",
        )
        self.ae = self._load_optional(
            "autoencoder",
            SentinelAutoencoder.load,
            self.model_dir,
            name="autoencoder_metrics",
        )
        self.ae_network = self._load_optional(
            "autoencoder_network",
            SentinelAutoencoder.load,
            self.model_dir,
            name="autoencoder_network",
        )
        self.xgb_network = self._load_optional(
            "xgboost_network",
            SentinelXGBoost.load,
            self.model_dir,
            name="xgboost_network",
        )
        self.bert = self._load_optional(
            "bert_log",
            SentinelBertLog.load,
            self.model_dir,
            name="bert_log",
        )

        try:
            self.ensemble = SentinelEnsemble.load(f"{self.model_dir}/ensemble_config.json")
            self.models_loaded["ensemble"] = True
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Ensemble artifact unavailable (%s); using defaults", exc)
            self.ensemble = SentinelEnsemble()

        logger.info("Model load state: %s", self.models_loaded)

    def _load_optional(self, state_key: str, loader, *args, **kwargs):
        try:
            model = loader(*args, **kwargs)
            self.models_loaded[state_key] = True
            return model
        except (FileNotFoundError, OSError) as exc:
            # Large artifacts such as BERT weights are intentionally excluded
            # from Git. Missing optional artifacts must degrade a modality, not
            # crash the entire API or hide the readiness state.
            logger.warning("Model artifact unavailable: %s (%s)", state_key, exc)
            return None

    def score_metric_record(self, record: dict) -> dict:
        if self.ae is None:
            return {"ae_score": None}
        return {"ae_score": float(self.ae.score(pd.DataFrame([record]))[0])}

    def score_log_record(self, record: dict) -> dict:
        if self.bert is None:
            return {"bert_score": None}
        return {"bert_score": float(self.bert.score(pd.DataFrame([record]))[0])}

    def process_record(self, record: dict, modality: str) -> Optional[dict]:
        """Score one record and create an alert only when the model flags it."""
        if modality not in {"metric", "log", "network"}:
            raise ValueError(f"Unknown modality: {modality}")

        self.stats["total_records_processed"] += 1
        attribution: dict[str, float] = {}
        enriched_record = record

        if modality == "metric":
            score = self.score_metric_record(record)["ae_score"]
            if score is None or self.ae is None or self.ae.threshold is None:
                return None
            fused = float(score)
            threshold = float(self.ae.threshold)

        elif modality == "log":
            score = self.score_log_record(record)["bert_score"]
            if score is None or self.bert is None or self.bert.threshold is None:
                return None
            fused = float(score)
            threshold = float(self.bert.threshold)

        else:
            df = add_network_features(pd.DataFrame([record]))
            enriched_record = df.iloc[0].to_dict()
            threshold = float(self.ensemble.network_threshold)

            if self.xgb_network is not None and self.ae_network is not None:
                xgb_score = float(self.xgb_network.score(df)[0])
                ae_score = float(self.ae_network.score(df)[0])
                fused = float(
                    self.ensemble.fuse_network_xgb(
                        xgb_scores=np.array([xgb_score]),
                        ae_scores=np.array([ae_score]),
                        ae_threshold=self.ae_network.threshold,
                    )[0]
                )
            elif self.if_network is not None and self.ae_network is not None:
                if_score = float(self.if_network.score(df)[0])
                ae_score = float(self.ae_network.score(df)[0])
                fused = float(
                    self.ensemble.fuse_network(
                        if_scores=np.array([if_score]),
                        ae_scores=np.array([ae_score]),
                        if_threshold=self.if_network.threshold,
                        ae_threshold=self.ae_network.threshold,
                    )[0]
                )
            elif self.if_network is not None:
                fused = float(self.if_network.score(df)[0])
                threshold = float(self.if_network.threshold)
            else:
                return None

        if fused < threshold:
            return None

        if modality == "metric" and self.ae is not None:
            try:
                attribution = autoencoder_reconstruction_attribution(self.ae, record)
            except Exception:
                logger.exception("Metric autoencoder attribution failed")
        elif modality == "network" and self.xgb_network is not None:
            try:
                attribution = xgboost_contribution_attribution(self.xgb_network, df)
            except Exception:
                logger.exception("XGBoost contribution attribution failed")
        elif modality == "network" and self.ae_network is not None:
            try:
                attribution = autoencoder_reconstruction_attribution(
                    self.ae_network,
                    enriched_record,
                )
            except Exception:
                logger.exception("Network autoencoder attribution failed")

        self.stats["total_anomalies_detected"] += 1
        anomaly_type = infer_anomaly_type(record, modality)
        mitre = self.mitre_mapper.map_to_dict(anomaly_type)
        top_features = attribution_top_features(attribution)

        if not top_features and modality == "metric":
            top_features = sorted(
                (key for key in self.feature_cols if key in record),
                key=lambda key: abs(float(record.get(key, 0) or 0)),
                reverse=True,
            )[:3]

        feature_values = {
            key: enriched_record.get(key, record.get(key))
            for key in top_features
        }

        narrative = (
            f"A {mitre['severity'].upper()} severity anomaly ({anomaly_type}) was detected "
            f"via {modality} analysis with a score of {fused:.3f} against a "
            f"decision threshold of {threshold:.3f}. "
            f"MITRE ATT&CK mapping: {mitre['technique']} ({mitre['technique_id']}) "
            f"under {mitre['tactic']}. {mitre['description']} "
            f"Recommended action: {mitre['recommended_action']}"
        )

        raw_record = {
            key: value
            for key, value in record.items()
            if key not in {"is_anomaly", "anomaly_type"}
        }
        alert_id = str(uuid.uuid4())
        alert = {
            "id": alert_id,
            "timestamp": record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "source": record.get("source", record.get("host", record.get("src_ip", "unknown"))),
            "modality": modality,
            "anomaly_type": anomaly_type,
            "fused_score": round(fused, 4),
            "severity": mitre["severity"],
            "is_acknowledged": False,
            "mitre_tactic": mitre["tactic"],
            "mitre_tactic_id": mitre["tactic_id"],
            "mitre_technique": mitre["technique"],
            "mitre_technique_id": mitre["technique_id"],
            "description": mitre["description"],
            "recommended_action": mitre["recommended_action"],
            "top_features": top_features,
            "feature_values": feature_values,
            "shap_attribution": attribution,
            "narrative": narrative,
            "raw_record": raw_record,
        }

        if self.alert_repository is not None:
            self.alert_repository.save(alert)
        else:
            self.alerts[alert_id] = alert
        return alert

    def get_alerts(self, limit: int = 100, severity: Optional[str] = None) -> list:
        if self.alert_repository is not None:
            return self.alert_repository.list(limit=limit, severity=severity)
        alerts = list(self.alerts.values())
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity]
        alerts.sort(key=lambda alert: alert["timestamp"], reverse=True)
        return alerts[:limit]

    def get_alert(self, alert_id: str) -> Optional[dict]:
        if self.alert_repository is not None:
            return self.alert_repository.get(alert_id)
        return self.alerts.get(alert_id)

    def acknowledge_alert(self, alert_id: str) -> bool:
        if self.alert_repository is not None:
            return self.alert_repository.acknowledge(alert_id)
        if alert_id not in self.alerts:
            return False
        self.alerts[alert_id]["is_acknowledged"] = True
        return True

    def get_stats(self) -> dict:
        elapsed = (datetime.now(timezone.utc) - self.stats["active_since"]).total_seconds()
        processed = self.stats["total_records_processed"]
        anomalies = self.stats["total_anomalies_detected"]
        return {
            "total_records_processed": processed,
            "total_anomalies_detected": anomalies,
            "anomaly_rate": round(anomalies / max(processed, 1), 4),
            "records_per_second": round(processed / max(elapsed, 1), 2),
            "active_since": self.stats["active_since"],
        }
