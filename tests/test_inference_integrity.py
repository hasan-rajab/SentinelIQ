"""Regression tests for inference integrity and label-leakage prevention."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.services.anomaly_rules import infer_anomaly_type
from backend.services.anomaly_service import AnomalyService
from backend.storage import AlertRepository
from ml.explainability.mitre_mapper import MitreMapper
from ml.fusion.ensemble import SentinelEnsemble


class DummyAutoencoder:
    def __init__(self, score: float, threshold: float = 1.0):
        self._score = score
        self.threshold = threshold
        self.feature_cols = [
            "cpu_percent",
            "mem_percent",
            "disk_read_mbps",
            "disk_write_mbps",
            "net_in_mbps",
            "net_out_mbps",
            "open_connections",
            "process_count",
        ]

    def score(self, _df):
        return np.array([self._score], dtype=float)


def metric_service(score: float, repository=None) -> AnomalyService:
    service = AnomalyService.__new__(AnomalyService)
    service.ae = DummyAutoencoder(score=score)
    service.bert = None
    service.if_network = None
    service.ae_network = None
    service.xgb_network = None
    service.ensemble = None
    service.feature_cols = service.ae.feature_cols
    service.mitre_mapper = MitreMapper()
    service.alert_repository = repository
    service.alerts = {}
    service.stats = {
        "total_records_processed": 0,
        "total_anomalies_detected": 0,
        "active_since": datetime.now(timezone.utc),
    }
    return service


class AnomalyRulesTests(unittest.TestCase):
    def test_metric_rules_cover_synthetic_patterns(self):
        self.assertEqual(infer_anomaly_type({"process_count": 900}, "metric"), "process_bomb")
        self.assertEqual(infer_anomaly_type({"open_connections": 6000}, "metric"), "connection_storm")
        self.assertEqual(infer_anomaly_type({"disk_write_mbps": 350}, "metric"), "disk_flood")
        self.assertEqual(infer_anomaly_type({"net_out_mbps": 250}, "metric"), "network_exfiltration")
        self.assertEqual(infer_anomaly_type({"mem_percent": 95}, "metric"), "memory_leak")
        self.assertEqual(infer_anomaly_type({"cpu_percent": 96}, "metric"), "cpu_spike")

    def test_network_rules_cover_synthetic_patterns(self):
        self.assertEqual(
            infer_anomaly_type({"bytes_out": 20_000_000, "dst_port": 443}, "network"),
            "data_exfiltration",
        )
        self.assertEqual(
            infer_anomaly_type(
                {"dst_port": 53, "protocol": "UDP", "packets": 300, "bytes_out": 5000},
                "network",
            ),
            "dns_tunneling",
        )
        self.assertEqual(
            infer_anomaly_type(
                {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.8", "dst_port": 445},
                "network",
            ),
            "lateral_movement",
        )
        self.assertEqual(
            infer_anomaly_type({"dst_port": 4444, "bytes_out": 500}, "network"),
            "c2_beacon",
        )
        self.assertEqual(
            infer_anomaly_type(
                {"dst_port": 22, "bytes_out": 60, "bytes_in": 0, "packets": 1},
                "network",
            ),
            "port_scan",
        )

    def test_log_rules_cover_attack_shapes(self):
        cases = {
            "sshd: Failed password for root": "brute_force",
            "sshd: Invalid user admin": "invalid_user",
            "GET /../../../etc/passwd": "path_traversal",
            "UNION SELECT * FROM users--": "sqli",
            "GET /shell.php?cmd=whoami": "web_shell",
            "sudo: command not allowed": "privilege_escalation",
            "sshd: authentication failure": "auth_failure",
            "GET /admin/config.php": "unauthorized_access",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(infer_anomaly_type({"message": message}, "log"), expected)

    def test_rules_ignore_simulator_labels(self):
        normal_record_with_fake_labels = {
            "cpu_percent": 20,
            "mem_percent": 40,
            "is_anomaly": True,
            "anomaly_type": "cpu_spike",
        }
        self.assertEqual(
            infer_anomaly_type(normal_record_with_fake_labels, "metric"),
            "unknown",
        )


class InferenceDecisionTests(unittest.TestCase):
    def _metric_record(self, cpu: float = 96.0) -> dict:
        return {
            "cpu_percent": cpu,
            "mem_percent": 50,
            "disk_read_mbps": 0,
            "disk_write_mbps": 0,
            "net_in_mbps": 1,
            "net_out_mbps": 1,
            "open_connections": 10,
            "process_count": 100,
        }

    def test_ground_truth_cannot_force_alert_when_model_score_is_normal(self):
        service = metric_service(score=0.25)
        record = self._metric_record()
        record.update({"is_anomaly": True, "anomaly_type": "cpu_spike"})
        self.assertIsNone(service.process_record(record, "metric"))
        self.assertEqual(service.stats["total_anomalies_detected"], 0)

    def test_model_can_alert_when_simulator_label_says_normal(self):
        service = metric_service(score=2.0)
        record = self._metric_record()
        record.update({"is_anomaly": False, "anomaly_type": None})
        alert = service.process_record(record, "metric")
        self.assertIsNotNone(alert)
        self.assertEqual(alert["anomaly_type"], "cpu_spike")
        self.assertNotIn("is_anomaly", alert["raw_record"])
        self.assertNotIn("anomaly_type", alert["raw_record"])

    def test_alert_persists_across_repository_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sentineliq.db"
            url = f"sqlite:///{db_path}"
            repository = AlertRepository(url)
            service = metric_service(score=2.0, repository=repository)
            alert = service.process_record(self._metric_record(), "metric")
            repository.engine.dispose()

            reopened = AlertRepository(url)
            stored = reopened.get(alert["id"])
            self.assertIsNotNone(stored)
            self.assertEqual(stored["id"], alert["id"])
            self.assertEqual(stored["anomaly_type"], "cpu_spike")
            reopened.engine.dispose()


class EnsemblePersistenceTests(unittest.TestCase):
    def test_network_threshold_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ensemble.json"
            ensemble = SentinelEnsemble(threshold=0.71, network_threshold=0.43)
            ensemble.save(str(path))
            loaded = SentinelEnsemble.load(str(path))
            self.assertAlmostEqual(loaded.threshold, 0.71)
            self.assertAlmostEqual(loaded.network_threshold, 0.43)


if __name__ == "__main__":
    unittest.main()
