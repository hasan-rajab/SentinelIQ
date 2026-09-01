"""SentinelIQ ensemble fusion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SentinelEnsemble:
    """Fuse modality/model anomaly scores with calibrated thresholds."""

    def __init__(
        self,
        weights: dict | None = None,
        strategy: str = "weighted_avg",
        threshold: float = 0.5,
        network_threshold: float = 0.5,
    ):
        self.weights = weights or {
            "isolation_forest": 0.30,
            "autoencoder": 0.30,
            "bert_log": 0.40,
        }
        if abs(sum(self.weights.values()) - 1.0) >= 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.strategy = strategy
        self.threshold = threshold
        self.network_threshold = network_threshold

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize batch scores; preserve single-record scores."""
        if len(scores) == 1:
            return scores
        minimum, maximum = scores.min(), scores.max()
        if maximum - minimum < 1e-8:
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    def fuse_network_xgb(
        self,
        xgb_scores: np.ndarray,
        ae_scores: np.ndarray,
        ae_threshold: float,
        xgb_weight: float = 0.7,
        ae_weight: float = 0.3,
    ) -> np.ndarray:
        """Fuse supervised XGBoost probabilities with AE reconstruction signal."""
        if abs(xgb_weight + ae_weight - 1.0) >= 1e-6:
            raise ValueError("Network weights must sum to 1.0")

        xgb_norm = np.clip(xgb_scores, 0, 1)
        ae_ratio = np.clip(ae_scores / max(ae_threshold, 1e-8), 0, 3) / 3
        fused = xgb_weight * xgb_norm + ae_weight * ae_ratio
        return np.clip(fused, 0, 1)

    def fuse_network(
        self,
        if_scores: np.ndarray,
        ae_scores: np.ndarray,
        if_threshold: float,
        ae_threshold: float,
        if_weight: float = 0.5,
        ae_weight: float = 0.5,
    ) -> np.ndarray:
        """Fuse IF and AE network scores by calibrated threshold ratios."""
        if abs(if_weight + ae_weight - 1.0) >= 1e-6:
            raise ValueError("Network weights must sum to 1.0")

        if_ratio = np.clip(if_scores / max(if_threshold, 1e-8), 0, 3)
        ae_ratio = np.clip(ae_scores / max(ae_threshold, 1e-8), 0, 3)
        fused = if_weight * if_ratio + ae_weight * ae_ratio
        return np.clip(fused / 2, 0, 1)

    def fuse(
        self,
        if_scores: Optional[np.ndarray] = None,
        ae_scores: Optional[np.ndarray] = None,
        bert_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a unified anomaly score for any available model signals."""
        available: dict[str, np.ndarray] = {}
        if if_scores is not None:
            available["isolation_forest"] = self._normalize(if_scores)
        if ae_scores is not None:
            available["autoencoder"] = self._normalize(ae_scores)
        if bert_scores is not None:
            available["bert_log"] = self._normalize(bert_scores)

        if not available:
            raise ValueError("At least one score array must be provided.")

        if self.strategy == "weighted_avg":
            total_weight = sum(self.weights[key] for key in available)
            return sum(
                self.weights[key] / total_weight * value
                for key, value in available.items()
            )
        if self.strategy == "max":
            return np.stack(list(available.values())).max(axis=0)
        if self.strategy == "vote":
            votes = np.stack([(score >= 0.5).astype(int) for score in available.values()])
            return votes.mean(axis=0)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict(
        self,
        if_scores: Optional[np.ndarray] = None,
        ae_scores: Optional[np.ndarray] = None,
        bert_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return (self.fuse(if_scores, ae_scores, bert_scores) >= self.threshold).astype(int)

    def fuse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if_scores = result["if_score"].values if "if_score" in result.columns else None
        ae_scores = result["ae_score"].values if "ae_score" in result.columns else None
        bert_scores = result["bert_score"].values if "bert_score" in result.columns else None
        result["fused_score"] = self.fuse(if_scores, ae_scores, bert_scores)
        result["is_anomaly_pred"] = (result["fused_score"] >= self.threshold).astype(int)
        return result

    def evaluate(self, y_true: np.ndarray, fused_scores: np.ndarray) -> dict:
        from sklearn.metrics import (
            average_precision_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )

        y_pred = (fused_scores >= self.threshold).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred)
        try:
            roc_auc = roc_auc_score(y_true, fused_scores)
            avg_precision = average_precision_score(y_true, fused_scores)
        except ValueError:
            roc_auc = avg_precision = 0.0

        return {
            "roc_auc": round(roc_auc, 4),
            "avg_precision": round(avg_precision, 4),
            "precision": round(report.get("1", {}).get("precision", 0), 4),
            "recall": round(report.get("1", {}).get("recall", 0), 4),
            "f1": round(report.get("1", {}).get("f1-score", 0), 4),
            "accuracy": round(report.get("accuracy", 0), 4),
            "confusion_matrix": matrix.tolist(),
            "strategy": self.strategy,
            "threshold": self.threshold,
            "network_threshold": self.network_threshold,
            "weights": self.weights,
        }

    def save(self, path: str) -> None:
        """Persist every parameter needed to reproduce serving decisions."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "weights": self.weights,
                    "strategy": self.strategy,
                    "threshold": self.threshold,
                    "network_threshold": self.network_threshold,
                },
                output_file,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "SentinelEnsemble":
        with open(path, encoding="utf-8") as input_file:
            config = json.load(input_file)
        return cls(**config)
