"""
SentinelIQ — Ensemble Fusion
Combines anomaly scores from IsolationForest, Autoencoder, and BERT
into a single unified score with configurable weighting strategy.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional


class SentinelEnsemble:
    """
    Fuses anomaly scores from three modality-specific models:
      - isolation_forest : metrics + network (tabular)
      - autoencoder      : metrics (reconstruction)
      - bert_log         : logs (sequence classification)

    Strategies:
      - weighted_avg : weighted average of normalized scores
      - max          : take the max score across models (high recall)
      - vote         : majority vote on binary predictions
    """

    def __init__(
        self,
        weights: dict = None,
        strategy: str = "weighted_avg",
        threshold: float = 0.5,
    ):
        self.weights = weights or {
            "isolation_forest": 0.30,
            "autoencoder": 0.30,
            "bert_log": 0.40,
        }
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.strategy = strategy
        self.threshold = threshold

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]. Single-value arrays pass through
        unchanged since there's no batch range to normalize against (this is
        the live single-record inference path)."""
        if len(scores) == 1:
            return scores
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-8:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    def fuse_network(
        self,
        if_scores: np.ndarray,
        ae_scores: np.ndarray,
        if_weight: float = 0.5,
        ae_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Dedicated fusion for network modality: combines Isolation Forest
        and Autoencoder scores. Kept separate from the metrics/log fuse()
        path since network has its own weighting and was added later
        to close the recall gap found in production validation.
        """
        assert abs(if_weight + ae_weight - 1.0) < 1e-6, "Network weights must sum to 1.0"
        if_norm = self._normalize(if_scores)
        ae_norm = self._normalize(ae_scores)
        return if_weight * if_norm + ae_weight * ae_norm

    def fuse(
        self,
        if_scores: Optional[np.ndarray] = None,
        ae_scores: Optional[np.ndarray] = None,
        bert_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns unified anomaly scores in [0, 1].
        Any missing modality is skipped and weights are renormalized.
        """
        available = {}
        if if_scores is not None:
            available["isolation_forest"] = self._normalize(if_scores)
        if ae_scores is not None:
            available["autoencoder"] = self._normalize(ae_scores)
        if bert_scores is not None:
            available["bert_log"] = self._normalize(bert_scores)

        if not available:
            raise ValueError("At least one score array must be provided.")

        if self.strategy == "weighted_avg":
            total_weight = sum(self.weights[k] for k in available)
            fused = sum(
                self.weights[k] / total_weight * available[k]
                for k in available
            )

        elif self.strategy == "max":
            fused = np.stack(list(available.values())).max(axis=0)

        elif self.strategy == "vote":
            votes = np.stack([
                (s >= 0.5).astype(int) for s in available.values()
            ])
            fused = votes.mean(axis=0)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return fused

    def predict(
        self,
        if_scores: Optional[np.ndarray] = None,
        ae_scores: Optional[np.ndarray] = None,
        bert_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns binary predictions: 1 = anomaly, 0 = normal."""
        fused = self.fuse(if_scores, ae_scores, bert_scores)
        return (fused >= self.threshold).astype(int)

    def fuse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects df with columns: if_score, ae_score, bert_score (any subset).
        Returns df with added columns: fused_score, is_anomaly_pred.
        """
        df = df.copy()
        if_scores  = df["if_score"].values  if "if_score"   in df.columns else None
        ae_scores  = df["ae_score"].values  if "ae_score"   in df.columns else None
        bert_scores = df["bert_score"].values if "bert_score" in df.columns else None

        df["fused_score"] = self.fuse(if_scores, ae_scores, bert_scores)
        df["is_anomaly_pred"] = (df["fused_score"] >= self.threshold).astype(int)
        return df

    def evaluate(self, y_true: np.ndarray, fused_scores: np.ndarray) -> dict:
        from sklearn.metrics import (
            classification_report, roc_auc_score,
            average_precision_score, confusion_matrix,
        )

        y_pred = (fused_scores >= self.threshold).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        try:
            roc_auc = roc_auc_score(y_true, fused_scores)
            avg_precision = average_precision_score(y_true, fused_scores)
        except ValueError:
            roc_auc = avg_precision = 0.0

        metrics = {
            "roc_auc": round(roc_auc, 4),
            "avg_precision": round(avg_precision, 4),
            "precision": round(report.get("1", {}).get("precision", 0), 4),
            "recall": round(report.get("1", {}).get("recall", 0), 4),
            "f1": round(report.get("1", {}).get("f1-score", 0), 4),
            "accuracy": round(report.get("accuracy", 0), 4),
            "confusion_matrix": cm.tolist(),
            "strategy": self.strategy,
            "threshold": self.threshold,
            "weights": self.weights,
        }

        print(f"\n[Ensemble] Evaluation Results (strategy={self.strategy}):")
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "weights"):
                print(f"  {k:20}: {v}")

        return metrics

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "weights": self.weights,
                "strategy": self.strategy,
                "threshold": self.threshold,
            }, f, indent=2)
        print(f"[Ensemble] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SentinelEnsemble":
        with open(path) as f:
            cfg = json.load(f)
        obj = cls(**cfg)
        print(f"[Ensemble] Loaded from {path}")
        return obj
