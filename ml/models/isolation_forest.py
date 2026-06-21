"""
SentinelIQ — Isolation Forest Model
Tabular anomaly detection for system metrics and network flows.
"""

import numpy as np
import pandas as pd
import joblib
import json
import yaml
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)


class SentinelIsolationForest:
    """
    Wraps sklearn IsolationForest with:
    - StandardScaler preprocessing
    - Configurable feature selection
    - Threshold tuning via contamination
    - Save / load with metadata
    """

    def __init__(self, config: dict, feature_cols: list):
        self.config = config
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=config.get("n_estimators", 200),
            max_samples=config.get("max_samples", "auto"),
            contamination=config.get("contamination", 0.075),
            max_features=config.get("max_features", 1.0),
            bootstrap=config.get("bootstrap", False),
            random_state=config.get("random_state", 42),
            n_jobs=-1,
        )
        self.threshold = None
        self.score_min = None
        self.score_max = None
        self.is_fitted = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return df[self.feature_cols].fillna(0).values

    def fit(self, df: pd.DataFrame):
        X = self._extract_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        # Compute anomaly scores on training data to set threshold
        scores = -self.model.score_samples(X_scaled)   # higher = more anomalous
        self.threshold = np.percentile(scores, 92)
        self.score_min = float(scores.min())
        self.score_max = float(scores.max())
        self.is_fitted = True
        print(f"[IsolationForest] Fitted on {len(df)} samples | threshold={self.threshold:.4f}")
        return self

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to [0, 1] using training score bounds."""
        if self.score_min is None or self.score_max is None:
            return scores
        return np.clip(
            (scores - self.score_min) / (self.score_max - self.score_min + 1e-8),
            0, 1,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns binary labels: 1 = anomaly, 0 = normal."""
        scores = self.score(df)
        return (scores >= self.threshold).astype(int)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Returns raw anomaly scores (higher = more anomalous)."""
        X = self._extract_features(df)
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)

    def evaluate(self, df: pd.DataFrame, y_true: np.ndarray) -> dict:
        scores = self.score(df)
        y_pred = (scores >= self.threshold).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "roc_auc": round(roc_auc, 4),
            "avg_precision": round(avg_precision, 4),
            "precision": round(report["1"]["precision"], 4) if "1" in report else 0,
            "recall": round(report["1"]["recall"], 4) if "1" in report else 0,
            "f1": round(report["1"]["f1-score"], 4) if "1" in report else 0,
            "accuracy": round(report["accuracy"], 4),
            "confusion_matrix": cm.tolist(),
            "threshold": round(float(self.threshold), 4),
            "n_samples": len(df),
            "n_anomalies_true": int(y_true.sum()),
            "n_anomalies_pred": int(y_pred.sum()),
        }

        print(f"\n[IsolationForest] Evaluation Results:")
        print(f"  ROC-AUC       : {metrics['roc_auc']}")
        print(f"  Avg Precision : {metrics['avg_precision']}")
        print(f"  Precision     : {metrics['precision']}")
        print(f"  Recall        : {metrics['recall']}")
        print(f"  F1            : {metrics['f1']}")
        print(f"  Accuracy      : {metrics['accuracy']}")

        return metrics

    def save(self, save_dir: str, name: str = "isolation_forest"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, f"{save_dir}/{name}_model.joblib")
        joblib.dump(self.scaler, f"{save_dir}/{name}_scaler.joblib")
        meta = {
            "feature_cols": self.feature_cols,
            "threshold": float(self.threshold),
            "score_min": self.score_min,
            "score_max": self.score_max,
            "config": self.config,
        }
        with open(f"{save_dir}/{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[IsolationForest] Saved to {save_dir}/{name}_*")

    @classmethod
    def load(cls, save_dir: str, name: str = "isolation_forest") -> "SentinelIsolationForest":
        with open(f"{save_dir}/{name}_meta.json") as f:
            meta = json.load(f)
        obj = cls(config=meta["config"], feature_cols=meta["feature_cols"])
        obj.model = joblib.load(f"{save_dir}/{name}_model.joblib")
        obj.scaler = joblib.load(f"{save_dir}/{name}_scaler.joblib")
        obj.threshold = meta["threshold"]
        obj.score_min = meta.get("score_min")
        obj.score_max = meta.get("score_max")
        obj.is_fitted = True
        print(f"[IsolationForest] Loaded from {save_dir}/{name}_*")
        return obj