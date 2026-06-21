"""
SentinelIQ — XGBoost Network Classifier
Supervised anomaly detection for network flows.
Trained on labeled data (is_anomaly) to classify known attack patterns:
  - port_scan, c2_beacon, data_exfiltration, dns_tunneling, lateral_movement

Complements the Autoencoder which handles unknown/novel attack patterns
via reconstruction error. Both feed into the network ensemble.
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("Install xgboost: pip install xgboost")


class SentinelXGBoost:
    """
    Supervised binary classifier for network flow anomaly detection.
    Uses predict_proba output as anomaly score so it integrates cleanly
    with the ensemble fusion layer.
    """

    def __init__(self, config: dict, feature_cols: list):
        self.config = config
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.score_min = 0.0
        self.score_max = 1.0
        self.is_fitted = False

        self.model = XGBClassifier(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            scale_pos_weight=config.get("scale_pos_weight", 10),
            eval_metric="aucpr",
            random_state=config.get("random_state", 42),
            n_jobs=-1,
        )

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return df[self.feature_cols].fillna(0).values

    def _fit_threshold(self, y_true: np.ndarray, scores: np.ndarray, min_recall: float = 0.75):
        """Set threshold via PR-curve, targeting min_recall for security use case."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)

        valid = recall[:-1] >= min_recall
        if valid.any():
            masked_f1 = np.where(valid, f1, 0.0)
            best_idx = int(np.argmax(masked_f1))
        else:
            best_idx = int(np.argmax(f1))

        self.threshold = float(thresholds[best_idx])
        print(
            f"[XGBoost] Threshold (PR-F1, min_recall={min_recall})={self.threshold:.5f} "
            f"| F1={f1[best_idx]:.4f} | Recall={recall[best_idx]:.4f}"
        )

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray,
            val_df: pd.DataFrame = None, y_val: np.ndarray = None):
        X_train = self.scaler.fit_transform(self._extract(train_df))

        eval_set = None
        if val_df is not None and y_val is not None:
            X_val = self.scaler.transform(self._extract(val_df))
            eval_set = [(X_train, y_train), (X_val, y_val)]

        print(f"[XGBoost] Training on {len(train_df)} samples "
              f"| anomalies={y_train.sum()} | normal={(y_train == 0).sum()}")

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        calib_X = self.scaler.transform(self._extract(val_df)) if val_df is not None else X_train
        calib_y = y_val if y_val is not None else y_train
        calib_scores = self.model.predict_proba(calib_X)[:, 1]
        self._fit_threshold(calib_y, calib_scores)

        self.is_fitted = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Returns anomaly probability scores in [0, 1]."""
        X = self.scaler.transform(self._extract(df))
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(int)

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        """Scores are already [0, 1] probabilities — pass through."""
        return np.clip(scores, 0, 1)

    def evaluate(self, df: pd.DataFrame, y_true: np.ndarray) -> dict:
        scores = self.score(df)
        y_pred = (scores >= self.threshold).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        roc_auc = roc_auc_score(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "roc_auc": round(roc_auc, 4),
            "avg_precision": round(avg_precision, 4),
            "precision": round(report.get("1", {}).get("precision", 0), 4),
            "recall": round(report.get("1", {}).get("recall", 0), 4),
            "f1": round(report.get("1", {}).get("f1-score", 0), 4),
            "accuracy": round(report.get("accuracy", 0), 4),
            "confusion_matrix": cm.tolist(),
            "threshold": round(float(self.threshold), 6),
            "n_samples": len(df),
            "n_anomalies_true": int(y_true.sum()),
            "n_anomalies_pred": int(y_pred.sum()),
        }

        print(f"\n[XGBoost] Evaluation Results:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k:20}: {v}")

        return metrics

    def feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, save_dir: str, name: str = "xgboost_network"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_model(f"{save_dir}/{name}_model.json")
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
        print(f"[XGBoost] Saved to {save_dir}/{name}_*")

    @classmethod
    def load(cls, save_dir: str, name: str = "xgboost_network") -> "SentinelXGBoost":
        with open(f"{save_dir}/{name}_meta.json") as f:
            meta = json.load(f)
        obj = cls(config=meta["config"], feature_cols=meta["feature_cols"])
        obj.model.load_model(f"{save_dir}/{name}_model.json")
        obj.scaler = joblib.load(f"{save_dir}/{name}_scaler.joblib")
        obj.threshold = meta["threshold"]
        obj.score_min = meta.get("score_min", 0.0)
        obj.score_max = meta.get("score_max", 1.0)
        obj.is_fitted = True
        print(f"[XGBoost] Loaded from {save_dir}/{name}_*")
        return obj