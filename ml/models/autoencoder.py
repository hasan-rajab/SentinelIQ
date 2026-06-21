"""
SentinelIQ — Autoencoder Model
Reconstruction-based anomaly detection for system metrics.
Anomalies produce higher reconstruction error than normal patterns.
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve,
)


# ── Architecture ──────────────────────────────────────────────────────────────
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()

        # Encoder: input_dim → hidden_dims[0] → ... → hidden_dims[-1]
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: mirror path back to input_dim
        dec_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        dec_layers = []
        for h in dec_dims:
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        # Replace last ReLU with identity for output layer
        dec_layers[-1] = nn.Identity()
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


# ── Wrapper ───────────────────────────────────────────────────────────────────
class SentinelAutoencoder:
    def __init__(self, config: dict, feature_cols: list):
        self.config = config
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.threshold = None
        self.score_min = None
        self.score_max = None
        self.score_clip = None
        self.is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = AutoencoderNet(
            input_dim=config.get("input_dim", len(feature_cols)),
            hidden_dims=config.get("hidden_dims", [64, 32, 16, 32, 64]),
            dropout=config.get("dropout", 0.2),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        self.criterion = nn.MSELoss()

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(X).to(self.device)

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].fillna(0).values

    def _fit_threshold(self, y_true: np.ndarray, scores: np.ndarray, min_recall: float = 0.90):
        """Set threshold via PR-curve F1 maximization on labeled validation data.
        Clips at normal-traffic p99*2 so extreme anomaly outliers do not push
        the threshold above borderline anomalies. min_recall enforces a recall
        floor so the PR-curve cannot sacrifice borderline cases for precision.
        """
        normal_scores = scores[y_true == 0]
        normal_p99 = float(np.percentile(normal_scores, 99))
        self.score_clip = normal_p99 * 2
        scores_clipped = np.clip(scores, 0, self.score_clip)

        precision, recall, thresholds = precision_recall_curve(y_true, scores_clipped)
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)

        valid = recall[:-1] >= min_recall
        if valid.any():
            best_idx = int(np.argmax(np.where(valid, f1_scores, 0.0)))
        else:
            best_idx = int(np.argmax(f1_scores))

        self.threshold = float(thresholds[best_idx])
        self.score_min = float(scores_clipped.min())
        self.score_max = float(scores_clipped.max())
        print(
            f"[Autoencoder] Threshold (PR-F1, min_recall={min_recall})={self.threshold:.5f} | "
            f"score range=[{self.score_min:.5f}, {self.score_max:.5f}] | "
            f"normal_p99={normal_p99:.5f} | score_clip={self.score_clip:.5f}"
        )

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        """Map raw reconstruction errors to [0, 1] using validation score bounds."""
        if self.score_min is None or self.score_max is None:
            return scores
        return np.clip(
            (scores - self.score_min) / (self.score_max - self.score_min + 1e-8),
            0, 1,
        )

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame = None, threshold_df: pd.DataFrame = None, min_recall: float = 0.90):
        X = self.scaler.fit_transform(self._extract(df))
        X_t = self._to_tensor(X)
        dataset = TensorDataset(X_t, X_t)
        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 64), shuffle=True, drop_last=True)

        epochs = self.config.get("epochs", 100)
        print(f"[Autoencoder] Training on {len(df)} samples | device={self.device} | epochs={epochs}")

        X_val_t = None
        if val_df is not None:
            X_val_scaled = self.scaler.transform(self._extract(val_df))
            X_val_t = self._to_tensor(X_val_scaled)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5,
        )

        self.net.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for xb, _ in loader:
                self.optimizer.zero_grad()
                recon = self.net(xb)
                loss = self.criterion(recon, xb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss_str = ""
            if X_val_t is not None:
                self.net.eval()
                with torch.no_grad():
                    val_recon = self.net(X_val_t)
                    val_loss = self.criterion(val_recon, X_val_t).item()
                scheduler.step(val_loss)
                val_loss_str = f" | val_loss={val_loss:.5f}"
                self.net.train()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | loss={total_loss/len(loader):.5f}{val_loss_str}")

        # Threshold calibration — uses threshold_df if provided, else val_df, else percentile on train
        calib_df = threshold_df if threshold_df is not None else val_df

        if calib_df is not None and "is_anomaly" in calib_df.columns and calib_df["is_anomaly"].sum() > 0:
            y_calib = calib_df["is_anomaly"].astype(int).values
            calib_scores = self._reconstruction_errors(
                self._to_tensor(self.scaler.transform(self._extract(calib_df)))
            )
            self._fit_threshold(y_calib, calib_scores, min_recall=min_recall)
        else:
            errors = self._reconstruction_errors(self._to_tensor(
                self.scaler.transform(self._extract(df))
            ))
            pct = self.config.get("threshold_percentile", 95)
            self.threshold = float(np.percentile(errors, pct))
            self.score_min = float(errors.min())
            self.score_max = float(errors.max())
            print(f"[Autoencoder] Fitted | threshold (p{pct})={self.threshold:.5f}")

        self.is_fitted = True
        return self

    def _reconstruction_errors(self, X_tensor: torch.Tensor) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            recon = self.net(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
        return errors

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self._extract(df))
        raw = self._reconstruction_errors(self._to_tensor(X))
        if hasattr(self, "score_clip") and self.score_clip is not None:
            raw = np.clip(raw, 0, self.score_clip)
        return raw

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(int)

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

        print(f"\n[Autoencoder] Evaluation Results:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k:20}: {v}")

        return metrics

    def save(self, save_dir: str, name: str = "autoencoder"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), f"{save_dir}/{name}_weights.pt")
        joblib.dump(self.scaler, f"{save_dir}/{name}_scaler.joblib")
        meta = {
            "feature_cols": self.feature_cols,
            "threshold": float(self.threshold),
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_clip": self.score_clip,
            "config": self.config,
        }
        with open(f"{save_dir}/{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Autoencoder] Saved to {save_dir}/{name}_*")

    @classmethod
    def load(cls, save_dir: str, name: str = "autoencoder") -> "SentinelAutoencoder":
        with open(f"{save_dir}/{name}_meta.json") as f:
            meta = json.load(f)
        obj = cls(config=meta["config"], feature_cols=meta["feature_cols"])
        obj.net.load_state_dict(torch.load(f"{save_dir}/{name}_weights.pt", map_location=obj.device))
        obj.scaler = joblib.load(f"{save_dir}/{name}_scaler.joblib")
        obj.threshold = meta["threshold"]
        obj.score_min = meta.get("score_min")
        obj.score_max = meta.get("score_max")
        obj.score_clip = meta.get("score_clip")
        obj.is_fitted = True
        print(f"[Autoencoder] Loaded from {save_dir}/{name}_*")
        return obj