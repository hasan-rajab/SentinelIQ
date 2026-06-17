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

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame = None):
        X = self.scaler.fit_transform(self._extract(df))
        X_t = self._to_tensor(X)
        dataset = TensorDataset(X_t, X_t)
        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 64), shuffle=True, drop_last=True)

        epochs = self.config.get("epochs", 50)
        print(f"[Autoencoder] Training on {len(df)} samples | device={self.device} | epochs={epochs}")

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
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | loss={total_loss/len(loader):.5f}")

        # Set threshold on training reconstruction errors
        errors = self._reconstruction_errors(X_t)
        pct = self.config.get("threshold_percentile", 95)
        self.threshold = np.percentile(errors, pct)
        self.is_fitted = True
        print(f"[Autoencoder] Fitted | threshold (p{pct})={self.threshold:.5f}")
        return self

    def _reconstruction_errors(self, X_tensor: torch.Tensor) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            recon = self.net(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
        return errors

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self._extract(df))
        return self._reconstruction_errors(self._to_tensor(X))

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
        obj.is_fitted = True
        print(f"[Autoencoder] Loaded from {save_dir}/{name}_*")
        return obj
