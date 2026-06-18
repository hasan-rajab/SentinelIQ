"""
SentinelIQ — Federated Client
Each client trains the Autoencoder locally on its own data partition
and only shares model weights with the server, never raw data.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import flwr as fl
from sklearn.metrics import roc_auc_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ml.models.autoencoder import SentinelAutoencoder, AutoencoderNet


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in open(path)])


def get_model_params(net: torch.nn.Module) -> list:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_model_params(net: torch.nn.Module, parameters: list):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


class SentinelClient(fl.client.NumPyClient):
    """
    A single federated node. Holds a private partition of metrics data
    and trains a local Autoencoder, sharing only weights with the server.
    """

    def __init__(self, client_id: int, data_path: str, config: dict, feature_cols: list):
        self.client_id = client_id
        self.config = config
        self.feature_cols = feature_cols

        df = load_jsonl(data_path)
        # Partition data by client_id (simulating data residing on different nodes)
        n_clients = config.get("n_clients", 3)
        partition_size = len(df) // n_clients
        start = client_id * partition_size
        end = start + partition_size if client_id < n_clients - 1 else len(df)
        self.df = df.iloc[start:end].reset_index(drop=True)

        self.y = self.df["is_anomaly"].astype(int).values
        self.normal_df = self.df[self.y == 0]

        ae_cfg = {**config, "input_dim": len(feature_cols)}
        self.model = SentinelAutoencoder(config=ae_cfg, feature_cols=feature_cols)

        print(f"[Client {client_id}] Loaded {len(self.df)} records | "
              f"{self.y.sum()} anomalies | training on {len(self.normal_df)} normal samples")

    def get_parameters(self, config) -> list:
        return get_model_params(self.model.net)

    def fit(self, parameters: list, config: dict) -> tuple:
        set_model_params(self.model.net, parameters)
        self.model.fit(self.normal_df)

        loss = self._compute_loss()
        return get_model_params(self.model.net), len(self.normal_df), {"loss": loss}

    def evaluate(self, parameters: list, config: dict) -> tuple:
        set_model_params(self.model.net, parameters)

        scores = self.model.score(self.df)
        if self.model.threshold is None:
            self.model.threshold = float(np.percentile(scores, 92))

        y_pred = (scores >= self.model.threshold).astype(int)

        try:
            roc_auc = roc_auc_score(self.y, scores)
        except ValueError:
            roc_auc = 0.0
        f1 = f1_score(self.y, y_pred, zero_division=0)
        loss = self._compute_loss()

        print(f"[Client {self.client_id}] Eval — roc_auc={roc_auc:.4f} | f1={f1:.4f}")

        return loss, len(self.df), {"roc_auc": roc_auc, "f1": f1}

    def _compute_loss(self) -> float:
        X = self.model.scaler.transform(self.normal_df[self.feature_cols].fillna(0).values) \
            if hasattr(self.model.scaler, "mean_") else \
            self.normal_df[self.feature_cols].fillna(0).values
        X_t = torch.FloatTensor(X).to(self.model.device)
        errors = self.model._reconstruction_errors(X_t)
        return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--server", type=str, default="localhost:8080")
    parser.add_argument("--data", type=str, default="data/simulated/metrics.jsonl")
    parser.add_argument("--config", type=str, default="configs/federated_config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    feature_cols = model_cfg["isolation_forest"]["features"]
    ae_config = {**model_cfg["autoencoder"], **cfg.get("client", {})}

    client = SentinelClient(
        client_id=args.client_id,
        data_path=args.data,
        config=ae_config,
        feature_cols=feature_cols,
    )

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()