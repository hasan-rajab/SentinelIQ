"""
SentinelIQ — Federated Simulation Runner
Runs server + N clients in a single process using Flower's simulation API.
This is the Kaggle-friendly way to demo federated learning without
needing separate processes or open ports.
"""

import sys
import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.client import Client, ClientApp
from flwr.server import ServerApp, ServerConfig
from flwr.simulation import run_simulation
from flwr.common import Context

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ml.models.autoencoder import SentinelAutoencoder
from federated.strategy import SentinelFedAvg
from federated.client import SentinelClient, get_model_params


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in open(path)])


def make_client_fn(data_path: str, model_cfg: dict, fed_cfg: dict, feature_cols: list):
    def client_fn(context: Context) -> Client:
        client_id = int(context.node_config.get("partition-id", 0))
        ae_config = {**model_cfg["autoencoder"], **fed_cfg.get("client", {})}
        client = SentinelClient(
            client_id=client_id,
            data_path=data_path,
            config=ae_config,
            feature_cols=feature_cols,
        )
        return client.to_client()
    return client_fn


def run_federated_simulation(
    data_path: str = "data/simulated/metrics.jsonl",
    model_config_path: str = "configs/model_config.yaml",
    federated_config_path: str = "configs/federated_config.yaml",
    save_dir: str = "ml/saved_models/federated",
):
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(federated_config_path) as f:
        fed_cfg = yaml.safe_load(f)

    feature_cols = model_cfg["isolation_forest"]["features"]
    n_clients = fed_cfg["client"]["n_clients"]
    n_rounds = fed_cfg["server"]["rounds"]

    print(f"\n{'='*60}")
    print(f"  SentinelIQ Federated Simulation")
    print(f"  Clients: {n_clients} | Rounds: {n_rounds}")
    print(f"{'='*60}\n")

    client_fn = make_client_fn(data_path, model_cfg, fed_cfg, feature_cols)
    client_app = ClientApp(client_fn=client_fn)

    def weighted_average(metrics):
        total = sum(n for n, _ in metrics)
        roc_auc = sum(m.get("roc_auc", 0) * n for n, m in metrics) / total
        f1 = sum(m.get("f1", 0) * n for n, m in metrics) / total
        return {"roc_auc": roc_auc, "f1": f1}

    def fit_weighted_average(metrics):
        total = sum(n for n, _ in metrics)
        loss = sum(m.get("loss", 0) * n for n, m in metrics) / total
        return {"loss": loss}

    strategy = SentinelFedAvg(
        save_dir=save_dir,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_weighted_average,
    )

    def server_fn(context: Context):
        config = ServerConfig(num_rounds=n_rounds)
        return fl.server.ServerAppComponents(strategy=strategy, config=config)

    server_app = ServerApp(server_fn=server_fn)

    backend_config = {"client_resources": fed_cfg["simulation"]["client_resources"]}

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config=backend_config,
    )

    print(f"\n{'='*60}")
    print("  Federated Simulation Complete")
    print(f"{'='*60}")
    for r in strategy.get_history():
        print(f"  {r}")

    # Save round history
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/round_history.json", "w") as f:
        json.dump(strategy.get_history(), f, indent=2)

    return strategy.get_history()


if __name__ == "__main__":
    run_federated_simulation()