"""
SentinelIQ — Federated Strategy
Custom FedAvg strategy for aggregating Autoencoder weights
across multiple federated nodes without sharing raw data.
"""

from typing import List, Tuple, Optional, Dict
import flwr as fl
from flwr.common import (
    FitRes, EvaluateRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np


class SentinelFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg that:
    - Logs per-round aggregated metrics
    - Tracks anomaly detection performance across rounds
    - Saves global model checkpoint after each round
    """

    def __init__(self, save_dir: str = "ml/saved_models/federated", **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.round_history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        n_clients = len(results)
        avg_loss = np.mean([r.metrics.get("loss", 0) for _, r in results])
        total_examples = sum(r.num_examples for _, r in results)

        print(f"\n[Round {server_round}] Aggregated {n_clients} clients | "
              f"avg_loss={avg_loss:.5f} | total_examples={total_examples}")

        self.round_history.append({
            "round": server_round,
            "n_clients": n_clients,
            "avg_loss": float(avg_loss),
            "total_examples": total_examples,
        })

        # Save checkpoint
        if aggregated_params is not None:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            ndarrays = parameters_to_ndarrays(aggregated_params)
            np.savez(f"{self.save_dir}/round_{server_round}_weights.npz", *ndarrays)

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if results:
            avg_roc_auc = np.mean([r.metrics.get("roc_auc", 0) for _, r in results])
            avg_f1 = np.mean([r.metrics.get("f1", 0) for _, r in results])
            print(f"[Round {server_round}] Eval — loss={aggregated_loss:.5f} | "
                  f"roc_auc={avg_roc_auc:.4f} | f1={avg_f1:.4f}")

        return aggregated_loss, metrics

    def get_history(self) -> list:
        return self.round_history