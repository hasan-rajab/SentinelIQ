"""
SentinelIQ — Federated Server
Coordinates federated training across multiple SentinelIQ nodes
using FedAvg aggregation. No raw data ever reaches the server.
"""

import sys
import argparse
from pathlib import Path

import flwr as fl

sys.path.append(str(Path(__file__).resolve().parents[1]))
from federated.strategy import SentinelFedAvg


def weighted_average(metrics: list) -> dict:
    """Aggregate evaluation metrics weighted by number of examples."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    roc_auc = sum(m.get("roc_auc", 0) * n for n, m in metrics) / total_examples
    f1 = sum(m.get("f1", 0) * n for n, m in metrics) / total_examples
    return {"roc_auc": roc_auc, "f1": f1}


def fit_weighted_average(metrics: list) -> dict:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    loss = sum(m.get("loss", 0) * n for n, m in metrics) / total_examples
    return {"loss": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--port", type=str, default="8080")
    parser.add_argument("--save-dir", type=str, default="ml/saved_models/federated")
    args = parser.parse_args()

    strategy = SentinelFedAvg(
        save_dir=args.save_dir,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_weighted_average,
    )

    print(f"\n{'='*60}")
    print(f"  SentinelIQ Federated Server")
    print(f"  Rounds: {args.rounds} | Min clients: {args.min_clients}")
    print(f"{'='*60}\n")

    history = fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print(f"\n{'='*60}")
    print("  Federated Training Complete")
    print(f"{'='*60}")
    print(f"\nRound history:")
    for round_info in strategy.get_history():
        print(f"  {round_info}")

    return history


if __name__ == "__main__":
    main()