from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from time import sleep
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print("Aggregate accuracy: ", sum(accuracies) / sum(examples))
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round):
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2
    }
    return config

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average, min_fit_clients = 6, min_available_clients=6, on_fit_config_fn=fit_config) #min_fit_clients, min_evaluate_clients=7

sleep(10)
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(
        num_rounds=20,
    ),
    strategy=strategy,
)
