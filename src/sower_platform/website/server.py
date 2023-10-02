import flwr as fl
import sys

port = sys.argv[1]
addr = "[::]:"+port
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3)

fl.server.start_server(strategy=strategy, server_address=addr, config=fl.server.ServerConfig(num_rounds=10))