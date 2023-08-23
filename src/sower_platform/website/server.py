import flwr as fl
import sys

port = sys.argv[1]
fl.server.start_server(server_address="[::]:{port}", config=fl.server.ServerConfig(num_rounds=3))