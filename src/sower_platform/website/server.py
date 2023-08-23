import flwr as fl
import sys

port = sys.argv[1]
addr = "[::]:"+port
fl.server.start_server(server_address=addr, config=fl.server.ServerConfig(num_rounds=3))