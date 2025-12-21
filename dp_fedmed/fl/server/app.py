"""Flower ServerApp entry point for federated learning."""

from flwr.server import ServerApp

from .factory import server_fn

# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
