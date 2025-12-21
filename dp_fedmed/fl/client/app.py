"""Flower ClientApp entry point for federated learning."""

from flwr.client import ClientApp

from .factory import client_fn

# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
