"""Flower ClientApp entry point for federated learning."""

from flwr.client import ClientApp
from flwr.common import Context
import flwr as fl

from .factory import client_fn as _client_fn


def client_fn(context: Context) -> fl.client.Client:
    """Wrapper for client_fn with required signature for ClientApp."""
    return _client_fn(context)


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
