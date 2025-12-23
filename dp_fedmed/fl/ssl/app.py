"""Flower ServerApp and ClientApp definitions for federated SSL pretraining.

This module defines the server and client apps that are used as entry points
for Flower's federation simulation.
"""

from flwr.server import ServerApp
from flwr.client import ClientApp

from dp_fedmed.fl.client.factory import create_client_fn, TrainingMode
from dp_fedmed.fl.ssl.server_factory import server_fn

DEFAULT_CONFIG = "configs/pretraining.toml"

# Create the server app
server_app = ServerApp(server_fn=server_fn)

# Create the client app with proper closure to fix the signature issue
# This is the fix for the original bug - using create_client_fn() instead of client_fn directly
client_app = ClientApp(client_fn=create_client_fn(DEFAULT_CONFIG, TrainingMode.SSL))
