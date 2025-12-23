"""Client-side components for federated learning with differential privacy.

Import components directly from submodules to avoid circular import issues:
    from dp_fedmed.fl.client.dp_client import DPFlowerClient
    from dp_fedmed.fl.client.factory import client_fn, create_client_fn, TrainingMode
    from dp_fedmed.fl.client.app import app
"""

__all__ = ["DPFlowerClient", "client_fn", "create_client_fn", "TrainingMode", "app"]
