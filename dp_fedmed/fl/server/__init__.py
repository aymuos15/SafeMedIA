"""Server-side components for federated learning with differential privacy.

Import components directly from submodules:
    from dp_fedmed.fl.base.strategy import DPStrategy
    from dp_fedmed.fl.server.aggregation import weighted_average
    from dp_fedmed.fl.server.factory import server_fn
    from dp_fedmed.fl.server.app import app
"""

__all__ = ["weighted_average", "server_fn", "app"]
