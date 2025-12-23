"""DP-FedMed: Differential Privacy + Federated Learning for Medical Imaging."""

import os

# Configure Ray before it initializes (via Flower simulation)
# Disable dashboard/metrics agent to prevent RPC connection errors
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
# Prevent Ray from overriding CUDA_VISIBLE_DEVICES when num_gpus=0
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

__version__ = "0.1.0"

from . import utils  # noqa: E402

__all__ = ["utils"]
