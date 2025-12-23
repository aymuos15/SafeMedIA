"""Base components for federated learning clients and strategies.

This module provides the core classes for DP-aware federated learning.
"""

from dp_fedmed.fl.base.client import BaseFlowerClient
from dp_fedmed.fl.base.strategy import DPStrategy
from dp_fedmed.fl.base.dataset import TupleDataset, UnlabeledImageDataset

__all__ = [
    "BaseFlowerClient",
    "DPStrategy",
    "TupleDataset",
    "UnlabeledImageDataset",
]
