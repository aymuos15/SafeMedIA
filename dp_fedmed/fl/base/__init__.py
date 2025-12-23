"""Base abstractions for federated learning clients and strategies.

This module provides abstract base classes that supervised and SSL
federated learning implementations inherit from.
"""

from dp_fedmed.fl.base.client import BaseFlowerClient
from dp_fedmed.fl.base.strategy import BaseDPStrategy
from dp_fedmed.fl.base.dataset import TupleDataset, UnlabeledImageDataset

__all__ = [
    "BaseFlowerClient",
    "BaseDPStrategy",
    "TupleDataset",
    "UnlabeledImageDataset",
]
