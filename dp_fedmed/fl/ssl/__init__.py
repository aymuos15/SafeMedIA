"""Self-supervised learning (SSL) module for federated pretraining.

This module provides SSL-specific implementations of Flower clients
and strategies for federated contrastive learning.
"""

from dp_fedmed.fl.ssl.client import SSLFlowerClient
from dp_fedmed.fl.ssl.strategy import DPFedAvgSSL
from dp_fedmed.fl.ssl.model import SSLUNet
from dp_fedmed.fl.ssl.transforms import get_ssl_transform
from dp_fedmed.fl.ssl.config import SSLConfig, AugmentationConfig
from dp_fedmed.fl.ssl.checkpoint import (
    save_pretrained_checkpoint,
    load_pretrained_encoder,
    load_encoder_for_downstream,
)

__all__ = [
    "SSLFlowerClient",
    "DPFedAvgSSL",
    "SSLUNet",
    "get_ssl_transform",
    "SSLConfig",
    "AugmentationConfig",
    "save_pretrained_checkpoint",
    "load_pretrained_encoder",
    "load_encoder_for_downstream",
]
