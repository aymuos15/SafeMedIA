"""SSL-specific augmentation transforms for medical images.

Uses Lightly's augmentation utilities with medical image-friendly settings.
"""

from typing import Any

from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.simsiam_transform import SimSiamTransform

# Registry of supported SSL transform classes
SSL_TRANSFORMS = {
    "simclr": SimCLRTransform,
    "moco": MoCoV2Transform,
    "simsiam": SimSiamTransform,
}


def get_ssl_transform(method: str, config: Any):
    """Get SSL transform for specified method.

    All SSL methods use the same augmentation parameters from config,
    only differing in the underlying transform class implementation.

    Args:
        method: SSL method ('simclr', 'moco', 'simsiam')
        config: Augmentation configuration with attributes:
            - input_size: tuple of (height, width)
            - color_jitter_prob: probability of color jitter
            - color_jitter_strength: strength of color jitter
            - crop_min_scale: minimum scale for random crop
            - gaussian_blur_prob: probability of gaussian blur
            - flip_prob: probability of vertical/horizontal flip
            - rotation_prob: probability of rotation
            - rotation_degrees: max rotation degrees
            - normalize_mean: normalization mean
            - normalize_std: normalization std

    Returns:
        Appropriate transform instance for the SSL method

    Raises:
        ValueError: If method is not supported
    """
    transform_cls = SSL_TRANSFORMS.get(method)
    if transform_cls is None:
        supported = list(SSL_TRANSFORMS.keys())
        raise ValueError(f"Unknown SSL method: {method}. Choose from {supported}")

    normalize_dict = {
        "mean": config.normalize_mean,
        "std": config.normalize_std,
    }

    return transform_cls(
        input_size=config.input_size[0],
        cj_prob=config.color_jitter_prob,
        cj_strength=config.color_jitter_strength,
        min_scale=config.crop_min_scale,
        random_gray_scale=0.2,
        gaussian_blur=config.gaussian_blur_prob,
        kernel_size=None,
        vf_prob=config.flip_prob,
        hf_prob=config.flip_prob,
        rr_prob=config.rotation_prob,
        rr_degrees=config.rotation_degrees,
        normalize=normalize_dict,
    )
