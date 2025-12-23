"""SSL-specific augmentation transforms for medical images.

Uses Lightly's augmentation utilities with medical image-friendly settings.
"""

from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.simsiam_transform import SimSiamTransform


def get_simclr_transform(config):
    """Get SimCLR-specific augmentation transform.

    Args:
        config: Augmentation configuration

    Returns:
        SimCLRTransform instance
    """
    normalize_dict = {
        "mean": config.normalize_mean,
        "std": config.normalize_std,
    }

    return SimCLRTransform(
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


def get_moco_transform(config) -> MoCoV2Transform:
    """Get MoCo-specific augmentation transform.

    Args:
        config: Augmentation configuration

    Returns:
        MoCoV2Transform instance
    """
    normalize_dict = {
        "mean": config.normalize_mean,
        "std": config.normalize_std,
    }

    return MoCoV2Transform(
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


def get_simsiam_transform(config) -> SimSiamTransform:
    """Get SimSiam-specific augmentation transform.

    Args:
        config: Augmentation configuration

    Returns:
        SimSiamTransform instance
    """
    normalize_dict = {
        "mean": config.normalize_mean,
        "std": config.normalize_std,
    }

    return SimSiamTransform(
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


def get_ssl_transform(method: str, config):
    """Get SSL transform for specified method.

    Args:
        method: SSL method ('simclr', 'moco', 'simsiam')
        config: Augmentation configuration

    Returns:
        Appropriate transform for the SSL method

    Raises:
        ValueError: If method is not supported
    """
    if method == "simclr":
        return get_simclr_transform(config)
    elif method == "moco":
        return get_moco_transform(config)
    elif method == "simsiam":
        return get_simsiam_transform(config)
    else:
        raise ValueError(
            f"Unknown SSL method: {method}. Choose from ['simclr', 'moco', 'simsiam']"
        )
