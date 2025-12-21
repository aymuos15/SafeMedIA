"""Data loading utilities for Cellpose/nnUNet format datasets.

This module provides functions to build data lists and transforms for
image segmentation datasets in nnUNet format.

Expected directory structure:
    data_dir/
        imagesTr/
            image_001.png
            image_002.png
            ...
        labelsTr/
            image_001.png
            image_002.png
            ...
        imagesTs/  (optional)
            ...
        labelsTs/  (optional)
            ...
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    ToTensord,
)


def build_data_list(
    data_dir: Union[str, Path], split: str = "train"
) -> List[Dict[str, str]]:
    """Build a list of image-label pairs for the dataset.

    Args:
        data_dir: Path to dataset root directory (nnUNet format)
        split: Either "train" or "test"

    Returns:
        List of dicts with "image" and "label" keys pointing to file paths
    """
    data_dir = Path(data_dir)

    if split == "train":
        images_dir = data_dir / "imagesTr"
        labels_dir = data_dir / "labelsTr"
    else:
        # Try test directory, fall back to train if not exists
        images_dir = data_dir / "imagesTs"
        labels_dir = data_dir / "labelsTs"
        if not images_dir.exists():
            images_dir = data_dir / "imagesTr"
            labels_dir = data_dir / "labelsTr"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Find all image files
    image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nii", ".nii.gz")
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    data_list = []
    for image_path in image_files:
        # Extract case ID from nnUNet naming: {case_id}_{modality}.ext -> case_id
        # e.g., "000_0000.png" -> "000"
        stem = image_path.stem
        if "_" in stem:
            case_id = stem.rsplit("_", 1)[0]
        else:
            case_id = stem

        # Find corresponding label file
        label_path = None
        for ext in image_extensions:
            candidate = labels_dir / f"{case_id}{ext}"
            if candidate.exists():
                label_path = candidate
                break

        # Also try exact match as fallback
        if label_path is None:
            exact_match = labels_dir / image_path.name
            if exact_match.exists():
                label_path = exact_match

        if label_path is not None:
            data_list.append({"image": str(image_path), "label": str(label_path)})

    if not data_list:
        raise ValueError(f"No valid image-label pairs found in {data_dir}")

    return data_list


def get_transforms(spatial_size: Tuple[int, int], is_train: bool = True) -> Compose:
    """Get MONAI transforms for training or testing.

    Args:
        spatial_size: Target image size (height, width)
        is_train: Whether to include augmentation transforms

    Returns:
        MONAI Compose transform pipeline
    """
    # Base transforms for both train and test
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(
            keys=["image", "label"],
            spatial_size=spatial_size,
            mode=["bilinear", "nearest"],
        ),
    ]

    if is_train:
        # Add augmentation for training
        augment_transforms = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ]
        transforms = base_transforms + augment_transforms
    else:
        transforms = base_transforms

    # Final conversion to tensors
    transforms.append(ToTensord(keys=["image", "label"]))

    return Compose(transforms)
