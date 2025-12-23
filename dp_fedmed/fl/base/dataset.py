"""Shared dataset wrappers for federated learning.

This module provides dataset adapters that work with Opacus DP
for both supervised and self-supervised learning scenarios.
"""

from typing import Optional, Callable, List, Sized, Any

import torch
import torch.utils.data


class TupleDataset(torch.utils.data.Dataset[Any]):
    """Wrapper to convert dict-style datasets to tuple format for Opacus.

    Opacus requires datasets to return tuples of (input, target).
    This wrapper converts MONAI-style datasets that return dicts
    like {"image": tensor, "label": tensor} to (image, label) tuples.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Any],
        input_key: str = "image",
        target_key: str = "label",
    ):
        """Initialize the wrapper.

        Args:
            dataset: Source dataset that returns dict items
            input_key: Key for input data in the dict
            target_key: Key for target data in the dict
        """
        self.dataset = dataset
        self.input_key = input_key
        self.target_key = target_key

    def __len__(self) -> int:
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        raise TypeError(f"Dataset {type(self.dataset)} is not Sized")

    def __getitem__(self, index: Any):
        item = self.dataset[index]
        if isinstance(item, dict):
            return item[self.input_key], item[self.target_key]
        return item


class UnlabeledImageDataset(torch.utils.data.Dataset[Any]):
    """Dataset wrapper for unlabeled images used in SSL pretraining.

    This dataset loads images without labels and applies SSL transforms
    that produce augmented view pairs for contrastive learning.
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        flatten_augmented_views: bool = False,
    ):
        """Initialize the unlabeled dataset.

        Args:
            image_paths: List of image file paths
            transform: Transform to apply to images (typically SSL augmentation)
            flatten_augmented_views: If True, stack (view1, view2) into single tensor
                                    for Opacus compatibility. Default False.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.flatten_augmented_views = flatten_augmented_views

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: Any):
        from PIL import Image
        import torchvision.transforms as T

        image_path = self.image_paths[index]

        # Load image as grayscale
        img = Image.open(image_path).convert("L")
        img_tensor = T.ToTensor()(img)

        # Apply transforms (produces augmented view pair for SSL)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # If flatten_augmented_views, stack views for Opacus compatibility
        if self.flatten_augmented_views and isinstance(img_tensor, (list, tuple)):
            # Stack: (view1, view2) -> [2, C, H, W]
            img_tensor = torch.stack(img_tensor, dim=0)
            # Return with dummy label tensor for DataLoader/Opacus compatibility
            return img_tensor, torch.tensor(0, dtype=torch.long)
        else:
            # Return with dummy label tensor for DataLoader/Opacus compatibility
            # Note: Opacus DPDataLoader requires tensors with proper dtype
            return img_tensor, torch.tensor(0, dtype=torch.long)


class ViewPairDataset(torch.utils.data.Dataset[Any]):
    """Dataset that returns explicit view pairs for SSL training.

    This is an alternative to UnlabeledImageDataset that separates
    the two views more explicitly, useful when you need to process
    views differently.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Any],
        transform1: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
    ):
        """Initialize the view pair dataset.

        Args:
            dataset: Base dataset that returns images
            transform1: Transform for first view
            transform2: Transform for second view (if None, uses transform1)
        """
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2 or transform1

    def __len__(self) -> int:
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        raise TypeError(f"Dataset {type(self.dataset)} is not Sized")

    def __getitem__(self, index: Any):
        item = self.dataset[index]

        # Handle dict-style items
        if isinstance(item, dict):
            image = item.get("image", item.get("input"))
        elif isinstance(item, (tuple, list)):
            image = item[0]
        else:
            image = item

        # Apply transforms to create two views
        view1 = self.transform1(image) if self.transform1 else image
        view2 = self.transform2(image) if self.transform2 else image

        # Dummy label tensor for Opacus DPDataLoader compatibility
        return (view1, view2), torch.tensor(0, dtype=torch.long)
