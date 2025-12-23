# Data API

The Data API provides dataset utilities for medical image segmentation in nnUNet format.

## Overview

DP-FedMed includes data loading utilities for medical imaging datasets with support for:

- Building data lists from nnUNet-formatted directories
- MONAI transform pipelines (resize, normalization, augmentation)
- Integration with PyTorch DataLoader
- Opacus-compatible tensor conversion

## Main Components

::: dp_fedmed.data.cellpose.build_data_list
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.data.cellpose.get_transforms
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Dataset Structure

Medical imaging datasets should follow nnUNet format:

```
data_dir/
├── imagesTr/           # Training images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── labelsTr/           # Training labels (masks)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── imagesTs/           # Test images (optional)
│   └── ...
└── labelsTs/           # Test labels (optional)
    └── ...
```

### Supported Formats

- Image extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.nii`, `.nii.gz`
- Label files must match image case IDs
- nnUNet naming convention: `{case_id}_{modality}.ext` (e.g., `000_0000.png`)

## Transforms Pipeline

### Training Transforms

```python
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    ToTensord,
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # Normalize to [0, 1]
    Resized(keys=["image", "label"], spatial_size=(256, 256), mode=["bilinear", "nearest"]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Horizontal flip
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # Vertical flip
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),     # 90° rotations
    EnsureTyped(keys=["image", "label"], data_type="tensor"),      # For Opacus compatibility
    ToTensord(keys=["image", "label"]),
])
```

### Test Transforms

```python
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    Resized(keys=["image", "label"], spatial_size=(256, 256), mode=["bilinear", "nearest"]),
    EnsureTyped(keys=["image", "label"], data_type="tensor"),
    ToTensord(keys=["image", "label"]),
])
```

## Opacus Compatibility

The `EnsureTyped` transform converts MONAI `MetaTensor` objects to plain `torch.Tensor`:

- **Why needed?** Opacus has issues with `MetaTensor` when handling empty batches during Poisson sampling
- **Error prevented**: `TypeError: zeros() received an invalid combination of arguments`
- **Applied at**: End of transform pipeline before training

## DataLoader Configuration

### For Differential Privacy

When using Opacus DP-SGD, DataLoaders must have:

```python
from torch.utils.data import DataLoader
from monai.data import Dataset

# Build data list
data_list = build_data_list(data_dir="/path/to/data", split="train")
transforms = get_transforms(spatial_size=(256, 256), is_train=True)

# Create MONAI Dataset
dataset = Dataset(data=data_list, transform=transforms)

# Create DataLoader (drop_last=True required for DP)
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,  # REQUIRED: Ensures all batches same size
    num_workers=2,
)
```

**Why `drop_last=True`?**
- Opacus requires fixed batch sizes for privacy accounting
- Variable-size batches have different privacy costs
- Dropping last incomplete batch ensures consistency

## Sample Data Format

Each sample returned by the transforms:

```python
{
    "image": torch.Tensor,  # Shape: (C, H, W), dtype: float32, range: [0, 1]
    "label": torch.Tensor,  # Shape: (C, H, W), dtype: float32
}
```

After batching by DataLoader:

```python
batch = {
    "image": torch.Tensor,  # Shape: (B, C, H, W)
    "label": torch.Tensor,  # Shape: (B, C, H, W)
}
```

## Integration with Federated Learning

The client factory function uses these utilities internally:

```python
from dp_fedmed.data import build_data_list, get_transforms
from monai.data import Dataset
from torch.utils.data import DataLoader

def create_client_dataloaders(data_dir, client_id, num_clients, batch_size, image_size):
    # Build full dataset
    data_list = build_data_list(data_dir, split="train")

    # Partition data for this client
    partition_size = len(data_list) // num_clients
    start_idx = client_id * partition_size
    end_idx = start_idx + partition_size
    client_data_list = data_list[start_idx:end_idx]

    # Create transforms
    train_transforms = get_transforms(spatial_size=(image_size, image_size), is_train=True)
    test_transforms = get_transforms(spatial_size=(image_size, image_size), is_train=False)

    # Create datasets
    train_dataset = Dataset(data=client_data_list, transform=train_transforms)
    test_data_list = build_data_list(data_dir, split="test")
    test_dataset = Dataset(data=test_data_list, transform=test_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

## See Also

- [Client API](client.md) - Using data loaders in training
- [Configuration Schema](../schemas/config.md) - Data configuration options
- [Tasks API](tasks.md) - Training loops with data loaders
