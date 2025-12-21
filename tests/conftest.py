"""Pytest fixtures for dp_fedmed tests."""

import pytest
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def dummy_model():
    """Create a small UNet for fast testing.

    Note: channels must be >= 16 for Opacus compatibility.
    Smaller models cause SIGFPE in Opacus per-sample gradient computation.
    """
    from dp_fedmed.models.unet2d import create_unet2d

    return create_unet2d(
        in_channels=1,
        out_channels=2,
        channels=(16, 32),  # Minimum viable for Opacus
        strides=(2,),
        num_res_units=1,
    )


@pytest.fixture
def dummy_dataloader():
    """Create synthetic data loader (no real images needed)."""
    # Create random tensors matching expected input format
    # Images: [B, C, H, W] = [8, 1, 64, 64]
    # Labels: [B, H, W] = [8, 64, 64] with values 0 or 1
    images = torch.rand(8, 1, 64, 64)
    labels = torch.randint(0, 2, (8, 64, 64))

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def dummy_dict_dataloader():
    """Create synthetic data loader with dict format (MONAI style)."""

    class DictDataset(Dataset):
        def __init__(self, size=8):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, index: int):
            return {
                "image": torch.rand(1, 64, 64),
                "label": torch.randint(0, 2, (64, 64)),
            }

    dataset = DictDataset()
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def optimizer(dummy_model):
    """Create optimizer for the dummy model."""
    from torch.optim.sgd import SGD

    return SGD(dummy_model.parameters(), lr=0.01)
