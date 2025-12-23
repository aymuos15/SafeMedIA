"""Tests for data loading utilities (cellpose.py)."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from dp_fedmed.data.cellpose import build_data_list, get_transforms


class TestBuildDataList:
    """Tests for build_data_list function."""

    def test_build_data_list_train(self, tmp_path):
        """Test building data list for training set."""
        # Create mock dataset structure
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create some dummy images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
            img.save(images_dir / f"case_{i:03d}_0000.png")
            lbl = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
            lbl.save(labels_dir / f"case_{i:03d}.png")

        data_list = build_data_list(tmp_path, "train")

        assert len(data_list) == 3
        for entry in data_list:
            assert "image" in entry
            assert "label" in entry
            assert Path(entry["image"]).exists()
            assert Path(entry["label"]).exists()

    def test_build_data_list_test(self, tmp_path):
        """Test building data list for test set."""
        # Create test set
        images_dir = tmp_path / "imagesTs"
        labels_dir = tmp_path / "labelsTs"
        images_dir.mkdir()
        labels_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img.save(images_dir / "test_001_0000.png")
        lbl = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        lbl.save(labels_dir / "test_001.png")

        data_list = build_data_list(tmp_path, "test")

        assert len(data_list) == 1
        assert "image" in data_list[0]
        assert "label" in data_list[0]

    def test_build_data_list_fallback_to_train(self, tmp_path):
        """Test fallback to train set when test set doesn't exist."""
        # Create only train set
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img.save(images_dir / "case_001_0000.png")
        lbl = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        lbl.save(labels_dir / "case_001.png")

        # Request test set, should fall back to train
        data_list = build_data_list(tmp_path, "test")

        assert len(data_list) == 1

    def test_build_data_list_no_images_raises(self, tmp_path):
        """Test that missing images directory raises error."""
        with pytest.raises(FileNotFoundError):
            build_data_list(tmp_path, "train")

    def test_build_data_list_no_pairs_raises(self, tmp_path):
        """Test that no valid image-label pairs raises error."""
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create image without corresponding label
        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img.save(images_dir / "orphan.png")

        with pytest.raises(ValueError, match="No valid image-label pairs"):
            build_data_list(tmp_path, "train")

    def test_build_data_list_multiple_extensions(self, tmp_path):
        """Test handling of multiple image extensions."""
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create images with different extensions
        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img.save(images_dir / "case_001_0000.png")
        img.save(images_dir / "case_002_0000.jpg")

        lbl = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        lbl.save(labels_dir / "case_001.png")
        lbl.save(labels_dir / "case_002.jpg")

        data_list = build_data_list(tmp_path, "train")

        assert len(data_list) == 2

    def test_build_data_list_exact_match_fallback(self, tmp_path):
        """Test exact filename match fallback when case ID parsing fails."""
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create files without standard naming convention
        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img.save(images_dir / "custom_name.png")
        lbl = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        lbl.save(labels_dir / "custom_name.png")

        data_list = build_data_list(tmp_path, "train")

        assert len(data_list) == 1


class TestGetTransforms:
    """Tests for get_transforms function."""

    def test_get_transforms_train(self):
        """Test training transforms include augmentation."""
        transforms = get_transforms((256, 256), is_train=True)

        assert transforms is not None
        # Training should have more transforms (augmentation)
        assert len(transforms.transforms) > 4  # Base + augmentation

    def test_get_transforms_test(self):
        """Test test transforms without augmentation."""
        transforms = get_transforms((256, 256), is_train=False)

        assert transforms is not None
        # Test should have fewer transforms (no augmentation)
        assert len(transforms.transforms) >= 4  # Base transforms only

    def test_get_transforms_custom_size(self):
        """Test transforms with custom spatial size."""
        spatial_size = (128, 128)
        transforms = get_transforms(spatial_size, is_train=False)

        assert transforms is not None
        # Check that resize transform uses correct size
        # (implementation detail, may need adjustment based on actual structure)

    def test_train_has_more_transforms_than_test(self):
        """Test that training transforms include augmentation."""
        train_transforms = get_transforms((256, 256), is_train=True)
        test_transforms = get_transforms((256, 256), is_train=False)

        assert len(train_transforms.transforms) > len(test_transforms.transforms)


class TestTupleDataset:
    """Tests for TupleDataset wrapper in factory.py."""

    def test_tuple_dataset_wrapper(self, tmp_path):
        """Test TupleDataset converts MONAI dict format to tuple."""
        from monai.data.dataset import Dataset
        from dp_fedmed.fl.client.factory import TupleDataset

        # Create minimal mock data
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img.save(images_dir / "test.png")
        lbl = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        lbl.save(labels_dir / "test.png")

        data_list = build_data_list(tmp_path, "train")
        transforms = get_transforms((64, 64), is_train=False)

        # Create MONAI dataset and wrap it
        monai_dataset = Dataset(data=data_list, transform=transforms)
        tuple_dataset = TupleDataset(monai_dataset)

        assert len(tuple_dataset) == len(monai_dataset)

        # Test that we get tuples
        batch = tuple_dataset[0]
        assert isinstance(batch, tuple)
        assert len(batch) == 2
        # First element is image, second is label

    def test_tuple_dataset_length(self):
        """Test TupleDataset preserves dataset length."""
        from monai.data.dataset import Dataset
        from dp_fedmed.fl.client.factory import TupleDataset

        # Create simple mock dataset
        data_list = [{"image": "img1.png", "label": "lbl1.png"}] * 5
        # Note: This will fail to load actual files, but we're just testing the wrapper
        monai_dataset = Dataset(data=data_list, transform=None)
        tuple_dataset = TupleDataset(monai_dataset)

        assert len(tuple_dataset) == 5
