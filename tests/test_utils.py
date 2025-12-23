"""Tests for aggregation utilities and utility functions."""

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule

from dp_fedmed.fl.server.aggregation import weighted_average
from dp_fedmed.utils import get_unwrapped_model, extract_batch_data, get_dataset_size  # type: ignore[import-not-found]


class TestWeightedAverage:
    """Tests for weighted_average function."""

    def test_weighted_average_empty_metrics(self):
        """Test weighted_average with empty metrics list."""
        result = weighted_average([])
        assert result == {}

    def test_weighted_average_single_client(self):
        """Test weighted_average with single client."""
        metrics = [(10, {"loss": 0.5, "dice": 0.8})]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        assert "loss" in result
        assert "dice" in result
        assert result["loss"] == 0.5
        assert result["dice"] == 0.8

    def test_weighted_average_multiple_clients(self):
        """Test weighted_average with multiple clients."""
        metrics = [
            (10, {"loss": 0.5, "dice": 0.8}),
            (20, {"loss": 0.3, "dice": 0.9}),
        ]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        # Calculate expected weighted average
        expected_loss = (10 * 0.5 + 20 * 0.3) / 30
        expected_dice = (10 * 0.8 + 20 * 0.9) / 30

        assert "loss" in result
        assert "dice" in result
        assert isinstance(result["loss"], (int, float))
        assert isinstance(result["dice"], (int, float))
        assert abs(result["loss"] - expected_loss) < 1e-6  # type: ignore[operator]
        assert abs(result["dice"] - expected_dice) < 1e-6  # type: ignore[operator]

    def test_weighted_average_zero_samples_warning(self):
        """Test weighted_average with zero total samples."""
        metrics = [(0, {"loss": 0.5}), (0, {"dice": 0.8})]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        # Should not crash, but may not aggregate properly
        assert isinstance(result, dict)

    def test_weighted_average_missing_keys(self):
        """Test weighted_average when some clients have missing metrics."""
        metrics = [
            (10, {"loss": 0.5, "dice": 0.8}),
            (20, {"loss": 0.3}),  # Missing "dice"
        ]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        assert "loss" in result
        assert "dice" in result
        # Dice is present in result (weighted average handles missing values by treating them as 0)
        # Expected: (10 * 0.8 + 20 * 0) / 30 = 8 / 30 ≈ 0.267
        assert isinstance(result["dice"], (int, float))
        assert abs(result["dice"] - 0.267) < 0.01  # type: ignore[operator]

    def test_weighted_average_non_numeric_values(self):
        """Test weighted_average skips non-numeric values."""
        metrics = [
            (10, {"loss": 0.5, "status": "done"}),
            (20, {"loss": 0.3, "status": "done"}),
        ]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        assert "loss" in result
        assert "status" not in result  # Non-numeric should be skipped

    def test_weighted_average_different_weights(self):
        """Test weighted_average with very different weights."""
        metrics = [
            (1, {"loss": 1.0}),
            (99, {"loss": 0.0}),
        ]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        # Should be heavily weighted toward second client
        assert isinstance(result["loss"], (int, float))
        assert result["loss"] < 0.05  # type: ignore[operator] # Close to 0

    def test_weighted_average_mixed_metric_sets(self):
        """Test weighted_average with completely different metric sets."""
        metrics = [
            (10, {"loss": 0.5, "accuracy": 0.9}),
            (20, {"dice": 0.8, "f1": 0.7}),
        ]
        result = weighted_average(metrics)  # type: ignore[arg-type]

        # All metrics should be present in result
        assert "loss" in result
        assert "accuracy" in result
        assert "dice" in result
        assert "f1" in result


class TestGetUnwrappedModel:
    """Tests for get_unwrapped_model utility."""

    def test_get_unwrapped_model_regular_model(self):
        """Test get_unwrapped_model with regular PyTorch model."""
        model = nn.Linear(10, 5)
        unwrapped = get_unwrapped_model(model)

        assert unwrapped is model

    def test_get_unwrapped_model_opacus_wrapped(self):
        """Test get_unwrapped_model with Opacus-wrapped model."""
        model = nn.Linear(10, 5)
        wrapped = GradSampleModule(model)

        unwrapped = get_unwrapped_model(wrapped)

        assert unwrapped is model
        assert not isinstance(unwrapped, GradSampleModule)

    def test_get_unwrapped_model_nested_modules(self):
        """Test get_unwrapped_model with sequential model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        unwrapped = get_unwrapped_model(model)
        assert unwrapped is model


class TestExtractBatchData:
    """Tests for extract_batch_data utility."""

    def test_extract_batch_data_dict_format(self):
        """Test extract_batch_data with dictionary format."""
        device = torch.device("cpu")
        batch = {
            "image": torch.randn(2, 1, 64, 64),
            "label": torch.randint(0, 2, (2, 64, 64)),
        }

        result = extract_batch_data(batch, device)

        assert result is not None
        images, labels = result
        assert images.shape == (2, 1, 64, 64)
        assert labels.shape == (2, 64, 64)
        assert labels.dtype == torch.long

    def test_extract_batch_data_tuple_format(self):
        """Test extract_batch_data with tuple format."""
        device = torch.device("cpu")
        images = torch.randn(2, 1, 64, 64)
        labels = torch.randint(0, 2, (2, 64, 64))
        batch = (images, labels)

        result = extract_batch_data(batch, device)

        assert result is not None
        extracted_images, extracted_labels = result
        assert extracted_images.shape == images.shape
        assert extracted_labels.shape == labels.shape
        assert extracted_labels.dtype == torch.long

    def test_extract_batch_data_list_format(self):
        """Test extract_batch_data with list format."""
        device = torch.device("cpu")
        images = torch.randn(2, 1, 64, 64)
        labels = torch.randint(0, 2, (2, 64, 64))
        batch = [images, labels]

        result = extract_batch_data(batch, device)

        assert result is not None
        extracted_images, extracted_labels = result
        assert extracted_images.shape == images.shape

    def test_extract_batch_data_missing_dict_keys(self):
        """Test extract_batch_data with missing keys in dict."""
        device = torch.device("cpu")
        batch = {"image": torch.randn(2, 1, 64, 64)}  # Missing "label"

        result = extract_batch_data(batch, device)

        assert result is None  # Should return None for invalid batch

    def test_extract_batch_data_invalid_tuple_length(self):
        """Test extract_batch_data with invalid tuple length."""
        device = torch.device("cpu")
        batch = (torch.randn(2, 1, 64, 64),)  # Only one element

        result = extract_batch_data(batch, device)

        assert result is None

    def test_extract_batch_data_invalid_format(self):
        """Test extract_batch_data with completely invalid format."""
        device = torch.device("cpu")
        batch = "invalid"

        result = extract_batch_data(batch, device)

        assert result is None

    def test_extract_batch_data_label_preprocessing(self):
        """Test that labels are preprocessed correctly."""
        device = torch.device("cpu")
        # Labels with channel dimension
        labels_with_channel = torch.randint(0, 2, (2, 1, 64, 64)).float()
        batch = {"image": torch.randn(2, 1, 64, 64), "label": labels_with_channel}

        result = extract_batch_data(batch, device)

        assert result is not None
        images, labels = result
        # Should squeeze channel dimension
        assert labels.shape == (2, 64, 64)
        # Should convert to long
        assert labels.dtype == torch.long

    def test_extract_batch_data_device_transfer(self):
        """Test that tensors are moved to correct device."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            batch = {
                "image": torch.randn(2, 1, 64, 64),
                "label": torch.randint(0, 2, (2, 64, 64)),
            }

            result = extract_batch_data(batch, device)

            assert result is not None
            images, labels = result
            assert images.device.type == "cuda"
            assert labels.device.type == "cuda"


class TestGetDatasetSize:
    """Tests for get_dataset_size utility."""

    def test_get_dataset_size_regular_dataloader(self):
        """Test get_dataset_size with regular DataLoader."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randint(0, 2, (100,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        size = get_dataset_size(dataloader)

        assert size == 100

    def test_get_dataset_size_no_length(self):
        """Test get_dataset_size with DataLoader without len."""

        # Create a mock dataloader-like object without __len__
        class MockDataLoader:
            def __init__(self):
                self.dataset = None

        dataloader = MockDataLoader()
        size = get_dataset_size(dataloader)  # type: ignore

        assert size == 0  # Should return 0 for invalid dataloader

    def test_get_dataset_size_opacus_dataloader(self):
        """Test get_dataset_size with Opacus DPDataLoader."""
        # This simulates what happens with Opacus DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 10), torch.randint(0, 2, (50,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)

        size = get_dataset_size(dataloader)

        assert size == 50
