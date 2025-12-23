"""Tests for DPFedAvg strategy and server-side aggregation."""

import json
from typing import Any, cast
import numpy as np
from unittest.mock import MagicMock, patch

from dp_fedmed.fl.server.strategy import DPFedAvg
from dp_fedmed.fl.checkpoint import UnifiedCheckpointManager


class TestDPFedAvgInitialization:
    """Tests for DPFedAvg initialization."""

    def test_default_initialization(self, tmp_path):
        """Test DPFedAvg initialization with defaults."""
        strategy = DPFedAvg(
            target_delta=1e-5,
            run_dir=tmp_path / "results",
            run_name="test",
        )

        assert strategy.privacy_accountant.target_delta == 1e-5
        assert strategy.run_name == "test"
        assert strategy.noise_multiplier == 1.0
        assert strategy.start_round == 1
        assert not strategy.is_mid_round_resume

    def test_initialization_with_custom_params(self, tmp_path):
        """Test DPFedAvg initialization with custom parameters."""
        strategy = DPFedAvg(
            target_delta=1e-6,
            run_dir=tmp_path / "results",
            run_name="custom",
            num_rounds=10,
            noise_multiplier=2.0,
            max_grad_norm=0.5,
            user_noise_multiplier=1.0,
            total_clients=20,
            start_round=3,
            local_epochs=10,
        )

        assert strategy.privacy_accountant.target_delta == 1e-6
        assert strategy.num_rounds == 10
        assert strategy.noise_multiplier == 2.0
        assert strategy.max_grad_norm == 0.5
        assert strategy.user_noise_multiplier == 1.0
        assert strategy.total_clients == 20
        assert strategy.start_round == 3
        assert strategy.local_epochs == 10

    def test_initialization_with_checkpoint_manager(self, tmp_path):
        """Test DPFedAvg initialization with external checkpoint manager."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        checkpoint_manager = UnifiedCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_name="external",
            num_rounds=5,
            target_delta=1e-5,
        )

        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
            checkpoint_manager=checkpoint_manager,
        )

        assert strategy.checkpoint_manager is checkpoint_manager

    def test_mid_round_resume_initialization(self, tmp_path):
        """Test DPFedAvg initialization for mid-round resume."""
        client_states = {0: MagicMock(), 1: MagicMock()}

        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
            start_round=5,
            is_mid_round_resume=True,
            client_resume_states=cast(Any, client_states),
        )

        assert strategy.is_mid_round_resume
        assert strategy.start_round == 5
        assert len(strategy.client_resume_states) == 2


class TestDPFedAvgAggregateFit:
    """Tests for aggregate_fit method."""

    def test_aggregate_fit_empty_results(self, tmp_path):
        """Test aggregate_fit with empty results returns None."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
        )

        params, metrics = strategy.aggregate_fit(
            server_round=1,
            results=[],
            failures=[],
        )

        assert params is None
        assert metrics == {}

    def test_aggregate_fit_records_privacy(self, tmp_path):
        """Test aggregate_fit records privacy metrics correctly."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
            noise_multiplier=1.0,
            total_clients=2,
        )

        # Create mock results
        mock_client1 = MagicMock()
        mock_client1.cid = "0"
        mock_fit_res1 = MagicMock()
        mock_fit_res1.metrics = {
            "epsilon": 0.5,
            "sample_rate": 0.01,
            "steps": 100,
            "loss": 0.5,
            "delta": 1e-5,
        }
        mock_fit_res1.num_examples = 500
        mock_fit_res1.parameters = MagicMock()
        mock_fit_res1.parameters.tensors = [np.random.randn(10, 10).tobytes()]
        mock_fit_res1.parameters.tensor_type = "numpy.ndarray"

        mock_client2 = MagicMock()
        mock_client2.cid = "1"
        mock_fit_res2 = MagicMock()
        mock_fit_res2.metrics = {
            "epsilon": 0.6,
            "sample_rate": 0.01,
            "steps": 100,
            "loss": 0.4,
            "delta": 1e-5,
        }
        mock_fit_res2.num_examples = 500
        mock_fit_res2.parameters = MagicMock()
        mock_fit_res2.parameters.tensors = [np.random.randn(10, 10).tobytes()]
        mock_fit_res2.parameters.tensor_type = "numpy.ndarray"

        results = [
            (mock_client1, mock_fit_res1),
            (mock_client2, mock_fit_res2),
        ]

        # Mock the parent aggregate_fit (FedAvg)
        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit",
            return_value=(MagicMock(), {}),
        ):
            params, metrics = strategy.aggregate_fit(
                server_round=1,
                results=results,
                failures=[],
            )

        # Check privacy metrics were recorded
        assert "round_sample_epsilon" in metrics
        assert "cumulative_sample_epsilon" in metrics
        assert "cumulative_user_epsilon" in metrics

        # Check client metrics were stored
        assert "0" in strategy.client_metrics
        assert "1" in strategy.client_metrics
        assert len(strategy.client_metrics["0"]) == 1
        assert strategy.client_metrics["0"][0]["round"] == 1

    def test_aggregate_fit_stores_server_round_data(self, tmp_path):
        """Test aggregate_fit stores server round data."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
        )

        mock_client = MagicMock()
        mock_client.cid = "0"
        mock_fit_res = MagicMock()
        mock_fit_res.metrics = {"epsilon": 0.5, "sample_rate": 0.01, "steps": 100}
        mock_fit_res.num_examples = 500
        mock_fit_res.parameters = MagicMock()
        mock_fit_res.parameters.tensors = [np.random.randn(10, 10).tobytes()]
        mock_fit_res.parameters.tensor_type = "numpy.ndarray"

        results = [(mock_client, mock_fit_res)]

        # Mock the parent aggregate_fit (FedAvg)
        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit",
            return_value=(MagicMock(), {}),
        ):
            strategy.aggregate_fit(server_round=1, results=results, failures=[])

        assert len(strategy.server_rounds) == 1
        assert strategy.server_rounds[0]["round"] == 1
        assert strategy.server_rounds[0]["num_clients"] == 1


class TestDPFedAvgAggregateEvaluate:
    """Tests for aggregate_evaluate method."""

    def test_aggregate_evaluate_empty_results(self, tmp_path):
        """Test aggregate_evaluate with empty results."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
        )

        loss, metrics = strategy.aggregate_evaluate(
            server_round=1,
            results=[],
            failures=[],
        )

        assert loss is None
        assert metrics == {}

    def test_aggregate_evaluate_computes_weighted_dice(self, tmp_path):
        """Test aggregate_evaluate computes weighted dice correctly."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
        )

        # First do a fit to populate client_metrics
        strategy.client_metrics["0"] = [{"round": 1, "train_loss": 0.5}]
        strategy.client_metrics["1"] = [{"round": 1, "train_loss": 0.4}]
        strategy.server_rounds = [{"round": 1}]

        mock_client1 = MagicMock()
        mock_client1.cid = "0"
        mock_eval_res1 = MagicMock()
        mock_eval_res1.metrics = {"dice": 0.8, "loss": 0.3}
        mock_eval_res1.num_examples = 100
        mock_eval_res1.loss = 0.3

        mock_client2 = MagicMock()
        mock_client2.cid = "1"
        mock_eval_res2 = MagicMock()
        mock_eval_res2.metrics = {"dice": 0.9, "loss": 0.2}
        mock_eval_res2.num_examples = 100
        mock_eval_res2.loss = 0.2

        results = [
            (mock_client1, mock_eval_res1),
            (mock_client2, mock_eval_res2),
        ]

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_evaluate",
            return_value=(0.25, {}),
        ):
            loss, metrics = strategy.aggregate_evaluate(
                server_round=1, results=results, failures=[]
            )

        # Weighted dice: (0.8*100 + 0.9*100) / 200 = 0.85
        assert "dice" in metrics
        dice = metrics["dice"]
        assert isinstance(dice, float)
        assert abs(dice - 0.85) < 1e-6


class TestDPFedAvgCheckpointing:
    """Tests for checkpoint saving functionality."""

    def test_save_checkpoints_no_parameters(self, tmp_path):
        """Test _save_checkpoints returns False when no parameters."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
        )

        # latest_parameters is None by default
        result = strategy._save_checkpoints(current_metric=0.8)

        assert result is False

    def test_save_checkpoints_success(self, tmp_path):
        """Test _save_checkpoints returns True on success."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
        )

        # Initialize checkpoint
        params = [np.random.randn(10, 10).astype(np.float32)]
        strategy.checkpoint_manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=5,
        )

        # Set latest parameters
        from flwr.common import ndarrays_to_parameters

        strategy.latest_parameters = ndarrays_to_parameters(params)
        strategy.current_round = 1

        result = strategy._save_checkpoints(current_metric=0.8)

        assert result is True


class TestDPFedAvgPrivacyIntegration:
    """Tests for privacy accountant integration."""

    def test_privacy_accountant_accumulates_across_rounds(self, tmp_path):
        """Test that privacy accountant accumulates epsilon across rounds."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
            noise_multiplier=1.0,
            total_clients=2,
        )

        epsilons = []

        for round_num in range(1, 4):
            mock_client = MagicMock()
            mock_client.cid = "0"
            mock_fit_res = MagicMock()
            mock_fit_res.metrics = {
                "epsilon": 0.5,
                "sample_rate": 0.01,
                "steps": 100,
            }
            mock_fit_res.num_examples = 500
            mock_fit_res.parameters = MagicMock()
            mock_fit_res.parameters.tensors = [np.random.randn(10, 10).tobytes()]
            mock_fit_res.parameters.tensor_type = "numpy.ndarray"

            results = [(mock_client, mock_fit_res)]

            with patch(
                "flwr.server.strategy.FedAvg.aggregate_fit",
                return_value=(MagicMock(), {}),
            ):
                _, metrics = strategy.aggregate_fit(
                    server_round=round_num, results=results, failures=[]
                )

            epsilons.append(metrics["cumulative_sample_epsilon"])

        # Epsilon should increase across rounds
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1]


class TestDPFedAvgConfigureFit:
    """Tests for configure_fit method."""

    def test_configure_fit_includes_noise_multiplier(self, tmp_path):
        """Test configure_fit includes noise_multiplier in config."""
        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
            noise_multiplier=2.5,
            local_epochs=10,
        )

        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 10
        mock_client = MagicMock()
        mock_client_manager.sample.return_value = [mock_client]

        mock_params = MagicMock()

        result = strategy.configure_fit(
            server_round=1,
            parameters=mock_params,
            client_manager=mock_client_manager,
        )

        # Result is list of (client, FitIns)
        assert len(result) == 1
        _, fit_ins = result[0]
        assert fit_ins.config["noise_multiplier"] == 2.5
        assert fit_ins.config["local_epochs"] == 10

    def test_configure_fit_mid_round_resume(self, tmp_path):
        """Test configure_fit signals mid-round resume."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        strategy = DPFedAvg(
            run_dir=tmp_path / "results",
            run_name="test",
            start_round=3,
            is_mid_round_resume=True,
        )

        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 10
        mock_client = MagicMock()
        mock_client_manager.sample.return_value = [mock_client]

        mock_params = MagicMock()

        result = strategy.configure_fit(
            server_round=1,  # Flower's internal round (1-indexed)
            parameters=mock_params,
            client_manager=mock_client_manager,
        )

        _, fit_ins = result[0]
        # actual_round = start_round + server_round - 1 = 3 + 1 - 1 = 3
        assert fit_ins.config["server_round"] == 3
        assert fit_ins.config["resume_from_checkpoint"] is True


class TestDPFedAvgSaveLogs:
    """Tests for save_logs functionality."""

    def test_save_logs_creates_files(self, tmp_path):
        """Test save_logs creates metrics.json and history.json."""
        run_dir = tmp_path / "results"
        run_dir.mkdir(parents=True)

        strategy = DPFedAvg(
            run_dir=run_dir,
            run_name="test",
            num_rounds=5,
            save_metrics=True,
        )

        # Add some round data
        strategy.server_rounds.append(
            {
                "round": 1,
                "sample_epsilon": 0.5,
                "cumulative_sample_epsilon": 0.5,
                "cumulative_user_epsilon": 0.0,
                "num_clients": 2,
                "aggregated_dice": 0.8,
                "aggregated_loss": 0.3,
            }
        )
        strategy.client_metrics["0"] = [{"round": 1, "train_loss": 0.5}]

        strategy.save_logs()

        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "history.json").exists()

        # Verify content
        with open(run_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["config"] == "test"
        assert metrics["final_dice"] == 0.8

        with open(run_dir / "history.json") as f:
            history = json.load(f)
        assert "server" in history
        assert "clients" in history

    def test_save_logs_disabled(self, tmp_path):
        """Test save_logs does nothing when save_metrics is False."""
        run_dir = tmp_path / "results"
        run_dir.mkdir(parents=True)

        strategy = DPFedAvg(
            run_dir=run_dir,
            run_name="test",
            save_metrics=False,
        )

        strategy.save_logs()

        assert not (run_dir / "metrics.json").exists()
        assert not (run_dir / "history.json").exists()
