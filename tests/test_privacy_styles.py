"""Integration tests for different privacy styles."""

import torch
import flwr as fl

from dp_fedmed.config import (
    DPStyle,
)
from dp_fedmed.fl.base.strategy import DPStrategy
from dp_fedmed.fl.server.factory import server_fn
from dp_fedmed.fl.client.dp_client import DPFlowerClient


def test_strategy_initialization_with_different_styles(tmp_path):
    """Test that DPStrategy can be initialized with different privacy styles."""
    run_dir = tmp_path / "run"

    # Test None style
    strategy = DPStrategy(
        target_delta=1e-5,
        run_dir=run_dir,
        num_rounds=1,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
    )
    assert strategy.privacy_accountant.target_delta == 1e-5


def test_server_factory_wraps_strategy_correctly(tmp_path, monkeypatch):
    """Test that server_fn correctly wraps strategy based on style."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def create_test_config(style):
        content = f"""
[data]
data_dir = "{data_dir}"

[model]
in_channels = 1
out_channels = 2
channels = [16, 32]
strides = [2]

[federated]
num_clients = 2
num_rounds = 1

[training]
local_epochs = 1
learning_rate = 0.01

[privacy]
style = "{style}"
target_delta = 1e-5
[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0
[privacy.user]
noise_multiplier = 0.5
max_grad_norm = 0.1
"""
        config_path = configs_dir / f"{style}.toml"
        config_path.write_text(content)
        return config_path

    # Test "none" style - should NOT be wrapped
    config_none = create_test_config("none")
    context_none = fl.common.Context(
        run_id=0,
        state=fl.common.RecordSet(),
        run_config={"config-file": str(config_none)},
        node_id=0,
        node_config={},
    )
    components_none = server_fn(context_none)
    assert isinstance(components_none.strategy, DPStrategy)

    # Test "user" style - SHOULD be wrapped
    config_user = create_test_config("user")
    context_user = fl.common.Context(
        run_id=1,
        state=fl.common.RecordSet(),
        run_config={"config-file": str(config_user)},
        node_id=0,
        node_config={},
    )
    components_user = server_fn(context_user)
    assert isinstance(
        components_user.strategy,
        fl.server.strategy.DifferentialPrivacyServerSideFixedClipping,
    )

    # Test "hybrid" style - SHOULD be wrapped
    config_hybrid = create_test_config("hybrid")
    context_hybrid = fl.common.Context(
        run_id=2,
        state=fl.common.RecordSet(),
        run_config={"config-file": str(config_hybrid)},
        node_id=0,
        node_config={},
    )
    components_hybrid = server_fn(context_hybrid)
    assert isinstance(
        components_hybrid.strategy,
        fl.server.strategy.DifferentialPrivacyServerSideFixedClipping,
    )


def test_client_initialization_with_different_styles(
    dummy_model, dummy_dataloader, tmp_path
):
    """Test that DPFlowerClient handles different styles correctly."""

    def get_client(style):
        model_config = {
            "in_channels": 1,
            "out_channels": 2,
            "channels": [16, 32],
            "strides": [2],
            "num_res_units": 1,
        }
        training_config = {"local_epochs": 1, "learning_rate": 0.01}
        privacy_config = {
            "style": style,
            "target_delta": 1e-5,
            "sample": {"noise_multiplier": 1.0, "max_grad_norm": 1.0},
        }
        return DPFlowerClient(
            train_loader=dummy_dataloader,
            test_loader=dummy_dataloader,
            model_config=model_config,
            training_config=training_config,
            privacy_config=privacy_config,
            device=torch.device("cpu"),
        )

    # None style
    client_none = get_client(DPStyle.NONE)
    assert client_none.privacy_config["style"] == DPStyle.NONE

    # Sample style
    client_sample = get_client(DPStyle.SAMPLE)
    assert client_sample.privacy_config["style"] == DPStyle.SAMPLE
