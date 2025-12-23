#!/usr/bin/env python3
"""Convert federated SSL checkpoint to downstream-compatible format.

This script converts the unified checkpoint format (with server.parameters as
numpy arrays) to a PyTorch state_dict format that can be loaded for downstream
supervised training.

Usage:
    python scripts/convert_ssl_checkpoint.py results/ssl/checkpoints/best.pt results/ssl/encoder.pt
"""

import argparse
from pathlib import Path

import torch

from dp_fedmed.fl.ssl.model import SSLUNet
from dp_fedmed.models.unet2d import create_unet2d, set_parameters


def convert_checkpoint(input_path: Path, output_path: Path) -> None:
    """Convert unified checkpoint to downstream-compatible format.

    Args:
        input_path: Path to unified checkpoint (best.pt or last.pt)
        output_path: Path to save converted checkpoint
    """
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(  # nosec B614 - loading trusted checkpoint
        input_path, map_location="cpu", weights_only=False
    )

    # Extract parameters from unified checkpoint (dict format)
    server_params = checkpoint["server"]["parameters"]
    round_info = checkpoint["round"]
    privacy_info = checkpoint["privacy"]

    # Create the SSLUNet model to load parameters into
    # Use the same config as pretraining
    base_model = create_unet2d(
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    ssl_model = SSLUNet(base_model, projection_dim=128, hidden_dim=256)

    # Load parameters into model
    set_parameters(ssl_model, server_params)

    # Extract just the base UNet encoder (without projection head)
    backbone = ssl_model.get_backbone()

    # Handle both dict and object formats for round/privacy
    current_round = (
        round_info["current"] if isinstance(round_info, dict) else round_info.current
    )
    best_dice = checkpoint["server"]["best_dice"]
    epsilon = (
        privacy_info["cumulative_sample_epsilon"]
        if isinstance(privacy_info, dict)
        else privacy_info.cumulative_sample_epsilon
    )
    delta = (
        privacy_info["target_delta"]
        if isinstance(privacy_info, dict)
        else privacy_info.target_delta
    )

    # Save in downstream-compatible format
    output_checkpoint = {
        "model_state_dict": backbone.state_dict(),
        "epoch": current_round,
        "metrics": {
            "val_loss": best_dice,
            "privacy": {
                "epsilon": epsilon,
                "delta": delta,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)
    print(f"Saved converted checkpoint to {output_path}")
    print(f"  Rounds trained: {current_round}")
    print(f"  Val loss: {best_dice:.4f}")
    print(f"  Epsilon: {epsilon:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert federated SSL checkpoint for downstream training"
    )
    parser.add_argument("input", type=Path, help="Input checkpoint path (best.pt)")
    parser.add_argument("output", type=Path, help="Output checkpoint path (encoder.pt)")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
