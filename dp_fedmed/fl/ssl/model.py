"""SSL model wrapper with projection head for contrastive learning."""

from typing import Tuple

import torch
import torch.nn as nn


class SSLUNet(nn.Module):
    """UNet wrapper with projection head for SSL pretraining.

    This model wraps a base UNet model and adds a projection head
    for contrastive learning (SimCLR, MoCo, SimSiam).
    """

    def __init__(
        self, base_model: nn.Module, projection_dim: int = 128, hidden_dim: int = 256
    ):
        """Initialize SSL UNet.

        Args:
            base_model: Base UNet model (encoder)
            projection_dim: Output dimension of projection head
            hidden_dim: Hidden dimension of projection head
        """
        super().__init__()
        self.backbone = base_model

        # Extract feature dimension from the base model
        feature_dim = self._get_feature_dim(base_model)

        # Use GroupNorm instead of BatchNorm for DP compatibility
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def _get_feature_dim(self, model: nn.Module) -> int:
        """Estimate feature dimension from model.

        Args:
            model: PyTorch model

        Returns:
            Estimated feature dimension
        """
        try:
            for module in reversed(list(model.modules())):
                if isinstance(module, nn.Conv2d):
                    return module.out_channels
        except (AttributeError, TypeError):
            # Fall through to default if model structure can't be inspected
            pass

        return 128  # Default estimate

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through backbone and projection head.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (backbone_features, projected_features)
        """
        features_flat = self._get_backbone_features(x)
        z = self.projection_head(features_flat)

        return features_flat, z

    def _get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        output = self.backbone(x)

        if isinstance(output, torch.Tensor) and len(output.shape) == 4:
            # Global average pooling: [B, C, H, W] -> [B, C]
            features = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
            features = features.view(features.size(0), -1)
            return features

        return output

    def get_backbone(self) -> nn.Module:
        """Get the backbone model without projection head.

        Returns:
            Backbone model
        """
        return self.backbone
