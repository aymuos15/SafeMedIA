"""FedAvg strategy with privacy budget tracking for SSL pretraining.

This module defines the DPFedAvgSSL strategy that extends BaseDPStrategy
for self-supervised learning with validation loss aggregation.
"""

from typing import Dict, List, Optional, Tuple

from flwr.common import Scalar
from loguru import logger

from dp_fedmed.fl.base.strategy import BaseDPStrategy


class DPFedAvgSSL(BaseDPStrategy):
    """FedAvg strategy with privacy tracking for SSL pretraining.

    This strategy extends BaseDPStrategy with SSL-specific functionality:
    - Validation loss aggregation
    - SSL-focused logging
    """

    def _get_primary_metric_name(self) -> str:
        """Get the primary evaluation metric name.

        Returns:
            'val_loss' for SSL pretraining
        """
        return "val_loss"

    def _aggregate_evaluation_metrics(
        self,
        results: List[Tuple],
        metrics_aggregated: Dict[str, Scalar],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate validation loss from client evaluation results.

        Args:
            results: List of (client_proxy, eval_result) tuples
            metrics_aggregated: Pre-aggregated metrics from parent class

        Returns:
            Tuple of (weighted_val_loss, updated_metrics_dict)
        """
        loss_values = []
        weights = []

        for client_proxy, eval_res in results:
            client_id = str(client_proxy.cid)
            metrics = eval_res.metrics or {}

            val_loss = float(metrics.get("val_loss", 0.0))
            loss_values.append(val_loss)
            weights.append(eval_res.num_examples)

            # Update client's last round entry with eval metrics
            server_round = self.current_round
            client_found = False
            for entry in reversed(self.client_metrics[client_id]):
                if entry["round"] == server_round:
                    entry["val_loss"] = val_loss
                    client_found = True
                    break
            if not client_found:
                logger.warning(
                    f"Round {server_round} not found in client_metrics for client {client_id}"
                )

        if loss_values:
            weighted_loss = sum(
                loss * weight for loss, weight in zip(loss_values, weights)
            ) / sum(weights)
            metrics_aggregated["val_loss"] = float(weighted_loss)
            return weighted_loss, metrics_aggregated

        return None, metrics_aggregated
