"""Metric aggregation utilities for federated learning.

This module contains helper functions for aggregating metrics across clients.
"""

from typing import List, Tuple

from flwr.common import Metrics
from loguru import logger


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    if not metrics:
        return {}

    total_samples = sum(num_samples for num_samples, _ in metrics)
    aggregated = {}

    metric_keys = set()
    for _, m in metrics:
        metric_keys.update(m.keys())

    for key in metric_keys:
        values = [
            (num_samples, m.get(key, 0.0)) for num_samples, m in metrics if key in m
        ]
        if values:
            numeric_values = []
            for n, v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append((n, float(v)))

            if numeric_values and total_samples > 0:
                weighted_sum = sum(n * v for n, v in numeric_values)
                aggregated[key] = weighted_sum / total_samples
            elif numeric_values:
                logger.warning(f"Cannot aggregate metric '{key}': total_samples is zero")

    return aggregated
