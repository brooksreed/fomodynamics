"""Acausal (offline) smoothing filters.

These filters use both past and future data points, providing better
smoothing but requiring all data to be available upfront.

Use cases:
- Post-processing recorded data
- Report generation
- Offline analysis
"""

from .moving_average import MovingAverage

__all__ = [
    "MovingAverage",
    # Future: "SavitzkyGolay", "GaussianSmooth", etc.
]
