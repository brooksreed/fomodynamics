"""Causal (online-compatible) filters.

These filters only use past and current data, making them suitable
for real-time / streaming applications.

Use cases:
- Real-time telemetry processing
- Online estimation
- Streaming data pipelines
"""

from .lowpass import ExponentialMovingAverage

__all__ = [
    "ExponentialMovingAverage",
    # Future: "KalmanFilter", "Butterworth", etc.
]
