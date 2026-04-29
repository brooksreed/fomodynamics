"""Centered moving average smoothing filter."""

import numpy as np
import pandas as pd

from ..base import AcausalFilter, FilterResult
from fmd.analysis.operations import circular_mean, wrap_angle


class MovingAverage(AcausalFilter):
    """Centered moving average smoothing.

    A simple acausal filter that averages data points within a symmetric
    window centered on each point. Uses both past and future data.

    Args:
        half_window: Number of points on each side of the center point.
                    Total window size is 2 * half_window + 1 (always odd).
                    Default is 2, giving a window of 5 points.
        circular: If True, treat values as angles on a circle (e.g. heading).
                  This avoids wrap-around artifacts (e.g. averaging 359° and 1°).
        period: Full-scale wrap value for circular data (default: 360.0 degrees).

    Example:
        >>> from fmd.analysis.filters.acausal import MovingAverage
        >>> ma = MovingAverage(half_window=2)  # window of 5 points
        >>> result = ma.apply(data)
        >>> smoothed = result.data
    """

    def __init__(self, half_window: int = 2, *, circular: bool = False, period: float = 360.0):
        if half_window < 0:
            raise ValueError("half_window must be >= 0")
        if period <= 0:
            raise ValueError("period must be > 0")
        self.half_window = half_window
        self.circular = circular
        self.period = float(period)

    @property
    def window_size(self) -> int:
        """Total window size (2 * half_window + 1)."""
        return 2 * self.half_window + 1

    @property
    def name(self) -> str:
        mode = "circular" if self.circular else "linear"
        extra = f", period={self.period:g}" if self.circular else ""
        return f"Moving Average ({mode}, half_window={self.half_window}, size={self.window_size}{extra})"

    def apply(self, data: np.ndarray | pd.Series) -> FilterResult:
        """Apply centered moving average to data.

        Edges are handled by using smaller windows (minimum valid approach).
        """
        is_series = isinstance(data, pd.Series)
        arr = np.asarray(data, dtype=float)

        n = len(arr)

        if not self.circular:
            # Vectorized linear case using pandas rolling
            s = pd.Series(arr)
            window_size = 2 * self.half_window + 1
            result = s.rolling(window_size, min_periods=1, center=True).mean().values
        else:
            result = np.empty(n)
            for i in range(n):
                # Determine window bounds (symmetric around i)
                start = max(0, i - self.half_window)
                end = min(n, i + self.half_window + 1)
                window = arr[start:end]

                valid = ~np.isnan(window)
                if not np.any(valid):
                    result[i] = np.nan
                    continue

                # Convert to radians, compute circular mean, convert back
                theta = (window[valid] / self.period) * (2 * np.pi)
                mean_rad = circular_mean(theta)

                # Convert back to original units and wrap to [-period/2, period/2)
                mean_original = (mean_rad / (2 * np.pi)) * self.period
                # Wrap to [-period/2, period/2) to match standard convention
                result[i] = float(wrap_angle(mean_original, low=-self.period/2, high=self.period/2))

        if is_series:
            result = pd.Series(result, index=data.index, name=data.name)

        return FilterResult(
            data=result,
            filter_name=self.name,
            filter_type=self.filter_type,
            params={
                "half_window": self.half_window,
                "window_size": self.window_size,
                "circular": self.circular,
                "period": self.period,
            },
        )

    def __repr__(self) -> str:
        if self.circular:
            return f"MovingAverage(half_window={self.half_window}, circular=True, period={self.period})"
        return f"MovingAverage(half_window={self.half_window})"
