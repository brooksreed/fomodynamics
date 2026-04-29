"""Causal lowpass filters."""

import numpy as np
import pandas as pd

from ..base import CausalFilter, FilterResult
from fmd.analysis.operations import wrap_angle


class ExponentialMovingAverage(CausalFilter):
    """Exponential moving average (EMA) lowpass filter.

    A simple causal filter that applies exponential smoothing.
    Only uses past and current data, making it suitable for online use.

    The filter equation is:
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    Args:
        alpha: Smoothing factor between 0 and 1.
               Higher alpha = less smoothing, faster response.
               Lower alpha = more smoothing, slower response.

               Common conversions:
               - From time constant τ: alpha = 1 - exp(-dt/τ)
               - From window size N: alpha ≈ 2 / (N + 1)
        circular: If True, treat values as angles on a circle (e.g. heading).
                  This avoids wrap-around artifacts (e.g. filtering 359°→1°).
        period: Full-scale wrap value for circular data (default: 360.0 degrees).

    Example:
        >>> from fmd.analysis.filters.causal import ExponentialMovingAverage
        >>> ema = ExponentialMovingAverage(alpha=0.1)
        >>> result = ema.apply(data)
        >>> filtered = result.data
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        circular: bool = False,
        period: float = 360.0,
        nan_policy: str = "propagate",
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if period <= 0:
            raise ValueError("period must be > 0")
        if nan_policy != "propagate":
            raise ValueError("nan_policy must be 'propagate'")
        self.alpha = alpha
        self.circular = circular
        self.period = float(period)
        self.nan_policy = nan_policy

    @property
    def name(self) -> str:
        mode = "circular" if self.circular else "linear"
        extra = f", period={self.period:g}" if self.circular else ""
        return f"Exponential Moving Average ({mode}, α={self.alpha:.3f}{extra})"

    @classmethod
    def from_window_size(
        cls,
        window_size: int,
        circular: bool = False,
        period: float = 360.0
    ) -> "ExponentialMovingAverage":
        """Create EMA with alpha derived from effective window size.

        Args:
            window_size: Effective averaging window (like SMA window)
            circular: Whether to use circular filtering (default: False)
            period: Period for circular filtering (default: 360.0)

        Returns:
            ExponentialMovingAverage with alpha ≈ 2 / (N + 1)
        """
        alpha = 2.0 / (window_size + 1)
        return cls(alpha=alpha, circular=circular, period=period)

    @classmethod
    def from_time_constant(cls, tau: float, dt: float) -> "ExponentialMovingAverage":
        """Create EMA with alpha derived from time constant.

        Args:
            tau: Time constant (in same units as dt)
            dt: Sample interval

        Returns:
            ExponentialMovingAverage with alpha = 1 - exp(-dt/τ)
        """
        alpha = 1.0 - np.exp(-dt / tau)
        return cls(alpha=alpha)

    def apply(self, data: np.ndarray | pd.Series) -> FilterResult:
        """Apply exponential moving average to data."""
        is_series = isinstance(data, pd.Series)
        arr = np.asarray(data, dtype=float)

        n = len(arr)
        result = np.empty(n)

        if not self.circular:
            # NaN policy (hard contract): propagate NaN, reset internal state, restart on next valid.
            y_prev = np.nan
            for i in range(n):
                x = arr[i]
                if np.isnan(x):
                    result[i] = np.nan
                    y_prev = np.nan
                    continue

                if np.isnan(y_prev):
                    # Restart from new sample
                    y_prev = x
                else:
                    y_prev = self.alpha * x + (1 - self.alpha) * y_prev
                result[i] = y_prev
        else:
            # Circular EMA via filtering unit vectors.
            theta = (arr / self.period) * (2 * np.pi)  # radians
            sin_x = np.sin(theta)
            cos_x = np.cos(theta)

            sin_prev = np.nan
            cos_prev = np.nan

            for i in range(n):
                x = arr[i]
                if np.isnan(x):
                    result[i] = np.nan
                    sin_prev = np.nan
                    cos_prev = np.nan
                    continue

                if np.isnan(sin_prev) or np.isnan(cos_prev):
                    # Restart from new sample
                    sin_prev = sin_x[i]
                    cos_prev = cos_x[i]
                else:
                    sin_prev = self.alpha * sin_x[i] + (1 - self.alpha) * sin_prev
                    cos_prev = self.alpha * cos_x[i] + (1 - self.alpha) * cos_prev

                angle = np.arctan2(sin_prev, cos_prev)
                angle_original = (angle / (2 * np.pi)) * self.period
                result[i] = wrap_angle(angle_original, low=-self.period / 2, high=self.period / 2)

        if is_series:
            result = pd.Series(result, index=data.index, name=data.name)

        return FilterResult(
            data=result,
            filter_name=self.name,
            filter_type=self.filter_type,
            params={
                "alpha": self.alpha,
                "circular": self.circular,
                "period": self.period,
                "nan_policy": self.nan_policy,
            },
        )

    def __repr__(self) -> str:
        if self.circular:
            return f"ExponentialMovingAverage(alpha={self.alpha}, circular=True, period={self.period})"
        return f"ExponentialMovingAverage(alpha={self.alpha})"
