"""Mathematical operations with proper handling of circular quantities.

This module re-exports core circular math from fmd.core and adds
pandas-specific interpolation functions for data analysis workflows.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# Re-export core circular math from fmd.core
from fmd.core import (
    wrap_angle,
    unwrap_angle,
    circular_subtract,
    circular_mean,
    angle_difference_to_vector,
)

def circular_std(angles: ArrayLike) -> float:
    """Compute circular standard deviation of angles (analysis-only).

    This function lives in fmd.analysis (not fmd.core) so fmd.core can remain
    numpy-only. Requires SciPy.
    """
    try:
        from scipy.stats import circstd
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "circular_std requires SciPy. Install with: uv sync"
        ) from e

    angles = np.asarray(angles, dtype=float)
    # Omit NaNs to behave like typical dropna() callers.
    return float(circstd(angles, high=np.pi, low=-np.pi, nan_policy="omit"))


def circular_interpolate(
    series: pd.Series,
    new_index: pd.Index,
    method: str = "linear",
) -> pd.Series:
    """Interpolate a Series of angles to a new index.

    Handles wraparound by interpolating on the unit circle (sin/cos)
    and converting back to angles. This is robust to wrap boundaries
    and NaNs.

    Args:
        series: Series with angle values in radians
        new_index: Target index for interpolation
        method: Interpolation method ('linear', 'cubic', etc.)

    Returns:
        Interpolated Series with angles wrapped to [-π, π]
    """
    # Interpolate on unit vectors to avoid unwrap issues and handle NaNs.
    union_index = series.index.union(new_index)

    sin_s = pd.Series(np.sin(series.values), index=series.index).reindex(union_index).interpolate(method=method)
    cos_s = pd.Series(np.cos(series.values), index=series.index).reindex(union_index).interpolate(method=method)

    sin_v = sin_s.reindex(new_index).values
    cos_v = cos_s.reindex(new_index).values

    # If sin/cos both near zero, direction is ambiguous (e.g. half-way between opposite angles).
    r = np.hypot(sin_v, cos_v)
    angle = np.arctan2(sin_v, cos_v)
    angle = np.where(r < 1e-8, np.nan, angle)

    return pd.Series(wrap_angle(angle), index=new_index)


def linear_interpolate(
    series: pd.Series,
    new_index: pd.Index,
    method: str = "linear",
) -> pd.Series:
    """Standard linear interpolation for non-circular quantities.

    Args:
        series: Series to interpolate
        new_index: Target index
        method: Interpolation method

    Returns:
        Interpolated Series
    """
    combined = series.reindex(series.index.union(new_index))
    interpolated = combined.interpolate(method=method)
    return interpolated.reindex(new_index)


def interpolate_with_max_gap(
    series: pd.Series,
    new_index: pd.Index,
    max_gap: float,
    is_circular: bool = False,
    method: str = "linear",
) -> pd.Series:
    """Interpolate with a maximum gap limit.

    Does not interpolate across gaps larger than max_gap.
    Returns NaN for values in large gaps.

    Args:
        series: Series to interpolate (index should be timestamps or numeric)
        new_index: Target index
        max_gap: Maximum gap size (in same units as index)
        is_circular: Whether values are circular (angles)
        method: Interpolation method

    Returns:
        Interpolated Series with NaN in large gaps
    """
    if len(series) < 2:
        return pd.Series(np.nan, index=new_index)

    # Do the interpolation first
    if is_circular:
        result = circular_interpolate(series, new_index, method)
    else:
        result = linear_interpolate(series, new_index, method)

    # Find gaps in original data
    orig_index = series.index.values
    orig_vals = orig_index.astype(float)
    gaps = np.diff(orig_vals)

    # Mark interpolated values that fall strictly inside large gaps as NaN.
    # Vectorized via searchsorted to avoid per-gap boolean masking.
    if len(gaps) == 0:
        return result

    gap_mask = gaps > max_gap
    if not np.any(gap_mask):
        return result

    new_vals = np.asarray(new_index.values, dtype=float)
    gap_starts = orig_vals[:-1][gap_mask]
    gap_ends = orig_vals[1:][gap_mask]

    # For exclusive (>, <) semantics:
    # start_idx: first new_vals > gap_start  => side="right"
    # end_idx:   first new_vals >= gap_end  => side="left"
    start_idxs = np.searchsorted(new_vals, gap_starts, side="right")
    end_idxs = np.searchsorted(new_vals, gap_ends, side="left")

    # Set NaNs in each [start:end) region.
    # Use iloc to keep index/shape stable even if new_index has duplicates.
    for s, e in zip(start_idxs, end_idxs):
        if e > s:
            result.iloc[s:e] = np.nan

    return result


__all__ = [
    # Core circular math (re-exported from fmd.core)
    "wrap_angle",
    "unwrap_angle",
    "circular_subtract",
    "circular_mean",
    "circular_std",
    "angle_difference_to_vector",
    # Pandas-specific interpolation
    "circular_interpolate",
    "linear_interpolate",
    "interpolate_with_max_gap",
]
