"""Signal processing utilities for resampling, filtering, and gap detection.

Provides:
- Anti-aliasing filters for downsampling
- Circular-aware filtering for angles
- Gap detection to avoid interpolating across data dropouts
- Sampling analysis utilities
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fmd.analysis.operations import wrap_angle, unwrap_angle
from fmd.analysis.core import detect_sample_rate
from fmd.analysis.time_grid import make_time_grid_inclusive


def butterworth_lowpass(
    data: ArrayLike,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter.

    Args:
        data: Input signal
        cutoff: Cutoff frequency in Hz
        fs: Sample rate in Hz
        order: Filter order (default 4)

    Returns:
        Filtered signal
    """
    from scipy.signal import butter, filtfilt

    data = np.asarray(data)

    # Handle edge case where cutoff >= Nyquist
    nyquist = fs / 2
    if cutoff >= nyquist:
        return data.copy()

    # Design filter
    b, a = butter(order, cutoff / nyquist, btype='low')

    # Apply zero-phase filtering
    # Handle short signals
    if len(data) <= 3 * order:
        return data.copy()

    return filtfilt(b, a, data)


def circular_lowpass(
    data: ArrayLike,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply low-pass filter to circular data (angles).

    Filters angles via unit vectors (sin/cos) to avoid wrap-around artifacts:
      angle -> (sin, cos) -> filter components -> atan2 -> wrap.

    Args:
        data: Angle data in radians
        cutoff: Cutoff frequency in Hz
        fs: Sample rate in Hz
        order: Filter order

    Returns:
        Filtered angles in [-π, π]
    """
    data = np.asarray(data, dtype=float)

    # Preserve NaNs by filtering contiguous valid segments only.
    out = np.full_like(data, np.nan, dtype=float)
    valid = ~np.isnan(data)
    if not np.any(valid):
        return out

    # Find contiguous valid runs.
    idx = np.flatnonzero(valid)
    # Split where gaps between valid indices are > 1
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)

    for run in runs:
        if run.size == 0:
            continue

        segment = data[run]
        sin_x = np.sin(segment)
        cos_x = np.cos(segment)

        sin_y = butterworth_lowpass(sin_x, cutoff, fs, order)
        cos_y = butterworth_lowpass(cos_x, cutoff, fs, order)

        out[run] = np.arctan2(sin_y, cos_y)

    return wrap_angle(out)


def find_gaps(
    timestamps: ArrayLike,
    threshold: float | None = None,
) -> list[tuple[int, int, float]]:
    """Find gaps in timestamp sequence.

    Args:
        timestamps: Array of timestamps
        threshold: Gap threshold (default: 3x median sample interval)

    Returns:
        List of (start_idx, end_idx, gap_size) tuples
    """
    timestamps = np.asarray(timestamps)

    if len(timestamps) < 2:
        return []

    diffs = np.diff(timestamps)

    if threshold is None:
        median_dt = np.median(diffs)
        threshold = 3.0 * median_dt

    gaps = []
    for i, dt in enumerate(diffs):
        if dt > threshold:
            gaps.append((i, i + 1, float(dt)))

    return gaps


def resample_to_rate(
    df: pd.DataFrame,
    target_rate: float,
    time_column: str = "time",
    circular_columns: list[str] | None = None,
    max_gap: float | None = None,
) -> pd.DataFrame:
    """Resample a DataFrame to a target sample rate.

    Uses appropriate methods for circular vs linear data.
    Applies anti-aliasing when downsampling.

    Args:
        df: Input DataFrame
        target_rate: Target sample rate in Hz
        time_column: Name of time column
        circular_columns: List of columns containing circular data
        max_gap: Maximum gap to interpolate across

    Returns:
        Resampled DataFrame
    """
    if circular_columns is None:
        circular_columns = []

    times = df[time_column].values
    if len(times) < 2:
        return df.copy()

    # Detect current rate
    current_rate = 1.0 / np.median(np.diff(times))

    # Create new time grid
    t_start = times[0]
    t_end = times[-1]
    dt = 1.0 / target_rate
    new_times = make_time_grid_inclusive(float(t_start), float(t_end), float(dt))

    result = {time_column: new_times}

    # Handle gaps: use default 3x-median threshold for gap detection,
    # then apply max_gap for masking interpolated regions (EDGE-8)
    gaps = find_gaps(times) if max_gap else []
    # Filter to only gaps exceeding max_gap
    if max_gap:
        gaps = [(s, e, sz) for s, e, sz in gaps if sz > max_gap]

    for col in df.columns:
        if col == time_column:
            continue

        is_circular = col in circular_columns

        if target_rate < current_rate:
            # Downsampling - use decimation with anti-aliasing
            ratio = float(current_rate / target_rate)
            factor = int(np.round(ratio))
            eps = max(1e-6, 1e-6 * abs(ratio))
            if factor < 1 or abs(ratio - factor) > eps:
                raise ValueError(
                    f"Downsampling requires an integer factor: current_rate/target_rate ~= {ratio:.6g}. "
                    f"Got factor={factor} (eps={eps:g})."
                )

            if is_circular:
                # Ensure endpoint exists in the decimated series by explicitly including the final sample.
                unwrapped = unwrap_angle(df[col].values)
                new_nyquist = current_rate / (2 * factor)
                cutoff = 0.8 * new_nyquist
                filtered = butterworth_lowpass(unwrapped, cutoff, current_rate)
                decimated = filtered[::factor]
                temp_times = times[::factor][: len(decimated)]
                if len(temp_times) == 0 or temp_times[-1] != t_end:
                    decimated = np.append(decimated, filtered[-1])
                    temp_times = np.append(temp_times, t_end)
                decimated = wrap_angle(decimated)
            else:
                new_nyquist = current_rate / (2 * factor)
                cutoff = 0.8 * new_nyquist
                filtered = butterworth_lowpass(df[col].values, cutoff, current_rate)
                decimated = filtered[::factor]
                temp_times = times[::factor][: len(decimated)]
                if len(temp_times) == 0 or temp_times[-1] != t_end:
                    decimated = np.append(decimated, filtered[-1])
                    temp_times = np.append(temp_times, t_end)

            interp_series = pd.Series(decimated, index=temp_times)

        else:
            # Upsampling or same rate - use interpolation
            interp_series = pd.Series(df[col].values, index=times)

        # Interpolate to new times
        combined = interp_series.reindex(interp_series.index.union(new_times))

        if is_circular:
            # Unwrap before interpolation
            unwrapped = unwrap_angle(combined.values)
            combined = pd.Series(unwrapped, index=combined.index)

        interpolated = combined.interpolate(method="linear")
        values = interpolated.reindex(new_times).values

        if is_circular:
            values = wrap_angle(values)

        # Mark gaps as NaN
        for gap_start_idx, gap_end_idx, _ in gaps:
            gap_start_time = times[gap_start_idx]
            gap_end_time = times[gap_end_idx]
            mask = (new_times > gap_start_time) & (new_times < gap_end_time)
            values[mask] = np.nan

        result[col] = values

    return pd.DataFrame(result)


def analyze_sampling(
    times: np.ndarray,
    gap_threshold: float | None = None,
) -> dict:
    """Analyze sampling characteristics of a time series.

    Args:
        times: Array of timestamps in seconds
        gap_threshold: Optional gap threshold in seconds (default: 3x median interval)

    Returns:
        Dictionary with analysis results:
        - total_samples: int
        - time_range: tuple[float, float]
        - duration: float
        - sample_rate: float | None
        - interval_stats: dict with mean, median, std, min, max
        - regularity_cv: float
        - gaps: list of (start_idx, end_idx, size) tuples
    """
    if len(times) < 2:
        raise ValueError("Need at least 2 samples for analysis")

    # Basic statistics
    total_samples = len(times)
    time_range = (float(times[0]), float(times[-1]))
    duration = time_range[1] - time_range[0]

    # Detect sample rate
    sample_rate = detect_sample_rate(times)

    # Interval statistics
    diffs = np.diff(times)
    interval_stats = {
        "mean": float(np.mean(diffs)),
        "median": float(np.median(diffs)),
        "std": float(np.std(diffs)),
        "min": float(np.min(diffs)),
        "max": float(np.max(diffs)),
    }

    # Regularity (coefficient of variation)
    regularity_cv = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else float('inf')

    # Find gaps
    gaps = find_gaps(times, threshold=gap_threshold)

    return {
        "total_samples": total_samples,
        "time_range": time_range,
        "duration": duration,
        "sample_rate": sample_rate,
        "interval_stats": interval_stats,
        "regularity_cv": regularity_cv,
        "gaps": gaps,
    }
