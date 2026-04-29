"""Core data containers for multi-rate vehicle telemetry.

Design principles:
- DataStream: Single-rate data with metadata (units, source rate)
- VehicleLog: Multi-rate container with lazy alignment
- All values in SI units with standard NED frame variable names
- Circular quantities (angles) handled automatically
"""

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd

from fmd.core.units import (
    VARIABLE_SCHEMA,
    get_display_name,
    convert_from_si,
    is_circular,
)
from fmd.analysis.operations import (
    circular_subtract,
    circular_mean,
    circular_interpolate,
    linear_interpolate,
    interpolate_with_max_gap,
    wrap_angle,
)
from fmd.analysis.time_grid import make_time_grid_inclusive


def detect_sample_rate(timestamps: np.ndarray) -> float | None:
    """Detect the nominal sample rate from timestamps.

    Args:
        timestamps: Array of timestamps (numeric, in seconds)

    Returns:
        Estimated sample rate in Hz, or None if irregular/insufficient data
    """
    if len(timestamps) < 2:
        return None

    diffs = np.diff(timestamps)

    # Use median to be robust to outliers/gaps
    median_dt = np.median(diffs)
    if median_dt <= 0:
        return None

    return 1.0 / median_dt


@dataclass
class DataStream:
    """Single-rate data container with metadata.

    Holds a DataFrame with a consistent sample rate, plus metadata
    about units and source. Provides circular-aware operations.

    Attributes:
        df: DataFrame with 'time' column and data columns
        name: Identifier for this stream (e.g., 'telemetry', 'imu')
        source_rate: Detected sample rate in Hz (None if irregular)
        units: Mapping of column name -> SI unit string
        metadata: Additional metadata (source file, etc.)
    """

    df: pd.DataFrame
    name: str
    source_rate: float | None = None
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect sample rate if not provided."""
        if self.source_rate is None and "time" in self.df.columns:
            self.source_rate = detect_sample_rate(self.df["time"].values)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        rate_str = f"{self.source_rate:.1f}Hz" if self.source_rate else "irregular"
        return f"DataStream('{self.name}', {len(self)} samples, {rate_str})"

    @property
    def columns(self) -> list[str]:
        """List of data columns (excluding 'time')."""
        return [c for c in self.df.columns if c != "time"]

    @property
    def time(self) -> pd.Series:
        """Time column as Series."""
        return self.df["time"]

    def _get_quantity_type(self, column: str) -> str:
        """Get the quantity type for a column."""
        if column in VARIABLE_SCHEMA:
            return VARIABLE_SCHEMA[column]
        return "dimensionless"

    def _is_circular(self, column: str) -> bool:
        """Check if a column contains circular data."""
        return is_circular(column)

    # =========================================================================
    # Circular-aware operations
    # =========================================================================

    def subtract(self, col_a: str, col_b: str, name: str | None = None) -> pd.Series:
        """Subtract two columns with automatic circular handling.

        If the columns are angles, uses circular subtraction.

        Args:
            col_a: First column name
            col_b: Second column name
            name: Name for the result Series (optional)

        Returns:
            col_a - col_b with proper handling for circular quantities

        Example:
            >>> stream.subtract('yaw', 'cog')  # Computes leeway correctly
        """
        a = self.df[col_a].values
        b = self.df[col_b].values

        known_a = col_a in VARIABLE_SCHEMA
        known_b = col_b in VARIABLE_SCHEMA

        # Hard contract: only use circular subtraction if BOTH columns are known circular.
        # If one is circular and the other is a known non-circular quantity, raise to avoid
        # silently hiding unit/type bugs. Unknown variables default to linear behavior.
        if known_a and known_b:
            circ_a = self._is_circular(col_a)
            circ_b = self._is_circular(col_b)
            if circ_a != circ_b:
                raise ValueError(
                    f"Cannot subtract mismatched quantity types: '{col_a}' (circular={circ_a}) "
                    f"and '{col_b}' (circular={circ_b})."
                )
            if circ_a and circ_b:
                result = circular_subtract(a, b)
            else:
                result = a - b
        else:
            # At least one column is not in VARIABLE_SCHEMA — default to linear
            # subtraction. Register columns in VARIABLE_SCHEMA for circular handling.
            result = a - b

        return pd.Series(result, index=self.df.index, name=name or f"{col_a}-{col_b}")

    def mean(self, column: str) -> float:
        """Compute mean with automatic circular handling.

        Args:
            column: Column name

        Returns:
            Mean value (circular mean for angles)
        """
        values = self.df[column].dropna().values

        if self._is_circular(column):
            return circular_mean(values)
        return float(np.mean(values))

    def std(self, column: str) -> float:
        """Compute standard deviation with circular handling for angles.

        Args:
            column: Column name

        Returns:
            Standard deviation
        """
        from fmd.analysis.operations import circular_std

        values = self.df[column].dropna().values

        if self._is_circular(column):
            return circular_std(values)
        return float(np.std(values))

    def interpolate_column(
        self,
        column: str,
        new_index: pd.Index,
        method: str = "linear",
        max_gap: float | None = None,
    ) -> pd.Series:
        """Interpolate a single column to a new index.

        Args:
            column: Column to interpolate
            new_index: Target index
            method: Interpolation method
            max_gap: Maximum gap to interpolate across (seconds)

        Returns:
            Interpolated Series
        """
        series = self.df.set_index("time")[column]
        is_circ = self._is_circular(column)

        if max_gap is not None:
            return interpolate_with_max_gap(
                series, new_index, max_gap, is_circular=is_circ, method=method
            )

        if is_circ:
            return circular_interpolate(series, new_index, method)
        return linear_interpolate(series, new_index, method)

    def resample_to(
        self,
        target_rate: float,
        method: str = "linear",
        max_gap: float | None = None,
    ) -> "DataStream":
        """Resample this stream to a new sample rate.

        Args:
            target_rate: Target sample rate in Hz
            method: Interpolation method
            max_gap: Maximum gap to interpolate across

        Returns:
            New DataStream at target rate
        """
        # Create new time index
        t_start = self.df["time"].min()
        t_end = self.df["time"].max()
        dt = 1.0 / target_rate
        new_times = make_time_grid_inclusive(float(t_start), float(t_end), float(dt))
        new_index = pd.Index(new_times)

        # Interpolate each column
        new_data = {"time": new_times}
        for col in self.columns:
            new_data[col] = self.interpolate_column(
                col, new_index, method, max_gap
            ).values

        return DataStream(
            df=pd.DataFrame(new_data),
            name=self.name,
            source_rate=target_rate,
            units=self.units.copy(),
            metadata={**self.metadata, "resampled_from": self.source_rate},
        )

    # =========================================================================
    # Display helpers
    # =========================================================================

    def to_display_units(self, column: str) -> tuple[pd.Series, str]:
        """Convert a column to display units.

        Args:
            column: Column name

        Returns:
            (converted_series, unit_string)
        """
        qty_type = self._get_quantity_type(column)
        values = self.df[column].values

        # Get conversion multiplier and unit from a reference value
        multiplier, unit = convert_from_si(1.0, qty_type)

        # Apply conversion to all values
        converted_values = values * multiplier

        return (
            pd.Series(converted_values, index=self.df.index, name=column),
            unit,
        )

    def get_display_name(self, column: str, domain: str = "generic") -> str:
        """Get display name for a column.

        Args:
            column: Column name
            domain: Domain context ('sailing', 'aviation', 'generic')

        Returns:
            Human-readable display name
        """
        return get_display_name(column, domain)


@dataclass
class VehicleLog:
    """Multi-rate data container with lazy alignment.

    Holds multiple DataStreams at different sample rates.
    Only resamples when explicitly requested via align_to().

    Attributes:
        streams: Dictionary of name -> DataStream
        domain: Domain for display purposes ('sailing', 'aviation', 'generic')
        metadata: Additional metadata
    """

    streams: dict[str, DataStream] = field(default_factory=dict)
    domain: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        stream_info = ", ".join(
            f"{name}: {len(s)} @ {s.source_rate or '?'}Hz"
            for name, s in self.streams.items()
        )
        return f"VehicleLog({stream_info})"

    def add_stream(self, stream: DataStream) -> None:
        """Add a DataStream to the log.

        Args:
            stream: DataStream to add
        """
        self.streams[stream.name] = stream

    def get_stream(self, name: str) -> DataStream:
        """Get a stream by name.

        Args:
            name: Stream name

        Returns:
            The DataStream

        Raises:
            KeyError: If stream not found
        """
        if name not in self.streams:
            available = list(self.streams.keys())
            raise KeyError(f"Stream '{name}' not found. Available: {available}")
        return self.streams[name]

    def align_to(
        self,
        target: str,
        *,
        mode: str = "nearest",
        method: str = "linear",
        max_gap: float | None = 1.0,
    ) -> pd.DataFrame:
        """Align all streams to a target stream's timestamps.

        Creates a single DataFrame with all columns from all streams,
        resampled to the target stream's time index.

        Args:
            target: Name of stream to use as time reference
            mode: Alignment mode:
                - "nearest": nearest-neighbor within max_gap tolerance (no interpolation)
                - "interpolate": interpolate values onto target timestamps (may invent data)
            method: Interpolation method (only used when mode="interpolate")
            max_gap: Maximum gap to match/interpolate across (seconds). In "nearest"
                mode, this is passed as merge tolerance; in "interpolate" mode, it
                limits interpolation across large gaps.

        Returns:
            DataFrame with all columns aligned to target timestamps

        Note:
            When columns from different streams share the same name, non-target
            columns are prefixed with the stream name (e.g., ``imu_time`` if
            both target and ``imu`` streams contain ``time``).
        """
        if target not in self.streams:
            available = list(self.streams.keys())
            raise KeyError(f"Target stream '{target}' not found. Available: {available}")

        if mode not in {"nearest", "interpolate"}:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'nearest' or 'interpolate'.")

        if mode == "interpolate":
            warnings.warn(
                "VehicleLog.align_to(mode='interpolate') will interpolate values onto the target "
                "timestamps and may invent data between samples. If you want raw-fidelity alignment "
                "without interpolation, use mode='nearest'.",
                UserWarning,
                stacklevel=2,
            )

        target_stream = self.streams[target]

        if mode == "nearest":
            # Fast path: nearest-neighbor alignment without interpolation.
            # Uses merge_asof which requires sorted keys.
            result = target_stream.df.sort_values("time").reset_index(drop=True).copy()

            for name, stream in self.streams.items():
                if name == target:
                    continue

                other = stream.df.sort_values("time").reset_index(drop=True)
                # Keep only time + data columns (exclude time from stream.columns already)
                cols = ["time"] + [c for c in stream.columns if c in other.columns]
                other = other[cols].copy()

                # Avoid column collisions (except 'time', which is the join key)
                rename_map: dict[str, str] = {}
                for col in other.columns:
                    if col == "time":
                        continue
                    rename_map[col] = col if col not in result.columns else f"{name}_{col}"
                if rename_map:
                    other = other.rename(columns=rename_map)

                merge_kwargs: dict[str, object] = {
                    "on": "time",
                    "direction": "nearest",
                }
                if max_gap is not None:
                    merge_kwargs["tolerance"] = float(max_gap)

                result = pd.merge_asof(result, other, **merge_kwargs)  # type: ignore[arg-type]

            return result

        # mode == "interpolate": interpolate onto the target index
        target_times = target_stream.df["time"].values
        target_index = pd.Index(target_times)

        # Start with target stream's data
        result = target_stream.df.copy()

        # Add columns from other streams
        for name, stream in self.streams.items():
            if name == target:
                continue

            for col in stream.columns:
                # Avoid column name collisions
                result_col = col if col not in result.columns else f"{name}_{col}"
                result[result_col] = stream.interpolate_column(
                    col, target_index, method, max_gap
                ).values

        return result

    @property
    def time_range(self) -> tuple[float, float]:
        """Get the overall time range across all streams.

        Returns:
            (min_time, max_time) in seconds
        """
        t_min = min(s.df["time"].min() for s in self.streams.values())
        t_max = max(s.df["time"].max() for s in self.streams.values())
        return (t_min, t_max)
