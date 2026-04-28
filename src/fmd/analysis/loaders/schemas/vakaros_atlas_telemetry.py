"""Schema for Vakaros Atlas vessel telemetry CSV files."""

import math
import warnings
import pandas as pd
import numpy as np
from ..base import Schema
from fmd.analysis.operations import wrap_angle


class VakarosAtlasTelemetrySchema(Schema):
    """
    Schema for Vakaros Atlas vessel telemetry data.

    Input columns:
    - timestamp: ISO8601 datetime string
    - latitude, longitude: decimal degrees
    - sog_kts: speed over ground in knots
    - cog: course over ground in degrees
    - hdg_true: true heading in degrees
    - heel: heel angle in degrees (starboard down positive)
    - trim: trim angle in degrees (bow up positive)

    Output (normalized):
    - time: seconds from start
    - latitude, longitude: decimal degrees
    - sog: speed over ground in m/s
    - cog: course over ground in radians
    - yaw: heading in radians (NED frame)
    - roll: heel in radians (NED frame)
    - pitch: trim in radians (NED frame)
    """

    @property
    def name(self) -> str:
        return "vakaros_atlas_telemetry"

    @property
    def required_columns(self) -> list[str]:
        return [
            "timestamp",
            "latitude",
            "longitude",
            "sog_kts",
            "cog",
            "hdg_true",
            "heel",
            "trim",
        ]

    @property
    def column_units(self) -> dict[str, tuple[str, str]]:
        """Source column units: column_name -> (quantity_type, source_unit)."""
        return {
            "sog_kts": ("speed", "kts"),
            "cog": ("angle", "deg"),
            "hdg_true": ("angle", "deg"),
            "heel": ("angle", "deg"),
            "trim": ("angle", "deg"),
            "latitude": ("position", "deg"),
            "longitude": ("position", "deg"),
        }

    @property
    def column_mapping(self) -> dict[str, str]:
        """Mapping from source column names to standard NED frame names."""
        return {
            "sog_kts": "sog",
            "hdg_true": "yaw",
            "heel": "roll",
            "trim": "pitch",
            # cog stays as cog
            # latitude, longitude stay as-is
        }

    @property
    def output_units(self) -> dict[str, str]:
        """SI units for output columns."""
        return {
            "time": "s",
            "latitude": "deg",
            "longitude": "deg",
            "sog": "m/s",
            "cog": "rad",
            "yaw": "rad",
            "roll": "rad",
            "pitch": "rad",
        }

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize vessel telemetry data.

        - Converts timestamp to 'time' (seconds from start of recording)
        - Converts units to SI (m/s, radians)
        - Renames columns to standard NED frame names
        """
        df = df.copy()

        # Parse timestamps and create time column (seconds from start)
        timestamps = pd.to_datetime(df["timestamp"])
        df["time"] = (timestamps - timestamps.iloc[0]).dt.total_seconds()

        # Convert speed: knots -> m/s
        df["sog"] = pd.to_numeric(df["sog_kts"], errors="coerce") * 0.514444

        # Convert angles: degrees -> radians, wrapped to [-π, π]
        deg_to_rad = math.pi / 180.0

        df["cog"] = wrap_angle(pd.to_numeric(df["cog"], errors="coerce") * deg_to_rad)
        df["yaw"] = wrap_angle(pd.to_numeric(df["hdg_true"], errors="coerce") * deg_to_rad)
        df["roll"] = wrap_angle(pd.to_numeric(df["heel"], errors="coerce") * deg_to_rad)
        df["pitch"] = wrap_angle(pd.to_numeric(df["trim"], errors="coerce") * deg_to_rad)

        # Position stays in degrees (standard for lat/lon)
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Warn about NaN values produced by coercion
        coerced_columns = ["sog", "cog", "yaw", "roll", "pitch", "latitude", "longitude"]
        for col in coerced_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                warnings.warn(f"Coerced {nan_count} invalid values to NaN in '{col}'")

        # Keep original timestamp for reference
        df["timestamp"] = timestamps

        # Select and order output columns
        output_cols = [
            "time",
            "timestamp",
            "latitude",
            "longitude",
            "sog",
            "cog",
            "yaw",
            "roll",
            "pitch",
        ]

        return df[output_cols]
