"""Schema for dynamic-simulator output files.

This schema handles CSV files produced by the dynamic-simulator package,
which outputs 6-DOF rigid body simulation data in NED frame with SI units.

State vector columns:
- time: simulation time (s)
- pos_n, pos_e, pos_d: NED position (m)
- vel_u, vel_v, vel_w: body-frame velocity (m/s)
- qw, qx, qy, qz: quaternion (scalar-first)
- omega_p, omega_q, omega_r: body angular velocity (rad/s)

Derived outputs (optional):
- roll, pitch, yaw: Euler angles (rad)
"""

import warnings
import numpy as np
import pandas as pd
from ..base import Schema


class DynamicSimulatorSchema(Schema):
    """Schema for dynamic-simulator output CSV format.

    All values are in SI units. Angles are in radians.
    Coordinate frame is NED (North-East-Down).
    """

    # Core state columns
    POSITION_COLS = ["pos_n", "pos_e", "pos_d"]
    VELOCITY_COLS = ["vel_u", "vel_v", "vel_w"]
    QUATERNION_COLS = ["qw", "qx", "qy", "qz"]
    ANGULAR_VEL_COLS = ["omega_p", "omega_q", "omega_r"]

    # Optional derived outputs
    EULER_COLS = ["roll", "pitch", "yaw"]

    @property
    def name(self) -> str:
        return "dynamic_simulator"

    @property
    def required_columns(self) -> list[str]:
        return (
            ["time"]
            + self.POSITION_COLS
            + self.VELOCITY_COLS
            + self.QUATERNION_COLS
            + self.ANGULAR_VEL_COLS
        )

    @property
    def optional_columns(self) -> list[str]:
        return self.EULER_COLS

    @property
    def output_units(self) -> dict[str, str]:
        """SI units for all columns."""
        return {
            "time": "s",
            "pos_n": "m",
            "pos_e": "m",
            "pos_d": "m",
            "vel_u": "m/s",
            "vel_v": "m/s",
            "vel_w": "m/s",
            "qw": "",
            "qx": "",
            "qy": "",
            "qz": "",
            "omega_p": "rad/s",
            "omega_q": "rad/s",
            "omega_r": "rad/s",
            "roll": "rad",
            "pitch": "rad",
            "yaw": "rad",
        }

    def matches(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame matches dynamic_simulator format.

        Requires all state columns. Quaternion columns are the key
        distinguishing feature from other formats.
        """
        cols = set(df.columns)

        # Must have quaternion columns (distinctive feature)
        if not all(c in cols for c in self.QUATERNION_COLS):
            return False

        # Must have all required columns
        return all(c in cols for c in self.required_columns)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate DataFrame against schema."""
        errors = super().validate(df)

        # Check quaternion normalization (should be close to 1)
        if all(c in df.columns for c in self.QUATERNION_COLS):
            quat_norm = np.sqrt(
                df["qw"] ** 2 + df["qx"] ** 2 + df["qy"] ** 2 + df["qz"] ** 2
            )
            if not np.allclose(quat_norm, 1.0, atol=1e-4):
                errors.append(
                    f"Quaternion not normalized: mean norm = {quat_norm.mean():.4f}"
                )

        return errors

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data (already in SI units, just ensure types).

        Also derives Euler angles from quaternion if not present.
        """
        df = df.copy()

        # Ensure numeric types
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Warn about NaN values produced by coercion
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                warnings.warn(f"Coerced {nan_count} invalid values to NaN in '{col}'")

        # Derive Euler angles if not present
        if "roll" not in df.columns and all(
            c in df.columns for c in self.QUATERNION_COLS
        ):
            roll, pitch, yaw = self._quat_to_euler(
                df["qw"].values,
                df["qx"].values,
                df["qy"].values,
                df["qz"].values,
            )
            df["roll"] = roll
            df["pitch"] = pitch
            df["yaw"] = yaw

        return df

    @staticmethod
    def _quat_to_euler(qw, qx, qy, qz):
        """Convert quaternion arrays to Euler angles (vectorized).

        Uses ZYX (yaw-pitch-roll) convention, NED frame.

        Returns:
            Tuple of (roll, pitch, yaw) arrays in radians
        """
        # Roll (rotation about x-axis)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation about y-axis)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (rotation about z-axis)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
