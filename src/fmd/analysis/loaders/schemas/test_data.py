"""Schema for generated/fixture test data format.

This schema is intended to match files produced by `generate-data`, which writes:
- a required `time` column
- 1+ numeric series columns named `data1`, `data2`, ...
"""

import warnings
import pandas as pd
from ..base import Schema


class TestDataSchema(Schema):
    """
    Schema for simple test data CSV format.

    Expected columns:
    - time: numeric time values
    - data1, data2: numeric data columns
    """

    @property
    def name(self) -> str:
        return "test_data"

    @property
    def required_columns(self) -> list[str]:
        # Only `time` is strictly required; data columns are validated via `matches`/`validate`.
        return ["time"]

    def matches(self, df: pd.DataFrame) -> bool:
        """Match `time` + one-or-more `dataN` columns (and nothing else)."""
        cols = list(df.columns)
        if "time" not in cols:
            return False

        data_cols = [c for c in cols if c.startswith("data") and c[4:].isdigit()]
        if len(data_cols) < 1:
            return False

        # Reject unknown columns to avoid false positives during auto-detection.
        allowed = {"time", *data_cols}
        return all(c in allowed for c in cols)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return validation errors (empty if valid)."""
        errors = super().validate(df)

        cols = list(df.columns)
        data_cols = [c for c in cols if c.startswith("data") and c[4:].isdigit()]
        if len(data_cols) < 1:
            errors.append("Expected at least one data column named like 'data1', 'data2', ...")

        unknown = [c for c in cols if c != "time" and c not in data_cols]
        if unknown:
            errors.append(f"Unexpected columns: {unknown}")

        return errors

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize test data.

        Already has 'time' column, just ensure numeric types.
        """
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Warn about NaN values produced by coercion
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                warnings.warn(f"Coerced {nan_count} invalid values to NaN in '{col}'")

        return df
