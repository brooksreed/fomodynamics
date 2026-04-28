"""Base classes for data file loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class LoadResult:
    """Result of loading a data file."""

    df: pd.DataFrame
    """Normalized DataFrame with 'time' column."""

    schema_name: str
    """Name of the schema used to load this data."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (source file, original columns, etc.)."""

    units: dict[str, str] = field(default_factory=dict)
    """Mapping of column name -> SI unit string."""


class Schema(ABC):
    """
    Base class for data schemas.

    A schema defines:
    - What columns are required/optional
    - How to detect if a DataFrame matches this schema
    - How to normalize the data to a standard format
    - Column units and mappings to standard names
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this schema."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Columns that must be present for this schema."""
        pass

    @property
    def optional_columns(self) -> list[str]:
        """Columns that may be present but aren't required."""
        return []

    @property
    def column_units(self) -> dict[str, tuple[str, str]]:
        """Source column units: column_name -> (quantity_type, source_unit).

        Example: {"sog_kts": ("speed", "kts"), "heel": ("angle", "deg")}

        Override in subclass to define unit conversions.
        """
        return {}

    @property
    def column_mapping(self) -> dict[str, str]:
        """Mapping from source column names to standard names.

        Example: {"heel": "roll", "trim": "pitch", "hdg_true": "yaw"}

        Override in subclass to map domain-specific names to standard NED frame.
        """
        return {}

    @property
    def output_units(self) -> dict[str, str]:
        """Output column SI units after normalization.

        Returns mapping of output column name -> SI unit string.
        Override in subclass or computed from column_units.
        """
        return {}

    def matches(self, df: pd.DataFrame) -> bool:
        """
        Check if a DataFrame matches this schema.

        Used for auto-detection. Override for more sophisticated matching.
        """
        return all(col in df.columns for col in self.required_columns)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """
        Validate that the DataFrame conforms to this schema.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        return errors

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data to standard format.

        Must ensure:
        - A 'time' column exists (numeric seconds, typically from start)
        - Values converted to SI units
        - Columns renamed to standard NED frame names where applicable
        - Numeric columns are properly typed
        - Returns a copy (doesn't modify input)
        """
        pass
