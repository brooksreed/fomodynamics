"""Schema registry for managing data schemas."""

from typing import Optional
import pandas as pd
from .base import Schema


class SchemaRegistry:
    """Registry for data schemas with detection and lookup."""

    def __init__(self):
        self._schemas: dict[str, Schema] = {}

    def register(self, schema: Schema) -> None:
        """Register a schema. Overwrites if name already exists."""
        self._schemas[schema.name] = schema

    def get(self, name: str) -> Schema:
        """Get a schema by name."""
        if name not in self._schemas:
            available = list(self._schemas.keys())
            raise ValueError(f"Unknown schema: '{name}'. Available: {available}")
        return self._schemas[name]

    def detect(self, df: pd.DataFrame) -> Optional[Schema]:
        """
        Find the first schema that matches the DataFrame.

        Returns None if no schema matches.
        """
        for schema in self._schemas.values():
            if schema.matches(df):
                return schema
        return None

    def list_schemas(self) -> list[str]:
        """List all registered schema names."""
        return list(self._schemas.keys())


# Global registry instance
registry = SchemaRegistry()
