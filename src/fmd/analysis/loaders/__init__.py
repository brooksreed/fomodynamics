"""
Data file loaders with schema-based detection and normalization.

Usage:
    from fmd.analysis.loaders import load_file, list_schemas

    # Auto-detect schema
    result = load_file("data.csv")
    print(f"Detected schema: {result.schema_name}")
    df = result.df

    # Explicit schema
    result = load_file("data.csv", schema="vakaros_atlas_telemetry")

    # Get DataStream for advanced operations
    stream = load_stream("data.csv")
    leeway = stream.subtract("yaw", "cog")  # Automatic circular handling

    # Multi-file loading
    log = load_vehicle_log({"telemetry": "vakaros.csv"})
    aligned = log.align_to("telemetry")  # Nearest-neighbor (no interpolation)
    aligned_interp = log.align_to("telemetry", mode="interpolate")  # May invent data
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from .base import LoadResult, Schema
from .registry import registry
from fmd.analysis.core import DataStream, VehicleLog

# Import and register all schemas
from .schemas.vakaros_atlas_telemetry import VakarosAtlasTelemetrySchema
from .schemas.test_data import TestDataSchema
from .schemas.dynamic_simulator import DynamicSimulatorSchema

# Register schemas (order doesn't matter for detection since we check all)
registry.register(TestDataSchema())
registry.register(VakarosAtlasTelemetrySchema())
registry.register(DynamicSimulatorSchema())


def load_file(
    filename: str,
    schema: Optional[str] = None,
    **read_kwargs,
) -> LoadResult:
    """
    Load a CSV data file and normalize it using the appropriate schema.

    Args:
        filename: Path to CSV file
        schema: Schema name to use (auto-detected if None)
        **read_kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        LoadResult containing:
        - df: Normalized DataFrame with 'time' column (SI units)
        - schema_name: Name of schema used
        - metadata: Additional info (source file, original columns)
        - units: Mapping of column name -> SI unit

    Raises:
        ValueError: If schema not found or validation fails

    Examples:
        >>> result = load_file("telemetry.csv")
        >>> result.df["sog"]  # Speed in m/s
        >>> result.df["yaw"]  # Heading in radians
    """
    path = Path(filename)

    # Read CSV
    df = pd.read_csv(filename, **read_kwargs)
    original_columns = list(df.columns)

    # Get schema (explicit or auto-detect)
    if schema:
        schema_obj = registry.get(schema)
    else:
        schema_obj = registry.detect(df)
        if not schema_obj:
            raise ValueError(
                f"Could not auto-detect schema for '{filename}'.\n"
                f"Columns found: {original_columns}\n"
                f"Available schemas: {registry.list_schemas()}\n"
                f"Use schema='...' to specify explicitly."
            )

    # Validate
    errors = schema_obj.validate(df)
    if errors:
        raise ValueError(
            f"Validation failed for '{filename}' using schema '{schema_obj.name}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # Normalize
    normalized = schema_obj.normalize(df)

    return LoadResult(
        df=normalized,
        schema_name=schema_obj.name,
        metadata={
            "source": str(path.resolve()),
            "original_columns": original_columns,
            "row_count": len(df),
        },
        units=schema_obj.output_units,
    )


def load_stream(
    filename: str,
    schema: Optional[str] = None,
    name: Optional[str] = None,
    **read_kwargs,
) -> DataStream:
    """
    Load a CSV file as a DataStream.

    DataStream provides circular-aware operations (subtract, mean, interpolate).

    Args:
        filename: Path to CSV file
        schema: Schema name to use (auto-detected if None)
        name: Name for the stream (default: derived from filename)
        **read_kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataStream with normalized data in SI units

    Examples:
        >>> stream = load_stream("telemetry.csv")
        >>> leeway = stream.subtract("yaw", "cog")  # Proper angle subtraction
        >>> stream.mean("yaw")  # Circular mean
    """
    result = load_file(filename, schema=schema, **read_kwargs)

    if name is None:
        name = Path(filename).stem

    return DataStream(
        df=result.df,
        name=name,
        units=result.units,
        metadata=result.metadata,
    )


def load_vehicle_log(
    files: dict[str, str],
    schemas: Optional[dict[str, str]] = None,
    domain: str = "generic",
) -> VehicleLog:
    """
    Load multiple files into a VehicleLog.

    Each file becomes a named DataStream in the log. Streams can have
    different sample rates; use align_to() to synchronize them.

    Args:
        files: Mapping of stream_name -> filename
        schemas: Optional mapping of stream_name -> schema_name
        domain: Domain for display ('sailing', 'aviation', 'generic')

    Returns:
        VehicleLog containing all streams

    Examples:
        >>> log = load_vehicle_log({
        ...     "telemetry": "vakaros.csv",
        ...     "imu": "ardupilot_imu.csv",
        ... })
        >>> aligned = log.align_to("telemetry")  # Nearest-neighbor (no interpolation)
        >>> aligned_interp = log.align_to("telemetry", mode="interpolate")  # May invent data
    """
    schemas = schemas or {}
    log = VehicleLog(domain=domain)

    for stream_name, filename in files.items():
        schema = schemas.get(stream_name)
        stream = load_stream(filename, schema=schema, name=stream_name)
        log.add_stream(stream)

    return log


def list_schemas() -> list[str]:
    """List all available schema names."""
    return registry.list_schemas()


# Expose key classes for advanced usage
__all__ = [
    "load_file",
    "load_stream",
    "load_vehicle_log",
    "list_schemas",
    "LoadResult",
    "Schema",
    "registry",
    "DataStream",
    "VehicleLog",
]
