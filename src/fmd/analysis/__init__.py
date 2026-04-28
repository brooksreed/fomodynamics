"""Data analysis tools for vehicle telemetry (sailing, aviation, robotics).

Key concepts:
- SI units internally (m/s, radians), converted for display (knots, degrees)
- NED coordinate frame (roll, pitch, yaw)
- Circular-aware operations for angles (automatic wrap handling)
- Multi-rate data support with lazy alignment

Quick start:
    from fmd.analysis import load_file, load_stream, plot_time_series

    # Load and plot (auto-detects schema, converts to SI)
    result = load_file("telemetry.csv")
    plot_time_series(result.df, domain="sailing")

    # Use DataStream for angle-aware operations
    stream = load_stream("telemetry.csv")
    leeway = stream.subtract("yaw", "cog")  # Correct circular subtraction
"""

from fmd.analysis.loaders import (
    load_file,
    load_stream,
    load_vehicle_log,
    list_schemas,
    LoadResult,
)
from fmd.analysis.core import DataStream, VehicleLog
from fmd.analysis.plots import plot_time_series, plot_polar
from fmd.analysis.wave_plots import (
    plot_wave_elevation_timeseries,
    plot_wave_encounter_spectrum,
    plot_waterfall,
    plot_wave_field_snapshot,
)
from fmd.analysis.generate import generate_random_data
from fmd.core.units import (
    QUANTITY_TYPES,
    VARIABLE_SCHEMA,
    VARIABLE_ALIASES,
    convert_to_si,
    convert_from_si,
    get_quantity_type,
    is_circular,
)
from fmd.analysis.operations import (
    wrap_angle,
    circular_subtract,
    circular_mean,
    circular_interpolate,
)
from fmd.analysis.processing import (
    butterworth_lowpass,
    circular_lowpass,
    find_gaps,
    resample_to_rate,
)
from fmd.analysis.closed_loop import (
    plot_config_overlay,
    plot_single_dashboard,
    compute_extended_metrics,
    compute_wave_vs_calm_metrics,
    format_metrics_table,
    find_interesting_window,
    compute_leeward_tip_depth,
)

__all__ = [
    # Loading
    "load_file",
    "load_stream",
    "load_vehicle_log",
    "list_schemas",
    "LoadResult",
    # Containers
    "DataStream",
    "VehicleLog",
    # Plotting
    "plot_time_series",
    "plot_polar",
    # Wave plotting
    "plot_wave_elevation_timeseries",
    "plot_wave_encounter_spectrum",
    "plot_waterfall",
    "plot_wave_field_snapshot",
    # Data generation
    "generate_random_data",
    # Units
    "QUANTITY_TYPES",
    "VARIABLE_SCHEMA",
    "VARIABLE_ALIASES",
    "convert_to_si",
    "convert_from_si",
    "get_quantity_type",
    "is_circular",
    # Operations
    "wrap_angle",
    "circular_subtract",
    "circular_mean",
    "circular_interpolate",
    # Processing
    "butterworth_lowpass",
    "circular_lowpass",
    "find_gaps",
    "resample_to_rate",
    # Closed-loop analysis
    "plot_config_overlay",
    "plot_single_dashboard",
    "compute_extended_metrics",
    "compute_wave_vs_calm_metrics",
    "format_metrics_table",
    "find_interesting_window",
    "compute_leeward_tip_depth",
]
