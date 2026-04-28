"""Plotting functions for time series data.

Supports both raw DataFrames and DataStreams, with automatic
unit conversion for display (SI -> knots, degrees, etc.).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype

from fmd.core.units import (
    VARIABLE_SCHEMA,
    convert_from_si,
    get_display_name,
)
from fmd.analysis.operations import wrap_angle


def _autosize_figure_to_screen(fig: plt.Figure) -> None:
    """Best-effort: expand the figure window to (roughly) fill the screen."""
    try:
        manager = plt.get_current_fig_manager()
        window = getattr(manager, "window", None)

        # TkAgg
        if hasattr(window, "state"):
            window.state("zoomed")
            return

        # Qt
        if hasattr(window, "showMaximized"):
            window.showMaximized()
            return

        # Fallback: set geometry to 90% of screen size (Tk-style API)
        if hasattr(window, "winfo_screenwidth") and hasattr(window, "winfo_screenheight") and hasattr(window, "geometry"):
            w = int(window.winfo_screenwidth() * 0.9)
            h = int(window.winfo_screenheight() * 0.9)
            window.geometry(f"{w}x{h}+0+0")
            return

        # Some backends provide this
        if hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
    except Exception:
        # If we can't resize (headless, non-GUI backend, etc.), just keep default sizing.
        return


def _get_display_info(column: str, domain: str = "generic") -> tuple[str, str, float]:
    """Get display name, unit, and conversion multiplier for a column.

    Args:
        column: Column name
        domain: Domain context

    Returns:
        (display_name, unit_string, multiplier_from_si)
    """
    display_name = get_display_name(column, domain)

    if column in VARIABLE_SCHEMA:
        qty_type = VARIABLE_SCHEMA[column]
        # convert_from_si returns (converted_value, unit). With value=1.0, the
        # converted_value is the multiplier from SI.
        multiplier, unit = convert_from_si(1.0, qty_type)
    else:
        multiplier = 1.0
        unit = ""

    return display_name, unit, multiplier


def radians_to_degrees(angles_rad: np.ndarray) -> np.ndarray:
    """Convert radians to degrees, preserving wrap convention [-180, 180).

    Args:
        angles_rad: Angles in radians (assumed to be in [-π, π])

    Returns:
        Angles in degrees, wrapped to [-180, 180)
    """
    angles_deg = np.degrees(angles_rad)
    # Wrap to [-180, 180) to ensure consistency
    return wrap_angle(angles_deg, low=-180.0, high=180.0)


def plot_time_series(
    data,
    *,
    x: pd.Series | None = None,
    x_label: str = "time",
    autosize: bool = False,
    domain: str = "generic",
    convert_units: bool = True,
    columns: list[str] | None = None,
):
    """Create subplots with time series for each numeric data field.

    Args:
        data: DataFrame or DataStream to plot
        x: Optional x-axis Series (default: 'time' column)
        x_label: Label for x-axis
        autosize: Whether to maximize window
        domain: Domain for display names ('sailing', 'aviation', 'generic')
        convert_units: Whether to convert from SI to display units
        columns: Specific columns to plot (default: all numeric)

    Notes:
    - By default uses the numeric `time` column on the x-axis.
    - Skips non-numeric columns (e.g. ISO timestamp strings).
    - When convert_units=True, displays values in user-friendly units
      (knots, degrees) instead of SI (m/s, radians).
    """
    # Handle DataStream input
    from fmd.analysis.core import DataStream

    if isinstance(data, DataStream):
        df = data.df
        if domain == "generic":
            # Inherit domain from DataStream if available
            domain = getattr(data, "domain", "generic")
    else:
        df = data

    if x is None:
        if "time" not in df.columns:
            raise ValueError("plot_time_series() requires a 'time' column (or pass x=...)")
        x = df["time"]

    if len(x) != len(df):
        raise ValueError("plot_time_series(): x must be the same length as df")

    # Determine which columns to plot
    if columns is not None:
        data_columns = [c for c in columns if c in df.columns and is_numeric_dtype(df[c])]
    else:
        data_columns = [
            col for col in df.columns
            if col != "time" and col != "timestamp" and is_numeric_dtype(df[col])
        ]

    num_plots = len(data_columns)

    if num_plots == 0:
        raise ValueError("No numeric columns to plot (besides 'time').")

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)

    if autosize:
        _autosize_figure_to_screen(fig)

    if num_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, data_columns):
        values = df[col].values

        # Get display info
        display_name, unit, multiplier = _get_display_info(col, domain)

        # Convert to display units if requested
        if convert_units and multiplier != 1.0:
            values = values * multiplier
            # For angles, ensure wrap convention is preserved ([-180, 180) degrees)
            if col in VARIABLE_SCHEMA and VARIABLE_SCHEMA[col] == "angle":
                values = wrap_angle(values, low=-180.0, high=180.0)
            label = f"{display_name} [{unit}]"
        else:
            label = display_name

        # Solid line with dots at each data point (makes sampling explicit)
        ax.plot(x, values, label=label, linestyle="-", marker=".", markersize=3)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    # If datetime-like x axis, make it readable as time-of-day.
    if is_datetime64_any_dtype(x):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        # Show time-of-day prominently for short ranges.
        formatter.formats = ["%H:%M:%S"] * 6
        formatter.zero_formats = ["%H:%M:%S"] * 6
        formatter.offset_formats = ["%Y-%m-%d"] * 6
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def plot_polar(
    data,
    *,
    angle_col: str,
    radius_col: str,
    domain: str = "generic",
    convert_units: bool = True,
    title: str | None = None,
):
    """Create a polar plot (useful for wind diagrams, etc.).

    Args:
        data: DataFrame or DataStream
        angle_col: Column name for angle (in radians if SI)
        radius_col: Column name for radius
        domain: Domain for display names
        convert_units: Whether to convert to display units
        title: Optional plot title
    """
    from fmd.analysis.core import DataStream

    if isinstance(data, DataStream):
        df = data.df
    else:
        df = data

    angles = df[angle_col].values
    radii = df[radius_col].values

    # Convert to display units
    radius_name, radius_unit, radius_mult = _get_display_info(radius_col, domain)

    if convert_units:
        # For polar plots, keep angles in radians for matplotlib
        # but convert radius to display units
        radii = radii * radius_mult

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    ax.scatter(angles, radii, alpha=0.5, s=5)

    if title:
        ax.set_title(title)

    if convert_units and radius_unit:
        ax.set_ylabel(f"{radius_name} [{radius_unit}]")

    plt.tight_layout()
    plt.show()
