"""Unified plotting system for BLUR.

Three-layer architecture:

- **L0 (Primitives)**: Takes axes + arrays. Knows only matplotlib styling.
  See ``_primitives.py``.
- **L1 (Simulation-aware)**: Takes arrays + metadata (names, units).
  Handles SI-to-display conversion, angle wrapping, multi-panel layout.
  See ``_simulation.py``.
- **L2 (Convenience)**: Takes result objects (SimulationResult, ClosedLoopResult).
  Unpacks results and delegates to L1 with model-specific defaults.
  See ``_convenience.py``.

Existing DataStream plotting (``plot_time_series``, ``plot_polar``) is
preserved in ``_datastream.py``.

Usage examples::

    # DataStream / DataFrame plotting (unchanged)
    from fmd.analysis.plots import plot_time_series, plot_polar

    # L1: simulation-aware plotting with arrays
    from fmd.analysis.plots import state_trajectory, estimation_error

    # L2: convenience plotting with result objects
    from fmd.analysis.plots import plot_simulation_result, plot_lqg_result
"""

# --- DataStream plotting (backward compat) ---
from fmd.analysis.plots._datastream import (
    plot_time_series,
    plot_polar,
    radians_to_degrees,
    _get_display_info,
    _autosize_figure_to_screen,
)

# --- Style ---
from fmd.analysis.plots._style import (
    BLUR_STYLE,
    apply_blur_style,
    style_axis,
    get_colors,
    savefig_and_close,
)

# --- Windowing ---
from fmd.analysis.plots._windowing import (
    compute_adaptive_window,
    apply_window,
)

# --- L0: Primitives ---
from fmd.analysis.plots._primitives import (
    time_series_panels,
    multi_trace_panel,
    error_with_envelope,
    bar_comparison,
    overlay_with_difference,
    step_indicator,
)

# --- L1: Simulation-aware ---
from fmd.analysis.plots._simulation import (
    state_trajectory,
    estimation_error,
    estimation_error_with_covariance,
    nees_plot,
    covariance_diagonal,
    control_effort,
    solver_diagnostics,
    trajectory_comparison,
    bar_comparison_by_variant,
    multi_trace_grid,
)

# --- L2: Convenience ---
from fmd.analysis.plots._convenience import (
    plot_simulation_result,
    plot_lqg_result,
    plot_closed_loop_result,
    plot_sweep_category,
    MOTH_3DOF_STATE_LABELS,
    MOTH_3DOF_STATE_TRANSFORMS,
    MOTH_3DOF_CONTROL_LABELS,
    MOTH_3DOF_CONTROL_TRANSFORMS,
)

__all__ = [
    # DataStream (backward compat)
    "plot_time_series",
    "plot_polar",
    "radians_to_degrees",
    "_get_display_info",
    "_autosize_figure_to_screen",
    # Style
    "BLUR_STYLE",
    "apply_blur_style",
    "style_axis",
    "get_colors",
    "savefig_and_close",
    # Windowing
    "compute_adaptive_window",
    "apply_window",
    # L0
    "time_series_panels",
    "multi_trace_panel",
    "error_with_envelope",
    "bar_comparison",
    "overlay_with_difference",
    "step_indicator",
    # L1
    "state_trajectory",
    "estimation_error",
    "estimation_error_with_covariance",
    "nees_plot",
    "covariance_diagonal",
    "control_effort",
    "solver_diagnostics",
    "trajectory_comparison",
    "bar_comparison_by_variant",
    "multi_trace_grid",
    # L2
    "plot_simulation_result",
    "plot_lqg_result",
    "plot_closed_loop_result",
    "plot_sweep_category",
    "MOTH_3DOF_STATE_LABELS",
    "MOTH_3DOF_STATE_TRANSFORMS",
    "MOTH_3DOF_CONTROL_LABELS",
    "MOTH_3DOF_CONTROL_TRANSFORMS",
]
