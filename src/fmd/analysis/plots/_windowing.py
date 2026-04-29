"""Adaptive axis windowing for multi-trace plots.

Computes axis bounds that focus on interesting dynamics while handling
divergent trajectories gracefully.
"""

import numpy as np


def compute_adaptive_window(
    traces,
    reference_value=0.0,
    *,
    physical_bounds=None,
    time_margin_factor=1.3,
    value_padding_factor=0.15,
    force_full_time=False,
):
    """Compute axis bounds that show interesting dynamics without divergence noise.

    Each trace dict must have:
        times: np.ndarray
        values: np.ndarray
        divergence_time: float | None
        stable: bool

    Args:
        traces: list of trace dicts
        reference_value: trim/reference value for the plotted quantity
        physical_bounds: (min, max) tuple of physically plausible values
        time_margin_factor: how far past last divergence to show (1.3 = 30% extra)
        value_padding_factor: fraction of value span to pad (0.15 = 15%)
        force_full_time: if True, always show full time range

    Returns:
        dict with xlim, ylim, has_divergent, divergence_times
    """
    if not traces:
        return {"xlim": None, "ylim": None, "has_divergent": False, "divergence_times": []}

    div_times = [
        t["divergence_time"] for t in traces
        if t.get("divergence_time") is not None
    ]
    has_divergent = len(div_times) > 0

    # --- Time window ---
    nonempty_ends = [t["times"][-1] for t in traces if len(t["times"]) > 0]
    if not nonempty_ends:
        return {"xlim": None, "ylim": None, "has_divergent": has_divergent, "divergence_times": div_times}
    full_duration = max(nonempty_ends)
    if has_divergent and not force_full_time:
        t_max = max(div_times) * time_margin_factor
        t_max = min(t_max, full_duration)
    else:
        t_max = full_duration

    # --- Value window: use only data within time window ---
    all_vals = []
    for tr in traces:
        times = tr["times"]
        values = tr["values"]
        mask = times <= t_max
        clipped = values[mask]
        finite = clipped[np.isfinite(clipped)]
        if len(finite) > 0:
            all_vals.append(finite)

    if not all_vals:
        return {
            "xlim": (0, t_max),
            "ylim": physical_bounds,
            "has_divergent": has_divergent,
            "divergence_times": div_times,
        }

    combined = np.concatenate(all_vals)
    v_min, v_max = float(np.nanmin(combined)), float(np.nanmax(combined))

    # Add padding
    span = v_max - v_min
    if span < 1e-10:
        span = max(abs(reference_value) * 0.1, 0.1)
    pad = value_padding_factor * span
    v_min -= pad
    v_max += pad

    # Clamp to physical bounds
    if physical_bounds is not None:
        v_min = max(v_min, physical_bounds[0])
        v_max = min(v_max, physical_bounds[1])

    return {
        "xlim": (0, t_max),
        "ylim": (v_min, v_max),
        "has_divergent": has_divergent,
        "divergence_times": div_times,
    }


def apply_window(ax, window, colors=None, traces=None):
    """Apply adaptive window bounds and divergence markers to an axes.

    Args:
        ax: matplotlib Axes
        window: dict returned by compute_adaptive_window
        colors: optional list of colors corresponding to traces
        traces: optional list of trace dicts (for divergence time markers)
    """
    if window.get("xlim"):
        ax.set_xlim(window["xlim"])
    if window.get("ylim"):
        ax.set_ylim(window["ylim"])

    # Add divergence time markers
    if traces and colors:
        for i, tr in enumerate(traces):
            dt = tr.get("divergence_time")
            if dt is not None:
                c = colors[i] if i < len(colors) else "red"
                ax.axvline(dt, color=c, linestyle=":", alpha=0.4, linewidth=0.8)

    if window.get("has_divergent"):
        n_div = len(window.get("divergence_times", []))
        ax.annotate(
            f"{n_div} diverge",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", fontsize=7, color="red", alpha=0.7,
        )
