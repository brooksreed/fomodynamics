"""L0: Array-level plotting primitives.

These functions take axes + arrays and know only about matplotlib styling.
No simulation or model concepts here.
"""

import numpy as np
import matplotlib.pyplot as plt

from fmd.analysis.plots._style import style_axis, get_colors


def _rolling_rms(x, window=20):
    """Rolling RMS with centered window (vectorized via pandas)."""
    import pandas as pd
    s = pd.Series(np.asarray(x) ** 2)
    return np.sqrt(s.rolling(window, min_periods=1, center=True).mean().values)


def time_series_panels(
    times,
    values_list,
    *,
    labels=None,
    ylabels=None,
    reference_lines=None,
    title=None,
    figsize=None,
    dpi=150,
):
    """N-panel stacked time series plot.

    Args:
        times: (N,) time array
        values_list: list of (N,) arrays, one per panel
        labels: optional list of legend labels per panel
        ylabels: optional list of y-axis labels per panel
        reference_lines: optional list of (value, label) tuples per panel;
            each element can be None, a single (value, label) tuple, or a list
            of (value, label) tuples
        title: optional suptitle
        figsize: optional (width, height)
        dpi: figure DPI

    Returns:
        (fig, axes) tuple
    """
    n_panels = len(values_list)
    if figsize is None:
        figsize = (10, 3 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    for i, (ax, values) in enumerate(zip(axes, values_list)):
        label = labels[i] if labels and i < len(labels) else None
        ax.plot(times, values, "b-", linewidth=0.8, label=label)

        if reference_lines and i < len(reference_lines) and reference_lines[i] is not None:
            refs = reference_lines[i]
            if isinstance(refs, tuple) and len(refs) == 2 and not isinstance(refs[0], tuple):
                refs = [refs]
            for ref_val, ref_label in refs:
                ax.axhline(ref_val, color="gray", linestyle="--", linewidth=0.8, label=ref_label)

        if ylabels and i < len(ylabels):
            ax.set_ylabel(ylabels[i])
        if label:
            ax.legend(fontsize=8)
        style_axis(ax)

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def multi_trace_panel(ax, traces, reference_value=None, ylabel=None, window=None, colors=None):
    """Overlay N traces on one axis.

    Args:
        ax: matplotlib Axes
        traces: list of dicts with keys: times, values, label, (optional: color)
        reference_value: optional horizontal reference line value
        ylabel: y-axis label
        window: optional dict from compute_adaptive_window
        colors: optional list of colors
    """
    if colors is None:
        colors = get_colors(len(traces))

    for i, tr in enumerate(traces):
        c = tr.get("color", colors[i] if i < len(colors) else None)
        ax.plot(tr["times"], tr["values"], color=c, linewidth=0.8, label=tr.get("label"))

    if reference_value is not None:
        ax.axhline(reference_value, color="gray", linestyle="--", alpha=0.5, label="reference")

    if window:
        from fmd.analysis.plots._windowing import apply_window
        apply_window(ax, window, colors, traces)

    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="best")
    style_axis(ax)


def error_with_envelope(ax, times, error, envelope=None, ylabel=None):
    """Error signal with optional envelope (2-sigma or rolling RMS).

    Args:
        ax: matplotlib Axes
        times: (N,) time array
        error: (N,) error signal
        envelope: (N,) envelope signal (e.g. 2-sigma bounds), or None for rolling RMS
        ylabel: y-axis label
    """
    if envelope is None:
        envelope = _rolling_rms(error, window=max(20, len(error) // 10))

    ax.plot(times, error, color="steelblue", linewidth=0.6, alpha=0.6, label="error")
    ax.plot(times, envelope, "r-", linewidth=1.5, label="envelope")
    ax.plot(times, -envelope, "r-", linewidth=1.5)
    ax.fill_between(times, -envelope, envelope, alpha=0.1, color="red")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)

    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    style_axis(ax)


def bar_comparison(ax, names, values, ylabel=None, colors=None):
    """Simple bar chart.

    Args:
        ax: matplotlib Axes
        names: list of category names
        values: list of values
        ylabel: y-axis label
        colors: optional list of colors per bar
    """
    n = len(names)
    if colors is None:
        colors = get_colors(n)

    ax.bar(range(n), values, color=colors[:n])
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax, grid=True)


def overlay_with_difference(times, series_a, series_b, ylabels=None, label_a="A", label_b="B", dpi=150):
    """N x 2 grid: left column overlays, right column differences.

    Args:
        times: (N,) time/step array
        series_a: list of (N,) arrays (one per state)
        series_b: list of (N,) arrays (one per state)
        ylabels: optional list of y-axis labels
        label_a: legend label for series A
        label_b: legend label for series B
        dpi: figure DPI

    Returns:
        (fig, axes) tuple where axes has shape (n_states, 2)
    """
    n_states = len(series_a)
    fig, axes = plt.subplots(n_states, 2, figsize=(14, 3 * n_states), dpi=dpi)
    if n_states == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_states):
        a_vals = series_a[i]
        b_vals = series_b[i]
        diff = a_vals - b_vals

        # Left: overlay
        axes[i, 0].plot(times, a_vals, "b-", linewidth=0.8, label=label_a)
        axes[i, 0].plot(times, b_vals, "r--", linewidth=0.8, label=label_b)
        if ylabels and i < len(ylabels):
            axes[i, 0].set_ylabel(ylabels[i])
        style_axis(axes[i, 0])
        if i == 0:
            axes[i, 0].legend(fontsize=8)
            axes[i, 0].set_title("Trajectory Overlay")

        # Right: difference
        axes[i, 1].plot(times, diff, "k-", linewidth=0.8)
        if ylabels and i < len(ylabels):
            axes[i, 1].set_ylabel(f"delta {ylabels[i]}")
        axes[i, 1].ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))
        style_axis(axes[i, 1])
        if i == 0:
            axes[i, 1].set_title(f"Difference ({label_a} - {label_b})")

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    fig.tight_layout()
    return fig, axes


def step_indicator(ax, times, flags, ylabel=None):
    """Binary step plot (e.g., MPC convergence flags).

    Args:
        ax: matplotlib Axes
        times: (N,) time/step array
        flags: (N,) boolean array
        ylabel: y-axis label
    """
    ax.step(times, np.array(flags, dtype=float), "g-", where="mid", linewidth=0.8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Failed", "Converged"])
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax)
