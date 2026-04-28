"""L1: Simulation-aware plotting functions.

All functions take arrays + metadata kwargs (state_labels, state_transforms,
etc.). They are model-agnostic -- model-specific defaults live in L2.
"""

import numpy as np
import matplotlib.pyplot as plt

from fmd.analysis.plots._style import style_axis, get_colors
from fmd.analysis.plots._primitives import (
    error_with_envelope,
    step_indicator,
)


def _apply_transforms(values, transforms, indices=None):
    """Apply per-state transforms (e.g., np.degrees) to columns of a 2D array.

    Args:
        values: (N, M) array
        transforms: list of callables or None, length M
        indices: optional list of column indices to process (default: all)

    Returns:
        transformed copy of values
    """
    result = np.array(values, dtype=float)
    if transforms is None:
        return result
    cols = indices if indices is not None else range(result.shape[1])
    for i in cols:
        if i < len(transforms) and transforms[i] is not None:
            result[:, i] = transforms[i](result[:, i])
    return result


def _prepend_t0(times, arrays):
    """If arrays have one more row than times, prepend t=0 to times."""
    if len(arrays) == len(times) + 1:
        return np.concatenate([[0.0], times])
    return times


def state_trajectory(
    times,
    states,
    *,
    state_names=None,
    state_labels=None,
    state_transforms=None,
    reference_state=None,
    title="State Trajectory",
    dpi=150,
):
    """Multi-panel state history with optional trim reference.

    Args:
        times: (N,) time array
        states: (N+1, M) or (N, M) state trajectory
        state_names: unused (kept for API symmetry with L2)
        state_labels: list of display labels per state (e.g. "theta (deg)")
        state_transforms: list of callables or None per state
        reference_state: (M,) trim/reference state (will be transformed)
        title: figure title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    t = _prepend_t0(times, states)
    n_states = states.shape[1]

    display_states = _apply_transforms(states, state_transforms)
    ref_display = None
    if reference_state is not None:
        ref_display = np.array(reference_state, dtype=float)
        if state_transforms:
            for i, tf in enumerate(state_transforms):
                if tf is not None and i < len(ref_display):
                    ref_display[i] = tf(ref_display[i])

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True, dpi=dpi)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, display_states[:, i], "b-", linewidth=0.8)
        if ref_display is not None:
            ax.axhline(ref_display[i], color="gray", linestyle="--", linewidth=0.8, label="trim")
        if state_labels and i < len(state_labels):
            ax.set_ylabel(state_labels[i])
        style_axis(ax)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def estimation_error(
    times,
    true_states,
    est_states,
    *,
    state_labels=None,
    state_transforms=None,
    title="Estimation Error (est - true) with Rolling RMS",
    dpi=150,
):
    """Multi-panel estimation error with rolling RMS envelope.

    Args:
        times: (N,) time array
        true_states: (N+1, M) or (N, M) true state trajectory
        est_states: (N+1, M) or (N, M) estimated state trajectory
        state_labels: list of display labels per state
        state_transforms: list of callables or None per state
        title: figure title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    t = _prepend_t0(times, true_states)
    n_states = true_states.shape[1]

    true_disp = _apply_transforms(true_states, state_transforms)
    est_disp = _apply_transforms(est_states, state_transforms)

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True, dpi=dpi)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        err = est_disp[:, i] - true_disp[:, i]
        ylabel = f"delta {state_labels[i]}" if state_labels and i < len(state_labels) else None
        error_with_envelope(ax, t, err, ylabel=ylabel)

    axes[0].legend(fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def estimation_error_with_covariance(
    times,
    errors,
    covariance_diags,
    *,
    state_labels=None,
    state_transforms=None,
    title="Estimation Error with 2-sigma Bounds",
    dpi=150,
):
    """Multi-panel estimation error with 2-sigma covariance bounds.

    Args:
        times: (N,) time array
        errors: (N, M) estimation errors
        covariance_diags: (N, M) diagonal elements of P
        state_labels: list of display labels per state
        state_transforms: list of callables or None per state
        title: figure title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    t = _prepend_t0(times, errors)
    n_states = errors.shape[1]

    # Apply transforms to errors and sqrt(diag) for sigma
    err_disp = _apply_transforms(errors, state_transforms)
    sigma = np.sqrt(np.abs(covariance_diags))
    sigma_disp = _apply_transforms(sigma, state_transforms)

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True, dpi=dpi)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        envelope = 2.0 * np.abs(sigma_disp[:, i])
        ylabel = f"delta {state_labels[i]}" if state_labels and i < len(state_labels) else None
        error_with_envelope(ax, t, err_disp[:, i], envelope=envelope, ylabel=ylabel)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def nees_plot(times, nees_values, n_states=5, title="Normalized Estimation Error Squared (NEES)", dpi=150):
    """NEES over time with chi-squared bounds.

    Args:
        times: (N,) time array
        nees_values: (N,) NEES values
        n_states: number of states (for chi-squared bounds)
        title: plot title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    from scipy.stats import chi2

    lower = chi2.ppf(0.025, n_states)
    upper = chi2.ppf(0.975, n_states)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)

    ax.plot(times, nees_values, "b-", linewidth=0.8, label="NEES")
    ax.axhline(n_states, color="green", linestyle="-", linewidth=1.0,
               label=f"E[NEES]={n_states}")
    ax.axhline(lower, color="red", linestyle="--", linewidth=0.8,
               label=f"lower 95% ({lower:.2f})")
    ax.axhline(upper, color="red", linestyle="--", linewidth=0.8,
               label=f"upper 95% ({upper:.2f})")
    ax.fill_between(times, lower, upper, alpha=0.08, color="green")

    style_axis(ax, xlabel="Time (s)", ylabel="NEES", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def covariance_diagonal(
    times,
    P_diags,
    *,
    state_labels=None,
    log_scale=True,
    title="Covariance Diagonal",
    dpi=150,
):
    """Per-state covariance diagonal elements over time.

    Args:
        times: (N,) time array
        P_diags: (N+1, M) or (N, M) diagonal elements of P
        state_labels: optional list of labels
        log_scale: whether to use log y-axis (default True)
        title: figure title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    t = _prepend_t0(times, P_diags)
    n_states = P_diags.shape[1]

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True, dpi=dpi)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if log_scale:
            ax.semilogy(t, P_diags[:, i], "b-", linewidth=0.8)
        else:
            ax.plot(t, P_diags[:, i], "b-", linewidth=0.8)
        ylabel = f"P[{i},{i}]"
        if state_labels and i < len(state_labels):
            ylabel = state_labels[i]
        ax.set_ylabel(ylabel)
        style_axis(ax)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title + (" (log scale)" if log_scale else ""))
    fig.tight_layout()
    return fig


def control_effort(
    times,
    controls,
    *,
    control_labels=None,
    control_transforms=None,
    bounds=None,
    title="Control Effort",
    dpi=150,
):
    """Multi-panel control inputs with optional saturation lines.

    Args:
        times: (N,) time array
        controls: (N, M) control inputs
        control_labels: list of display labels per control
        control_transforms: list of callables or None per control
        bounds: optional list of (min, max) tuples per control (in display units)
        title: figure title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    n_controls = controls.shape[1]
    display_controls = _apply_transforms(controls, control_transforms)

    fig, axes = plt.subplots(n_controls, 1, figsize=(10, 3 * n_controls), sharex=True, dpi=dpi)
    if n_controls == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, display_controls[:, i], "b-", linewidth=0.8)
        if bounds and i < len(bounds) and bounds[i] is not None:
            lo, hi = bounds[i]
            ax.axhline(lo, color="red", linestyle="--", linewidth=0.8, label="limit")
            ax.axhline(hi, color="red", linestyle="--", linewidth=0.8)
            ax.legend(fontsize=8)
        if control_labels and i < len(control_labels):
            ax.set_ylabel(control_labels[i])
        style_axis(ax)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def solver_diagnostics(solve_times, converged_flags, title="MPC Solve Performance", dpi=150):
    """2-panel MPC diagnostics: solve times + convergence.

    Args:
        solve_times: (N,) wall-clock solve times in seconds
        converged_flags: (N,) boolean convergence flags
        title: figure title
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=dpi)
    steps = np.arange(len(solve_times))

    # Solve times
    axes[0].plot(steps, np.array(solve_times) * 1000, "b-", linewidth=0.8)
    style_axis(axes[0], ylabel="Solve time (ms)", title=title)

    # Convergence
    step_indicator(axes[1], steps, converged_flags, ylabel="Status")
    axes[1].set_xlabel("MPC solve index")

    n_total = len(converged_flags)
    n_conv = sum(converged_flags)
    axes[1].set_title(f"Convergence: {n_conv}/{n_total} ({100 * n_conv / max(n_total, 1):.0f}%)")

    fig.tight_layout()
    return fig


def trajectory_comparison(
    times,
    states_a,
    states_b,
    *,
    label_a="A",
    label_b="B",
    state_labels=None,
    state_transforms=None,
    title=None,
    dpi=150,
):
    """Overlay + difference plot for trajectory equivalence.

    Args:
        times: (N,) time/step array
        states_a: (N, M) first trajectory
        states_b: (N, M) second trajectory
        label_a: legend label for trajectory A
        label_b: legend label for trajectory B
        state_labels: list of display labels per state
        state_transforms: list of callables or None per state
        title: optional suptitle
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    from fmd.analysis.plots._primitives import overlay_with_difference

    a_disp = _apply_transforms(states_a, state_transforms)
    b_disp = _apply_transforms(states_b, state_transforms)

    series_a = [a_disp[:, i] for i in range(a_disp.shape[1])]
    series_b = [b_disp[:, i] for i in range(b_disp.shape[1])]

    fig, axes = overlay_with_difference(
        times, series_a, series_b,
        ylabels=state_labels,
        label_a=label_a,
        label_b=label_b,
        dpi=dpi,
    )
    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
    return fig


def bar_comparison_by_variant(
    variant_errors,
    *,
    state_labels=None,
    state_transforms=None,
    title="Variant RMS Comparison",
    dpi=150,
):
    """Multi-panel bar chart comparing RMS errors by variant.

    Args:
        variant_errors: dict mapping variant name -> (M,) RMS error per state
        state_labels: list of display labels per state
        state_transforms: list of callables or None per state
        title: figure suptitle
        dpi: figure DPI

    Returns:
        matplotlib Figure
    """
    variants = list(variant_errors.keys())
    n_variants = len(variants)
    n_states = len(next(iter(variant_errors.values())))

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), dpi=dpi)
    if n_states == 1:
        axes = [axes]

    colors = get_colors(n_variants)

    for i, ax in enumerate(axes):
        vals = [variant_errors[v][i] for v in variants]
        # Apply transforms to RMS values
        if state_transforms and i < len(state_transforms) and state_transforms[i] is not None:
            vals = [state_transforms[i](v) for v in vals]

        ax.bar(range(n_variants), vals, color=colors[:n_variants])
        ax.set_xticks(range(n_variants))
        ax.set_xticklabels(variants, fontsize=8)
        if state_labels and i < len(state_labels):
            ax.set_title(state_labels[i], fontsize=10)
        style_axis(ax)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def multi_trace_grid(
    traces_by_panel,
    *,
    layout=None,
    reference_values=None,
    physical_bounds=None,
    use_adaptive_window=True,
    panel_labels=None,
    title=None,
    figsize=None,
    dpi=150,
):
    """Grid of panels for sweep comparisons.

    Args:
        traces_by_panel: list of list-of-trace-dicts (one per panel)
        layout: (rows, cols) tuple. Default: (N, 1)
        reference_values: list of reference values per panel
        physical_bounds: list of (min, max) tuples per panel
        use_adaptive_window: whether to apply adaptive windowing
        panel_labels: list of y-axis labels per panel
        title: optional suptitle
        figsize: optional figure size
        dpi: figure DPI

    Returns:
        (fig, axes) tuple
    """
    from fmd.analysis.plots._windowing import compute_adaptive_window, apply_window
    from fmd.analysis.plots._primitives import multi_trace_panel

    n_panels = len(traces_by_panel)
    if layout is None:
        layout = (n_panels, 1)
    rows, cols = layout
    if figsize is None:
        figsize = (6 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, traces in enumerate(traces_by_panel):
        r, c = divmod(idx, cols)
        if r >= rows:
            break
        ax = axes[r, c]

        ref_val = reference_values[idx] if reference_values and idx < len(reference_values) else None
        p_bounds = physical_bounds[idx] if physical_bounds and idx < len(physical_bounds) else None

        window = None
        if use_adaptive_window and traces:
            window = compute_adaptive_window(traces, ref_val or 0.0, physical_bounds=p_bounds)

        ylabel = panel_labels[idx] if panel_labels and idx < len(panel_labels) else None
        multi_trace_panel(ax, traces, reference_value=ref_val, ylabel=ylabel, window=window)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes
