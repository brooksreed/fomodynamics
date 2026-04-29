"""L2: Convenience wrappers that accept result objects.

These functions know about specific result types (SimulationResult,
ClosedLoopResult, etc.) and model-specific display constants. They unpack
results and delegate to L1 functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from fmd.analysis.plots._simulation import (
    state_trajectory,
    estimation_error,
    estimation_error_with_covariance,
    covariance_diagonal,
    control_effort,
)
from fmd.analysis.plots._style import FMD_STYLE, savefig_and_close

# ---------------------------------------------------------------------------
# Moth 3DOF display constants
# ---------------------------------------------------------------------------

MOTH_3DOF_STATE_LABELS = ["pos_d (m)", "theta (deg)", "w (m/s)", "q (deg/s)", "u (m/s)"]
MOTH_3DOF_STATE_TRANSFORMS = [None, np.degrees, None, np.degrees, None]
MOTH_3DOF_CONTROL_LABELS = ["Main flap (deg)", "Rudder elevator (deg)"]
MOTH_3DOF_CONTROL_TRANSFORMS = [np.degrees, np.degrees]



# ---------------------------------------------------------------------------
# L2 functions
# ---------------------------------------------------------------------------


def plot_simulation_result(
    result,
    *,
    state_labels=None,
    state_transforms=None,
    reference_state=None,
    title="Simulation Result",
    save_path=None,
):
    """Plot a SimulationResult or RichSimulationResult.

    Args:
        result: SimulationResult or RichSimulationResult
        state_labels: display labels per state (auto-detected from RichSimulationResult)
        state_transforms: callables per state
        reference_state: trim/reference state
        title: figure title
        save_path: if given, save figure to this path and close

    Returns:
        matplotlib Figure
    """
    times = np.array(result.times)
    states = np.array(result.states)

    # Try to get state names from RichSimulationResult
    if state_labels is None and hasattr(result, "state_names"):
        state_labels = list(result.state_names)

    fig = state_trajectory(
        times,
        states,
        state_labels=state_labels,
        state_transforms=state_transforms,
        reference_state=reference_state,
        title=title,
    )

    if save_path:
        savefig_and_close(fig, save_path)
    return fig


def plot_lqg_result(result, *, save_dir=None):
    """Generate Moth foiling and estimation dashboard figures from a ClosedLoopResult.

    Produces two figures:
    - ``"foiling_dashboard"``: Physical state of the boat (heights, pitch,
      flap, forces, strut depth, foil span)
    - ``"estimation_dashboard"``: EKF tracking and filter health (state
      tracking, estimation error with 2-sigma, covariance, innovations)

    The result must be a self-contained ClosedLoopResult with ``params``,
    ``force_log``, ``measurements_clean``, ``measurements_noisy``,
    and ``innovations`` fields populated.

    Args:
        result: ClosedLoopResult from ``simulate_closed_loop`` with ``params`` provided.
        save_dir: If given, save both figures to this directory as PNGs.

    Returns:
        dict mapping figure name to matplotlib Figure.
    """
    import os

    times = np.array(result.times)
    true_states = np.array(result.true_states)
    est_states = np.array(result.est_states)
    controls = np.array(result.controls)
    params = result.params
    force_log = result.force_log
    heel_angle = result.heel_angle
    cov_diags = np.array(result.covariance_diagonals)
    est_errors = np.array(result.estimation_errors)
    n_steps = len(times)

    figs = {}

    # -----------------------------------------------------------------------
    # Figure 1: Foiling Dashboard (7 vertical subplots)
    # -----------------------------------------------------------------------
    with plt.rc_context(FMD_STYLE):
        fig_foil, axes_f = plt.subplots(7, 1, figsize=(12, 16), sharex=True)

        # 1. Heights (m above water)
        # pos_d is NED depth: negate to get height above surface
        ax = axes_f[0]
        cg_height = -true_states[:n_steps, 0]
        ax.plot(times, cg_height, linewidth=1.0, label="CG")
        # Bowsprit tip height: rotate bowsprit body position through pitch
        # delta_D (NED down) = -bp_x * sin(theta) + bp_z * cos(theta)
        # bowsprit height = cg_height - delta_D
        bp = np.array(params.bowsprit_position)
        theta = true_states[:n_steps, 1]
        delta_d = -bp[0] * np.sin(theta) + bp[2] * np.cos(theta)
        bp_height = cg_height - delta_d
        ax.plot(times, bp_height, linewidth=1.0, label="Bowsprit")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Height (m)")
        ax.legend(fontsize=7, frameon=False)
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # 2. Pitch (deg)
        ax = axes_f[1]
        ax.plot(times, np.degrees(true_states[:n_steps, 1]), linewidth=1.0)
        ax.set_ylabel("Pitch (deg)")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # 3. Main flap (deg)
        ax = axes_f[2]
        ax.plot(times, np.degrees(controls[:, 0]), linewidth=1.0)
        ax.set_ylabel("Main flap (deg)")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # 4. Rudder elevator (deg)
        ax = axes_f[3]
        ax.plot(times, np.degrees(controls[:, 1]), linewidth=1.0)
        ax.set_ylabel("Elevator (deg)")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # 5. Vertical forces (N)
        ax = axes_f[4]
        if force_log is not None:
            ax.plot(times, -force_log.main_foil_force[:, 2], linewidth=1.0, label="Main foil")
            ax.plot(times, -force_log.rudder_force[:, 2], linewidth=1.0, label="Rudder")
            ax.legend(fontsize=7, frameon=False)
        ax.set_ylabel("Vertical force (N)")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # 6. Strut depth (m)
        ax = axes_f[5]
        pos_d = true_states[:n_steps, 0]
        # Apply CG offset to get effective foil positions (matches dynamics model)
        total_mass = params.hull_mass + params.sailor_mass
        cg_offset = params.sailor_mass * np.array(params.sailor_position) / total_mass
        foil_pos = np.array(params.main_foil_position) - cg_offset
        rudder_pos = np.array(params.rudder_position) - cg_offset
        from fmd.simulator.components.moth_forces import compute_foil_ned_depth
        main_depth = np.asarray(compute_foil_ned_depth(
            pos_d, foil_pos[0], foil_pos[2], theta, heel_angle))
        rudder_depth = np.asarray(compute_foil_ned_depth(
            pos_d, rudder_pos[0], rudder_pos[2], theta, heel_angle))
        ax.plot(times, main_depth, linewidth=1.0, label="Main (center)")
        ax.plot(times, rudder_depth, linewidth=1.0, label="Rudder (center)")
        # Leeward tip: rises by (span/2)*sin(heel) above the T-junction
        main_tip_depth = main_depth - (params.main_foil_span / 2.0) * np.sin(heel_angle)
        rudder_tip_depth = rudder_depth - (params.rudder_span / 2.0) * np.sin(heel_angle)
        ax.plot(times, main_tip_depth, linewidth=1.0, linestyle="--", label="Main (lee tip)")
        ax.plot(times, rudder_tip_depth, linewidth=1.0, linestyle="--", label="Rudder (lee tip)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Foil depth (m)")
        ax.invert_yaxis()
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # 7. Foil submersion (geometric % submerged vs depth factor model)
        ax = axes_f[6]
        main_max_sub = (params.main_foil_span / 2.0) * np.sin(max(heel_angle, 0.01))
        rudder_max_sub = (params.rudder_span / 2.0) * np.sin(max(heel_angle, 0.01))
        main_geom = np.clip(main_depth / main_max_sub, 0, 1)
        rudder_geom = np.clip(rudder_depth / rudder_max_sub, 0, 1)
        ax.plot(times, main_geom, linewidth=1.0, label="Main (geom)")
        ax.plot(times, rudder_geom, linewidth=1.0, label="Rudder (geom)")

        # Depth factor from model (numpy approximation of the smooth model)
        from fmd.simulator.components.moth_forces import compute_depth_factor
        import jax.numpy as jnp_
        main_df = np.array([
            float(compute_depth_factor(
                jnp_.array(main_depth[i]),
                params.main_foil_span, heel_angle,
            ))
            for i in range(n_steps)
        ])
        rudder_df = np.array([
            float(compute_depth_factor(
                jnp_.array(rudder_depth[i]),
                params.rudder_span, heel_angle,
            ))
            for i in range(n_steps)
        ])
        ax.plot(times, main_df, linewidth=1.0, linestyle="--", label="Main (model)")
        ax.plot(times, rudder_df, linewidth=1.0, linestyle="--", label="Rudder (model)")
        ax.set_ylabel("Foil submersion / depth factor")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        fig_foil.tight_layout()
        figs["foiling_dashboard"] = fig_foil

    # -----------------------------------------------------------------------
    # Figure 2: Estimation Dashboard (8 vertical subplots)
    # -----------------------------------------------------------------------
    with plt.rc_context(FMD_STYLE):
        fig_est, axes_e = plt.subplots(8, 1, figsize=(12, 18), sharex=True)

        # State display config: (index, label, is_angle)
        state_info = [
            (0, "Height (m)", False),
            (1, "Pitch (deg)", True),
            (2, "w (m/s)", False),
            (3, "q (deg/s)", True),
            (4, "u (m/s)", False),
        ]

        # Subplots 1-5: State tracking (true, estimated, and measurements for
        # states that have measurement channels)
        meas_clean = result.measurements_clean
        meas_noisy = result.measurements_noisy

        for plot_idx, (si, label, is_angle) in enumerate(state_info):
            ax = axes_e[plot_idx]
            transform = np.degrees if is_angle else lambda x: x

            true_vals = transform(true_states[:n_steps, si])
            est_vals = transform(est_states[:n_steps, si])
            ax.plot(times, true_vals, linewidth=1.0, label="True")
            ax.plot(times, est_vals, linewidth=1.0, linestyle="--", alpha=0.8, label="Estimated")

            # Show clean/noisy measurements if available and this state has
            # a corresponding measurement channel.
            if meas_clean is not None and meas_noisy is not None:
                state_map = getattr(result, "measurement_state_index_map", None)
                meas_names = getattr(result, "measurement_output_names", None)
                meas_idx = None
                if state_map and meas_names:
                    # Use model-provided mapping (output_name → state_index)
                    for mi, mname in enumerate(meas_names):
                        if state_map.get(mname) == si:
                            meas_idx = mi
                            break
                elif not state_map:
                    # Fallback for result objects without state_index_map
                    n_meas = meas_clean.shape[1] if meas_clean.ndim > 1 else 1
                    if n_meas >= 5 and si < 5:
                        meas_idx = si
                if meas_idx is not None:
                    ax.plot(times, transform(meas_clean[:, meas_idx]), linewidth=0.5, alpha=0.5, label="Meas (clean)")
                    ax.scatter(times[::10], transform(meas_noisy[::10, meas_idx]), s=3, alpha=0.3, label="Meas (noisy)")

            ax.set_ylabel(label)
            ax.legend(fontsize=6, frameon=False, loc="upper right")
            ax.margins(x=0, y=0.05)
            ax.grid(True, alpha=0.3)

        # Subplot 6: Estimation error + 2-sigma for all states
        ax = axes_e[5]
        colors = plt.colormaps.get_cmap("tab10").resampled(10)
        for si, label_short, is_angle in state_info:
            transform = np.degrees if is_angle else lambda x: x
            short_name = label_short.split(" ")[0]
            err = transform(est_errors[:n_steps, si])
            sigma = transform(np.sqrt(cov_diags[:n_steps, si]))
            c = colors(si)
            ax.plot(times, err, linewidth=0.8, color=c, label=short_name)
            ax.fill_between(times, -2 * sigma, 2 * sigma, color=c, alpha=0.08)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Est. error")
        ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper right")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # Subplot 7: Covariance diagonal (per-state sigma^2)
        ax = axes_e[6]
        for si, label_short, _ in state_info:
            short_name = label_short.split(" ")[0]
            ax.plot(times, cov_diags[:n_steps, si], linewidth=0.8, label=short_name)
        ax.set_ylabel("Variance")
        ax.set_yscale("log")
        ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper right")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        # Subplot 8: Innovation sequence
        ax = axes_e[7]
        innovations = result.innovations
        if innovations is not None and len(innovations) > 0:
            n_innov = innovations.shape[1] if innovations.ndim > 1 else 1
            for j in range(min(n_innov, 5)):
                ax.plot(times, innovations[:, j], linewidth=0.6, alpha=0.7, label=f"ch {j}")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.legend(fontsize=6, frameon=False, ncol=min(n_innov, 5))
        ax.set_ylabel("Innovation")
        ax.set_xlabel("Time (s)")
        ax.margins(x=0, y=0.05)
        ax.grid(True, alpha=0.3)

        fig_est.tight_layout()
        figs["estimation_dashboard"] = fig_est

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            savefig_and_close(fig, os.path.join(save_dir, f"{name}.png"))

    return figs


# Alias for the new ClosedLoopResult type (same duck-typed interface)
plot_closed_loop_result = plot_lqg_result


def plot_sweep_category(
    results,
    *,
    state_indices,
    state_transforms=None,
    reference_state=None,
    physical_bounds=None,
    panel_config=None,
    title=None,
    dpi=150,
):
    """Generic sweep category plot.

    Builds trace dicts from a list of results and delegates to multi_trace_grid.

    Args:
        results: list of objects with .result (SimulationResult), .config, and
                 optional .divergence_time, .stable attributes
        state_indices: list of state column indices to plot
        state_transforms: list of callables or None per state index
        reference_state: optional reference state array
        physical_bounds: dict or list of (min, max) per state index
        panel_config: optional list of dicts with panel-level overrides
        title: figure suptitle
        dpi: figure DPI

    Returns:
        (fig, axes) tuple
    """
    from fmd.analysis.plots._simulation import multi_trace_grid

    traces_by_panel = []
    ref_values = []
    p_bounds = []

    for si_idx, state_idx in enumerate(state_indices):
        transform = None
        if state_transforms and si_idx < len(state_transforms):
            transform = state_transforms[si_idx]

        traces = []
        for r in results:
            if r.result is None:
                continue
            times = np.array(r.result.times)
            states = np.array(r.result.states)
            values = states[:, state_idx]
            if transform is not None:
                values = transform(values)
            traces.append({
                "times": times,
                "values": values,
                "divergence_time": getattr(r, "divergence_time", None),
                "stable": getattr(r, "stable", True),
                "label": getattr(r.config, "description", ""),
            })
        traces_by_panel.append(traces)

        ref_val = None
        if reference_state is not None:
            ref_val = float(reference_state[state_idx])
            if transform is not None:
                ref_val = float(transform(ref_val))
        ref_values.append(ref_val)

        if isinstance(physical_bounds, dict):
            p_bounds.append(None)
        elif isinstance(physical_bounds, list) and si_idx < len(physical_bounds):
            p_bounds.append(physical_bounds[si_idx])
        else:
            p_bounds.append(None)

    fig, axes = multi_trace_grid(
        traces_by_panel,
        reference_values=ref_values,
        physical_bounds=p_bounds,
        title=title,
        dpi=dpi,
    )
    return fig, axes
