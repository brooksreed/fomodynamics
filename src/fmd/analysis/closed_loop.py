"""Shared analysis utilities for closed-loop simulation results.

Provides reusable plotting, metrics computation, and formatting functions
for comparing control configurations across wave conditions.

All functions accept ClosedLoopResult objects and numpy arrays. No JAX
dependency in this module — all computation is done with numpy.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# State indices (Moth 3DOF)
# ---------------------------------------------------------------------------

POS_D = 0
THETA = 1
W = 2
Q = 3
U = 4

# Default colors for config overlays
DEFAULT_COLORS = {
    "speed_pitch_height": "C0",
    "speed_pitch_wand_lqg": "C1",
    "wand_only_lqg": "C2",
    "mechanical_wand": "C3",
}


# ---------------------------------------------------------------------------
# Foil tip depth
# ---------------------------------------------------------------------------

def _get_foil_position(params) -> np.ndarray:
    """Compute main foil position relative to system CG.

    Accounts for CG offset due to sailor mass placement.

    Returns:
        3-element array [x, y, z] in FRD body frame (m).
    """
    total_mass = params.hull_mass + params.sailor_mass
    cg_offset = params.sailor_mass * np.array(params.sailor_position) / total_mass
    return np.array(params.main_foil_position) - cg_offset


def compute_leeward_tip_depth(
    pos_d: np.ndarray,
    theta: np.ndarray,
    foil_x: float,
    foil_z: float,
    foil_span: float,
    heel_angle: float,
) -> np.ndarray:
    """Compute leeward foil tip depth in NED frame.

    Delegates to the canonical ``compute_foil_ned_depth`` from moth_forces
    and applies the leeward tip offset for a heeled boat.

    Note: This computes geometric depth relative to the still water surface
    (pos_d=0). Wave surface elevation is not accounted for, so in wave
    conditions the depth is relative to the mean water level, not the
    local wave surface.

    Args:
        pos_d: CG vertical position (m), positive down.
        theta: Pitch angle (rad).
        foil_x: Foil x-position relative to system CG (m), FRD body frame.
        foil_z: Foil z-position relative to system CG (m), FRD body frame.
        foil_span: Foil wingspan (m).
        heel_angle: Static heel angle (rad).

    Returns:
        Leeward tip depth (m). Positive = submerged.
    """
    # Foil center depth (pure numpy implementation matching moth_forces formula)
    center_depth = (
        pos_d
        + foil_z * np.cos(heel_angle) * np.cos(theta)
        - foil_x * np.sin(theta)
    )
    # Leeward tip rises toward surface
    half_span_rise = (foil_span / 2.0) * np.sin(heel_angle)
    return center_depth - half_span_rise


# ---------------------------------------------------------------------------
# Extended metrics
# ---------------------------------------------------------------------------

def compute_extended_metrics(
    result,
    trim_state: np.ndarray,
    aux: dict,
    dt: float = 0.005,
    steady_state_start: float = 5.0,
) -> dict:
    """Compute extended performance metrics from a ClosedLoopResult.

    This is a superset of ``moth_metrics.compute_metrics()``. Returns
    ride height, pitch, speed, foil tip depth, control effort, control
    rate, settling time, and NaN flag.

    Args:
        result: ClosedLoopResult from a simulation run.
        trim_state: Trim state vector (numpy array, 5-element).
        aux: Auxiliary trajectory dict from compute_aux_trajectory.
        dt: Simulation timestep (s).
        steady_state_start: Time (s) after which to compute steady-state
            statistics. Default 5.0.

    Returns:
        Dict with comprehensive metrics.
    """
    true_states = np.asarray(result.true_states[1:])  # skip initial
    controls = np.asarray(result.controls)
    times = np.asarray(result.times)

    has_nan = bool(np.any(np.isnan(true_states)) or np.any(np.isnan(controls)))

    n_steps = len(times)
    ss_idx = int(steady_state_start / dt) if dt > 0 else 0
    ss_idx = min(ss_idx, n_steps - 1)

    # Full-run state arrays
    pos_d = true_states[:, POS_D]
    theta = true_states[:, THETA]
    u_fwd = true_states[:, U]

    # Steady-state slices
    pos_d_ss = pos_d[ss_idx:]
    theta_ss = theta[ss_idx:]
    u_ss = u_fwd[ss_idx:]
    controls_ss = controls[ss_idx:]

    # Errors from trim
    pos_d_err = pos_d - trim_state[POS_D]
    theta_err = theta - trim_state[THETA]

    # --- Ride height stats ---
    ride_height = {
        "mean": float(np.mean(pos_d_ss)),
        "std": float(np.std(pos_d_ss)),
        "min": float(np.min(pos_d_ss)),
        "max": float(np.max(pos_d_ss)),
        "rms_error": float(np.sqrt(np.mean(pos_d_err[ss_idx:] ** 2))),
    }

    # --- Pitch stats ---
    pitch = {
        "mean_rad": float(np.mean(theta_ss)),
        "std_rad": float(np.std(theta_ss)),
        "min_rad": float(np.min(theta_ss)),
        "max_rad": float(np.max(theta_ss)),
        "rms_error_rad": float(np.sqrt(np.mean(theta_err[ss_idx:] ** 2))),
        "mean_deg": float(np.degrees(np.mean(theta_ss))),
        "std_deg": float(np.degrees(np.std(theta_ss))),
    }

    # --- Speed stats ---
    speed = {
        "mean": float(np.mean(u_ss)),
        "std": float(np.std(u_ss)),
        "min": float(np.min(u_ss)),
        "max": float(np.max(u_ss)),
    }

    # --- Foil tip depth ---
    # Use auxiliary data if available, otherwise skip
    foil_tip_depth = {}
    # Compute from state if we have geometry (check for heel_angle on result)
    heel_angle = getattr(result, "heel_angle", 0.0)
    params = getattr(result, "params", None)
    if params is not None:
        foil_pos = _get_foil_position(params)

        tip_depth = compute_leeward_tip_depth(
            pos_d, theta, foil_pos[0], foil_pos[2],
            params.main_foil_span, heel_angle,
        )
        tip_depth_ss = tip_depth[ss_idx:]
        breached = tip_depth_ss < 0  # negative depth = above water
        # Count breach events (transitions from submerged to breached)
        breach_transitions = np.diff(breached.astype(int))
        breach_count = int(np.sum(breach_transitions == 1))

        foil_tip_depth = {
            "min": float(np.min(tip_depth_ss)),
            "max": float(np.max(tip_depth_ss)),
            "mean": float(np.mean(tip_depth_ss)),
            "breach_fraction": float(np.mean(breached)),
            "breach_count": breach_count,
        }

    # --- Control effort ---
    n_controls = controls.shape[1]
    control_effort = {}
    control_names = ["main_flap", "rudder_elevator"]
    for ci in range(n_controls):
        name = control_names[ci] if ci < len(control_names) else f"control_{ci}"
        ctrl_ss = controls_ss[:, ci]
        control_effort[name] = {
            "mean_rad": float(np.mean(ctrl_ss)),
            "std_rad": float(np.std(ctrl_ss)),
            "max_abs_rad": float(np.max(np.abs(ctrl_ss))),
            "mean_deg": float(np.degrees(np.mean(ctrl_ss))),
            "std_deg": float(np.degrees(np.std(ctrl_ss))),
            "max_abs_deg": float(np.degrees(np.max(np.abs(ctrl_ss)))),
        }

    # --- Control rate ---
    control_rate = {}
    if n_steps > 1:
        du = np.diff(controls, axis=0) / dt
        du_ss = du[max(ss_idx - 1, 0):]
        for ci in range(n_controls):
            name = control_names[ci] if ci < len(control_names) else f"control_{ci}"
            control_rate[name] = {
                "rms_rad_per_s": float(np.sqrt(np.mean(du_ss[:, ci] ** 2))),
                "rms_deg_per_s": float(np.degrees(np.sqrt(np.mean(du_ss[:, ci] ** 2)))),
            }

    # --- Settling time ---
    settling_time = float(times[-1])  # default: never settled
    threshold = 0.01
    within = np.abs(pos_d_err) < threshold
    for i in range(len(within)):
        if np.all(within[i:]):
            settling_time = float(times[i])
            break

    # --- Overall control effort (scalar, backward-compatible) ---
    if n_steps > 1:
        du_all = np.diff(controls, axis=0)
        overall_control_effort = float(np.sqrt(np.mean(du_all ** 2)))
    else:
        overall_control_effort = 0.0

    return {
        "ride_height": ride_height,
        "pitch": pitch,
        "speed": speed,
        "foil_tip_depth": foil_tip_depth,
        "control_effort": control_effort,
        "control_rate": control_rate,
        "settling_time": settling_time,
        "overall_control_effort": overall_control_effort,
        "rms_pos_d": ride_height["rms_error"],
        "rms_theta": pitch["rms_error_rad"],
        "has_nan": has_nan,
    }


# ---------------------------------------------------------------------------
# Wave vs calm comparison metrics
# ---------------------------------------------------------------------------

def compute_wave_vs_calm_metrics(
    wave_result,
    calm_result,
    wave_aux: dict,
    calm_aux: dict,
    trim_state: np.ndarray,
    dt: float = 0.005,
) -> dict:
    """Compute metrics comparing wave and calm simulation results.

    Returns per-state deltas (wave_mean - calm_mean, wave_std, calm_std),
    speed windows, and drag decomposition.

    Args:
        wave_result: ClosedLoopResult from wave simulation.
        calm_result: ClosedLoopResult from calm simulation.
        wave_aux: Auxiliary trajectory dict for wave run.
        calm_aux: Auxiliary trajectory dict for calm run.
        trim_state: Trim state vector.
        dt: Simulation timestep (s).

    Returns:
        Dict with state_statistics, control_statistics, drag_decomposition,
        and speed_equilibrium.
    """
    wave_states = np.asarray(wave_result.true_states[1:])
    calm_states = np.asarray(calm_result.true_states[1:])
    wave_controls = np.asarray(wave_result.controls)
    calm_controls = np.asarray(calm_result.controls)
    times = np.asarray(wave_result.times)

    # State statistics
    state_names = [("pos_d", POS_D), ("theta", THETA), ("w", W), ("q", Q), ("u", U)]
    state_stats = {}
    for name, idx in state_names:
        wave_vals = wave_states[:, idx]
        calm_vals = calm_states[:, idx]
        state_stats[name] = {
            "wave_mean": float(np.mean(wave_vals)),
            "wave_std": float(np.std(wave_vals)),
            "wave_min": float(np.min(wave_vals)),
            "wave_max": float(np.max(wave_vals)),
            "calm_mean": float(np.mean(calm_vals)),
            "calm_std": float(np.std(calm_vals)),
            "delta_mean": float(np.mean(wave_vals) - np.mean(calm_vals)),
        }

    # Control statistics
    control_names = [("main_flap", 0), ("rudder_elevator", 1)]
    control_stats = {}
    for name, idx in control_names:
        wave_vals = np.degrees(wave_controls[:, idx])
        calm_vals = np.degrees(calm_controls[:, idx])
        control_stats[name] = {
            "wave_mean_deg": float(np.mean(wave_vals)),
            "wave_std_deg": float(np.std(wave_vals)),
            "calm_mean_deg": float(np.mean(calm_vals)),
            "calm_std_deg": float(np.std(calm_vals)),
        }

    # Drag decomposition
    drag_keys = [
        "main_drag_aero", "rudder_drag_aero",
        "main_strut_drag", "rudder_strut_drag",
    ]
    drag_decomp = {}
    for key in drag_keys:
        if key in wave_aux and key in calm_aux:
            drag_decomp[key] = {
                "wave_mean": float(np.mean(wave_aux[key])),
                "calm_mean": float(np.mean(calm_aux[key])),
                "difference": float(np.mean(wave_aux[key]) - np.mean(calm_aux[key])),
            }

    # Additional drag/buoyancy keys
    for key in ["main_strut_immersion", "rudder_strut_immersion",
                "hull_drag", "hull_buoyancy"]:
        if key in wave_aux and key in calm_aux:
            drag_decomp[key] = {
                "wave_mean": float(np.mean(wave_aux[key])),
                "calm_mean": float(np.mean(calm_aux[key])),
            }

    # Speed equilibrium: mean speed in time windows
    duration = float(times[-1]) if len(times) > 0 else 0.0
    windows = [(0, 5), (5, 10), (10, 20), (20, 30)]
    speed_windows = {}
    for t0, t1 in windows:
        if t0 >= duration:
            continue
        mask = (times >= t0) & (times < t1)
        if not np.any(mask):
            continue
        speed_windows[f"{t0}-{t1}s"] = {
            "wave_mean_ms": float(np.mean(wave_states[mask, U])),
            "calm_mean_ms": float(np.mean(calm_states[mask, U])),
        }

    return {
        "state_statistics": state_stats,
        "control_statistics": control_stats,
        "drag_decomposition": drag_decomp,
        "speed_equilibrium": speed_windows,
    }


# ---------------------------------------------------------------------------
# Metrics table formatting
# ---------------------------------------------------------------------------

def format_metrics_table(
    metrics: dict[tuple[str, str], dict],
) -> str:
    """Format a metrics dictionary into a printable text table.

    Args:
        metrics: Mapping of ``(config_name, condition)`` to metrics dict.
            Each metrics dict should have keys like ``rms_pos_d``,
            ``rms_theta``, ``overall_control_effort``, ``settling_time``,
            ``has_nan``. Also accepts the nested format from
            ``compute_extended_metrics`` (auto-detected).

    Returns:
        Formatted text table as a string.
    """

    # Build rows
    rows = []
    for (config, condition), m in sorted(metrics.items()):
        # Handle both flat and nested metric formats
        rms_pos_d = m.get("rms_pos_d", 0.0)
        rms_theta = m.get("rms_theta", 0.0)
        ctrl_effort = m.get("overall_control_effort", m.get("control_effort", 0.0))
        if isinstance(ctrl_effort, dict):
            ctrl_effort = 0.0  # nested format, skip scalar
        settling = m.get("settling_time", 0.0)
        has_nan = m.get("has_nan", False)

        # Foil tip breach fraction
        tip = m.get("foil_tip_depth", {})
        breach_frac = tip.get("breach_fraction", 0.0) if isinstance(tip, dict) else 0.0

        rows.append({
            "config": config,
            "condition": condition,
            "rms_pos_d": rms_pos_d,
            "rms_theta": rms_theta,
            "overall_control_effort": ctrl_effort,
            "settling_time": settling,
            "breach_fraction": breach_frac,
            "has_nan": has_nan,
        })

    # Format header and rows
    header = (
        f"{'Config':<30} {'Cond':<15} {'RMS pos_d':>10} {'RMS theta':>10} "
        f"{'Ctrl Effort':>12} {'Settle (s)':>10} {'Breach %':>10} {'NaN':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        line = (
            f"{r['config']:<30} {r['condition']:<15} "
            f"{r['rms_pos_d']:>10.4f} {r['rms_theta']:>10.4f} "
            f"{r['overall_control_effort']:>12.4f} {r['settling_time']:>10.2f} "
            f"{r['breach_fraction']*100:>9.1f}% "
            f"{'YES' if r['has_nan'] else 'no':>5}"
        )
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interesting window finder
# ---------------------------------------------------------------------------

def find_interesting_window(
    results: dict,
    auxs: dict,
    trim_state: np.ndarray,
    window_size: float = 5.0,
    dt: float = 0.005,
) -> tuple[float, float]:
    """Find the most interesting time window across all configs.

    Computes a per-timestep interest score based on ride height excursion,
    control activity, and breach proximity. Returns the start and end
    times of the most interesting window.

    Args:
        results: Dict mapping config name to ClosedLoopResult.
        auxs: Dict mapping config name to aux trajectory dict.
        trim_state: Trim state vector.
        window_size: Window duration (s).
        dt: Simulation timestep (s).

    Returns:
        Tuple of (t_start, t_end) for the most interesting window.
    """
    # Determine time array from first result
    first_key = next(iter(results))
    times = np.asarray(results[first_key].times)
    n_steps = len(times)
    window_steps = int(window_size / dt)

    if n_steps <= window_steps:
        return (float(times[0]), float(times[-1]))

    # Compute per-timestep interest score (sum across all configs)
    interest = np.zeros(n_steps)
    for name, result in results.items():
        states = np.asarray(result.true_states[1:])
        controls = np.asarray(result.controls)

        # Ride height excursion from trim
        pos_d_err = np.abs(states[:, POS_D] - trim_state[POS_D])
        interest += pos_d_err / max(np.std(pos_d_err), 1e-8)

        # Control activity (absolute value of control rate)
        if len(controls) > 1:
            du = np.abs(np.diff(controls, axis=0))
            du_padded = np.concatenate([du, du[-1:]], axis=0)
            interest += np.sum(du_padded, axis=1) / max(np.std(du_padded), 1e-8)

        # Pitch excursion
        theta_err = np.abs(states[:, THETA] - trim_state[THETA])
        interest += theta_err / max(np.std(theta_err), 1e-8)

    # Sliding window sum
    cumsum = np.cumsum(interest)
    cumsum = np.insert(cumsum, 0, 0.0)
    window_scores = cumsum[window_steps:] - cumsum[:-window_steps]

    best_start_idx = int(np.argmax(window_scores))
    t_start = float(times[best_start_idx])
    t_end = t_start + window_size

    # Clamp to available time range
    t_end = min(t_end, float(times[-1]))

    return (t_start, t_end)


# ---------------------------------------------------------------------------
# Plotting: Config overlay
# ---------------------------------------------------------------------------

def _time_slice(times: np.ndarray, t_start, t_end):
    """Return boolean mask for times in [t_start, t_end]."""
    mask = np.ones(len(times), dtype=bool)
    if t_start is not None:
        mask &= times >= t_start
    if t_end is not None:
        mask &= times <= t_end
    return mask


def plot_config_overlay(
    results: dict,
    trim_state: np.ndarray,
    *,
    aux_dict: dict | None = None,
    wave_params=None,
    colors: dict[str, str] | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    title: str | None = None,
) -> Figure:
    """6-panel overlay comparison plot for multiple configs.

    Panels:
        1. Wave elevation (if waves, from aux)
        2. pos_d (ride height)
        3. theta (pitch, degrees)
        4. Forward speed (u)
        5. Control effort (main flap, degrees)
        6. Foil tip depth (if params available)

    Args:
        results: Dict mapping config name to ClosedLoopResult.
        trim_state: Trim state vector.
        aux_dict: Optional dict mapping config name to aux trajectory dict.
        wave_params: Optional wave params (used for title annotation).
        colors: Optional dict mapping config name to matplotlib color.
        t_start: Start of time window for plotting.
        t_end: End of time window for plotting.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    has_waves = (
        aux_dict is not None
        and any("wave_eta_main" in aux for aux in aux_dict.values())
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), tight_layout=True)
    if title is None:
        title = "Config Comparison"
        if wave_params is not None:
            title += f" (Hs={wave_params.significant_wave_height:.2f}m)"
    fig.suptitle(title, fontsize=14)

    for name, result in results.items():
        color = colors.get(name, None)
        times = np.asarray(result.times)
        mask = _time_slice(times, t_start, t_end)
        t_plot = times[mask]
        states = np.asarray(result.true_states[1:])[mask]
        controls = np.asarray(result.controls)[mask]

        # Panel 1: Wave elevation
        ax = axes[0, 0]
        if has_waves and aux_dict and name in aux_dict:
            aux = aux_dict[name]
            if "wave_eta_main" in aux:
                ax.plot(t_plot, aux["wave_eta_main"][mask],
                        color=color, alpha=0.7, label=f"{name}")

        # Panel 2: pos_d
        ax = axes[0, 1]
        ax.plot(t_plot, states[:, POS_D], color=color, alpha=0.8, label=name)

        # Panel 3: theta (degrees)
        ax = axes[1, 0]
        ax.plot(t_plot, np.degrees(states[:, THETA]),
                color=color, alpha=0.8, label=name)

        # Panel 4: forward speed
        ax = axes[1, 1]
        ax.plot(t_plot, states[:, U], color=color, alpha=0.8, label=name)

        # Panel 5: main flap (degrees)
        ax = axes[2, 0]
        ax.plot(t_plot, np.degrees(controls[:, 0]),
                color=color, alpha=0.8, label=name)

        # Panel 6: foil tip depth
        ax = axes[2, 1]
        params = getattr(result, "params", None)
        heel_angle = getattr(result, "heel_angle", 0.0)
        if params is not None:
            foil_pos = _get_foil_position(params)
            tip_depth = compute_leeward_tip_depth(
                states[:, POS_D], states[:, THETA],
                foil_pos[0], foil_pos[2],
                params.main_foil_span, heel_angle,
            )
            ax.plot(t_plot, tip_depth, color=color, alpha=0.8, label=name)

    # Format panels
    axes[0, 0].set_ylabel("Wave elevation (m)")
    axes[0, 0].set_title("Wave Elevation at Main Foil")
    if not has_waves:
        axes[0, 0].text(0.5, 0.5, "No waves", transform=axes[0, 0].transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
    if has_waves:
        axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axhline(trim_state[POS_D], color="k", linestyle="--", alpha=0.3)
    axes[0, 1].set_ylabel("pos_d (m)")
    axes[0, 1].set_title("Ride Height")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].axhline(np.degrees(trim_state[THETA]), color="k", linestyle="--", alpha=0.3)
    axes[1, 0].set_ylabel("theta (deg)")
    axes[1, 0].set_title("Pitch")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axhline(trim_state[U], color="k", linestyle="--", alpha=0.3)
    axes[1, 1].set_ylabel("u (m/s)")
    axes[1, 1].set_title("Forward Speed")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Main flap (deg)")
    axes[2, 0].set_title("Main Flap Control")
    axes[2, 0].legend(fontsize=7)
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].axhline(0, color="k", linestyle="--", alpha=0.3)
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Tip depth (m)")
    axes[2, 1].set_title("Leeward Foil Tip Depth")
    axes[2, 1].invert_yaxis()
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].grid(True, alpha=0.3)

    return fig


# ---------------------------------------------------------------------------
# Plotting: Single config dashboard
# ---------------------------------------------------------------------------

def plot_single_dashboard(
    result,
    trim_state: np.ndarray,
    *,
    aux: dict | None = None,
    wave_params=None,
    calm_result=None,
    calm_aux: dict | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    title: str | None = None,
) -> Figure:
    """6-panel wave-aware dashboard for a single configuration.

    Panels:
        1. Wave elevation (if waves, from aux)
        2. pos_d with optional calm overlay
        3. theta (degrees) with optional calm overlay
        4. Forward speed (u) with optional calm overlay
        5. Control effort (both channels, degrees)
        6. Foil tip depth with breach markers

    Args:
        result: ClosedLoopResult for the primary run.
        trim_state: Trim state vector.
        aux: Optional auxiliary trajectory dict.
        wave_params: Optional wave params for title annotation.
        calm_result: Optional ClosedLoopResult for calm baseline.
        calm_aux: Optional auxiliary trajectory for calm baseline.
        t_start: Start of time window for plotting.
        t_end: End of time window for plotting.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    times = np.asarray(result.times)
    mask = _time_slice(times, t_start, t_end)
    t_plot = times[mask]
    states = np.asarray(result.true_states[1:])[mask]
    controls = np.asarray(result.controls)[mask]

    has_waves = aux is not None and "wave_eta_main" in aux

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), tight_layout=True)
    if title is None:
        title = "Closed-Loop Dashboard"
    fig.suptitle(title, fontsize=14)

    # Panel 1: Wave elevation
    ax = axes[0, 0]
    if has_waves:
        ax.plot(t_plot, aux["wave_eta_main"][mask], label="Main foil", alpha=0.8)
        ax.plot(t_plot, aux["wave_eta_rudder"][mask], label="Rudder", alpha=0.8)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Calm water (no waves)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.set_ylabel("Wave elevation (m)")
    ax.set_title("Wave Elevation at Foils")
    ax.grid(True, alpha=0.3)

    # Panel 2: pos_d
    ax = axes[0, 1]
    ax.plot(t_plot, states[:, POS_D], label="pos_d", alpha=0.8)
    ax.axhline(trim_state[POS_D], color="k", linestyle="--", alpha=0.5, label="trim")
    if calm_result is not None:
        calm_states = np.asarray(calm_result.true_states[1:])[mask]
        ax.plot(t_plot, calm_states[:, POS_D], label="calm", alpha=0.4,
                color="gray", linestyle="--")
    ax.set_ylabel("pos_d (m)")
    ax.set_title("Ride Height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: theta
    ax = axes[1, 0]
    ax.plot(t_plot, np.degrees(states[:, THETA]), label="theta", alpha=0.8)
    ax.axhline(np.degrees(trim_state[THETA]), color="k", linestyle="--",
               alpha=0.5, label="trim")
    if calm_result is not None:
        ax.plot(t_plot, np.degrees(calm_states[:, THETA]), label="calm",
                alpha=0.4, color="gray", linestyle="--")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Pitch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: forward speed
    ax = axes[1, 1]
    ax.plot(t_plot, states[:, U], label="u", alpha=0.8)
    ax.axhline(trim_state[U], color="k", linestyle="--", alpha=0.5, label="trim")
    if calm_result is not None:
        ax.plot(t_plot, calm_states[:, U], label="calm", alpha=0.4,
                color="gray", linestyle="--")
    ax.set_ylabel("u (m/s)")
    ax.set_title("Forward Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: controls
    ax = axes[2, 0]
    ax.plot(t_plot, np.degrees(controls[:, 0]), label="Main flap", alpha=0.8)
    ax.plot(t_plot, np.degrees(controls[:, 1]), label="Rudder elev", alpha=0.8)
    if calm_result is not None:
        calm_controls = np.asarray(calm_result.controls)[mask]
        ax.plot(t_plot, np.degrees(calm_controls[:, 0]),
                label="Flap (calm)", alpha=0.4, color="gray", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control (deg)")
    ax.set_title("Control Effort")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: foil tip depth
    ax = axes[2, 1]
    params = getattr(result, "params", None)
    heel_angle = getattr(result, "heel_angle", 0.0)
    if params is not None:
        foil_pos = _get_foil_position(params)

        tip_depth = compute_leeward_tip_depth(
            states[:, POS_D], states[:, THETA],
            foil_pos[0], foil_pos[2],
            params.main_foil_span, heel_angle,
        )
        ax.plot(t_plot, tip_depth, label="Lee tip", alpha=0.8)
        ax.axhline(0, color="r", linestyle="--", alpha=0.5, label="surface")

        # Mark breach events
        breached = tip_depth < 0
        if np.any(breached):
            ax.fill_between(t_plot, tip_depth, 0, where=breached,
                            alpha=0.2, color="red", label="breach")

        if calm_result is not None:
            calm_tip = compute_leeward_tip_depth(
                calm_states[:, POS_D], calm_states[:, THETA],
                foil_pos[0], foil_pos[2],
                params.main_foil_span, heel_angle,
            )
            ax.plot(t_plot, calm_tip, label="Lee tip (calm)",
                    alpha=0.4, color="gray", linestyle="--")

        ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip depth (m)")
    ax.set_title("Leeward Foil Tip Depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig
