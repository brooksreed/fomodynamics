#!/usr/bin/env python3
"""Canonical wand vs PID comparison under SF Bay moderate waves.

Runs a single-seed time series and a 50-seed Monte Carlo, comparing:
- ``create_mechanical_wand_config``  (passive mechanical wand-to-flap linkage)
- ``create_pid_wand_config``         (PID on closed-form wand-derived height)

Both controllers see the same wave field per seed (paired comparison).
Outputs metrics.json + report_guidelines.txt + plots/*.png in the
report folder. The accompanying interpretation_skill.md tells an agent
how to write report.md from these artifacts.

Usage (from fmd repo root with the fmd editable install active):
    JAX_PLATFORMS=cpu .venv/bin/python docs/reports/wand_vs_pid_waves/run.py
    JAX_PLATFORMS=cpu .venv/bin/python docs/reports/wand_vs_pid_waves/run.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import attrs
import numpy as np

# Ensure src/ is importable when running outside an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from fmd.simulator import _config  # noqa: F401, must be first jax import

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmd.analysis.closed_loop import (
    compute_extended_metrics,
    compute_leeward_tip_depth,
    plot_single_dashboard,
)
from fmd.simulator.closed_loop_pipeline import simulate_closed_loop
from fmd.simulator.environment import Environment
from fmd.simulator.integrator import SimulationResult, compute_aux_trajectory
from fmd.simulator.moth_3d import ConstantSchedule, Moth3D
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.moth_scenarios import (
    create_mechanical_wand_config,
    create_pid_wand_config,
)
from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.params.presets import WAVE_SF_BAY_MODERATE
from fmd.simulator.sweep import sweep_closed_loop, stack_envs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

U_FORWARD = 10.0          # m/s (nominal foiling speed)
HEEL_ANGLE = np.deg2rad(30.0)
DT = 0.005                # s
DURATION = 60.0           # s
N_SEEDS_FULL = 50
N_SEEDS_QUICK = 5
SINGLE_SEED = 0
STEADY_START = 10.0       # s — skip the first 10s for steady-state stats

# Deeper PID target: 30 cm below the foil-tip-at-surface depth, leaves a
# realistic safety margin so the leeward foil tip stays comfortably
# below the local wave surface under SF Bay moderate waves
# (Hs ≈ 0.5 m, peak excursion ~0.25 m).
PID_DEEPER_MARGIN_M = 0.30

# Canonical controller list, used everywhere the plot/aggregation code
# iterates over kinds.
CONTROLLERS: tuple[str, ...] = ("mechanical", "pid_natural", "pid_deeper")
CONTROLLER_COLORS = {
    "mechanical": "C0",
    "pid_natural": "C3",
    "pid_deeper": "C2",
}
CONTROLLER_LABELS = {
    "mechanical": "mechanical wand",
    "pid_natural": "PID (natural trim)",
    "pid_deeper": "PID (deeper trim)",
}

REPORT_GUIDELINES = """\
## Physics Guidance for wand_vs_pid_waves Report

### NED Frame Convention
- pos_d is positive DOWN in NED. A pos_d that becomes MORE NEGATIVE means
  the boat is RISING (climbing above the still-water reference). Trim
  pos_d is typically around -0.95 to -1.0 m (boat flying that distance
  above the surface, on its main foil).
- A breach event is when the leeward foil tip emerges above the water
  surface (tip_depth < 0).

### What this report compares
- ``mechanical_wand``: WandSensor + PassthroughEstimator +
  MechanicalWandController. The wand-to-flap conversion is a pure
  trig/lever linkage (``WandLinkage``). It is fast (no sensor lag,
  no controller dynamics) but has fixed gain and zero integrator —
  any steady-state offset must be tuned via the linkage hardware.
- ``pid_wand``: WandSensor + PassthroughEstimator + PIDController. The
  wand angle is inverted to a closed-form ride-height estimate
  (assuming trim attitude) and fed through PID(Kp, Ki, Kd) on height
  error. The integrator allows zero steady-state offset under bias;
  Kd damps oscillation. The price is potentially higher flap activity
  and risk of saturation under aggressive gains.

### Expected qualitative differences
- PID should have *lower* ride-height RMS than the mechanical wand
  in waves (the integrator rejects the wave-induced steady bias).
- PID should have *higher* flap activity and a higher fraction of
  time saturated than the mechanical wand (it acts harder).
- PID should have *fewer* breach events than the mechanical wand
  (better height tracking).
- Both should stay foiling on average (mean foil tip depth > 0 over
  the steady-state window).
- Pitch tracking and forward-speed loss are secondary metrics; expect
  both controllers to behave similarly there (they both leave the
  rudder elevator at trim).

### NED-sign pitfalls
- "Lower ride-height RMS" means a smaller std of pos_d about trim —
  it does NOT mean pos_d is more negative.
- A breach happens when tip_depth crosses ZERO going negative.
  ``breach_count`` is the number of zero-crossings into negative.
- Mean depth factor > 0 means the foil is on average submerged.

### What to flag (do NOT silently accept)
- If PID has higher ride-height RMS than the mechanical wand, the
  gains likely need tuning (or there is a sign error in the
  closed-form inversion).
- If mean foil tip depth < 0, both controllers are flying off the
  foil on average — the wave amplitude is too large for the wand
  controller envelope.
- Asymmetric saturation (flap pinned to one limit) indicates a
  steady bias the integrator cannot recover from within the
  saturation envelope.

### Report structure (follow interpretation_skill.md for full detail)
1. Summary: 3-5 bullets, lead with the headline RMS / breach numbers.
2. Setup: trim state, wave preset, controller gains, sim parameters.
3. Single-seed time series (seed=0): pos_d, theta, u, flap, depth factor.
4. Monte Carlo (50 seeds): box/violin plots of RMS, breach count,
   flap activity. Use metrics.json for exact numbers.
5. Mechanism: why does PID help (or not)? Where does it saturate?
6. Tuning suggestions: which gain to raise, which wave preset is
   harder, what would benefit from a wave-aware sensor model.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_jsonable(x: Any) -> Any:
    """Convert numpy arrays / scalars / nested containers to JSON-safe types."""
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return _to_jsonable(np.asarray(x).tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _foil_breach_count(
    pos_d: np.ndarray,
    theta: np.ndarray,
    params,
    heel_angle,
    wave_eta: np.ndarray | None = None,
):
    """Count zero-crossings of leeward foil tip depth into negative (breach onsets).

    When ``wave_eta`` (main-foil wave surface elevation, length-matched to
    ``pos_d``) is supplied, the depth is measured against the
    *instantaneous* free surface rather than the still-water reference —
    a tip momentarily emerges only when it actually crosses the local
    wave surface, not just the mean water level.
    """
    total_mass = params.hull_mass + params.sailor_mass
    cg_offset = params.sailor_mass * np.asarray(params.sailor_position) / total_mass
    foil_pos = np.asarray(params.main_foil_position) - cg_offset
    tip_depth = compute_leeward_tip_depth(
        pos_d, theta, foil_pos[0], foil_pos[2],
        params.main_foil_span, heel_angle, wave_eta=wave_eta,
    )
    breached = tip_depth < 0
    transitions = np.diff(breached.astype(int))
    breach_count = int(np.sum(transitions == 1))
    breach_fraction = float(np.mean(breached))
    return breach_count, breach_fraction, tip_depth


def _mean_depth_factor(pos_d: np.ndarray, theta: np.ndarray, params, heel_angle):
    """Average depth_factor (1.0 = fully submerged) of the main foil over time."""
    from fmd.simulator.components.moth_forces import (
        compute_depth_factor,
        compute_foil_ned_depth,
    )

    total_mass = params.hull_mass + params.sailor_mass
    cg_offset = params.sailor_mass * np.asarray(params.sailor_position) / total_mass
    foil_pos = np.asarray(params.main_foil_position) - cg_offset

    depth = np.asarray(
        compute_foil_ned_depth(
            jnp.asarray(pos_d), float(foil_pos[0]), float(foil_pos[2]),
            jnp.asarray(theta), heel_angle,
        )
    )
    dfs = np.array([
        float(compute_depth_factor(
            jnp.asarray(d), params.main_foil_span, heel_angle,
        ))
        for d in depth
    ])
    return dfs


def _saturation_fraction(controls: np.ndarray, u_min: np.ndarray, u_max: np.ndarray):
    """Fraction of timesteps where the main flap is at either limit."""
    flap = controls[:, 0]
    eps = 1e-6
    saturated = (flap <= u_min[0] + eps) | (flap >= u_max[0] - eps)
    return float(np.mean(saturated))


# ---------------------------------------------------------------------------
# Single-seed time series
# ---------------------------------------------------------------------------


def _target_pos_d_for_kind(controller_kind: str, *, trim_state: np.ndarray) -> float:
    """Per-controller ride-height setpoint.

    Mirrors the dispatch in :func:`_build_controller_for_kind` — keep
    the two in sync. Returned value is what each controller is *trying*
    to hold, used by ``ride_height_rms_around_target``.

    - ``mechanical``: no explicit setpoint; the mechanical linkage's
      static equilibrium is approximately at the natural trim by
      construction (``pullrod_offset`` tuned to that point), so we
      report the natural trim for symmetry with the metric definition.
    - ``pid_natural``: PID's default target is ``lqr.trim.state[0]``
      (the natural-trim ride height).
    - ``pid_deeper``: ``compute_tip_at_surface_pos_d() + PID_DEEPER_MARGIN_M``,
      matching the override passed into ``create_pid_wand_config``.
    """
    if controller_kind == "mechanical":
        return float(trim_state[0])
    if controller_kind == "pid_natural":
        return float(trim_state[0])
    if controller_kind == "pid_deeper":
        return float(compute_tip_at_surface_pos_d() + PID_DEEPER_MARGIN_M)
    raise ValueError(f"Unknown controller_kind: {controller_kind}")


def _build_controller_for_kind(lqr, controller_kind: str):
    """Dispatch to the right factory for one of the three canonical kinds."""
    if controller_kind == "mechanical":
        return create_mechanical_wand_config(
            lqr, params=MOTH_BIEKER_V3, heel_angle=HEEL_ANGLE,
        )
    if controller_kind == "pid_natural":
        return create_pid_wand_config(
            lqr, params=MOTH_BIEKER_V3, heel_angle=HEEL_ANGLE, dt=DT,
        )
    if controller_kind == "pid_deeper":
        # Deeper-trim PID: parks the leeward foil tip 30 cm *below* the
        # still-water surface (NED depth positive = submerged), so the
        # foil keeps a safety margin against ventilation under SF Bay
        # moderate waves.  In NED, "deeper" means a MORE POSITIVE pos_d
        # (less negative altitude), so we ADD the margin to the
        # tip-at-surface pos_d, not subtract.  The plan write-up has the
        # sign inverted; we flip it here.
        target_pos_d = compute_tip_at_surface_pos_d() + PID_DEEPER_MARGIN_M
        return create_pid_wand_config(
            lqr, params=MOTH_BIEKER_V3, heel_angle=HEEL_ANGLE, dt=DT,
            target_pos_d=float(target_pos_d),
        )
    raise ValueError(f"Unknown controller_kind: {controller_kind}")


def run_single_seed(lqr, *, controller_kind: str, wave_seed: int):
    """Run one 60-second simulation with the given controller and wave seed.

    Returns the closed-loop result together with the ``Moth3D`` system
    and the wave-bearing ``Environment`` so callers can post-compute aux
    trajectories (e.g., ``wave_eta_main`` for the dashboard wave panel).
    """
    sensor, estimator, controller = _build_controller_for_kind(lqr, controller_kind)

    moth = Moth3D(
        MOTH_BIEKER_V3,
        u_forward=ConstantSchedule(U_FORWARD),
        heel_angle=HEEL_ANGLE,
    )
    trim_state = jnp.array(lqr.trim.state)
    trim_control = jnp.array(lqr.trim.control)
    x0_true = trim_state  # start at trim, let waves perturb us
    x0_est = trim_state
    P0 = jnp.eye(5) * 0.1

    wave_params = attrs.evolve(WAVE_SF_BAY_MODERATE, mean_direction=np.pi,
                               seed=int(wave_seed))
    env = Environment.with_waves(wave_params)

    result = simulate_closed_loop(
        system=moth,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        x0_est=x0_est,
        P0=P0,
        dt=DT,
        duration=DURATION,
        rng_key=jax.random.PRNGKey(wave_seed),
        params=MOTH_BIEKER_V3,
        env=env,
        trim_state=trim_state,
        trim_control=trim_control,
        u_trim=trim_control,
    )
    # Cache heel_angle for downstream geometry helpers (used by
    # compute_extended_metrics / plot_single_dashboard)
    result.heel_angle = HEEL_ANGLE
    return result, moth, env


# ---------------------------------------------------------------------------
# Monte Carlo sweep
# ---------------------------------------------------------------------------


def run_monte_carlo(lqr, *, controller_kind: str, n_seeds: int):
    """Run a vmap-batched 60s sweep over n_seeds wave realizations."""
    sensor, estimator, controller = _build_controller_for_kind(lqr, controller_kind)

    moth = Moth3D(
        MOTH_BIEKER_V3,
        u_forward=ConstantSchedule(U_FORWARD),
        heel_angle=HEEL_ANGLE,
    )
    trim_state = jnp.array(lqr.trim.state)
    P0 = jnp.eye(5) * 0.1
    x0_true_batch = jnp.tile(trim_state, (n_seeds, 1))
    x0_est_batch = jnp.tile(trim_state, (n_seeds, 1))

    # One Environment per seed (wave field built deterministically from
    # WaveParams.seed)
    envs = [
        Environment.with_waves(
            attrs.evolve(
                WAVE_SF_BAY_MODERATE, mean_direction=np.pi, seed=int(s)
            )
        )
        for s in range(n_seeds)
    ]
    env_batch = stack_envs(envs)

    # Batched system (same params for every seed — broadcast)
    from fmd.simulator.sweep import stack_systems
    system_batch = stack_systems([moth] * n_seeds)

    result = sweep_closed_loop(
        system=system_batch,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true_batch,
        x0_est=x0_est_batch,
        P0=P0,
        dt=DT,
        duration=DURATION,
        # Measurement-noise RNG seed for the closed-loop pipeline; the
        # wave seeds come from WaveParams.seed (built above). Changing
        # this only affects sensor-noise realisations.
        rng_key=jax.random.PRNGKey(1234),
        env=env_batch,
    )
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _get_main_foil_position(params) -> np.ndarray:
    """Main-foil position relative to CG (accounts for sailor CG offset).

    Mirrors ``fmd.analysis.closed_loop._get_foil_position`` so the
    wave-eta query uses the same eff_main_{x,z} as the Moth3D model.
    """
    total_mass = params.hull_mass + params.sailor_mass
    cg_offset = params.sailor_mass * np.asarray(params.sailor_position) / total_mass
    return np.asarray(params.main_foil_position) - cg_offset


def _wave_eta_main_per_seed(
    true_states: np.ndarray,
    *,
    seeds: list[int],
    foil_pos: np.ndarray,
) -> np.ndarray:
    """Compute per-seed main-foil wave elevation, shape (N, T).

    Mirrors the elevation query inside ``Moth3D.compute_aux``: the main
    foil's NED north position is ``u * t + eff_x * cos(theta) + eff_z *
    sin(theta)`` and the surface elevation at that point + time is the
    wave-aware reference for the breach metric.
    """
    n_seeds, n_steps, _ = true_states.shape
    times = np.arange(n_steps) * DT
    eta_per_seed = np.zeros((n_seeds, n_steps))
    for k in range(n_seeds):
        wave_params = attrs.evolve(
            WAVE_SF_BAY_MODERATE, mean_direction=np.pi, seed=int(seeds[k])
        )
        env = Environment.with_waves(wave_params)
        theta_k = true_states[k, :, 1]
        u_k = true_states[k, :, 4]
        cos_th = np.cos(theta_k)
        sin_th = np.sin(theta_k)
        # x_main_ned = u * t + eff_main_x * cos(theta) + eff_main_z * sin(theta)
        # where eff_main_{x,z} = foil_pos relative to CG (already in body frame).
        x_main_ned = u_k * times + foil_pos[0] * cos_th + foil_pos[2] * sin_th
        # env.wave_field.elevation is JAX-traceable but works fine with vmap;
        # here we just do a loop in t for simplicity (50 × 12000 = 600k calls).
        eta_main = np.asarray(
            jax.vmap(lambda x, t: env.wave_field.elevation(x, 0.0, t))(
                jnp.asarray(x_main_ned), jnp.asarray(times)
            )
        )
        eta_per_seed[k] = eta_main
    return eta_per_seed


def aggregate_mc(result, *, trim_state: np.ndarray, trim_control: np.ndarray,
                 controller_kind: str,
                 u_min: np.ndarray, u_max: np.ndarray,
                 wave_eta_main: np.ndarray | None = None,
                 target_pos_d: float | None = None):
    """Aggregate the four required metric groups across MC seeds.

    Args:
        result: MC sweep result (batched ClosedLoopResult).
        trim_state, trim_control: scalar trim used to compute deviations.
        controller_kind: tag stored in the aggregate.
        u_min, u_max: control bounds for the saturation fraction.
        wave_eta_main: optional (N, T) wave elevation at the main-foil
            position. When supplied, the breach metric is computed
            against the *instantaneous* free surface (wave-aware). When
            None, the still-water reference is used (legacy / wave-blind).
        target_pos_d: Optional per-controller ride-height setpoint.
            When provided, ``ride_height_rms_around_target`` is computed
            as RMS deviation from this setpoint (intra-controller
            stability — the right cross-setpoint comparison). When
            ``None``, falls back to ``trim_state[0]`` so the metric
            equals ``ride_height_rms`` by construction.
    """
    true_states = np.asarray(result.true_states[:, 1:, :])    # (N, T, n)
    controls = np.asarray(result.controls)                    # (N, T, m)
    n_seeds, n_steps, _ = true_states.shape
    ss_idx = int(STEADY_START / DT)
    ss_idx = min(ss_idx, n_steps - 1)

    pos_d = true_states[:, :, 0]
    theta = true_states[:, :, 1]
    u_fwd = true_states[:, :, 4]

    flap_trim = float(trim_control[0])
    # Per-controller setpoint for the intra-target RMS. Mechanical has no
    # explicit setpoint; pid_natural's default is the natural trim; only
    # pid_deeper differs. Falling back to ``trim_state[0]`` makes the new
    # metric equal the existing ``ride_height_rms`` for mechanical /
    # pid_natural by construction, which is the desired semantics.
    target_pos_d_val = (
        float(target_pos_d) if target_pos_d is not None else float(trim_state[0])
    )

    per_seed = {
        "ride_height_rms": [],
        "ride_height_rms_around_target": [],
        "ride_height_std": [],
        "ride_height_mean": [],
        "depth_factor_mean": [],
        "breach_count": [],
        "breach_fraction": [],
        "flap_rms": [],
        "flap_saturation_fraction": [],
        "pitch_rms_error": [],
        "speed_loss_mean": [],
    }
    for k in range(n_seeds):
        pd_k = pos_d[k]
        th_k = theta[k]
        u_k = u_fwd[k]
        ctl_k = controls[k]

        pd_err = pd_k - float(trim_state[0])
        # Deviation from the controller's *own* setpoint (intra-target
        # stability — fair across controllers whose setpoints differ).
        pd_err_target = pd_k - target_pos_d_val
        th_err = th_k - float(trim_state[1])

        # Ride height
        per_seed["ride_height_rms"].append(float(np.sqrt(np.mean(pd_err[ss_idx:] ** 2))))
        per_seed["ride_height_rms_around_target"].append(
            float(np.sqrt(np.mean(pd_err_target[ss_idx:] ** 2)))
        )
        per_seed["ride_height_std"].append(float(np.std(pd_k[ss_idx:])))
        per_seed["ride_height_mean"].append(float(np.mean(pd_k[ss_idx:])))

        # Depth factor + breaches (steady-state window only)
        dfs = _mean_depth_factor(pd_k, th_k, MOTH_BIEKER_V3, HEEL_ANGLE)
        per_seed["depth_factor_mean"].append(float(np.mean(dfs[ss_idx:])))
        eta_k_ss = (
            wave_eta_main[k, ss_idx:] if wave_eta_main is not None else None
        )
        bc_ss, bf_ss, _ = _foil_breach_count(
            pd_k[ss_idx:], th_k[ss_idx:], MOTH_BIEKER_V3, HEEL_ANGLE,
            wave_eta=eta_k_ss,
        )
        per_seed["breach_count"].append(int(bc_ss))
        per_seed["breach_fraction"].append(float(bf_ss))

        # Flap activity: RMS of deviation from trim flap (not absolute flap).
        # Subtracting flap_trim makes the metric name match its semantics —
        # it measures variation about the trim point, not the L2-norm of
        # the absolute command.
        flap = ctl_k[ss_idx:, 0]
        flap_dev = flap - flap_trim
        per_seed["flap_rms"].append(float(np.sqrt(np.mean(flap_dev ** 2))))
        per_seed["flap_saturation_fraction"].append(
            _saturation_fraction(ctl_k[ss_idx:], u_min, u_max)
        )

        # Pitch tracking
        per_seed["pitch_rms_error"].append(
            float(np.sqrt(np.mean(th_err[ss_idx:] ** 2)))
        )

        # Speed loss (relative to trim u)
        per_seed["speed_loss_mean"].append(
            float(float(trim_state[4]) - np.mean(u_k[ss_idx:]))
        )

    # Aggregate
    agg = {}
    for k, v in per_seed.items():
        arr = np.asarray(v)
        agg[k] = {
            "per_seed": v,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    agg["controller"] = controller_kind
    agg["n_seeds"] = int(n_seeds)
    agg["duration_s"] = float(DURATION)
    agg["steady_state_start_s"] = float(STEADY_START)
    agg["target_pos_d_m"] = target_pos_d_val
    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_single_dashboards(out_dir, results_single, systems_single, envs_single,
                           trim_state):
    """Per-controller wave-aware dashboard (pos_d, theta, u, flap, tip depth).

    Computes the auxiliary trajectory (including ``wave_eta_main`` /
    ``wave_eta_rudder``) per controller and threads it through
    ``plot_single_dashboard`` so the wave-elevation panel is populated.
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results_single.items():
        system = systems_single[name]
        env = envs_single[name]
        # Build a SimulationResult slice that matches compute_aux_trajectory's
        # expected per-step layout (times, states, controls all length T).
        sim_result = SimulationResult(
            times=np.asarray(result.times),
            states=np.asarray(result.true_states[1:]),
            controls=np.asarray(result.controls),
        )
        aux = compute_aux_trajectory(system, sim_result, env=env)
        fig = plot_single_dashboard(
            result,
            trim_state=np.asarray(trim_state),
            aux=aux,
            wave_params=WAVE_SF_BAY_MODERATE,
            title=f"{name} (seed={SINGLE_SEED}, SF Bay moderate)",
            t_start=0.0,
            t_end=DURATION,
        )
        path = plots_dir / f"dashboard_{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"  saved {path}")


def make_overlay_plots(out_dir, results_single):
    """Single-seed comparison overlays: ride height, flap command, wand angle.

    Generalised to N controllers via the module-level ``CONTROLLERS``
    list and ``CONTROLLER_COLORS`` / ``CONTROLLER_LABELS`` maps. Adding
    a fourth controller only requires extending those three constants.
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    first_kind = next(iter(results_single))
    times = np.asarray(results_single[first_kind].times)

    def _plot_overlay(extract, ylabel, title, fname, *, invert_y=False):
        fig, ax = plt.subplots(figsize=(11, 4))
        for name, result in results_single.items():
            y = extract(result)
            if y is None:
                continue
            ax.plot(
                times, y,
                label=CONTROLLER_LABELS.get(name, name),
                color=CONTROLLER_COLORS.get(name, None),
                alpha=0.8,
            )
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        if invert_y:
            ax.invert_yaxis()
        p = plots_dir / fname
        fig.savefig(p, dpi=110); plt.close(fig); print(f"  saved {p}")

    _plot_overlay(
        lambda r: np.asarray(r.true_states[1:, 0]),
        "pos_d (m, NED)",
        f"Ride height — seed {SINGLE_SEED}, SF Bay moderate",
        "compare_ride_height.png",
        invert_y=True,
    )
    _plot_overlay(
        lambda r: np.degrees(np.asarray(r.controls[:, 0])),
        "Main flap (deg)",
        f"Flap command — seed {SINGLE_SEED}, SF Bay moderate",
        "compare_flap_command.png",
    )
    _plot_overlay(
        lambda r: (
            np.degrees(np.asarray(r.measurements_clean[:, 0]))
            if r.measurements_clean is not None else None
        ),
        "Wand angle (deg)",
        f"Wand angle — seed {SINGLE_SEED}, SF Bay moderate",
        "compare_wand_angle.png",
    )


_BOX_FACECOLORS = {
    "mechanical": "#cce5ff",
    "pid_natural": "#ffcccc",
    "pid_deeper": "#ccffcc",
}


def _add_box(ax, mc_agg, metric_key, *, kinds=CONTROLLERS):
    """Boxplot one metric across the listed kinds onto ``ax``.

    Generalises the prior two-controller boxplot helper. Returns the
    matplotlib BoxPlotArtist for downstream face-color tweaks.
    """
    data = [mc_agg[k][metric_key]["per_seed"] for k in kinds]
    labels = [CONTROLLER_LABELS.get(k, k) for k in kinds]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for i, k in enumerate(kinds):
        bp["boxes"][i].set_facecolor(_BOX_FACECOLORS.get(k, "#dddddd"))
    # Overlay strip plot
    for i, d in enumerate(data, start=1):
        jitter = np.random.RandomState(0).uniform(-0.07, 0.07, len(d))  # fixed seed for reproducible jitter
        ax.scatter(np.full(len(d), i) + jitter, d, alpha=0.4, s=15,
                   color="black")
    return bp


def make_mc_distribution_plots(out_dir, mc_agg):
    """Side-by-side box+strip plots for the four MC metric groups (N kinds)."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = mc_agg[CONTROLLERS[0]]["n_seeds"]

    def _box(metric_key, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(8, 5))
        _add_box(ax, mc_agg, metric_key)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3)
        p = plots_dir / fname
        fig.savefig(p, dpi=110); plt.close(fig); print(f"  saved {p}")

    _box("ride_height_rms", "Ride height RMS vs natural trim (m)",
         f"Ride height RMS vs natural trim over {n_seeds} seeds",
         "mc_ride_height_rms.png")
    _box("ride_height_rms_around_target",
         "Ride height RMS vs own setpoint (m)",
         f"Ride height RMS vs own setpoint over {n_seeds} seeds "
         "(intra-controller stability)",
         "mc_ride_height_rms_around_target.png")
    _box("breach_count", "Breach count (per 60s)",
         f"Foil tip breach count over {n_seeds} seeds",
         "mc_breach_distribution.png")
    _box("flap_rms", "Main flap RMS (rad)",
         f"Main flap RMS over {n_seeds} seeds", "mc_flap_activity.png")

    # Two-panel for pitch & speed
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    _add_box(axs[0], mc_agg, "pitch_rms_error")
    axs[0].set_ylabel("Pitch RMS error (rad)")
    axs[0].set_title("Pitch tracking"); axs[0].grid(True, alpha=0.3)
    _add_box(axs[1], mc_agg, "speed_loss_mean")
    axs[1].set_ylabel("Speed loss (m/s)")
    axs[1].set_title("Forward speed drop from trim")
    axs[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "mc_pitch_speed.png"
    fig.savefig(p, dpi=110); plt.close(fig); print(f"  saved {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help=f"Use {N_SEEDS_QUICK} seeds instead of {N_SEEDS_FULL}.")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = N_SEEDS_QUICK if args.quick else N_SEEDS_FULL
    print(f"[wand_vs_pid_waves] mode={'quick' if args.quick else 'full'} "
          f"n_seeds={n_seeds} duration={DURATION}s")

    # 1. Trim + LQR design (LQR only used for trim state/control + bounds)
    t0 = time.time()
    lqr = design_moth_lqr(
        params=MOTH_BIEKER_V3, u_forward=U_FORWARD,
        dt=DT, heel_angle=HEEL_ANGLE,
    )
    trim_state = np.asarray(lqr.trim.state)
    trim_control = np.asarray(lqr.trim.control)
    print(f"  trim: pos_d={trim_state[0]:.4f}m theta={np.degrees(trim_state[1]):.3f}deg "
          f"u={trim_state[4]:.3f}m/s "
          f"flap_trim={np.degrees(trim_control[0]):.3f}deg "
          f"elev_trim={np.degrees(trim_control[1]):.3f}deg "
          f"(elapsed {time.time()-t0:.1f}s)")

    # 2. Single-seed time series
    print("\n[1/4] Single-seed time series ...")
    t0 = time.time()
    results_single: dict[str, Any] = {}
    systems_single: dict[str, Any] = {}
    envs_single: dict[str, Any] = {}
    for kind in CONTROLLERS:
        result, system, env = run_single_seed(
            lqr, controller_kind=kind, wave_seed=SINGLE_SEED,
        )
        results_single[kind] = result
        systems_single[kind] = system
        envs_single[kind] = env
    print(f"  done in {time.time()-t0:.1f}s")

    # 3. Single-seed metrics
    print("\n[2/4] Single-seed metrics + dashboards ...")
    single_metrics: dict[str, Any] = {}
    for name, result in results_single.items():
        m = compute_extended_metrics(
            result, trim_state, aux={}, dt=DT, steady_state_start=STEADY_START,
        )
        single_metrics[name] = m

    # 4. Monte Carlo sweep
    print(f"\n[3/4] Monte Carlo sweep, n_seeds={n_seeds} ...")
    t0 = time.time()
    mc_results = {}
    for kind in CONTROLLERS:
        print(f"  running {kind} ...")
        t_k = time.time()
        mc_results[kind] = run_monte_carlo(
            lqr, controller_kind=kind, n_seeds=n_seeds,
        )
        print(f"    {kind} done in {time.time()-t_k:.1f}s")
    print(f"  total MC elapsed {time.time()-t0:.1f}s")

    # 5. Aggregate MC stats
    print("\n[4/4] Aggregating MC, generating plots, writing files ...")
    moth = Moth3D(
        MOTH_BIEKER_V3,
        u_forward=ConstantSchedule(U_FORWARD),
        heel_angle=HEEL_ANGLE,
    )
    u_min = np.asarray(moth.control_lower_bounds)
    u_max = np.asarray(moth.control_upper_bounds)
    # Pre-compute wave eta at the main-foil position per seed for the
    # wave-aware breach metric. Same wave seeds are used for both
    # controllers (paired comparison), so we compute eta only once.
    foil_pos = _get_main_foil_position(MOTH_BIEKER_V3)
    seeds = list(range(n_seeds))
    # Both controllers see identical true states *only* if the dynamics
    # respond identically, which they don't — but the wave field is
    # solely determined by (seed, foil x_ned) and x_ned depends on the
    # state (u, theta). For consistency we compute eta per controller.
    mc_agg = {}
    for kind in CONTROLLERS:
        true_states_k = np.asarray(mc_results[kind].true_states[:, 1:, :])
        eta_main = _wave_eta_main_per_seed(
            true_states_k, seeds=seeds, foil_pos=foil_pos
        )
        mc_agg[kind] = aggregate_mc(
            mc_results[kind],
            trim_state=trim_state,
            trim_control=trim_control,
            controller_kind=kind,
            u_min=u_min,
            u_max=u_max,
            wave_eta_main=eta_main,
            target_pos_d=_target_pos_d_for_kind(kind, trim_state=trim_state),
        )

    # 6. Plots
    make_single_dashboards(
        out_dir, results_single, systems_single, envs_single, trim_state,
    )
    make_overlay_plots(out_dir, results_single)
    make_mc_distribution_plots(out_dir, mc_agg)

    # 7. metrics.json
    payload = {
        "setup": {
            "u_forward_ms": U_FORWARD,
            "heel_angle_deg": float(np.degrees(HEEL_ANGLE)),
            "dt_s": DT,
            "duration_s": DURATION,
            "n_seeds": int(n_seeds),
            "steady_state_start_s": STEADY_START,
            "wave_preset": "WAVE_SF_BAY_MODERATE (Hs=0.5m, Tp=3.0s, JONSWAP)",
            "wave_direction_rad": float(np.pi),
            "moth_preset": "MOTH_BIEKER_V3",
            "trim_state": {
                "pos_d_m": float(trim_state[0]),
                "theta_deg": float(np.degrees(trim_state[1])),
                "w_ms": float(trim_state[2]),
                "q_degs": float(np.degrees(trim_state[3])),
                "u_ms": float(trim_state[4]),
            },
            "trim_control": {
                "flap_deg": float(np.degrees(trim_control[0])),
                "elevator_deg": float(np.degrees(trim_control[1])),
            },
            "controls_bounds_deg": {
                "flap_min_deg": float(np.degrees(u_min[0])),
                "flap_max_deg": float(np.degrees(u_max[0])),
            },
        },
        "single_seed": {
            "wave_seed": SINGLE_SEED,
            "metrics": _to_jsonable(single_metrics),
        },
        "monte_carlo": _to_jsonable(mc_agg),
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  saved {metrics_path}")

    # 8. report_guidelines.txt
    guidelines_path = out_dir / "report_guidelines.txt"
    with open(guidelines_path, "w") as f:
        f.write(REPORT_GUIDELINES)
    print(f"  saved {guidelines_path}")

    # 9. Print headline
    print("\n--- headline numbers (steady-state, mean over seeds) ---")
    for kind in CONTROLLERS:
        a = mc_agg[kind]
        print(
            f"  {kind:12s} "
            f"rms_vs_trim={a['ride_height_rms']['mean']:.4f}m "
            f"rms_vs_target={a['ride_height_rms_around_target']['mean']:.4f}m "
            f"breach_count={a['breach_count']['mean']:.2f} "
            f"flap_RMS={a['flap_rms']['mean']:.4f}rad "
            f"flap_sat={a['flap_saturation_fraction']['mean']*100:.1f}% "
            f"pitch_RMS={a['pitch_rms_error']['mean']:.4f}rad "
            f"speed_loss={a['speed_loss_mean']['mean']:.3f}m/s "
            f"depth_factor_mean={a['depth_factor_mean']['mean']:.3f}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
