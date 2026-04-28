#!/usr/bin/env python3
"""Moth LQG calm water analysis: baseline perturbation recovery + surface breach.

Runs two calm-water LQG scenarios and produces:
1. Dashboard PNGs for each scenario (foiling + estimation dashboards)
2. metrics.json with settling time, overshoot, breach fraction, depth factor stats
3. report_guidelines.txt for the /generate-plot-interpretation-report skill

Usage:
    env JAX_PLATFORMS=cpu uv run python examples/moth_lqg_calm_water.py
    env JAX_PLATFORMS=cpu uv run python examples/moth_lqg_calm_water.py --output-dir docs/reports/lqg_calm_water
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmd.simulator import _config  # noqa: F401


def resolve_output_dir(base_path: str) -> Path:
    """Add date suffix to output path, incrementing version if same-day dir exists.

    docs/reports/foo  ->  docs/reports/foo_2026-03-19
    (if that exists)  ->  docs/reports/foo_2026-03-19_v2
    """
    today = datetime.now().strftime("%Y-%m-%d")
    candidate = Path(f"{base_path}_{today}")
    if not candidate.exists():
        return candidate
    version = 2
    while Path(f"{base_path}_{today}_v{version}").exists():
        version += 1
    return Path(f"{base_path}_{today}_v{version}")


REPORT_GUIDELINES = """
## Physics Guidance for Calm Water LQG Report

### NED Frame Convention
- pos_d is positive DOWN in NED. pos_d becoming MORE NEGATIVE means the boat is RISING.
- A negative pos_d_dot means the boat is moving upward (gaining altitude).
- pos_d at trim is typically around -1.39m (boat flying ~1.39m above the surface reference).
  The exact value depends on model version, trim configuration, and target_pos_d -- check metrics.json.

### LQR R-Weight Rationale
The controller uses lqr_R_diag=(10, 100), meaning:
- Main flap actuation is 10x the baseline cost (R=1): moderately penalizes flap movement
  to reduce chatter while still allowing responsive depth control.
- Rudder elevator actuation is 100x the baseline: strongly penalizes rudder movement
  since pitch control is less critical than depth control and rudder oscillation
  creates drag without proportional benefit.
- This asymmetry reflects the physical reality: main flap directly controls lift (and
  hence depth), while the rudder elevator primarily controls pitch trim. Aggressive
  rudder actuation creates drag with diminishing returns on stability.

### Baseline Scenario (Perturbation Recovery)
- Tests the controller's ability to reject an initial state perturbation.
- Perturbation: +0.05m pos_d (pushed 5cm deeper), -2 deg pitch, zero velocities.
- target_pos_d is set 30cm below the surface breach point (in NED: 30cm more positive
  than the tip-at-surface depth), providing comfortable margin from ventilation.
- No process noise: deterministic dynamics with noisy measurements only.
- Key metrics: settling time (time to reach and stay within 2% of steady state),
  overshoot (maximum excursion beyond target during recovery), final error.

### Surface Breach Scenario (Foil Ventilation)
- Tests controller behavior when the foil tip is at the water surface.
- target_pos_d = compute_tip_at_surface_pos_d(): the depth where the leeward foil
  tip is exactly at the waterline. At this depth, the depth_factor is ~0.5 meaning
  about half the foil span is above the surface.
- Process noise W_true_diag=(1e-7, 1e-8, 1e-6, 1e-6, 1e-8) creates realistic
  stochastic disturbances that push the foil in and out of the water.
- Key metrics: breach fraction (% of time foil tip is above surface), depth_factor
  statistics (mean, std, min — indicates effective foil immersion), control effort.

### Depth Factor Interpretation
- depth_factor = 1.0: foil is fully submerged, full lift production
- depth_factor ~ 0.5: about half the foil span is above the surface
- depth_factor ~ 0.0: foil is fully ventilated (above water), no lift
- depth_factor dropping below ~0.3 is dangerous: lift drops faster than the boat
  can recover, leading to potential "foil stall" where the boat sinks uncontrollably.
- In the surface breach scenario, expect depth_factor to oscillate around 0.5.
  Oscillation amplitude indicates controller's ability to manage ventilation risk.

### Ventilation Physics
When the foil tip breaches the surface:
1. Lift is reduced proportionally to the exposed span (depth_factor < 1)
2. The boat begins to sink (pos_d increases in NED) since lift < weight
3. As the boat sinks, more foil re-enters the water (depth_factor increases)
4. Lift recovers and the boat rises again
5. This creates a self-excited oscillation modulated by the controller

The controller must balance:
- Too aggressive: overshoots past surface, causing ventilation
- Too conservative: slow recovery, large depth excursions
- The R-weight asymmetry (10, 100) helps by allowing the main flap to respond
  quickly while keeping rudder movement smooth.

### EKF Convergence
- Initial covariance P0 = 0.1 * I means the filter starts with moderate uncertainty.
- Covariance trace should decrease rapidly in the first 1-2 seconds as the filter
  converges. A trace that doesn't converge indicates observability issues.
- The estimation error (true - estimated) should be small relative to the state
  magnitudes after convergence.

### Report Structure
1. **Summary**: 3-5 bullet-point key findings at the top.
2. **Setup**: trim state for each scenario (pos_d, theta, flap, elevator at 10 m/s),
   controller design (R-weights with rationale), perturbation specification,
   target_pos_d values (and how they were computed).
3. **Baseline Scenario Analysis**: perturbation recovery dynamics, settling time,
   overshoot, steady-state error, EKF convergence.
4. **Surface Breach Scenario Analysis**: ventilation behavior, depth_factor
   statistics, breach fraction, control effort, state oscillations.
5. **Cross-Scenario Comparison**: how controller performance differs between
   comfortable operating depth and surface-breach conditions.
6. **Conclusions**: controller robustness assessment, R-weight effectiveness,
   recommendations for tuning or operating envelope.
"""


def _build_baseline_config():
    """Build the baseline (perturbation recovery) scenario config."""
    from fmd.simulator.moth_scenarios import ScenarioConfig
    from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d

    surface_pos_d = compute_tip_at_surface_pos_d()
    target_pos_d = surface_pos_d + 0.30  # 30cm deeper than surface breach in NED

    return ScenarioConfig(
        name="baseline",
        lqr_R_diag=(10, 100),
        target_pos_d=target_pos_d,
        target_theta=0.0,
        perturbation=(0.05, np.radians(-2.0), 0.0, 0.0, 0.0),
        duration=10.0,
    )


def _build_breach_config():
    """Build the surface breach scenario config."""
    from fmd.simulator.moth_scenarios import ScenarioConfig
    from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d

    return ScenarioConfig(
        name="surface_breach",
        lqr_R_diag=(10, 100),
        target_pos_d=compute_tip_at_surface_pos_d(),
        target_theta=0.0,
        W_true_diag=(1e-7, 1e-8, 1e-6, 1e-6, 1e-8),
        duration=30.0,
    )


def run_scenario(config):
    """Run a single LQG scenario, return (config, result)."""
    from fmd.simulator.moth_scenarios import run_scenario as _run_scenario

    print(f"\n--- Running scenario: {config.name} ---")
    print(f"  duration: {config.duration}s, dt: {config.dt}")
    print(f"  target_pos_d: {config.target_pos_d}")
    print(f"  target_theta: {config.target_theta}")
    print(f"  lqr_R_diag: {config.lqr_R_diag}")
    if config.perturbation is not None:
        print(f"  perturbation: {config.perturbation}")
    if config.W_true_diag is not None:
        print(f"  W_true_diag: {config.W_true_diag}")

    result = _run_scenario(config)
    print(f"  {len(result.times)} timesteps completed")
    return result


def generate_dashboards(output_dir, scenario_results):
    """Generate foiling + estimation dashboard PNGs for each scenario.

    Args:
        output_dir: Root output directory.
        scenario_results: dict mapping scenario name to (config, result).

    Returns:
        List of saved file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    from fmd.analysis.plots import plot_lqg_result, savefig_and_close

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, (config, result) in scenario_results.items():
        print(f"\n--- Generating dashboards: {name} ---")
        figs = plot_lqg_result(result)
        for fig_name, fig in figs.items():
            path = plots_dir / f"{name}_{fig_name}.png"
            savefig_and_close(fig, path)
            print(f"  Saved: {path}")
            paths.append(path)

    return paths


def _compute_settling_time(times, values, target, tolerance_frac=0.02):
    """Compute settling time: first time the signal enters and stays within tolerance band.

    Args:
        times: Time array.
        values: Signal values.
        target: Target value.
        tolerance_frac: Fraction of initial error for tolerance band.

    Returns:
        Settling time in seconds, or None if never settles.
    """
    initial_error = abs(values[0] - target)
    if initial_error < 1e-10:
        return 0.0
    band = tolerance_frac * initial_error

    # Walk backward from end to find last time outside band
    within = np.abs(values - target) <= band
    if not np.any(within):
        return None

    # Find the last index that is outside the band
    outside_indices = np.where(~within)[0]
    if len(outside_indices) == 0:
        return float(times[0])

    last_outside = outside_indices[-1]
    if last_outside >= len(times) - 1:
        return None  # never settled

    return float(times[last_outside + 1])


def _compute_overshoot(values, target, initial_value):
    """Compute maximum overshoot past target (as fraction of initial error).

    Returns overshoot fraction (0.0 = no overshoot, 0.1 = 10% overshoot).
    """
    initial_error = initial_value - target
    if abs(initial_error) < 1e-10:
        return 0.0

    # Overshoot is when the value goes past target in the opposite direction
    if initial_error > 0:
        # Started above target, overshoot is going below
        overshoot = target - np.min(values)
    else:
        # Started below target, overshoot is going above
        overshoot = np.max(values) - target

    return max(0.0, float(overshoot / abs(initial_error)))


def extract_baseline_metrics(config, result):
    """Extract metrics for the baseline perturbation recovery scenario."""
    from fmd.simulator.moth_3d import POS_D, THETA, W, Q, U

    times = np.asarray(result.times)
    states = np.asarray(result.true_states[:-1])
    est_states = np.asarray(result.est_states[:-1])
    controls = np.asarray(result.controls)
    trim_state = np.asarray(result.trim_state)
    cov_traces = np.asarray(result.covariance_traces)

    metrics = {}

    # Settling time and overshoot for pos_d and theta
    for name, idx in [("pos_d", POS_D), ("theta", THETA)]:
        vals = states[:, idx]
        target = float(trim_state[idx])
        settling = _compute_settling_time(times, vals, target)
        overshoot = _compute_overshoot(vals, target, float(vals[0]))
        metrics[f"{name}_settling_time_s"] = settling
        metrics[f"{name}_overshoot_frac"] = overshoot

    # Final state error from trim
    final_state = np.asarray(result.true_states[-1])
    final_error = final_state - trim_state
    metrics["final_error"] = {
        "pos_d_m": float(final_error[POS_D]),
        "theta_deg": float(np.degrees(final_error[THETA])),
        "w_ms": float(final_error[W]),
        "q_degs": float(np.degrees(final_error[Q])),
        "u_ms": float(final_error[U]),
    }

    # EKF convergence
    metrics["ekf_convergence"] = {
        "initial_cov_trace": float(cov_traces[0]),
        "final_cov_trace": float(cov_traces[-1]),
        "convergence_ratio": float(cov_traces[-1] / cov_traces[0]) if cov_traces[0] > 0 else 0.0,
    }

    # Final estimation error
    final_est_error = np.asarray(result.estimation_errors[-1])
    metrics["final_estimation_error"] = {
        "pos_d_m": float(final_est_error[POS_D]),
        "theta_deg": float(np.degrees(final_est_error[THETA])),
        "w_ms": float(final_est_error[W]),
        "q_degs": float(np.degrees(final_est_error[Q])),
        "u_ms": float(final_est_error[U]),
    }

    # Control effort
    for ctrl_name, idx in [("main_flap", 0), ("rudder_elevator", 1)]:
        vals_deg = np.degrees(controls[:, idx])
        metrics[f"{ctrl_name}_effort"] = {
            "mean_deg": float(np.mean(vals_deg)),
            "std_deg": float(np.std(vals_deg)),
            "max_abs_deg": float(np.max(np.abs(vals_deg))),
        }

    # Trim state
    metrics["trim_state"] = {
        "pos_d_m": float(trim_state[POS_D]),
        "theta_deg": float(np.degrees(trim_state[THETA])),
        "w_ms": float(trim_state[W]),
        "q_degs": float(np.degrees(trim_state[Q])),
        "u_ms": float(trim_state[U]),
    }

    # Trim control
    trim_control = np.asarray(result.trim_control)
    metrics["trim_control"] = {
        "main_flap_deg": float(np.degrees(trim_control[0])),
        "rudder_elevator_deg": float(np.degrees(trim_control[1])),
    }

    return metrics


def extract_breach_metrics(config, result):
    """Extract metrics for the surface breach scenario."""
    from fmd.simulator.moth_3d import POS_D, THETA, W, Q, U
    from fmd.simulator.components.moth_forces import compute_leeward_tip_depth

    times = np.asarray(result.times)
    states = np.asarray(result.true_states[:-1])
    controls = np.asarray(result.controls)
    trim_state = np.asarray(result.trim_state)
    cov_traces = np.asarray(result.covariance_traces)

    metrics = {}

    # Shared geometry computations
    params = config.params
    total_mass = params.hull_mass + params.sailor_mass
    cg_offset = params.sailor_mass * params.sailor_position / total_mass
    main_foil_pos = params.main_foil_position - cg_offset
    rudder_pos = params.rudder_position - cg_offset

    # Breach fraction: compute foil tip depth trajectory
    tip_depths = compute_leeward_tip_depth(
        pos_d=result.true_states[:, 0],
        eff_pos_x=main_foil_pos[0],
        eff_pos_z=main_foil_pos[2],
        theta=result.true_states[:, 1],
        heel_angle=config.heel_angle,
        foil_span=params.main_foil_span,
    )
    tip_depths = np.asarray(tip_depths)
    breach_frac = float(np.mean(tip_depths < 0.0) * 100.0)
    metrics["breach_fraction_pct"] = breach_frac

    # Depth factor stats: compute from state trajectory using the depth factor model
    from fmd.simulator.components.moth_forces import (
        compute_foil_ned_depth,
        compute_depth_factor,
    )
    import jax.numpy as jnp

    pos_d = np.asarray(result.true_states[:-1, 0])
    theta_arr = np.asarray(result.true_states[:-1, 1])

    main_depth = np.asarray(compute_foil_ned_depth(
        pos_d, main_foil_pos[0], main_foil_pos[2], theta_arr, config.heel_angle,
    ))
    rudder_depth = np.asarray(compute_foil_ned_depth(
        pos_d, rudder_pos[0], rudder_pos[2], theta_arr, config.heel_angle,
    ))

    main_df = np.array([
        float(compute_depth_factor(
            jnp.array(main_depth[i]), params.main_foil_span, config.heel_angle,
        ))
        for i in range(len(main_depth))
    ])
    rudder_df = np.array([
        float(compute_depth_factor(
            jnp.array(rudder_depth[i]), params.rudder_span, config.heel_angle,
        ))
        for i in range(len(rudder_depth))
    ])

    metrics["depth_factor"] = {
        "main_mean": float(np.mean(main_df)),
        "main_std": float(np.std(main_df)),
        "main_min": float(np.min(main_df)),
        "main_max": float(np.max(main_df)),
        "rudder_mean": float(np.mean(rudder_df)),
        "rudder_std": float(np.std(rudder_df)),
        "rudder_min": float(np.min(rudder_df)),
    }

    # State statistics (full duration)
    state_stats = {}
    for name, idx in [("pos_d", POS_D), ("theta", THETA), ("w", W), ("q", Q), ("u", U)]:
        vals = states[:, idx]
        state_stats[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    metrics["state_statistics"] = state_stats

    # Control effort
    for ctrl_name, idx in [("main_flap", 0), ("rudder_elevator", 1)]:
        vals_deg = np.degrees(controls[:, idx])
        metrics[f"{ctrl_name}_effort"] = {
            "mean_deg": float(np.mean(vals_deg)),
            "std_deg": float(np.std(vals_deg)),
            "max_abs_deg": float(np.max(np.abs(vals_deg))),
            "range_deg": float(np.ptp(vals_deg)),
        }

    # State oscillation metrics (std of detrended signal)
    for name, idx in [("pos_d", POS_D), ("theta", THETA)]:
        vals = states[:, idx]
        # Detrend by removing running mean (window = 1s)
        dt = float(times[1] - times[0])
        window = max(1, int(1.0 / dt))
        if len(vals) > window:
            # Simple moving average detrend
            kernel = np.ones(window) / window
            smoothed = np.convolve(vals, kernel, mode="same")
            oscillation_std = float(np.std(vals - smoothed))
        else:
            oscillation_std = float(np.std(vals))
        metrics[f"{name}_oscillation_std"] = oscillation_std

    # EKF convergence
    metrics["ekf_convergence"] = {
        "initial_cov_trace": float(cov_traces[0]),
        "final_cov_trace": float(cov_traces[-1]),
        "convergence_ratio": float(cov_traces[-1] / cov_traces[0]) if cov_traces[0] > 0 else 0.0,
    }

    # Trim state
    metrics["trim_state"] = {
        "pos_d_m": float(trim_state[POS_D]),
        "theta_deg": float(np.degrees(trim_state[THETA])),
        "w_ms": float(trim_state[W]),
        "q_degs": float(np.degrees(trim_state[Q])),
        "u_ms": float(trim_state[U]),
    }

    # Trim control
    trim_control = np.asarray(result.trim_control)
    metrics["trim_control"] = {
        "main_flap_deg": float(np.degrees(trim_control[0])),
        "rudder_elevator_deg": float(np.degrees(trim_control[1])),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Moth LQG calm water analysis")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots and metrics",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline scenario")
    parser.add_argument("--skip-breach", action="store_true", help="Skip surface breach scenario")
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = resolve_output_dir("docs/reports/lqg_calm_water")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build scenario configs
    scenario_results = {}

    if not args.skip_baseline:
        print("=" * 60)
        print("BASELINE SCENARIO (perturbation recovery)")
        print("=" * 60)
        baseline_config = _build_baseline_config()
        baseline_result = run_scenario(baseline_config)
        scenario_results["baseline"] = (baseline_config, baseline_result)

    if not args.skip_breach:
        print("\n" + "=" * 60)
        print("SURFACE BREACH SCENARIO (foil ventilation)")
        print("=" * 60)
        breach_config = _build_breach_config()
        breach_result = run_scenario(breach_config)
        scenario_results["surface_breach"] = (breach_config, breach_result)

    # Generate dashboards
    if scenario_results:
        print("\n" + "=" * 60)
        print("GENERATING DASHBOARDS")
        print("=" * 60)
        dashboard_paths = generate_dashboards(output_dir, scenario_results)
        print(f"\nGenerated {len(dashboard_paths)} dashboards")

    # Extract metrics
    all_metrics = {}
    if "baseline" in scenario_results:
        print("\n" + "=" * 60)
        print("EXTRACTING BASELINE METRICS")
        print("=" * 60)
        config, result = scenario_results["baseline"]
        all_metrics["baseline"] = extract_baseline_metrics(config, result)

    if "surface_breach" in scenario_results:
        print("\n" + "=" * 60)
        print("EXTRACTING SURFACE BREACH METRICS")
        print("=" * 60)
        config, result = scenario_results["surface_breach"]
        all_metrics["surface_breach"] = extract_breach_metrics(config, result)

    # Add scenario configurations to metrics
    for name, (config, result) in scenario_results.items():
        if name in all_metrics:
            all_metrics[name]["scenario_config"] = {
                "name": config.name,
                "duration_s": float(config.duration),
                "dt_s": float(config.dt),
                "target_pos_d": float(config.target_pos_d),
                "target_theta": float(config.target_theta),
                "lqr_R_diag": list(config.lqr_R_diag),
                "perturbation": list(config.perturbation) if config.perturbation else None,
                "W_true_diag": list(config.W_true_diag) if config.W_true_diag else None,
                "heel_angle_deg": float(np.degrees(config.heel_angle)),
            }
            all_metrics[name]["scenario_config"]["params_preset"] = "MOTH_BIEKER_V3"
            # Add LQR gains matrix K[i,j]
            if hasattr(result, 'lqr_K') and result.lqr_K is not None:
                all_metrics[name]["scenario_config"]["lqr_K"] = result.lqr_K.tolist()

    # Save metrics.json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # Save report_guidelines.txt
    guidelines_path = output_dir / "report_guidelines.txt"
    with open(guidelines_path, "w") as f:
        f.write(REPORT_GUIDELINES)
    print(f"Report guidelines saved to {guidelines_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nTo generate the narrative report:")
    print(f"  /generate-plot-interpretation-report {output_dir}")


if __name__ == "__main__":
    main()
