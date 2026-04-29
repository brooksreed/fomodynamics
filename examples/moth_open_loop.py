#!/usr/bin/env python3
"""Moth 3DOF open-loop analysis: equilibrium + impulse response characterization.

Runs both surge-enabled and surge-disabled modes by default, producing:
1. Per-mode plots in <output-dir>/<mode>/plots/
2. Consolidated metrics.json covering both modes
3. report_guidelines.txt for the /generate-plot-interpretation-report skill

Usage:
    env JAX_PLATFORMS=cpu uv run python examples/moth_open_loop.py
    env JAX_PLATFORMS=cpu uv run python examples/moth_open_loop.py --output-dir docs/reports/moth_open_loop
    env JAX_PLATFORMS=cpu uv run python examples/moth_open_loop.py --surge-only
    env JAX_PLATFORMS=cpu uv run python examples/moth_open_loop.py --no-surge-only
    env JAX_PLATFORMS=cpu uv run python examples/moth_open_loop.py --cases impulse
"""

from __future__ import annotations

# Report guidelines for /generate-plot-interpretation-report skill.
REPORT_GUIDELINES = """
When writing interpretation reports for these open-loop simulation results:

TERMINOLOGY:
- Use hull height above water (bow/stern, positive = above water) for vertical
  position discussion. The plots show this directly. Never use raw pos_d values
  in the narrative — pos_d is NED (positive-down) and causes sign confusion.
- At trim (10 m/s): bow height ≈ 0.48m, stern height ≈ 0.46m above water.

NARRATIVE FOCUS:
- Emphasize the foiling-regime dynamics in the first 1-2 seconds after each
  perturbation. This is where the model physics are most accurate: pitch-heave
  coupling, non-minimum-phase behavior (elevator cases), force balance transients.
- Ventilation (foils emerging) and hull contact (hull entering water) are boundary
  effects modeled with rough approximations. Note them briefly when they activate
  (e.g., "ventilation arrests the rise at bow height ~0.55m") but do not analyze
  those regimes in detail.

SURGE PHYSICS (when surge is enabled):
- Speed changes are dominated by gravity body-frame projection:
  gravity_fx = -m*g*sin(theta). Nose-up pitch retards forward motion; nose-down
  pitch accelerates it. This is potential energy conversion and acts immediately.
- Strut immersion drag is the secondary effect: more strut in water = more drag.
  Rising (hull height increases) reduces strut immersion and drag; sinking increases
  them. This builds over 1-3s as hull height changes accumulate.
- NEVER attribute speed changes to "frontal area" or "reduced drag at deeper
  submergence" — these are incorrect.

STRUCTURE:
- Each impulse case: describe the initial foiling dynamics (0-1s), then briefly
  note boundary effects if they occur. Compare to other cases where informative.
- For surge-enabled: include a comparison column to surge-disabled in key metrics.

REPORT ORGANIZATION:
- The report covers two modes: surge_enabled (5-state model with speed dynamics)
  and surge_disabled (4-state model with fixed forward speed).
- Each mode has its own plots/ subdirectory and per-case artifacts.
- The top-level metrics.json consolidates both modes for cross-comparison.
- Lead with a summary comparing the two modes, then detail each.
"""

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


from fmd.simulator.validation import SimCase
from fmd.simulator.moth_validation import run_case, case_trim_equilibrium
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.integrator import result_with_meta
from fmd.simulator.control import PiecewiseConstantControl
import jax.numpy as jnp


def plot_case(result, case_name, output_dir, trim_state=None, rich_result=None,
              params=None):
    """Plot simulation case with hull height, states, and aux panels.

    Layout (4 rows x 2 cols when params and aux available):
      Row 1: Hull height (bow/stern/surface) | Foil tip depth (main/rudder)
      Row 2: theta | u
      Row 3: w | q
      Row 4: Lift force | Depth factor

    Args:
        result: SimulationResult with times and states.
        case_name: Name used for plot title and filename.
        output_dir: Directory to save plot to.
        trim_state: Optional trim state for reference line overlay.
        rich_result: Optional RichSimulationResult with aux outputs.
        params: Optional MothParams for hull height and foil tip geometry.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    states = np.array(result.states)
    times = np.array(result.times)

    has_df = rich_result is not None and "main_df" in rich_result.outputs
    has_geom = params is not None

    # Fixed layout: geom row + theta/u + w/q + (optional) aux
    nrows = 1 + 2 + (1 if has_df else 0)  # geom + 2 state rows + aux
    if not has_geom:
        nrows = 2 + (1 if has_df else 0)  # fallback: just states + aux

    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))

    if has_geom:
        from fmd.simulator.components.moth_forces import (
            compute_foil_ned_depth, compute_leeward_tip_depth,
        )

        pos_d = states[:, 0]
        theta = states[:, 1]
        heel_angle = np.deg2rad(30.0)

        # CG offset (same pattern as _convenience.py)
        total_mass = params.hull_mass + params.sailor_mass
        cg_offset = params.sailor_mass * np.array(params.sailor_position) / total_mass
        foil_pos = np.array(params.main_foil_position) - cg_offset
        rudder_pos = np.array(params.rudder_position) - cg_offset

        # Hull bow/stern buoyancy points (hull_length/4 fore and aft of hull CG)
        buoyancy_fwd_x = params.hull_length / 4 - cg_offset[0]
        buoyancy_aft_x = -params.hull_length / 4 - cg_offset[0]
        hull_z = params.hull_cg_above_bottom - cg_offset[2]

        # Row 1 left: Hull height above water (positive = above)
        ax_hull = axes[0, 0]
        bow_height = -np.asarray(compute_foil_ned_depth(
            pos_d, buoyancy_fwd_x, hull_z, theta, 0.0))
        stern_height = -np.asarray(compute_foil_ned_depth(
            pos_d, buoyancy_aft_x, hull_z, theta, 0.0))
        ax_hull.plot(times, bow_height, 'b-', linewidth=1, label='bow')
        ax_hull.plot(times, stern_height, 'r-', linewidth=1, label='stern')
        ax_hull.axhline(0, color='gray', linewidth=0.5, linestyle='--',
                        label='surface')
        if trim_state is not None:
            trim_bow = -float(compute_foil_ned_depth(
                trim_state[0], buoyancy_fwd_x, hull_z, trim_state[1], 0.0))
            ax_hull.axhline(trim_bow, color='r', linestyle='--', alpha=0.3,
                            label='trim bow')
        ax_hull.set_ylabel('Hull Height (m)')
        ax_hull.set_xlabel('Time (s)')
        ax_hull.grid(True, alpha=0.3)
        ax_hull.legend()

        # Row 1 right: Foil tip depth (positive = submerged)
        ax_tip = axes[0, 1]
        main_tip = np.asarray(compute_leeward_tip_depth(
            pos_d, foil_pos[0], foil_pos[2], theta, heel_angle,
            params.main_foil_span))
        rudder_tip = np.asarray(compute_leeward_tip_depth(
            pos_d, rudder_pos[0], rudder_pos[2], theta, heel_angle,
            params.rudder_span))
        ax_tip.plot(times, main_tip, 'g-', linewidth=1, label='main tip')
        ax_tip.plot(times, rudder_tip, 'm-', linewidth=1, label='rudder tip')
        ax_tip.axhline(0, color='gray', linewidth=0.5, linestyle='--',
                       label='surface')
        ax_tip.set_ylabel('Foil Tip Depth (m)')
        ax_tip.set_xlabel('Time (s)')
        ax_tip.invert_yaxis()
        ax_tip.grid(True, alpha=0.3)
        ax_tip.legend()

        state_row_offset = 1  # states start at row 1
    else:
        state_row_offset = 0  # no geom row, states start at row 0

    # Row 2: theta | u
    ax_theta = axes[state_row_offset, 0]
    ax_theta.plot(times, states[:, 1], 'b-', linewidth=1)
    if trim_state is not None:
        ax_theta.axhline(y=trim_state[1], color='r', linestyle='--',
                         alpha=0.5, label='trim')
    ax_theta.set_ylabel('theta (rad)')
    ax_theta.set_xlabel('Time (s)')
    ax_theta.grid(True, alpha=0.3)
    ax_theta.legend()

    ax_u = axes[state_row_offset, 1]
    if states.shape[1] >= 5:
        ax_u.plot(times, states[:, 4], 'b-', linewidth=1)
    if trim_state is not None and len(trim_state) >= 5:
        ax_u.axhline(y=trim_state[4], color='r', linestyle='--',
                     alpha=0.5, label='trim')
    ax_u.set_ylabel('u (m/s)')
    ax_u.set_xlabel('Time (s)')
    ax_u.grid(True, alpha=0.3)
    ax_u.legend()

    # Row 3: w | q
    ax_w = axes[state_row_offset + 1, 0]
    ax_w.plot(times, states[:, 2], 'b-', linewidth=1)
    if trim_state is not None:
        ax_w.axhline(y=trim_state[2], color='r', linestyle='--',
                     alpha=0.5, label='trim')
    ax_w.set_ylabel('w (m/s)')
    ax_w.set_xlabel('Time (s)')
    ax_w.grid(True, alpha=0.3)
    ax_w.legend()

    ax_q = axes[state_row_offset + 1, 1]
    ax_q.plot(times, states[:, 3], 'b-', linewidth=1)
    if trim_state is not None:
        ax_q.axhline(y=trim_state[3], color='r', linestyle='--',
                     alpha=0.5, label='trim')
    ax_q.set_ylabel('q (rad/s)')
    ax_q.set_xlabel('Time (s)')
    ax_q.grid(True, alpha=0.3)
    ax_q.legend()

    # Row 4 (if aux): Lift force | Depth factor
    if has_df:
        aux_row = state_row_offset + 2

        ax_lift = axes[aux_row, 0]
        ax_lift.plot(times, rich_result.outputs["main_lift_aero"], 'g-',
                     linewidth=1, label='main lift')
        ax_lift.plot(times, rich_result.outputs["rudder_lift_aero"], 'm-',
                     linewidth=1, label='rudder lift')
        ax_lift.set_ylabel('Lift Force (N)')
        ax_lift.set_xlabel('Time (s)')
        ax_lift.grid(True, alpha=0.3)
        ax_lift.legend()

        ax_df = axes[aux_row, 1]
        ax_df.plot(times, rich_result.outputs["main_df"], 'g-', linewidth=1,
                   label='main foil')
        ax_df.plot(times, rich_result.outputs["rudder_df"], 'm-', linewidth=1,
                   label='rudder')
        ax_df.set_ylabel('Depth Factor')
        ax_df.set_xlabel('Time (s)')
        ax_df.set_ylim(-0.05, 1.1)
        ax_df.grid(True, alpha=0.3)
        ax_df.legend()

    fig.suptitle(case_name, fontsize=14)
    fig.tight_layout()

    filepath = str(Path(output_dir) / f'{case_name}.png')
    fig.savefig(filepath, dpi=150)
    print(f"  Saved plot: {filepath}")
    plt.close(fig)


def print_diagnostics(diag, case_name):
    """Print diagnostic summary to stdout."""
    print(f"\n--- {case_name} ---")
    print(f"  NaN: {diag.has_nan}, Inf: {diag.has_inf}")
    for i, name in enumerate(diag.state_names):
        print(
            f"  {name}: min={diag.state_min[i]:.4f}, "
            f"max={diag.state_max[i]:.4f}, "
            f"drift={diag.state_drift[i]:.6f}"
        )


def write_aux_artifacts(rich_result, case_name, output_dir, system=None):
    """Write aux output CSV and summary JSON.

    Args:
        rich_result: RichSimulationResult with aux outputs.
        case_name: Name of the simulation case.
        output_dir: Directory to save artifacts to.
        system: Optional dynamic system for aux_names ordering.
    """
    if not rich_result.outputs:
        return

    times = rich_result.times
    if system is not None and hasattr(system, 'aux_names'):
        # Use aux_names order for logical grouping
        aux_keys = [k for k in system.aux_names if k in rich_result.outputs]
        # Append any non-aux keys (e.g., wave outputs) sorted
        aux_keys += sorted(k for k in rich_result.outputs if k not in system.aux_names)
    else:
        aux_keys = sorted(rich_result.outputs.keys())

    output_dir = Path(output_dir)

    # CSV
    csv_path = output_dir / f'{case_name}_aux_outputs.csv'
    header = "time," + ",".join(aux_keys)
    data = np.column_stack([times] + [rich_result.outputs[k] for k in aux_keys])
    np.savetxt(str(csv_path), data, delimiter=",", header=header, comments="")
    print(f"  Saved aux CSV: {csv_path}")

    # Summary JSON
    summary = {}
    for key in aux_keys:
        arr = rich_result.outputs[key]
        idx_min = int(np.argmin(arr))
        idx_max = int(np.argmax(arr))
        summary[key] = {
            "min": float(arr[idx_min]),
            "min_time": float(times[idx_min]),
            "max": float(arr[idx_max]),
            "max_time": float(times[idx_max]),
        }

    json_path = output_dir / f'{case_name}_aux_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved aux JSON: {json_path}")


def extract_case_metrics(result, rich_result, case_name, trim_state):
    """Extract per-case metrics for the consolidated metrics.json.

    Returns a dict with state statistics and aux summary for one case.
    """
    states = np.array(result.states)
    times = np.array(result.times)
    state_names = ['pos_d', 'theta', 'w', 'q', 'u']

    state_stats = {}
    for i, name in enumerate(state_names[:states.shape[1]]):
        col = states[:, i]
        idx_min = int(np.argmin(col))
        idx_max = int(np.argmax(col))
        state_stats[name] = {
            "initial": float(col[0]),
            "final": float(col[-1]),
            "min": float(col[idx_min]),
            "min_time": float(times[idx_min]),
            "max": float(col[idx_max]),
            "max_time": float(times[idx_max]),
            "drift": float(col[-1] - col[0]),
            "std": float(np.std(col)),
        }

    metrics = {"state_statistics": state_stats}

    # Add aux summary if available
    if rich_result is not None and rich_result.outputs:
        aux_summary = {}
        for key in ["main_df", "rudder_df", "main_lift_aero", "rudder_lift_aero"]:
            if key in rich_result.outputs:
                arr = rich_result.outputs[key]
                aux_summary[key] = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
        metrics["aux_summary"] = aux_summary

    return metrics


def write_state_summary(result, case_name, output_dir):
    """Write per-case state min/max/drift summary JSON."""
    states = np.array(result.states)
    times = np.array(result.times)
    state_names = ['pos_d', 'theta', 'w', 'q', 'u']

    summary = {}
    for i, name in enumerate(state_names[:states.shape[1]]):
        col = states[:, i]
        idx_min = int(np.argmin(col))
        idx_max = int(np.argmax(col))
        summary[name] = {
            "initial": float(col[0]),
            "final": float(col[-1]),
            "min": float(col[idx_min]),
            "min_time": float(times[idx_min]),
            "max": float(col[idx_max]),
            "max_time": float(times[idx_max]),
            "drift": float(col[-1] - col[0]),
        }

    json_path = Path(output_dir) / f'{case_name}_state_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved state JSON: {json_path}")


def write_trim_summary(trim, output_dir, surge_enabled=False):
    """Write trim point summary JSON."""
    state_names = ['pos_d', 'theta', 'w', 'q', 'u']
    control_names = ['flap', 'elevator']

    summary = {
        "state": {name: float(trim.state[i])
                  for i, name in enumerate(state_names[:len(trim.state)])},
        "control": {name: float(trim.control[i])
                    for i, name in enumerate(control_names[:len(trim.control)])},
        "control_deg": {name: float(np.degrees(trim.control[i]))
                        for i, name in enumerate(control_names[:len(trim.control)])},
        "residual": float(trim.residual),
        "surge_enabled": surge_enabled,
    }

    json_path = Path(output_dir) / 'trim_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved trim JSON: {json_path}")


def run_equilibrium(output_dir, surge_enabled=False):
    """Run trim equilibrium case. Returns list of (result, diag, trim, case, rich)."""
    print("\n=== Equilibrium Cases ===")
    case = case_trim_equilibrium(u_forward=10.0, duration=10.0)
    case.surge_enabled = surge_enabled
    result, diag, trim, moth = run_case(case)

    rich = result_with_meta(moth, result)

    print_diagnostics(diag, case.name)
    plot_case(result, case.name, output_dir, trim.state, rich_result=rich,
             params=MOTH_BIEKER_V3)
    write_aux_artifacts(rich, case.name, output_dir, system=moth)
    write_state_summary(result, case.name, output_dir)
    write_trim_summary(trim, output_dir, surge_enabled=surge_enabled)
    return [(result, diag, trim, case, rich)]


def run_impulse_cases(output_dir, surge_enabled=False):
    """Run impulse response cases. Returns list of (result, diag, trim, case, rich).

    Expected qualitative behavior (no-surge):

    Common theme: the model has no pos_d restoring force when foils are well
    submerged (df~1.0). Pitch changes from impulses cause steady heave drift
    via kinematics (pos_d_dot = -u*sin(theta) + w*cos(theta)). Two one-sided
    restoring boundaries exist:
      - Ventilation ceiling: when the boat rises high enough for foil tips
        to approach the surface (~pos_d=-1.75 at 15 deg heel), lift loss
        arrests the upward motion.
      - Hull contact floor: when the boat sinks to hull_contact_depth
        (~0.94m above water), hull buoyancy/drag arrests downward motion.

    Impulse sign conventions (main foil at x=+0.55m forward of CG,
    rudder at x=-1.755m aft of CG):
      - +flap: more lift + nose-up moment (minimum-phase heave)
      - -flap: less lift + nose-down moment
      - +elevator: more lift at tail + nose-DOWN moment (non-minimum-phase heave)
      - -elevator: less lift at tail + nose-UP moment

    flap_pos/elev_neg: pitch nose up -> rise -> hit ventilation ceiling.
    flap_neg/elev_pos: pitch nose down -> sink -> hit hull contact floor.
    Elevator cases show non-minimum phase in heave (direct lift opposes
    the pitch-induced drift). Flap cases do not (lift and pitch moment
    are in the same direction because the main foil is forward of CG).

    Trim at 10 m/s is near pos_d~-1.3m, theta~0.3 deg.
    """
    print("\n=== Impulse Response Cases ===")
    results = []

    # Find trim first (use a temporary moth just for trim)
    trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
    trim_control = trim.control

    impulse_configs = [
        ("flap_pos", 0, np.deg2rad(3.0)),    # +3 deg flap
        ("flap_neg", 0, np.deg2rad(-3.0)),   # -3 deg flap
        ("elev_pos", 1, np.deg2rad(2.0)),    # +2 deg elevator
        ("elev_neg", 1, np.deg2rad(-2.0)),   # -2 deg elevator
    ]

    for name, ctrl_idx, delta in impulse_configs:
        # Build impulse control schedule
        impulse_control = trim_control.copy()
        impulse_control[ctrl_idx] += delta

        # Piecewise: impulse for 0.1s then back to trim
        schedule = PiecewiseConstantControl(
            times=jnp.array([0.0, 0.1]),
            controls=jnp.array([impulse_control, trim_control]),
        )

        case = SimCase(
            name=f"impulse_{name}",
            u_forward=10.0,
            duration=5.0,
            dt=0.005,
            control_schedule=schedule,
            surge_enabled=surge_enabled,
            description=f"Impulse {name}: delta={np.degrees(delta):.1f} deg for 0.1s",
        )

        result, diag, _, moth = run_case(case)
        rich = result_with_meta(moth, result)

        print_diagnostics(diag, case.name)
        plot_case(result, case.name, output_dir, trim.state, rich_result=rich,
                 params=MOTH_BIEKER_V3)
        write_aux_artifacts(rich, case.name, output_dir, system=moth)
        write_state_summary(result, case.name, output_dir)
        results.append((result, diag, trim, case, rich))

    return results


def run_mode(output_dir, surge_enabled, cases):
    """Run all requested cases for a single surge mode.

    Args:
        output_dir: Root output directory for this mode.
        surge_enabled: Whether surge dynamics are enabled.
        cases: Which case set to run ('equilibrium', 'impulse', or 'all').

    Returns:
        Dict of case_name -> (result, diag, trim, case, rich) for metrics extraction.
    """
    mode_label = "surge_enabled" if surge_enabled else "surge_disabled"
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"RUNNING MODE: {mode_label}")
    print("=" * 60)

    all_results = {}

    if cases in ('equilibrium', 'all'):
        eq_results = run_equilibrium(str(plots_dir), surge_enabled=surge_enabled)
        for result, diag, trim, case, rich in eq_results:
            all_results[case.name] = (result, diag, trim, case, rich)

    if cases in ('impulse', 'all'):
        imp_results = run_impulse_cases(str(plots_dir), surge_enabled=surge_enabled)
        for result, diag, trim, case, rich in imp_results:
            all_results[case.name] = (result, diag, trim, case, rich)

    # PNGs stay in plots/; move data artifacts (CSV, JSON) to mode root.
    import shutil
    for f in plots_dir.iterdir():
        if f.suffix != '.png':
            shutil.move(str(f), str(Path(output_dir) / f.name))

    return all_results


def extract_mode_metrics(mode_results, surge_enabled):
    """Extract metrics for all cases in a mode.

    Args:
        mode_results: Dict of case_name -> (result, diag, trim, case, rich).
        surge_enabled: Whether surge was enabled.

    Returns:
        Dict with trim info and per-case metrics.
    """
    mode_metrics = {"surge_enabled": surge_enabled, "cases": {}}

    for case_name, (result, diag, trim, case, rich) in mode_results.items():
        case_metrics = extract_case_metrics(result, rich, case_name, trim.state)
        case_metrics["description"] = case.description if case.description else case_name
        case_metrics["duration_s"] = case.duration
        case_metrics["has_nan"] = bool(diag.has_nan)
        case_metrics["has_inf"] = bool(diag.has_inf)
        mode_metrics["cases"][case_name] = case_metrics

    # Add trim info from the first case that has it
    for case_name, (result, diag, trim, case, rich) in mode_results.items():
        state_names = ['pos_d', 'theta', 'w', 'q', 'u']
        control_names = ['flap', 'elevator']
        mode_metrics["trim"] = {
            "state": {name: float(trim.state[i])
                      for i, name in enumerate(state_names[:len(trim.state)])},
            "control_rad": {name: float(trim.control[i])
                            for i, name in enumerate(control_names[:len(trim.control)])},
            "control_deg": {name: float(np.degrees(trim.control[i]))
                            for i, name in enumerate(control_names[:len(trim.control)])},
            "residual": float(trim.residual),
        }
        break

    return mode_metrics


def main():
    parser = argparse.ArgumentParser(description="Moth 3DOF open-loop analysis")
    parser.add_argument(
        '--cases', type=str, default='all',
        choices=['equilibrium', 'impulse', 'all'],
        help='Which case set to run',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for plots and metrics',
    )
    parser.add_argument(
        '--surge-only', action='store_true', default=False,
        help='Run only surge-enabled mode',
    )
    parser.add_argument(
        '--no-surge-only', action='store_true', default=False,
        help='Run only surge-disabled mode',
    )
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = resolve_output_dir("docs/reports/moth_open_loop")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which modes to run
    if args.surge_only and args.no_surge_only:
        parser.error("Cannot specify both --surge-only and --no-surge-only")

    modes = []
    if args.surge_only:
        modes = [True]
    elif args.no_surge_only:
        modes = [False]
    else:
        modes = [True, False]  # default: both

    all_metrics = {}

    for surge_enabled in modes:
        mode_label = "surge_enabled" if surge_enabled else "surge_disabled"
        mode_dir = output_dir / mode_label

        mode_results = run_mode(str(mode_dir), surge_enabled, args.cases)
        all_metrics[mode_label] = extract_mode_metrics(mode_results, surge_enabled)

    # Save consolidated metrics.json at top level
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # Save report_guidelines.txt at top level
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


if __name__ == '__main__':
    main()
