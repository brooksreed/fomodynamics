"""Multi-speed Moth thrust-table calibration: sweep + I/O helpers.

The sweep (:func:`calibrate_moth_thrust_table`) wraps
:func:`fmd.simulator.trim_casadi.calibrate_moth_thrust` in a loop over target
speeds, with optional warm-starting from a JSON seed cache. Each converged
solution is written back to the cache, so subsequent runs at the same speeds
start from a good initial guess.

Each speed is solved independently — the seed cache provides per-speed
warm-starts across runs (no speed-to-speed propagation within a single
sweep). For true intra-run continuation, call
:func:`fmd.simulator.trim_casadi.find_casadi_trim_sweep` directly.

The output helpers (:func:`print_results`,
:func:`print_snippet_and_comparison`, :func:`save_csv`,
:func:`generate_plot`, :func:`write_outputs`, :func:`regenerate_outputs`)
back the bundled ``scripts/calibrate_thrust_table.py`` driver and are
exposed here so downstream wrappers can layer on additional artifacts
without duplicating the public output logic.

Example::

    from fmd.simulator.params import MOTH_BIEKER_V3
    from fmd.simulator.trim_calibration import (
        calibrate_moth_thrust_table,
        write_outputs,
    )

    results = calibrate_moth_thrust_table(
        MOTH_BIEKER_V3,
        speeds=range(6, 21),
        seed_path="trim_seeds.json",
    )
    write_outputs(
        list(range(6, 21)), results, "./calib_today",
        params=MOTH_BIEKER_V3, params_summary="MOTH_BIEKER_V3, heel=30°",
    )
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from fmd.simulator.trim_casadi import (
    CalibrationTrimResult,
    calibrate_moth_thrust,
    validate_thrust_sweep,
)
from fmd.simulator.trim_report import (
    TrimSweepReport,
    casadi_results_to_report,
    load_report,
    render_markdown,
    save_report,
)
from fmd.simulator.params.moth import MothParams


_Z_LEN = 8  # decision-variable length: [pos_d, theta, w, q, u, flap, elev, thrust]


def _seed_key(speed: float) -> str:
    return f"{float(speed):.1f}"


def _load_seeds(path: str | os.PathLike) -> dict[str, list[float]]:
    """Load the seed cache. Returns {} if the file is missing or malformed."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            raw = json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"WARNING: failed to load seeds from {p}: {exc}", flush=True)
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_seeds(seeds: dict[str, list[float]], path: str | os.PathLike) -> None:
    """Write the seed cache to disk, creating parents as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(seeds, f, indent=2)


def _z0_from_result(result: CalibrationTrimResult) -> list[float]:
    """Extract the 8-vector decision variable from a converged result."""
    z0 = list(result.trim.state) + list(result.trim.control) + [result.thrust]
    return [round(float(v), 8) for v in z0]


def calibrate_moth_thrust_table(
    params: MothParams,
    speeds: Iterable[float],
    *,
    seed_path: str | os.PathLike | None = None,
    save_seeds: bool = True,
    heel_angle: float = np.deg2rad(30.0),
    verbose: bool = True,
) -> list[CalibrationTrimResult]:
    """Calibrate the Moth thrust table across a range of speeds.

    For each speed, solves a single-speed calibration via
    :func:`calibrate_moth_thrust`. Optionally warm-starts each solve from a
    JSON seed cache mapping ``"<speed>" -> [z0_8vec]``. After the sweep,
    writes converged solutions back to the cache (preserving entries for
    speeds not run in this call).

    Args:
        params: Moth parameters (e.g. ``MOTH_BIEKER_V3``).
        speeds: Target forward speeds in m/s.
        seed_path: Path to a JSON seed cache. If ``None`` (default) no seed
            file is read or written.
        save_seeds: When ``seed_path`` is given, controls whether converged
            solutions are written back. Defaults to ``True``.
        heel_angle: Static heel angle in radians. Default 30°, matching the
            single-speed calibration default.
        verbose: When ``True``, print a one-line per-speed status as the
            sweep progresses.

    Returns:
        List of ``CalibrationTrimResult`` in the same order as ``speeds``.
        Failed solves are still included; check ``r.trim.success``.
    """
    speeds_list = [float(s) for s in speeds]

    seeds: dict[str, list[float]] = {}
    if seed_path is not None:
        seeds = _load_seeds(seed_path)
        if verbose and seeds:
            print(f"Loaded {len(seeds)} seeds from {seed_path}", flush=True)

    results: list[CalibrationTrimResult] = []
    for speed in speeds_list:
        z0 = None
        seed = seeds.get(_seed_key(speed))
        if seed is not None and len(seed) == _Z_LEN:
            z0 = np.asarray(seed, dtype=float)

        result = calibrate_moth_thrust(
            params,
            target_u=speed,
            heel_angle=heel_angle,
            z0=z0,
        )
        results.append(result)

        if verbose:
            tr = result.trim
            status = "OK" if tr.success else "FAIL"
            print(
                f"  {speed:5.1f} m/s: thrust={result.thrust:7.1f} N, "
                f"theta={np.degrees(tr.state[1]):6.3f} deg, "
                f"residual={tr.residual:.2e}, status={status}",
                flush=True,
            )

    if seed_path is not None and save_seeds:
        for speed, result in zip(speeds_list, results):
            if result.trim.success:
                seeds[_seed_key(speed)] = _z0_from_result(result)
        _save_seeds(seeds, seed_path)
        if verbose:
            print(f"Saved seeds to {seed_path}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Output helpers — back the bundled scripts/calibrate_thrust_table.py driver.
# Exposed so downstream wrappers can layer on additional artifacts without
# duplicating the public output logic.
# ---------------------------------------------------------------------------


def print_results(
    speeds: list[float], results: list[CalibrationTrimResult]
) -> None:
    """Print a per-speed status line for each result."""
    for speed, r in zip(speeds, results):
        tr = r.trim
        status = "OK" if tr.success else "FAIL"
        warns = ", ".join(r.warnings) if r.warnings else "none"
        print(
            f"  {speed:5.1f} m/s: thrust={r.thrust:7.1f} N, "
            f"theta={np.degrees(tr.state[1]):6.3f} deg, "
            f"pos_d={tr.state[0]:7.4f} m, "
            f"flap={np.degrees(tr.control[0]):6.3f} deg, "
            f"elev={np.degrees(tr.control[1]):6.3f} deg, "
            f"residual={tr.residual:.2e}, status={status}, warns=[{warns}]",
            flush=True,
        )


def print_snippet_and_comparison(
    speeds: list[float],
    results: list[CalibrationTrimResult],
    params: MothParams,
) -> None:
    """Print a paste-into-presets snippet and an old-vs-new comparison table."""
    new_speeds = tuple(float(s) for s in speeds)
    new_thrusts = tuple(r.thrust for r in results)

    print("\n" + "=" * 70)
    print("PYTHON SNIPPET (paste into presets.py):")
    print("=" * 70)
    speeds_str = ", ".join(f"{s:.1f}" for s in new_speeds)
    thrusts_str = ", ".join(f"{t:.1f}" for t in new_thrusts)

    coeff_10: float | None = None
    for s, t in zip(new_speeds, new_thrusts):
        if abs(s - 10.0) < 0.01:
            coeff_10 = t
            break
    if coeff_10 is None and new_thrusts:
        coeff_10 = new_thrusts[len(new_thrusts) // 2]

    print(f"    sail_thrust_speeds=({speeds_str},),")
    print(f"    sail_thrust_values=({thrusts_str},),")
    if coeff_10 is not None:
        print(f"    sail_thrust_coeff={coeff_10:.1f},  # fallback at 10 m/s")
    print()

    warns = validate_thrust_sweep(new_speeds, new_thrusts)
    if warns:
        print("SWEEP WARNINGS:")
        for w in warns:
            print(f"  - {w}")
    else:
        print("Sweep validation: PASS (monotonic, no sharp jumps)")

    old_speeds = params.sail_thrust_speeds
    old_values = params.sail_thrust_values
    if old_speeds and old_values:
        print("\n" + "=" * 70)
        print("OLD vs NEW COMPARISON:")
        print("=" * 70)
        print(
            f"{'Speed':>8} {'Old (N)':>10} {'New (N)':>10} "
            f"{'Delta (N)':>10} {'Delta %':>10}"
        )
        print("-" * 50)
        old_dict = dict(zip(old_speeds, old_values))
        for s, new_t in zip(new_speeds, new_thrusts):
            if s in old_dict:
                old_t = old_dict[s]
                delta = new_t - old_t
                pct = 100 * delta / old_t if old_t != 0 else float("inf")
                print(
                    f"{s:8.1f} {old_t:10.1f} {new_t:10.1f} "
                    f"{delta:+10.1f} {pct:+9.1f}%"
                )
            else:
                print(f"{s:8.1f} {'N/A':>10} {new_t:10.1f}")


def save_csv(
    speeds: list[float],
    results: list[CalibrationTrimResult],
    output_dir: str | os.PathLike,
) -> Path:
    """Write per-speed calibration log to ``thrust_table.csv``."""
    csv_path = Path(output_dir) / "thrust_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "speed_ms",
                "thrust_N",
                "residual",
                "theta_deg",
                "pos_d_m",
                "flap_deg",
                "elev_deg",
                "success",
                "warnings",
            ]
        )
        for speed, r in zip(speeds, results):
            tr = r.trim
            writer.writerow(
                [
                    f"{speed:.1f}",
                    f"{r.thrust:.2f}",
                    f"{tr.residual:.6e}",
                    f"{np.degrees(tr.state[1]):.4f}",
                    f"{tr.state[0]:.4f}",
                    f"{np.degrees(tr.control[0]):.4f}",
                    f"{np.degrees(tr.control[1]):.4f}",
                    str(tr.success),
                    "; ".join(r.warnings) if r.warnings else "",
                ]
            )
    return csv_path


def generate_plot(
    new_speeds: tuple[float, ...],
    new_thrusts: tuple[float, ...],
    output_dir: str | os.PathLike,
    params: MothParams,
) -> Path | None:
    """Generate ``plots/thrust_table_comparison.png`` overlaying old vs new.

    Returns the plot path on success, or ``None`` if matplotlib is not
    available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    old_speeds = params.sail_thrust_speeds
    old_values = params.sail_thrust_values
    if old_speeds and old_values:
        ax.plot(old_speeds, old_values, "rs--", label="Old (preset)", markersize=6)

    ax.plot(new_speeds, new_thrusts, "bo-", label="New (calibrated)", markersize=6)

    if new_speeds:
        ref_speed = 10.0
        ref_idx = min(range(len(new_speeds)), key=lambda i: abs(new_speeds[i] - ref_speed))
        ref_thrust = new_thrusts[ref_idx]
        ref_u = new_speeds[ref_idx]
        u_range = np.linspace(min(new_speeds), max(new_speeds), 50)
        u2_ref = ref_thrust * (u_range / ref_u) ** 2
        ax.plot(u_range, u2_ref, "k:", alpha=0.5, label=f"u^2 ref (from {ref_u:.0f} m/s)")

    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Moth Sail Thrust Calibration (CasADi)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = plots_dir / "thrust_table_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def write_outputs(
    speeds: list[float],
    results: list[CalibrationTrimResult],
    output_dir: str | os.PathLike,
    *,
    params: MothParams,
    params_summary: str,
) -> TrimSweepReport:
    """Write CSV + JSON + Markdown report + plot for a finished sweep.

    Returns the constructed ``TrimSweepReport`` so callers (e.g. wrappers
    layering on additional artifacts) can reuse it without rebuilding.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_csv(speeds, results, output_dir)
    print(f"Saved: {csv_path}")

    report = casadi_results_to_report(
        results,
        [float(s) for s in speeds],
        params_summary=params_summary,
    )
    json_path, md_path = save_report(report, output_dir)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")

    new_speeds = tuple(float(s) for s in speeds)
    new_thrusts = tuple(r.thrust for r in results)
    plot_path = generate_plot(new_speeds, new_thrusts, output_dir, params)
    if plot_path is not None:
        print(f"Saved: {plot_path}")
    else:
        print("Matplotlib not available, skipping plot.")

    return report


def regenerate_outputs(
    output_dir: str | os.PathLike, params: MothParams
) -> TrimSweepReport:
    """Regenerate Markdown + plot from a saved ``results.json`` (no re-solve)."""
    output_dir = Path(output_dir)
    print(f"Loading results from: {output_dir}")
    report = load_report(output_dir)

    md_path = output_dir / "report.md"
    md_path.write_text(render_markdown(report))
    print(f"Regenerated: {md_path}")

    new_speeds = tuple(report.speeds)
    new_thrusts = tuple(r["thrust"] for r in report.results)
    plot_path = generate_plot(new_speeds, new_thrusts, output_dir, params)
    if plot_path is not None:
        print(f"Regenerated: {plot_path}")

    return report
