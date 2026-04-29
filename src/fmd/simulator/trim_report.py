"""Unified trim sweep report generator (JSON + Markdown).

Supports both CasADi and SciPy trim results, producing consistent
output format for comparison and validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class TrimSweepReport:
    """Metadata + per-speed results for a trim sweep."""

    solver: str                     # "casadi" or "scipy"
    timestamp: str                  # ISO 8601
    params_summary: str             # e.g. "MOTH_BIEKER_V3, heel=30°"
    speeds: list[float]
    results: list[dict]             # per-speed result dicts
    metadata: dict = field(default_factory=dict)


def casadi_results_to_report(
    results: list,
    speeds: list[float],
    params_summary: str = "",
) -> TrimSweepReport:
    """Adapt list[CasadiTrimResult] or list[CalibrationTrimResult] to TrimSweepReport.

    Duck-types: if a result has a ``.trim`` attribute (CalibrationTrimResult),
    unwraps it and captures the outer warnings.  Otherwise treats it as a
    plain CasadiTrimResult.
    """
    rows = []
    for speed, r in zip(speeds, results):
        # Unwrap CalibrationTrimResult if present
        if hasattr(r, "trim"):
            outer_warnings = list(r.warnings)
            tr = r.trim
        else:
            outer_warnings = []
            tr = r

        row = {
            "speed": speed,
            "thrust": tr.thrust,
            "theta_deg": float(np.degrees(tr.state[1])),
            "pos_d": float(tr.state[0]),
            "flap_deg": float(np.degrees(tr.control[0])),
            "elev_deg": float(np.degrees(tr.control[1])),
            "max_residual": tr.residual,
            "success": tr.success,
            "warnings": outer_warnings or tr.warnings,
            "solve_time_s": tr.solve_time,
            "phases": [],
        }
        for p in tr.phases:
            row["phases"].append({
                "phase": p.phase,
                "status": p.status,
                "iterations": p.iterations,
                "wall_time_s": p.wall_time_s,
                "objective": p.objective,
                "max_residual": p.max_residual,
            })
        rows.append(row)

    return TrimSweepReport(
        solver="casadi",
        timestamp=datetime.now(timezone.utc).isoformat(),
        params_summary=params_summary,
        speeds=list(speeds),
        results=rows,
    )


def scipy_results_to_report(
    results: list,
    speeds: list[float],
    params_summary: str = "",
) -> TrimSweepReport:
    """Adapt list[CalibrationTrimResult] to TrimSweepReport."""
    rows = []
    for speed, r in zip(speeds, results):
        trim = r.trim
        row = {
            "speed": speed,
            "thrust": r.thrust,
            "theta_deg": float(np.degrees(trim.state[1])) if trim.state is not None else None,
            "pos_d": float(trim.state[0]) if trim.state is not None else None,
            "flap_deg": float(np.degrees(trim.control[0])) if trim.control is not None else None,
            "elev_deg": float(np.degrees(trim.control[1])) if trim.control is not None else None,
            "max_residual": float(trim.residual) if trim.residual is not None else None,
            "success": trim.success,
            "solve_time_s": None,
            "phases": [],
        }
        if hasattr(trim, "solver_info") and trim.solver_info:
            for info in trim.solver_info:
                row["phases"].append({
                    "phase": info.label,
                    "status": "Converged" if info.success else info.message,
                    "iterations": info.nit,
                    "wall_time_s": info.wall_time_s,
                    "objective": info.final_obj,
                    "max_residual": None,
                })
        rows.append(row)

    return TrimSweepReport(
        solver="scipy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        params_summary=params_summary,
        speeds=list(speeds),
        results=rows,
    )


def render_markdown(report: TrimSweepReport) -> str:
    """Generate markdown report from TrimSweepReport."""
    lines = []

    # Header
    n_success = sum(1 for r in report.results if r["success"])
    n_total = len(report.results)
    lines.append(f"# Trim Sweep Report — {report.solver.upper()}")
    lines.append("")
    lines.append(f"- **Solver**: {report.solver}")
    lines.append(f"- **Timestamp**: {report.timestamp}")
    if report.params_summary:
        lines.append(f"- **Params**: {report.params_summary}")
    speed_range = f"{min(report.speeds):.0f}–{max(report.speeds):.0f} m/s"
    lines.append(f"- **Speed range**: {speed_range}")
    lines.append(f"- **Success rate**: {n_success}/{n_total}")
    lines.append("")

    # Main results table
    lines.append("## Results")
    lines.append("")
    lines.append("| Speed (m/s) | Thrust (N) | Theta (°) | pos_d (m) | Flap (°) | Elev (°) | Max Residual | Status |")
    lines.append("|-------------|-----------|-----------|-----------|----------|----------|-------------|--------|")
    for r in report.results:
        status = "OK" if r["success"] else "FAIL"
        theta = f"{r['theta_deg']:+.2f}" if r["theta_deg"] is not None else "—"
        pos_d = f"{r['pos_d']:.4f}" if r["pos_d"] is not None else "—"
        flap = f"{r['flap_deg']:+.2f}" if r["flap_deg"] is not None else "—"
        elev = f"{r['elev_deg']:+.2f}" if r["elev_deg"] is not None else "—"
        res = f"{r['max_residual']:.2e}" if r["max_residual"] is not None else "—"
        thrust = f"{r['thrust']:.1f}" if r["thrust"] is not None else "—"
        lines.append(
            f"| {r['speed']:11.0f} | {thrust:>9s} | {theta:>9s} | {pos_d:>9s} | "
            f"{flap:>8s} | {elev:>8s} | {res:>11s} | {status:>6s} |"
        )
    lines.append("")

    # Per-phase detail table (if any phases present)
    has_phases = any(r.get("phases") for r in report.results)
    if has_phases:
        lines.append("## Per-Phase Details")
        lines.append("")
        lines.append("| Speed | Phase | Status | Iters | Time (s) | Objective | Max Residual |")
        lines.append("|-------|-------|--------|-------|----------|-----------|-------------|")
        for r in report.results:
            for p in r.get("phases", []):
                status = p["status"][:25] if p["status"] else "—"
                iters = str(p["iterations"]) if p["iterations"] is not None else "—"
                wtime = f"{p['wall_time_s']:.3f}" if p["wall_time_s"] is not None else "—"
                obj = f"{p['objective']:.4e}" if p["objective"] is not None else "—"
                res = f"{p['max_residual']:.2e}" if p["max_residual"] is not None else "—"
                lines.append(
                    f"| {r['speed']:5.0f} | {p['phase']:20s} | {status:25s} | "
                    f"{iters:>5s} | {wtime:>8s} | {obj:>9s} | {res:>11s} |"
                )
        lines.append("")

    # Warnings
    failures = [r for r in report.results if not r["success"]]
    if failures:
        lines.append("## Warnings")
        lines.append("")
        for r in failures:
            lines.append(f"- **{r['speed']:.0f} m/s**: Failed to converge (residual={r['max_residual']:.2e})")
        lines.append("")

    # Sanity warnings (from successful solves)
    warned = [r for r in report.results if r.get("warnings") and r["success"]]
    if warned:
        lines.append("## Sanity Warnings")
        lines.append("")
        for r in warned:
            lines.append(f"- **{r['speed']:.0f} m/s**: {'; '.join(r['warnings'])}")
        lines.append("")

    return "\n".join(lines)


def load_report(input_dir: str | Path) -> TrimSweepReport:
    """Load a TrimSweepReport from results.json in input_dir."""
    input_dir = Path(input_dir)
    json_path = input_dir / "results.json"
    with open(json_path) as f:
        data = json.load(f)
    return TrimSweepReport(**data)


def save_report(report: TrimSweepReport, output_dir: str | Path) -> tuple[Path, Path]:
    """Write results.json + report.md to output_dir.

    Returns (json_path, md_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON — serialize with numpy-safe converter
    json_path = output_dir / "results.json"

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=_default)

    # Markdown
    md_path = output_dir / "report.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(report))

    return json_path, md_path
