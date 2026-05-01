#!/usr/bin/env python3
"""CLI for the Moth sail thrust-table calibration sweep.

Thin wrapper around :mod:`fmd.simulator.trim_calibration`. Runs
:func:`calibrate_moth_thrust_table` across a list of speeds and writes:

  - ``thrust_table.csv``                 — per-speed log
  - ``results.json`` + ``report.md``     — TrimSweepReport
  - ``plots/thrust_table_comparison.png`` — old-vs-new thrust curve
  - stdout: a Python snippet ready to paste into ``presets.py``

The library helpers in :mod:`fmd.simulator.trim_calibration` are importable,
so downstream wrappers (e.g. private repos that want to layer on extra
artifacts) can reuse them without re-implementing CSV/plot/report logic.

Examples::

    uv run python scripts/calibrate_thrust_table.py
    uv run python scripts/calibrate_thrust_table.py --speeds 8 10 12
    uv run python scripts/calibrate_thrust_table.py --no-seeds
    uv run python scripts/calibrate_thrust_table.py --from-dir <prior-output-dir>
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_calibration import (
    calibrate_moth_thrust_table,
    print_results,
    print_snippet_and_comparison,
    regenerate_outputs,
    write_outputs,
)


def _default_output_dir() -> Path:
    return Path.cwd() / f"moth_trim_calibration_{date.today().isoformat()}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Moth sail thrust lookup table (CasADi).",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=None,
        help="Speeds to calibrate at (m/s). Default: 6..20 in 1 m/s steps.",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        default="trim_seeds.json",
        help="Path to JSON seed cache (default: ./trim_seeds.json).",
    )
    parser.add_argument(
        "--no-seeds",
        action="store_true",
        help="Disable seed loading and saving.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory (default: ./moth_trim_calibration_<YYYY-MM-DD> "
            "in CWD)."
        ),
    )
    parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help=(
            "Regenerate report.md and plot from an existing results.json "
            "without re-solving."
        ),
    )
    args = parser.parse_args(argv)

    params = MOTH_BIEKER_V3

    if args.from_dir:
        regenerate_outputs(args.from_dir, params=params)
        print("Done!")
        return

    speeds = args.speeds or [float(s) for s in range(6, 21)]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    seed_path = None if args.no_seeds else args.seed_file

    print(f"Calibrating thrust table at {len(speeds)} speeds: {speeds}", flush=True)
    print("Solver: CasADi/IPOPT two-phase", flush=True)
    print(f"Output: {output_dir}", flush=True)

    results = calibrate_moth_thrust_table(
        params,
        speeds,
        seed_path=seed_path,
        verbose=True,
    )

    print_results(speeds, results)
    print_snippet_and_comparison(speeds, results, params)

    write_outputs(
        speeds,
        results,
        output_dir,
        params=params,
        params_summary="MOTH_BIEKER_V3, heel=30°",
    )

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    converged = sum(1 for r in results if r.trim.success)
    print(f"Converged: {converged}/{len(speeds)}")


if __name__ == "__main__":
    main()
