#!/usr/bin/env python3
"""Plot the Moth 3DOF 10-second smoke test.

Produces a multi-panel time-history plot of the key states using the
unified plotting library.

Expected behavior (as of 2026-02-27, 9c7ed6f):
    Starts from default_state (pos_d=-0.25, theta=0, zero controls) which is
    NOT at trim (trim requires theta~1.8 deg, flap~-1.6 deg, elev~0.8 deg).

    Phase 1 (0-1.75s): Force/moment imbalance causes nose-up pitch and
    oscillation. The boat rises from pos_d=-0.25 to -0.70 (0.45m above
    initial ride height).

    Transition (~1.5s): At pos_d~-0.70 the main foil tip approaches the
    surface. Depth factor drops from ~1.0 to 0.23 (77% ventilated), causing
    massive lift loss that arrests the upward motion.

    Phase 2 (1.75s+): Settles at pos_d~-0.697, df~0.65 (35% ventilated),
    theta~2.1 deg. This is a spurious stable equilibrium created by the
    smooth ventilation model — partial ventilation provides a restoring force
    (rise -> more ventilation -> less lift -> sink back). Physically, a moth
    0.7m above the water with a ventilated foil would crash back down.

Usage:
    uv run python examples/plot_moth_smoke_test.py
    uv run python examples/plot_moth_smoke_test.py --output-dir results/custom/
"""

import argparse
import os
from datetime import datetime

import numpy as np

# Ensure JAX float64 before any imports
from fmd.simulator import _config  # noqa: F401

import matplotlib.pyplot as plt
from fmd.simulator import Moth3D, simulate
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.analysis.plots import (
    plot_simulation_result,
    savefig_and_close,
    MOTH_3DOF_STATE_LABELS,
    MOTH_3DOF_STATE_TRANSFORMS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the Moth 3DOF 10-second smoke test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/moth-smoke-test/<timestamp>/)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = args.output_dir or f"results/moth-smoke-test/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    moth = Moth3D(MOTH_BIEKER_V3)
    initial = moth.default_state()
    result = simulate(moth, initial, dt=0.005, duration=10.0)

    fig = plot_simulation_result(
        result,
        state_labels=MOTH_3DOF_STATE_LABELS,
        state_transforms=MOTH_3DOF_STATE_TRANSFORMS,
        title="Moth 3DOF Smoke Test -- 10 s open-loop (zero controls)",
    )

    filepath = os.path.join(output_dir, "moth_smoke_test_plot.png")
    savefig_and_close(fig, filepath)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    main()
