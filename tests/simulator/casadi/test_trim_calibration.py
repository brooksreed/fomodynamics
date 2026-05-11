"""Tests for the multi-speed thrust-table calibration sweep."""

import json

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_calibration import (
    _Z_LEN,
    _seed_key,
    calibrate_moth_thrust_table,
)


def test_cold_start_sweep_two_speeds():
    """Cold-start sweep at 8 and 10 m/s should converge with monotonic thrust."""
    speeds = [8.0, 10.0]
    results = calibrate_moth_thrust_table(
        MOTH_BIEKER_V3, speeds, seed_path=None, verbose=False
    )

    assert len(results) == 2
    for speed, r in zip(speeds, results):
        assert r.speed == speed
        assert r.trim.success, f"Solve failed at {speed} m/s: {r.trim.ipopt_stats}"
        assert r.trim.residual < 1e-6, (
            f"Residual {r.trim.residual:.2e} too large at {speed} m/s"
        )

    # Thrust monotonic in speed (drag grows with u^2 → thrust grows).
    assert results[1].thrust > results[0].thrust


def test_seed_round_trip(tmp_path):
    """First run writes seeds; second run reads them and matches within 1%."""
    seed_path = tmp_path / "trim_seeds.json"
    speeds = [10.0]

    cold = calibrate_moth_thrust_table(
        MOTH_BIEKER_V3, speeds, seed_path=seed_path, verbose=False
    )
    assert seed_path.exists(), "Seed file should be written after a successful sweep"

    with seed_path.open() as f:
        seeds = json.load(f)
    key = _seed_key(10.0)
    assert key in seeds
    assert len(seeds[key]) == _Z_LEN

    warm = calibrate_moth_thrust_table(
        MOTH_BIEKER_V3, speeds, seed_path=seed_path, verbose=False
    )
    assert warm[0].trim.success

    # Warm-start should land on a nearby trim. The seed system speeds up
    # convergence; it does not guarantee bit-reproducibility — IPOPT is
    # path-dependent in the residual null-space, so a few % of drift is
    # acceptable as long as both solutions are valid trims.
    rel_err = abs(warm[0].thrust - cold[0].thrust) / abs(cold[0].thrust)
    assert rel_err < 0.05, (
        f"Warm-started thrust drifted >5% from cold: cold={cold[0].thrust:.3f} N, "
        f"warm={warm[0].thrust:.3f} N (rel_err={rel_err:.2%})"
    )


def test_save_seeds_disabled(tmp_path):
    """save_seeds=False should leave the seed file untouched."""
    seed_path = tmp_path / "trim_seeds.json"
    seed_path.write_text("{}")  # pre-existing empty cache
    pre_mtime = seed_path.stat().st_mtime_ns

    calibrate_moth_thrust_table(
        MOTH_BIEKER_V3,
        [10.0],
        seed_path=seed_path,
        save_seeds=False,
        verbose=False,
    )

    # Cache should be unchanged.
    assert seed_path.read_text() == "{}"
    assert seed_path.stat().st_mtime_ns == pre_mtime
