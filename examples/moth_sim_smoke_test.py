#!/usr/bin/env python3
"""Moth 3DOF simulation smoke test.

Runs a 10-second simulation with default parameters and reports
key metrics to verify the model is working correctly.

Usage:
    uv run python examples/moth_sim_smoke_test.py
"""

import sys
import numpy as np

# Ensure JAX float64 before any imports
from fmd.simulator import _config  # noqa: F401

import jax.numpy as jnp
from fmd.simulator import Moth3D, simulate
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.moth_3d import POS_D, THETA, W, Q


def main():
    print("=" * 60)
    print("Moth 3DOF Simulation Smoke Test")
    print("=" * 60)
    print()

    # Create model
    moth = Moth3D(MOTH_BIEKER_V3)
    print(f"Model: Moth3D with MOTH_BIEKER_V3 params")
    print(f"  Total mass: {moth.total_mass:.1f} kg")
    print(f"  Iyy: {moth.iyy:.1f} kg*m^2")
    print(f"  Forward speed: {moth.get_forward_speed(0.0):.1f} m/s")
    print()

    # Initial conditions
    initial = moth.default_state()
    print(f"Initial state: pos_d={float(initial[POS_D]):.2f} m, "
          f"theta={np.degrees(float(initial[THETA])):.1f} deg, "
          f"w={float(initial[W]):.2f} m/s, "
          f"q={float(initial[Q]):.2f} rad/s")
    print()

    # Run 10s simulation
    dt = 0.005
    duration = 10.0
    print(f"Running {duration}s simulation (dt={dt}s)...")
    result = simulate(moth, initial, dt=dt, duration=duration)
    print(f"  Steps: {result.times.shape[0]}")
    print()

    # Extract states
    pos_d = np.array(result.states[:, POS_D])
    theta = np.array(result.states[:, THETA])
    w = np.array(result.states[:, W])
    q = np.array(result.states[:, Q])

    # Check for NaN/Inf
    all_states = np.array(result.states)
    has_nan = np.any(np.isnan(all_states))
    has_inf = np.any(np.isinf(all_states))

    print("=== Results ===")
    print()
    print(f"NaN detected: {has_nan}")
    print(f"Inf detected: {has_inf}")
    print()

    # Ride height stats (convert to more intuitive units)
    print(f"Ride height (pos_d):")
    print(f"  Min:   {np.min(pos_d):.4f} m")
    print(f"  Max:   {np.max(pos_d):.4f} m")
    print(f"  Mean:  {np.mean(pos_d):.4f} m")
    print(f"  Final: {pos_d[-1]:.4f} m")
    print()

    print(f"Pitch angle (theta):")
    print(f"  Min:   {np.degrees(np.min(theta)):.2f} deg")
    print(f"  Max:   {np.degrees(np.max(theta)):.2f} deg")
    print(f"  Mean:  {np.degrees(np.mean(theta)):.2f} deg")
    print(f"  Final: {np.degrees(theta[-1]):.2f} deg")
    print()

    print(f"Heave velocity (w):")
    print(f"  Min:   {np.min(w):.4f} m/s")
    print(f"  Max:   {np.max(w):.4f} m/s")
    print(f"  Final: {w[-1]:.4f} m/s")
    print()

    print(f"Pitch rate (q):")
    print(f"  Min:   {np.degrees(np.min(q)):.2f} deg/s")
    print(f"  Max:   {np.degrees(np.max(q)):.2f} deg/s")
    print(f"  Final: {np.degrees(q[-1]):.2f} deg/s")
    print()

    # Final state
    print(f"Final state: pos_d={pos_d[-1]:.4f} m, "
          f"theta={np.degrees(theta[-1]):.2f} deg, "
          f"w={w[-1]:.4f} m/s, "
          f"q={np.degrees(q[-1]):.2f} deg/s")
    print()

    # Bounded check
    bounded = (
        np.max(np.abs(pos_d)) < 100.0
        and np.max(np.abs(theta)) < 10.0
        and np.max(np.abs(w)) < 100.0
        and np.max(np.abs(q)) < 100.0
    )
    print(f"Simulation bounded: {bounded}")
    print()

    # Quick derivative check at initial state
    deriv = moth.forward_dynamics(initial, moth.default_control())
    print(f"Derivatives at initial state:")
    print(f"  pos_d_dot = {float(deriv[POS_D]):.4f} m/s")
    print(f"  theta_dot = {np.degrees(float(deriv[THETA])):.4f} deg/s")
    print(f"  w_dot     = {float(deriv[W]):.4f} m/s^2")
    print(f"  q_dot     = {np.degrees(float(deriv[Q])):.4f} deg/s^2")
    print()

    # Summary
    print("=" * 60)
    if has_nan or has_inf:
        print("FAIL: Non-finite values detected!")
        return 1
    elif not bounded:
        print("WARN: Simulation diverged (states unbounded)")
        return 1
    else:
        print("PASS: Simulation completed successfully")
        print("  - All states finite")
        print("  - States bounded")
        return 0


if __name__ == "__main__":
    sys.exit(main())
