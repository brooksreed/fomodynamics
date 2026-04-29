#!/usr/bin/env python3
"""Example: Simple pendulum simulation.

Demonstrates the SimplePendulum class - a classic 2D dynamics example.
Shows nonlinear oscillation and energy conservation.

Usage:
    uv run python examples/pendulum.py
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from fmd.simulator import SimplePendulum, simulate, LogWriter, result_with_meta
from fmd.simulator.params import SimplePendulumParams


def main():
    # Pendulum parameters
    length = 1.0  # 1 meter

    # Create pendulum with params
    params = SimplePendulumParams(length=length)
    pendulum = SimplePendulum(params)

    # Initial state: 45 degrees from vertical, at rest
    initial_angle = np.radians(45)
    initial_state = jnp.array([initial_angle, 0.0])

    # Simulation parameters
    dt = 0.01  # 100 Hz
    duration = 10.0  # 10 seconds (several oscillations)

    # Theoretical period for small angles
    theoretical_period = float(pendulum.period_small_angle())

    print(f"Simple Pendulum Simulation")
    print(f"  Length: {length} m")
    print(f"  Initial angle: {np.degrees(initial_angle):.1f} degrees")
    print(f"  Theoretical period (small angle): {theoretical_period:.3f} s")
    print(f"  Duration: {duration} s")

    # Run simulation
    result = simulate(pendulum, initial_state, dt, duration)

    # Convert to numpy for analysis
    states = np.asarray(result.states)
    times = np.asarray(result.times)

    # Compute energy at start and end
    initial_energy = float(pendulum.energy(states[0]))
    final_energy = float(pendulum.energy(states[-1]))
    energy_error = abs(final_energy - initial_energy) / initial_energy * 100

    print(f"\nEnergy conservation check:")
    print(f"  Initial energy: {initial_energy:.6f} J/kg")
    print(f"  Final energy:   {final_energy:.6f} J/kg")
    print(f"  Error: {energy_error:.4f}%")

    # Estimate period from zero crossings
    theta = states[:, 0]
    zero_crossings = np.where(np.diff(np.sign(theta)) < 0)[0]
    if len(zero_crossings) >= 2:
        periods = np.diff(times[zero_crossings])
        measured_period = np.mean(periods)
        print(f"\nMeasured period: {measured_period:.3f} s")
        print(f"Period ratio (measured/theoretical): {measured_period/theoretical_period:.4f}")

    # Convert to rich result with metadata for logging
    rich_result = result_with_meta(pendulum, result)

    # Write results to CSV
    output_dir = Path(__file__).parent.parent / "data"
    writer = LogWriter(output_dir=output_dir)

    output_file = writer.write(rich_result, "pendulum.csv")
    print(f"\nResults written to: {output_file}")


if __name__ == "__main__":
    main()
