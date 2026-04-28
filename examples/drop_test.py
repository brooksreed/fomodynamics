#!/usr/bin/env python3
"""Example: Drop test simulation.

Demonstrates the fmd.simulator package by simulating a 1kg cube
dropped from 100m height under gravity.

The output CSV is compatible with fmd.analysis and can be loaded using:

    from fmd.analysis import load_stream
    stream = load_stream("drop_test.csv")

Usage:
    uv run python examples/drop_test.py
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from fmd.simulator import (
    RigidBody6DOF,
    simulate,
    LogWriter,
    create_state,
    result_with_meta,
    Gravity,
)


def main():
    # Physical parameters
    mass = 1.0  # kg
    side = 0.1  # 10cm cube

    # Inertia tensor for a uniform cube: I = (1/6) * m * s^2
    inertia_diagonal = (1/6) * mass * side**2
    inertia = jnp.array([inertia_diagonal, inertia_diagonal, inertia_diagonal])

    # Create rigid body with gravity
    body = RigidBody6DOF(
        mass=mass,
        inertia=inertia,
        components=[Gravity(mass=mass)],
    )

    # Initial state: 100m above ground, at rest
    # In NED frame, negative D means above the reference (ground)
    initial_state = create_state(
        position=jnp.array([0.0, 0.0, -100.0]),  # 100m up
        velocity=jnp.array([0.0, 0.0, 0.0]),
        angular_velocity=jnp.array([0.0, 0.0, 0.0]),
    )

    # Simulation parameters
    dt = 0.01  # 100 Hz
    duration = 4.5  # seconds (should hit ground around 4.5s)

    print(f"Simulating {duration}s drop from 100m...")
    print(f"  Mass: {mass} kg")
    print(f"  Time step: {dt} s")
    print(f"  Expected impact time: {(2*100/9.81)**0.5:.2f} s")

    # Run simulation
    result = simulate(body, initial_state, dt, duration)

    # Convert to rich result with metadata for logging
    rich_result = result_with_meta(body, result)

    # Write results to CSV
    output_dir = Path(__file__).parent.parent / "data"
    writer = LogWriter(output_dir=output_dir)

    output_file = writer.write(rich_result, "drop_test.csv")
    print(f"\nResults written to: {output_file}")

    # Also write metadata
    meta_file = writer.write_metadata("drop_test", extra={
        "description": "Drop test - 1kg cube from 100m",
        "simulation_dt": dt,
        "duration": duration,
    })
    print(f"Metadata written to: {meta_file}")

    # Print summary (convert to numpy for display)
    states = np.asarray(result.states)
    times = np.asarray(result.times)
    final_state = states[-1]
    print(f"\nFinal state (t={times[-1]:.2f}s):")
    print(f"  Position D: {final_state[2]:.2f} m (0 = ground)")
    print(f"  Velocity W: {final_state[5]:.2f} m/s")

    # Verify against analytical solution
    g = 9.80665
    t = duration
    expected_d = -100 + 0.5 * g * t**2
    expected_vw = g * t
    print(f"\nExpected (analytical):")
    print(f"  Position D: {expected_d:.2f} m")
    print(f"  Velocity W: {expected_vw:.2f} m/s")


if __name__ == "__main__":
    main()
