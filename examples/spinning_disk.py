#!/usr/bin/env python3
"""Example: Thrown spinning disk (frisbee) simulation.

Demonstrates 3D translation and rotation with gyroscopic effects.
A disk is thrown with initial velocity and spin, showing how
angular momentum stabilizes the trajectory.

Usage:
    uv run python examples/spinning_disk.py
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from fmd.simulator import RigidBody6DOF, simulate, LogWriter, create_state, result_with_meta, Gravity
from fmd.core import euler_to_quat


def main():
    # Disk parameters (standard frisbee)
    mass = 0.175  # kg
    radius = 0.135  # m

    # Inertia tensor for thin disk
    # Ixx = Iyy = (1/4) m r^2  (perpendicular to disk)
    # Izz = (1/2) m r^2        (spin axis, through center)
    Ixx = 0.25 * mass * radius**2
    Iyy = Ixx
    Izz = 0.5 * mass * radius**2

    print("Spinning Disk (Frisbee) Simulation")
    print(f"  Mass: {mass} kg")
    print(f"  Radius: {radius} m")
    print(f"  Inertia: Ixx=Iyy={Ixx:.6f}, Izz={Izz:.6f} kg·m²")

    # Create rigid body with gravity
    disk = RigidBody6DOF(
        mass=mass,
        inertia=jnp.array([Ixx, Iyy, Izz]),
        components=[Gravity(mass=mass)],
    )

    # Initial conditions: thrown like a frisbee
    release_height = 1.5  # m above ground (negative D in NED)
    throw_speed = 12.0    # m/s forward
    throw_angle = 10.0    # degrees upward
    tilt_angle = 10.0     # degrees of initial tilt (causes precession)
    spin_rate = 30.0      # rad/s (~300 RPM, typical frisbee spin)

    # Compute initial velocity
    throw_angle_rad = np.radians(throw_angle)
    vel_forward = throw_speed * np.cos(throw_angle_rad)
    vel_up = throw_speed * np.sin(throw_angle_rad)

    # Initial state
    initial_state = create_state(
        position=jnp.array([0.0, 0.0, -release_height]),  # Above ground (NED: -D = up)
        velocity=jnp.array([vel_forward, 0.0, -vel_up]),  # Forward and up (NED: -W = up)
        quaternion=jnp.array(euler_to_quat([0.0, np.radians(tilt_angle), 0.0])),  # Tilted about pitch
        angular_velocity=jnp.array([0.0, 0.0, spin_rate]),  # Spinning about body z-axis
    )

    print(f"\nInitial conditions:")
    print(f"  Release height: {release_height} m")
    print(f"  Throw speed: {throw_speed} m/s at {throw_angle}° upward")
    print(f"  Initial tilt: {tilt_angle}°")
    print(f"  Spin rate: {spin_rate} rad/s ({spin_rate * 60 / (2*np.pi):.0f} RPM)")

    # Simulation parameters
    dt = 0.001  # 1000 Hz (need fine timestep for gyroscopic dynamics)
    duration = 2.5  # seconds

    # Run simulation
    result = simulate(disk, initial_state, dt, duration)

    # Convert to numpy for analysis
    states = np.asarray(result.states)
    times = np.asarray(result.times)

    # Find when disk hits ground (pos_d >= 0)
    ground_idx = np.where(states[:, 2] >= 0)[0]
    if len(ground_idx) > 0:
        flight_time = times[ground_idx[0]]
        landing_pos = states[ground_idx[0], 0:3]
        print(f"\nFlight results:")
        print(f"  Flight time: {flight_time:.3f} s")
        print(f"  Range (North): {landing_pos[0]:.2f} m")
        print(f"  Drift (East): {landing_pos[1]:.2f} m")
    else:
        print(f"\nDisk still in flight after {duration}s")

    # Check angular momentum conservation (magnitude)
    # L = I * omega in body frame
    def angular_momentum_magnitude(state):
        omega = state[10:13]
        L_body = np.array([Ixx, Iyy, Izz]) * omega
        return np.linalg.norm(L_body)

    L_initial = angular_momentum_magnitude(states[0])
    L_final = angular_momentum_magnitude(states[-1])
    L_error = abs(L_final - L_initial) / L_initial * 100

    print(f"\nAngular momentum check:")
    print(f"  Initial |L|: {L_initial:.6f} kg·m²/s")
    print(f"  Final |L|:   {L_final:.6f} kg·m²/s")
    print(f"  Error: {L_error:.4f}%")

    # Convert to rich result with metadata for logging
    rich_result = result_with_meta(disk, result)

    # Write results to CSV
    output_dir = Path(__file__).parent.parent / "data"
    writer = LogWriter(output_dir=output_dir)

    output_file = writer.write(rich_result, "spinning_disk.csv")
    print(f"\nResults written to: {output_file}")

    # Also save metadata
    meta_file = writer.write_metadata("spinning_disk", extra={
        "description": "Thrown spinning disk (frisbee) simulation",
        "mass_kg": mass,
        "radius_m": radius,
        "throw_speed_mps": throw_speed,
        "spin_rate_rps": spin_rate,
    })
    print(f"Metadata written to: {meta_file}")


if __name__ == "__main__":
    main()
