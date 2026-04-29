"""Trajectory generators for control and tracking.

This module provides factory functions for generating common reference
trajectories used in control testing and benchmarking.

Trajectories are returned as arrays that can be used with TrajectoryLQRController
or TVLQRController for trajectory tracking.

Numerical Stability Note
------------------------
When using these trajectories with LQR controllers, be aware of numerical
stability constraints. Aggressive LQR tuning on small vehicles can create
very fast closed-loop dynamics that require small simulation timesteps.

See the ``fmd.simulator.lqr`` module docstring for tuning guidance.

Example:
    from fmd.simulator import PlanarQuadrotor, simulate
    from fmd.simulator.trajectories import circle_trajectory_2d
    from fmd.simulator.lqr import TrajectoryLQRController
    from fmd.simulator.params import PLANAR_QUAD_CRAZYFLIE

    quad = PlanarQuadrotor(PLANAR_QUAD_CRAZYFLIE)

    # Generate circular trajectory
    times, x_refs, u_refs = circle_trajectory_2d(
        radius=1.0,
        period=5.0,
        center=(0.0, 1.0),
        num_points=100,
        hover_thrust=PLANAR_QUAD_CRAZYFLIE.hover_thrust_per_rotor,
    )

    # Create controller (use conservative R for dt=10ms stability)
    Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.01]))
    R = jnp.diag(jnp.array([10.0, 10.0]))  # Higher R for stability

    controller = TrajectoryLQRController.from_trajectory(
        quad, times, x_refs, u_refs, Q, R
    )
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import Tuple


# =============================================================================
# 2D Planar Quadrotor Trajectories
# =============================================================================


def circle_trajectory_2d(
    radius: float,
    period: float,
    center: Tuple[float, float] = (0.0, 1.0),
    num_points: int = 100,
    hover_thrust: float = 0.0,
    mass: float = 1.0,
    g: float = 9.8,
) -> Tuple[Array, Array, Array]:
    """Generate circular trajectory for 2D planar quadrotor.

    Creates a trajectory where the quadrotor flies in a vertical circle
    in the x-z plane.

    State convention: [x, z, theta, x_dot, z_dot, theta_dot]
    Control convention: [T1, T2] individual rotor thrusts

    Args:
        radius: Circle radius (m)
        period: Time to complete one circle (s)
        center: Center of circle (x, z) in meters, default (0, 1)
        num_points: Number of trajectory points
        hover_thrust: Thrust per rotor at hover (N), computed if 0
        mass: Vehicle mass (kg) for feedforward computation
        g: Gravitational acceleration (m/s^2)

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 6)
        u_refs: Reference controls (num_points, 2)
    """
    omega = 2 * np.pi / period
    times = np.linspace(0, period, num_points)

    # Compute hover thrust if not provided
    if hover_thrust <= 0:
        hover_thrust = mass * g / 2

    # Position along circle
    x = center[0] + radius * np.cos(omega * times)
    z = center[1] + radius * np.sin(omega * times)

    # Velocity (first derivative)
    x_dot = -radius * omega * np.sin(omega * times)
    z_dot = radius * omega * np.cos(omega * times)

    # Acceleration (second derivative)
    x_ddot = -radius * omega**2 * np.cos(omega * times)
    z_ddot = -radius * omega**2 * np.sin(omega * times)

    # Required pitch angle from acceleration
    # x_ddot = -(T/m) * sin(theta), z_ddot = (T/m) * cos(theta) - g
    # For small angles: theta ≈ -x_ddot * m / T_total
    # More accurately, we solve for theta that achieves desired acceleration

    # Total thrust needed: T = m * sqrt(x_ddot^2 + (z_ddot + g)^2)
    T_total = mass * np.sqrt(x_ddot**2 + (z_ddot + g)**2)

    # Pitch angle: theta = atan2(-x_ddot, z_ddot + g)
    theta = np.arctan2(-x_ddot, z_ddot + g)

    # Pitch rate from differentiation
    # d(theta)/dt = d/dt[atan2(-x_ddot, z_ddot + g)]
    x_dddot = radius * omega**3 * np.sin(omega * times)
    z_dddot = -radius * omega**3 * np.cos(omega * times)

    numerator = -x_dddot * (z_ddot + g) - (-x_ddot) * z_dddot
    denominator = x_ddot**2 + (z_ddot + g)**2 + 1e-10
    theta_dot = numerator / denominator

    # Assemble state trajectory
    x_refs = np.stack([x, z, theta, x_dot, z_dot, theta_dot], axis=1)

    # Control: split total thrust equally (assumes symmetric hover)
    # For more accuracy, would need to account for pitch moment
    u_refs = np.stack([T_total / 2, T_total / 2], axis=1)

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def figure_eight_trajectory_2d(
    width: float,
    height: float,
    period: float,
    center: Tuple[float, float] = (0.0, 1.0),
    num_points: int = 100,
    mass: float = 1.0,
    g: float = 9.8,
) -> Tuple[Array, Array, Array]:
    """Generate figure-8 trajectory for 2D planar quadrotor.

    Creates a lemniscate (figure-8) trajectory in the x-z plane.
    Uses parametric form: x = width * sin(t), z = height * sin(2t) / 2

    Args:
        width: Half-width of figure-8 (m)
        height: Half-height of figure-8 (m)
        period: Time to complete one figure-8 (s)
        center: Center point (x, z) in meters
        num_points: Number of trajectory points
        mass: Vehicle mass (kg)
        g: Gravitational acceleration (m/s^2)

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 6)
        u_refs: Reference controls (num_points, 2)
    """
    omega = 2 * np.pi / period
    times = np.linspace(0, period, num_points)

    # Lemniscate parametrization
    x = center[0] + width * np.sin(omega * times)
    z = center[1] + height * np.sin(2 * omega * times) / 2

    # Velocities
    x_dot = width * omega * np.cos(omega * times)
    z_dot = height * omega * np.cos(2 * omega * times)

    # Accelerations
    x_ddot = -width * omega**2 * np.sin(omega * times)
    z_ddot = -2 * height * omega**2 * np.sin(2 * omega * times)

    # Total thrust and pitch angle
    T_total = mass * np.sqrt(x_ddot**2 + (z_ddot + g)**2)
    theta = np.arctan2(-x_ddot, z_ddot + g)

    # Pitch rate (numerical differentiation for simplicity)
    theta_dot = np.gradient(theta, times)

    # Assemble state trajectory
    x_refs = np.stack([x, z, theta, x_dot, z_dot, theta_dot], axis=1)

    # Control
    u_refs = np.stack([T_total / 2, T_total / 2], axis=1)

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def step_trajectory_2d(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    duration: float,
    transition_time: float = 0.0,
    num_points: int = 100,
    mass: float = 1.0,
    g: float = 9.8,
) -> Tuple[Array, Array, Array]:
    """Generate step/setpoint change trajectory for 2D planar quadrotor.

    Creates a trajectory that transitions from start_pos to end_pos.
    If transition_time > 0, uses smooth interpolation; otherwise instant step.

    Args:
        start_pos: Starting position (x, z) in meters
        end_pos: Ending position (x, z) in meters
        duration: Total trajectory duration (s)
        transition_time: Time for smooth transition (s), 0 for instant
        num_points: Number of trajectory points
        mass: Vehicle mass (kg)
        g: Gravitational acceleration (m/s^2)

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 6)
        u_refs: Reference controls (num_points, 2)
    """
    times = np.linspace(0, duration, num_points)
    hover_thrust = mass * g / 2

    x_refs = np.zeros((num_points, 6))
    u_refs = np.zeros((num_points, 2))

    for i, t in enumerate(times):
        if transition_time <= 0 or t <= 0:
            # Instant step at t=0
            alpha = 0.0 if t < duration / 2 else 1.0
        else:
            # Smooth transition using sigmoid-like function
            t_mid = duration / 2
            alpha = 0.5 * (1 + np.tanh(4 * (t - t_mid) / transition_time))

        x = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
        z = start_pos[1] + alpha * (end_pos[1] - start_pos[1])

        x_refs[i] = [x, z, 0.0, 0.0, 0.0, 0.0]
        u_refs[i] = [hover_thrust, hover_thrust]

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


# =============================================================================
# Cartpole Trajectories
# =============================================================================


def cartpole_swing_up_trajectory(
    duration: float,
    num_points: int = 100,
    max_angle: float = np.pi,
    swing_duration_fraction: float = 0.7,
) -> Tuple[Array, Array, Array]:
    """Generate swing-up reference trajectory for cartpole.

    Creates a trajectory that swings the pole from hanging (theta=pi or -pi)
    to upright (theta=0). Uses smooth interpolation.

    State convention: [x, x_dot, theta, theta_dot]
    Control convention: [F] horizontal force

    Note: This is a reference trajectory only. Actual swing-up requires
    energy-based or nonlinear control to track this reference.

    Args:
        duration: Total trajectory duration (s)
        num_points: Number of trajectory points
        max_angle: Starting angle (rad), pi = hanging down
        swing_duration_fraction: Fraction of time for swing phase

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 4)
        u_refs: Reference controls (num_points, 1)
    """
    times = np.linspace(0, duration, num_points)
    swing_time = duration * swing_duration_fraction

    x_refs = np.zeros((num_points, 4))
    u_refs = np.zeros((num_points, 1))

    for i, t in enumerate(times):
        if t < swing_time:
            # Swing phase: interpolate from hanging to upright
            # Use smooth S-curve
            progress = t / swing_time
            smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))

            theta = max_angle * (1 - smooth_progress)
            theta_dot = -max_angle * np.pi / swing_time * np.sin(np.pi * progress) / 2
        else:
            # Stabilization phase: hold upright
            theta = 0.0
            theta_dot = 0.0

        x_refs[i] = [0.0, 0.0, theta, theta_dot]
        u_refs[i] = [0.0]  # Feedforward force (could be computed from dynamics)

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def cartpole_stabilization_trajectory(
    duration: float,
    num_points: int = 100,
    cart_position: float = 0.0,
) -> Tuple[Array, Array, Array]:
    """Generate stabilization reference trajectory for cartpole.

    Creates a constant trajectory at the upright equilibrium.
    Useful for testing LQR stabilization.

    Args:
        duration: Total trajectory duration (s)
        num_points: Number of trajectory points
        cart_position: Desired cart x position (m)

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 4)
        u_refs: Reference controls (num_points, 1)
    """
    times = np.linspace(0, duration, num_points)

    x_refs = np.zeros((num_points, 4))
    x_refs[:, 0] = cart_position  # x = cart_position
    # theta = 0 (upright), all velocities = 0

    u_refs = np.zeros((num_points, 1))

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def cartpole_trapezoidal_tracking(
    distance: float = 1.0,
    max_velocity: float = 0.5,
    accel_time: float = 0.5,
    pause_time: float = 0.5,
    num_points: int = 301,
    mass_cart: float = 1.0,
    mass_pole: float = 0.1,
    g: float = 9.8,
    smoothing_factor: float = 5.0,
    include_feedforward: bool = False,
) -> Tuple[Array, Array, Array]:
    """Generate trapezoidal velocity profile trajectory for cartpole tracking.

    Creates a trajectory where the cart moves from position 0 to +distance
    and back to 0, with a trapezoidal velocity profile. The pole angle is
    computed using the quasi-static approximation theta ≈ -arctan(x_ddot / g),
    which represents the pole lean required to generate the reaction force
    for cart acceleration.

    This trajectory is designed to demonstrate TVLQR's advantage over LQR:
    since theta varies during acceleration/deceleration phases, TVLQR computes
    different optimal gains at each point, while LQR uses a fixed gain.

    State convention: [x, x_dot, theta, theta_dot]
    Control convention: [F] horizontal force

    The motion profile consists of:
    1. Accelerate to max_velocity (pole leans backward)
    2. Cruise at max_velocity (pole returns to vertical)
    3. Decelerate to stop at +distance (pole leans forward)
    4. Pause at +distance
    5. Accelerate back toward origin (pole leans forward)
    6. Cruise at -max_velocity (pole vertical)
    7. Decelerate to stop at origin (pole leans backward)

    Physics:
        For the cart to accelerate at a = x_ddot while the pole remains
        quasi-stationary, the pole must lean at angle theta such that
        the horizontal component of gravity on the pole provides the
        reaction force: m_pole * g * sin(theta) ≈ m_pole * x_ddot

        This gives: theta ≈ -arctan(x_ddot / g)

        The negative sign indicates the pole leans opposite to the
        acceleration direction (lean back when accelerating forward).

    Args:
        distance: Distance to travel in each direction (m). Must be positive.
        max_velocity: Maximum cart velocity during cruise phase (m/s)
        accel_time: Time to accelerate/decelerate (s)
        pause_time: Time to pause at the far position (s)
        num_points: Number of trajectory points
        mass_cart: Cart mass for feedforward computation (kg)
        mass_pole: Pole mass for feedforward computation (kg)
        g: Gravitational acceleration (m/s^2)
        smoothing_factor: Controls smoothness of velocity transitions.
            Higher values = sharper transitions. Default 5.0 gives
            smooth transitions with reasonable peak accelerations.
        include_feedforward: If True, compute approximate feedforward control
            as F_ff = (m_cart + m_pole) * x_ddot

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 4) with [x, x_dot, theta, theta_dot]
        u_refs: Reference controls (num_points, 1)

    Raises:
        ValueError: If distance <= 0

    Example:
        >>> times, x_refs, u_refs = cartpole_trapezoidal_tracking(
        ...     distance=1.0, max_velocity=0.5, accel_time=0.5
        ... )
        >>> # Pole angle varies during acceleration phases
        >>> np.abs(x_refs[50, 2]) > 0.01  # theta != 0 during accel
        True
    """
    if distance <= 0:
        raise ValueError(f"distance must be positive, got {distance}")

    # Compute phase durations
    # Acceleration distance: d_accel = 0.5 * a * t^2 where a = v_max / t_accel
    accel = max_velocity / accel_time
    accel_distance = 0.5 * accel * accel_time**2  # = 0.5 * v_max * t_accel

    # Cruise distance: total distance - 2 * accel_distance
    cruise_distance = distance - 2 * accel_distance
    if cruise_distance < 0:
        # Not enough distance for full accel/decel, reduce max velocity
        # d = 2 * 0.5 * a * t^2 = a * t^2, so t = sqrt(d/a)
        # But we keep accel_time fixed, so we reduce max_velocity
        max_velocity = np.sqrt(distance * accel)
        accel_distance = distance / 2
        cruise_distance = 0.0

    cruise_time = cruise_distance / max_velocity if max_velocity > 0 else 0.0

    # Total time for one direction: accel + cruise + decel
    one_way_time = 2 * accel_time + cruise_time

    # Total round-trip duration
    total_duration = 2 * one_way_time + pause_time

    times = np.linspace(0, total_duration, num_points)
    dt = times[1] - times[0] if num_points > 1 else 1.0

    # Build smooth trapezoidal velocity profile using tanh transitions
    # This avoids discontinuities in acceleration

    def smooth_trapezoid_velocity(t):
        """Compute velocity at time t with smooth tanh transitions."""
        # Phase boundaries
        t1 = accel_time                      # End of first accel
        t2 = accel_time + cruise_time        # End of first cruise
        t3 = 2 * accel_time + cruise_time    # End of first decel (at +distance)
        t4 = t3 + pause_time                 # End of pause
        t5 = t4 + accel_time                 # End of return accel
        t6 = t5 + cruise_time                # End of return cruise
        t7 = total_duration                  # End of return decel (at origin)

        k = smoothing_factor  # Sharpness of transitions

        # Forward motion (positive velocity)
        # Ramp up from 0 to max_velocity
        v_accel_up = max_velocity * 0.5 * (1 + np.tanh(k * (t - accel_time/2) / accel_time))
        # Ramp down from max_velocity to 0
        v_decel_down = max_velocity * 0.5 * (1 - np.tanh(k * (t - (t2 + accel_time/2)) / accel_time))

        # Forward phase: blend accel up and decel down
        v_forward = np.minimum(v_accel_up, v_decel_down)

        # Return motion (negative velocity)
        t_return = t - t4  # Time since start of return
        t_ret_1 = accel_time
        t_ret_2 = accel_time + cruise_time

        # Ramp to -max_velocity
        v_ret_accel = -max_velocity * 0.5 * (1 + np.tanh(k * (t_return - accel_time/2) / accel_time))
        # Ramp from -max_velocity to 0
        v_ret_decel = -max_velocity * 0.5 * (1 - np.tanh(k * (t_return - (t_ret_2 + accel_time/2)) / accel_time))

        # Return phase: blend (take max since both are negative, so "minimum magnitude")
        v_return = np.maximum(v_ret_accel, v_ret_decel)

        # Combine phases with smooth switching
        # Before pause: forward motion
        # After pause: return motion
        switch_to_return = 0.5 * (1 + np.tanh(k * (t - (t3 + pause_time/2)) / max(pause_time, 0.01)))

        # During pause, both v_forward and v_return should be near zero
        v = v_forward * (1 - switch_to_return) + v_return * switch_to_return

        return v

    # Compute velocity profile
    x_dot = np.array([smooth_trapezoid_velocity(t) for t in times])

    # Integrate velocity to get position
    x = np.zeros_like(times)
    for i in range(1, len(times)):
        x[i] = x[i-1] + 0.5 * (x_dot[i] + x_dot[i-1]) * dt

    # Compute acceleration from velocity gradient
    x_ddot = np.gradient(x_dot, times)

    # Compute theta from quasi-static approximation: theta = -arctan(x_ddot / g)
    theta = -np.arctan(x_ddot / g)

    # Compute theta_dot from theta gradient
    theta_dot = np.gradient(theta, times)

    # Assemble state trajectory
    x_refs = np.stack([x, x_dot, theta, theta_dot], axis=1)

    # Feedforward control
    if include_feedforward:
        # Approximate feedforward: F = (m_c + m_p) * x_ddot
        u_refs = ((mass_cart + mass_pole) * x_ddot).reshape(-1, 1)
    else:
        u_refs = np.zeros((num_points, 1))

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def cartpole_sinusoidal_tracking(
    amplitude: float = 0.5,
    period: float = 3.0,
    duration: float = 6.0,
    num_points: int = 301,
    mass_cart: float = 1.0,
    mass_pole: float = 0.1,
    include_feedforward: bool = False,
) -> Tuple[Array, Array, Array]:
    """Generate sinusoidal cart tracking trajectory for cartpole.

    Creates a trajectory where the cart follows a sinusoidal position while
    keeping the pole balanced upright (theta=0). Useful for demonstrating
    trajectory tracking controllers (LQR, TVLQR, iLQR).

    State convention: [x, x_dot, theta, theta_dot]
    Control convention: [F] horizontal force

    The reference trajectory has:
    - x(t) = amplitude * sin(2*pi*t / period)
    - x_dot(t) = amplitude * (2*pi/period) * cos(2*pi*t / period)
    - theta(t) = 0 (upright)
    - theta_dot(t) = 0

    Note on Physical Feasibility:
        The theta=0 reference is an idealization. From the linearized cartpole
        equations, maintaining theta=0 requires x_ddot=0. For sinusoidal cart
        motion with non-zero acceleration, the pole must deviate slightly from
        vertical. This reference represents the *desired* behavior, not what's
        physically achievable. Controllers will find the best compromise between
        position tracking and pole stability.

    Feedforward Control:
        When include_feedforward=True, computes approximate feedforward as:
            F_ff = (m_cart + m_pole) * x_ddot
        This assumes theta ≈ 0 and helps feedback controllers track better.
        For aggressive trajectories, the actual required force differs due to
        pole dynamics.

    Args:
        amplitude: Sinusoidal amplitude for cart position (m)
        period: Period of oscillation (s)
        duration: Total trajectory duration (s)
        num_points: Number of trajectory points
        mass_cart: Cart mass for feedforward computation (kg)
        mass_pole: Pole mass for feedforward computation (kg)
        include_feedforward: If True, compute approximate feedforward control

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 4)
        u_refs: Reference controls (num_points, 1)
    """
    omega = 2 * np.pi / period
    times = np.linspace(0, duration, num_points)

    # Cart position follows sinusoid
    x = amplitude * np.sin(omega * times)
    x_dot = amplitude * omega * np.cos(omega * times)

    # Pole stays upright (idealized reference)
    theta = np.zeros_like(times)
    theta_dot = np.zeros_like(times)

    # Assemble state trajectory
    x_refs = np.stack([x, x_dot, theta, theta_dot], axis=1)

    # Feedforward control
    if include_feedforward:
        # Cart acceleration
        x_ddot = -amplitude * omega**2 * np.sin(omega * times)
        # Approximate feedforward assuming theta ≈ 0: F = (m_c + m_p) * x_ddot
        u_refs = ((mass_cart + mass_pole) * x_ddot).reshape(-1, 1)
    else:
        u_refs = np.zeros((num_points, 1))

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


# =============================================================================
# Ground Vehicle Trajectories (6-state: X, Y, psi, vx, vy, omega)
# =============================================================================


def lane_change_trajectory(
    initial_speed: float,
    lane_offset: float,
    maneuver_distance: float,
    total_distance: float,
    num_points: int = 100,
) -> Tuple[Array, Array, Array]:
    """Generate lane change trajectory for a 6-state ground vehicle model.

    Creates a smooth lane change maneuver at constant longitudinal speed.
    Uses a sigmoid function for lateral displacement.

    State convention: [X, Y, psi, vx, vy, omega]
    Control convention: [delta, a] steering angle, acceleration

    Args:
        initial_speed: Longitudinal velocity (m/s)
        lane_offset: Lateral displacement for lane change (m)
        maneuver_distance: Distance over which to complete lane change (m)
        total_distance: Total trajectory distance (m)
        num_points: Number of trajectory points

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 6)
        u_refs: Reference controls (num_points, 2)
    """
    duration = total_distance / initial_speed
    times = np.linspace(0, duration, num_points)

    x_refs = np.zeros((num_points, 6))
    u_refs = np.zeros((num_points, 2))

    # Longitudinal position increases linearly with time
    X = initial_speed * times

    # Lateral position follows sigmoid during maneuver
    # Lane change starts at X = (total_distance - maneuver_distance) / 2
    x_start = (total_distance - maneuver_distance) / 2
    x_end = x_start + maneuver_distance

    for i, (t, x) in enumerate(zip(times, X)):
        if x < x_start:
            Y = 0.0
            psi = 0.0
            vy = 0.0
            omega = 0.0
            delta = 0.0
        elif x > x_end:
            Y = lane_offset
            psi = 0.0
            vy = 0.0
            omega = 0.0
            delta = 0.0
        else:
            # Sigmoid transition
            progress = (x - x_start) / maneuver_distance
            smooth = 0.5 * (1 + np.tanh(6 * (progress - 0.5)))

            Y = lane_offset * smooth

            # Approximate heading from path tangent
            # dY/dX = lane_offset * d(smooth)/d(progress) / maneuver_distance
            d_smooth = 3 * (1 - np.tanh(6 * (progress - 0.5))**2) / maneuver_distance
            dY_dX = lane_offset * d_smooth

            psi = np.arctan(dY_dX)

            # Approximate steering for kinematic model (simplified)
            vy = initial_speed * np.sin(psi)
            omega = 0.0  # Approximate
            delta = psi / 3  # Rough approximation

        x_refs[i] = [x, Y, psi, initial_speed, vy, omega]
        u_refs[i] = [delta, 0.0]  # No longitudinal acceleration

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def circular_track_trajectory(
    radius: float,
    speed: float,
    laps: float = 1.0,
    num_points: int = 100,
) -> Tuple[Array, Array, Array]:
    """Generate circular track trajectory for a 6-state ground vehicle model.

    Creates a trajectory following a circular track at constant speed.

    Args:
        radius: Track radius (m)
        speed: Constant longitudinal speed (m/s)
        laps: Number of laps to complete
        num_points: Number of trajectory points

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 6)
        u_refs: Reference controls (num_points, 2)
    """
    circumference = 2 * np.pi * radius
    duration = laps * circumference / speed
    omega_track = speed / radius  # Angular velocity around track

    times = np.linspace(0, duration, num_points)

    # Position around circle
    angle = omega_track * times
    X = radius * np.cos(angle)
    Y = radius * np.sin(angle)

    # Heading tangent to circle (perpendicular to radius)
    psi = angle + np.pi / 2

    # Velocities in body frame
    vx = np.full_like(times, speed)
    vy = np.zeros_like(times)  # No sideslip for ideal tracking
    omega_body = np.full_like(times, omega_track)  # Yaw rate = track angular velocity

    # Steering angle for circular motion (Ackermann approximation)
    # For bicycle model: delta ≈ L / R for small angles
    # This is a rough approximation; actual value depends on model parameters
    delta_approx = 0.33 / radius  # Assume wheelbase ~0.33m

    x_refs = np.stack([X, Y, psi, vx, vy, omega_body], axis=1)
    u_refs = np.stack([np.full_like(times, delta_approx), np.zeros_like(times)], axis=1)

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


def slalom_trajectory(
    initial_speed: float,
    amplitude: float,
    wavelength: float,
    total_distance: float,
    num_points: int = 100,
) -> Tuple[Array, Array, Array]:
    """Generate slalom trajectory for a 6-state ground vehicle model.

    Creates a sinusoidal weaving trajectory.

    Args:
        initial_speed: Longitudinal velocity (m/s)
        amplitude: Lateral amplitude of weave (m)
        wavelength: Distance between peaks (m)
        total_distance: Total trajectory distance (m)
        num_points: Number of trajectory points

    Returns:
        times: Time points (num_points,)
        x_refs: Reference states (num_points, 6)
        u_refs: Reference controls (num_points, 2)
    """
    duration = total_distance / initial_speed
    times = np.linspace(0, duration, num_points)

    X = initial_speed * times
    k = 2 * np.pi / wavelength
    Y = amplitude * np.sin(k * X)

    # Heading from path tangent
    dY_dX = amplitude * k * np.cos(k * X)
    psi = np.arctan(dY_dX)

    # Velocities
    vx = np.full_like(times, initial_speed)
    vy = initial_speed * np.sin(psi)  # Approximation
    omega = np.gradient(psi, times)

    # Steering (approximation)
    delta = psi / 3

    x_refs = np.stack([X, Y, psi, vx, vy, omega], axis=1)
    u_refs = np.stack([delta, np.zeros_like(times)], axis=1)

    return jnp.array(times), jnp.array(x_refs), jnp.array(u_refs)


# =============================================================================
# Utility Functions
# =============================================================================


def hold_final_value(
    times: Array,
    x_refs: Array,
    u_refs: Array,
    extension_duration: float,
    num_extension_points: int = 10,
) -> Tuple[Array, Array, Array]:
    """Extend trajectory by holding final value.

    Useful for ensuring the controller has a defined reference
    after the main trajectory completes.

    Args:
        times: Original time points
        x_refs: Original reference states
        u_refs: Original reference controls
        extension_duration: Duration to extend (s)
        num_extension_points: Number of points to add

    Returns:
        Extended times, x_refs, u_refs
    """
    times_np = np.asarray(times)
    x_refs_np = np.asarray(x_refs)
    u_refs_np = np.asarray(u_refs)

    t_final = times_np[-1]
    extension_times = np.linspace(
        t_final,
        t_final + extension_duration,
        num_extension_points + 1
    )[1:]  # Exclude duplicate first point

    extension_x = np.tile(x_refs_np[-1], (num_extension_points, 1))
    extension_u = np.tile(u_refs_np[-1], (num_extension_points, 1))

    new_times = np.concatenate([times_np, extension_times])
    new_x_refs = np.concatenate([x_refs_np, extension_x])
    new_u_refs = np.concatenate([u_refs_np, extension_u])

    return jnp.array(new_times), jnp.array(new_x_refs), jnp.array(new_u_refs)


def resample_trajectory(
    times: Array,
    x_refs: Array,
    u_refs: Array,
    new_num_points: int,
) -> Tuple[Array, Array, Array]:
    """Resample trajectory to different number of points.

    Uses linear interpolation.

    Args:
        times: Original time points
        x_refs: Original reference states
        u_refs: Original reference controls
        new_num_points: Desired number of points

    Returns:
        Resampled times, x_refs, u_refs
    """
    times_np = np.asarray(times)
    x_refs_np = np.asarray(x_refs)
    u_refs_np = np.asarray(u_refs)

    new_times = np.linspace(times_np[0], times_np[-1], new_num_points)

    # Interpolate each state dimension
    n_states = x_refs_np.shape[1]
    n_controls = u_refs_np.shape[1]

    new_x_refs = np.zeros((new_num_points, n_states))
    new_u_refs = np.zeros((new_num_points, n_controls))

    for i in range(n_states):
        new_x_refs[:, i] = np.interp(new_times, times_np, x_refs_np[:, i])

    for i in range(n_controls):
        new_u_refs[:, i] = np.interp(new_times, times_np, u_refs_np[:, i])

    return jnp.array(new_times), jnp.array(new_x_refs), jnp.array(new_u_refs)
