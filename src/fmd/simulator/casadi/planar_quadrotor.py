"""CasADi Planar Quadrotor (2D) dynamics.

A quadrotor restricted to the x-z plane with pitch rotation.
Two rotors provide thrust forces T1 (right) and T2 (left).

This is the CasADi implementation that exactly matches PlanarQuadrotorJax for
MPC applications and equivalence testing.
"""

from __future__ import annotations

import casadi as cs

from fmd.simulator.casadi.base import CasadiDynamicSystem
from fmd.simulator.params import PlanarQuadrotorParams


class PlanarQuadrotorCasadiExact(CasadiDynamicSystem):
    """Planar quadrotor (2D) dynamics - CasADi implementation.

    Exactly matches PlanarQuadrotorJax for equivalence testing.

    State vector (6 elements):
        [0] x         - Horizontal position (m)
        [1] z         - Vertical position (m), positive = up
        [2] theta     - Pitch angle (rad), positive = nose up
        [3] x_dot     - Horizontal velocity (m/s)
        [4] z_dot     - Vertical velocity (m/s)
        [5] theta_dot - Pitch rate (rad/s)

    Control vector (2 elements):
        [0] T1 - Right rotor thrust (N)
        [1] T2 - Left rotor thrust (N)

    Dynamics:
        Total thrust: T = T1 + T2
        Moment: M = (T1 - T2) * arm_length

        x_ddot = -(T/m) * sin(theta)
        z_ddot = (T/m) * cos(theta) - g
        theta_ddot = M / I

    Attributes:
        mass: Vehicle mass (kg)
        arm_length: Distance from rotor to center of mass (m)
        inertia_pitch: Pitch moment of inertia (kg*m^2)
        g: Gravitational acceleration (m/s^2)
    """

    state_names: tuple[str, ...] = ("x", "z", "theta", "x_dot", "z_dot", "theta_dot")
    control_names: tuple[str, ...] = ("T1", "T2")

    def __init__(self, params: PlanarQuadrotorParams):
        """Initialize planar quadrotor model from parameters.

        Args:
            params: PlanarQuadrotorParams instance with validated model parameters.
        """
        self.mass = params.mass
        self.arm_length = params.arm_length
        self.inertia_pitch = params.inertia_pitch
        self.g = params.g

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute state derivative: dx/dt = f(x, u, t).

        Args:
            x: Current state [x, z, theta, x_dot, z_dot, theta_dot] as CasADi symbolic
            u: Control input [T1, T2] as CasADi symbolic
            t: Current time (unused for this time-invariant system)

        Returns:
            State derivative [x_dot, z_dot, theta_dot, x_ddot, z_ddot, theta_ddot] as cs.SX
        """
        # Extract velocities
        x_dot = x[3]
        z_dot = x[4]
        theta_dot = x[5]
        theta = x[2]

        # Extract thrusts
        T1_val = u[0]
        T2_val = u[1]

        # Total thrust and moment
        T_total = T1_val + T2_val
        moment = (T1_val - T2_val) * self.arm_length

        # Trigonometric functions
        cos_theta = cs.cos(theta)
        sin_theta = cs.sin(theta)

        # Accelerations
        # Thrust acts along body -Z axis (up in body frame)
        # In world frame: ax = -T*sin(theta), az = T*cos(theta)
        x_ddot = -(T_total / self.mass) * sin_theta
        z_ddot = (T_total / self.mass) * cos_theta - self.g
        theta_ddot = moment / self.inertia_pitch

        return cs.vertcat(x_dot, z_dot, theta_dot, x_ddot, z_ddot, theta_ddot)
