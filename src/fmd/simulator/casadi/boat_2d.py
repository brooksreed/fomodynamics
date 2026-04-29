"""CasADi 2D planar boat dynamics with Coriolis coupling.

A physically-consistent 3-DOF maneuvering model (Fossen-style, without added mass).
Body-frame velocities with proper rotational coupling terms.

This is the CasADi implementation that exactly matches Boat2DJax for
MPC applications and equivalence testing.
"""

from __future__ import annotations

import casadi as cs

from fmd.simulator.casadi.base import CasadiDynamicSystem
from fmd.simulator.params import Boat2DParams


class Boat2DCasadiExact(CasadiDynamicSystem):
    """2D planar boat with coupled rigid-body dynamics - CasADi implementation.

    Exactly matches Boat2DJax for equivalence testing.

    State vector (6 elements):
        [0] x   - North position (m)
        [1] y   - East position (m)
        [2] psi - Heading angle from North, CW positive (rad), wrapped to [-π, π]
        [3] u   - Surge velocity in body +x (m/s)
        [4] v   - Sway velocity in body +y (m/s)
        [5] r   - Yaw rate about body +z (rad/s)

    Control vector (2 elements):
        [0] thrust     - Forward thrust force (N)
        [1] yaw_moment - Yaw moment about z (N·m)

    Equations of motion (coupled planar rigid body):
        x_dot   = u·cos(psi) - v·sin(psi)
        y_dot   = u·sin(psi) + v·cos(psi)
        psi_dot = r
        u_dot   = thrust/m - (D_u/m)·u + r·v      # Coriolis: +rv
        v_dot   = -(D_v/m)·v - r·u                # Coriolis: -ru
        r_dot   = yaw_moment/Izz - (D_r/Izz)·r

    Attributes:
        mass: Vehicle mass (kg)
        izz: Yaw moment of inertia (kg*m^2)
        drag_surge: Surge damping coefficient (kg/s)
        drag_sway: Sway damping coefficient (kg/s)
        drag_yaw: Yaw damping coefficient (kg*m^2/s)
    """

    state_names: tuple[str, ...] = ("x", "y", "psi", "u", "v", "r")
    control_names: tuple[str, ...] = ("thrust", "yaw_moment")

    def __init__(self, params: Boat2DParams):
        """Initialize boat model from parameters.

        Args:
            params: Boat2DParams instance with validated model parameters.
        """
        self.mass = params.mass
        self.izz = params.izz
        self.drag_surge = params.drag_surge
        self.drag_sway = params.drag_sway
        self.drag_yaw = params.drag_yaw

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute state derivative with Coriolis coupling.

        Args:
            x: Current state [x, y, psi, u, v, r] as CasADi symbolic
            u: Control input [thrust, yaw_moment] as CasADi symbolic
            t: Current time (unused for this time-invariant system)

        Returns:
            State derivative [x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot] as cs.SX
        """
        # Extract state
        psi = x[2]
        u_vel = x[3]  # surge velocity (renamed to avoid conflict with control u)
        v_vel = x[4]  # sway velocity
        r = x[5]      # yaw rate

        # Extract control
        thrust = u[0]
        yaw_moment = u[1]

        cos_psi = cs.cos(psi)
        sin_psi = cs.sin(psi)

        # Kinematics (body → NED)
        x_dot = u_vel * cos_psi - v_vel * sin_psi
        y_dot = u_vel * sin_psi + v_vel * cos_psi
        psi_dot = r

        # Dynamics with Coriolis coupling
        u_dot = thrust / self.mass - (self.drag_surge / self.mass) * u_vel + r * v_vel
        v_dot = -(self.drag_sway / self.mass) * v_vel - r * u_vel
        r_dot = yaw_moment / self.izz - (self.drag_yaw / self.izz) * r

        return cs.vertcat(x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot)

    def post_step(self, x: cs.SX) -> cs.SX:
        """Post-process state: wrap heading to [-pi, pi].

        This matches Boat2DJax.post_step exactly using arctan2.

        Args:
            x: State after integration step

        Returns:
            State with wrapped heading angle
        """
        psi = x[2]
        psi_wrapped = cs.arctan2(cs.sin(psi), cs.cos(psi))

        # Reconstruct state with wrapped psi
        return cs.vertcat(x[0], x[1], psi_wrapped, x[3], x[4], x[5])
