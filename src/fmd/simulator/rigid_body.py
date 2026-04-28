"""JAX 6-DOF Rigid Body dynamics.

Implements full 6 degrees of freedom rigid body dynamics using
quaternion attitude representation in NED frame - JAX implementation.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from typing import Tuple, List

from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.quaternion import (
    quat_derivative,
    quat_normalize,
    quat_to_euler,
    rotate_vector,
)
from fmd.simulator.components.base import JaxForceElement


# State vector indices
POS_N = 0   # North position
POS_E = 1   # East position
POS_D = 2   # Down position
VEL_U = 3   # Body-frame forward velocity
VEL_V = 4   # Body-frame right velocity
VEL_W = 5   # Body-frame down velocity
QUAT_W = 6  # Quaternion scalar
QUAT_X = 7  # Quaternion x
QUAT_Y = 8  # Quaternion y
QUAT_Z = 9  # Quaternion z
OMEGA_P = 10  # Roll rate
OMEGA_Q = 11  # Pitch rate
OMEGA_R = 12  # Yaw rate

NUM_STATES = 13


class RigidBody6DOFJax(JaxDynamicSystem):
    """6 degrees of freedom rigid body dynamics - JAX implementation.

    State vector (13 elements):
        [0:3]   pos_ned    - Position in NED frame [N, E, D] (m)
        [3:6]   vel_body   - Velocity in body frame [u, v, w] (m/s)
        [6:10]  quat       - Quaternion [qw, qx, qy, qz]
        [10:13] omega_body - Angular velocity in body frame [p, q, r] (rad/s)

    Equations of motion:
        ṗ = R(q) * v           (position derivative)
        v̇ = F/m - ω × v        (velocity derivative)
        q̇ = 0.5 * Ω ⊗ q        (quaternion derivative)
        ω̇ = I⁻¹(M - ω × Iω)    (angular velocity derivative)

    Uses the "Force Accumulator" pattern where external components
    provide forces and moments.

    Attributes:
        mass: Body mass (kg)
        inertia: 3x3 inertia tensor in body frame (kg·m²)
        inertia_inv: Inverse of inertia tensor
        components: Tuple of JaxForceElement components
    """

    mass: float
    inertia: Array
    inertia_inv: Array
    components: Tuple[JaxForceElement, ...]

    # Static metadata
    state_names: Tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "pos_n", "pos_e", "pos_d",
            "vel_u", "vel_v", "vel_w",
            "qw", "qx", "qy", "qz",
            "omega_p", "omega_q", "omega_r",
        ),
    )
    control_names: Tuple[str, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        mass: float,
        inertia: Array,
        components: List[JaxForceElement] | Tuple[JaxForceElement, ...] = None,
    ):
        """Initialize 6-DOF rigid body.

        Args:
            mass: Body mass (kg)
            inertia: 3x3 inertia tensor in body frame (kg·m²)
                    or 3-element array [Ixx, Iyy, Izz] for diagonal inertia
            components: List of JaxForceElement components (e.g., JaxGravity)
        """
        self.mass = mass

        # Handle diagonal inertia shorthand
        inertia = jnp.asarray(inertia)
        if inertia.shape == (3,):
            self.inertia = jnp.diag(inertia)
        else:
            self.inertia = inertia

        self.inertia_inv = jnp.linalg.inv(self.inertia)
        self.components = tuple(components) if components else ()

    @classmethod
    def from_values(
        cls,
        mass: float,
        inertia: Array,
        components: Tuple[JaxForceElement, ...] = (),
    ) -> "RigidBody6DOFJax":
        """Create rigid body directly from values (JAX-traceable).

        Use this constructor when differentiating through body parameters,
        as it avoids validation overhead.

        Args:
            mass: Body mass (kg)
            inertia: 3x3 inertia tensor or [Ixx, Iyy, Izz] diagonal
            components: Tuple of JaxForceElement components

        Returns:
            RigidBody6DOFJax instance
        """
        obj = object.__new__(cls)

        # Handle diagonal inertia
        inertia = jnp.asarray(inertia)
        if inertia.shape == (3,):
            inertia = jnp.diag(inertia)

        object.__setattr__(obj, "mass", mass)
        object.__setattr__(obj, "inertia", inertia)
        object.__setattr__(obj, "inertia_inv", jnp.linalg.inv(inertia))
        object.__setattr__(obj, "components", tuple(components))
        object.__setattr__(
            obj,
            "state_names",
            (
                "pos_n", "pos_e", "pos_d",
                "vel_u", "vel_v", "vel_w",
                "qw", "qx", "qy", "qz",
                "omega_p", "omega_q", "omega_r",
            ),
        )
        object.__setattr__(obj, "control_names", ())
        return obj

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative ẋ = f(x, u, t).

        Implements the 6-DOF equations of motion.

        Args:
            state: Current state vector (13 elements)
            control: Control input vector
            t: Current time (default 0.0)
            env: Optional Environment with wave/wind/current fields.

        Returns:
            State derivative vector (13 elements)
        """
        # Extract state components
        vel_body = state[VEL_U:VEL_W + 1]
        quat = state[QUAT_W:QUAT_Z + 1]
        omega = state[OMEGA_P:OMEGA_R + 1]

        # Accumulate forces and moments from components
        total_force = jnp.zeros(3)
        total_moment = jnp.zeros(3)

        for component in self.components:
            force, moment = component.compute(state, control, t, env=env)
            total_force = total_force + force
            total_moment = total_moment + moment

        # Position derivative: ṗ = R(q) * v
        # Transform body velocity to NED frame
        pos_dot = rotate_vector(quat, vel_body)

        # Velocity derivative: v̇ = F/m - ω × v
        # (Coriolis term from rotating reference frame)
        vel_dot = total_force / self.mass - jnp.cross(omega, vel_body)

        # Quaternion derivative: q̇ = 0.5 * Ω ⊗ q
        quat_dot = quat_derivative(quat, omega)

        # Angular velocity derivative: ω̇ = I⁻¹(M - ω × Iω)
        # (Euler's equation for rigid body rotation)
        omega_dot = self.inertia_inv @ (
            total_moment - jnp.cross(omega, self.inertia @ omega)
        )

        # Assemble state derivative
        return jnp.concatenate([
            pos_dot,
            vel_dot,
            quat_dot,
            omega_dot,
        ])

    def default_state(self) -> Array:
        """Return default initial state (at origin, no velocity, level attitude)."""
        return jnp.array([
            0.0, 0.0, 0.0,      # position
            0.0, 0.0, 0.0,      # velocity
            1.0, 0.0, 0.0, 0.0,  # quaternion (identity)
            0.0, 0.0, 0.0,      # angular velocity
        ])

    def default_control(self) -> Array:
        """Return default control (empty)."""
        return jnp.zeros(0)

    def post_step(self, state: Array) -> Array:
        """Normalize quaternion after integration step.

        Args:
            state: State vector after integration step

        Returns:
            State with normalized quaternion
        """
        quat = state[QUAT_W:QUAT_Z + 1]
        quat_normalized = quat_normalize(quat)
        return state.at[QUAT_W:QUAT_Z + 1].set(quat_normalized)

    def get_euler_angles(self, state: Array) -> Array:
        """Extract Euler angles from state.

        Args:
            state: State vector

        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        quat = state[QUAT_W:QUAT_Z + 1]
        return quat_to_euler(quat)

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Position indices for symplectic integration.

        For 6-DOF rigid body:
        - Position NED (indices 0, 1, 2)
        - Quaternion attitude (indices 6, 7, 8, 9)

        Both are "position-like" in that they are updated from velocities.
        The quaternion evolves according to angular velocity via q_dot = 0.5 * omega tensor * q.
        """
        return (POS_N, POS_E, POS_D, QUAT_W, QUAT_X, QUAT_Y, QUAT_Z)

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Velocity indices for symplectic integration.

        For 6-DOF rigid body:
        - Body velocity (indices 3, 4, 5)
        - Angular velocity (indices 10, 11, 12)

        These are updated first based on forces/moments, then positions are updated.
        """
        return (VEL_U, VEL_V, VEL_W, OMEGA_P, OMEGA_Q, OMEGA_R)


def create_state_jax(
    position: Array = None,
    velocity: Array = None,
    quaternion: Array = None,
    angular_velocity: Array = None,
) -> Array:
    """Helper to create a 6-DOF state vector.

    Args:
        position: NED position [N, E, D] in meters
        velocity: Body velocity [u, v, w] in m/s
        quaternion: Attitude [qw, qx, qy, qz]
        angular_velocity: Body rates [p, q, r] in rad/s

    Returns:
        13-element state vector
    """
    pos = position if position is not None else jnp.zeros(3)
    vel = velocity if velocity is not None else jnp.zeros(3)
    quat = quaternion if quaternion is not None else jnp.array([1.0, 0.0, 0.0, 0.0])
    omega = angular_velocity if angular_velocity is not None else jnp.zeros(3)

    return jnp.concatenate([pos, vel, quat, omega])
