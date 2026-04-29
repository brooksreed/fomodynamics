"""JAX gravity force component.

Implements constant gravitational force in the NED frame.
This is the JAX equivalent of fmd.simulator.components.gravity.Gravity.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import jax.numpy as jnp
from jax import Array
from fmd.simulator.components.base import JaxForceElement
from fmd.simulator.quaternion import rotate_vector_inverse
from fmd.simulator.params.base import STANDARD_GRAVITY


class JaxGravity(JaxForceElement):
    """Gravitational force component (JAX implementation).

    Computes the gravitational force acting on a rigid body.
    In NED frame, gravity acts in the +Z direction (down).

    The force is transformed to body frame using the body's orientation
    quaternion from the state vector.

    Attributes:
        mass: Mass of the body (kg)
        g: Gravitational acceleration (m/s^2), default 9.80665

    Example:
        gravity = JaxGravity(mass=10.0)
        force, moment = gravity.compute(state=state, control=jnp.array([]))
    """

    mass: float
    g: float = STANDARD_GRAVITY

    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute gravitational force in body frame.

        The state vector is expected to have quaternion at indices 6:10
        in the order [qw, qx, qy, qz].

        Args:
            state: State vector with quaternion at [6:10]
            control: Control vector (unused)
            t: Current time (unused, included for interface consistency)

        Returns:
            Tuple of (force, moment) in body frame.
            Gravity produces no moment about the center of mass.
        """
        # Extract quaternion from state
        quat = state[6:10]

        # Gravity vector in NED frame: [0, 0, mg] (down is positive Z)
        gravity_ned = jnp.array([0.0, 0.0, self.mass * self.g])

        # Transform to body frame
        force_body = rotate_vector_inverse(quat, gravity_ned)

        # Gravity produces no moment about CoM
        moment_body = jnp.zeros(3)

        return force_body, moment_body
