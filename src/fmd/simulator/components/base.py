"""Base class for JAX force and moment components.

JaxForceElement is the JAX equivalent of blur.simulator.components.ForceElement.
All components are Equinox modules (PyTrees) that can be JIT-compiled and
differentiated.

The signature convention is (state, control, t=0.0) with time as an optional
parameter, since most robotic systems are time-invariant.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from abc import abstractmethod


class JaxForceElement(eqx.Module):
    """Abstract base class for JAX force/moment generating components.

    Each JaxForceElement computes forces and moments acting on a rigid body.
    Forces and moments are returned in the body frame.

    All methods must be JIT-compatible: no Python control flow,
    no side effects, pure JAX operations only.

    Conventions:
        - Coordinate frame: NED (North-East-Down)
        - Forces/moments in body frame
        - State indices: POS 0-2, VEL 3-5, QUAT 6-9, OMEGA 10-12
        - Signature: (state, control, t=0.0) with time optional

    Example:
        class MyForce(JaxForceElement):
            strength: float

            def compute(self, state, control, t=0.0):
                force = jnp.array([self.strength, 0.0, 0.0])
                moment = jnp.zeros(3)
                return force, moment
    """

    @abstractmethod
    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute force and moment vectors in body frame.

        This method must be JIT-compatible: no Python control flow,
        no side effects, pure JAX operations only.

        Args:
            state: Full state vector of the rigid body (13 elements for 6-DOF)
            control: Control input vector
            t: Current simulation time (default 0.0 for time-invariant systems)
            env: Optional Environment with wave/wind/current fields.

        Returns:
            Tuple of (force, moment) vectors in body frame coordinates.
            Force: [Fx, Fy, Fz] in Newtons, shape (3,)
            Moment: [Mx, My, Mz] in Newton-meters, shape (3,)
        """
        pass

    @staticmethod
    def zero_force_moment() -> tuple[Array, Array]:
        """Return zero force and moment vectors.

        Convenience method for components that need to return zeros.

        Returns:
            Tuple of zero force and moment vectors, each shape (3,)
        """
        return jnp.zeros(3), jnp.zeros(3)
