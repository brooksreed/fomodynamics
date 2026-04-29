"""JAX simple pendulum dynamics.

A point mass on a massless rod swinging under gravity.
This is the JAX equivalent of fmd.simulator.pendulum.SimplePendulum.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import SimplePendulumParams
from fmd.simulator.params.base import STANDARD_GRAVITY


# State vector indices
THETA = 0  # Angle from vertical (rad)
THETA_DOT = 1  # Angular velocity (rad/s)


class SimplePendulumJax(JaxDynamicSystem):
    """Simple pendulum (point mass on massless rod) - JAX implementation.

    State vector (2 elements):
        [0] theta     - Angle from vertical (rad), positive = clockwise
        [1] theta_dot - Angular velocity (rad/s)

    Equation of motion:
        theta_ddot = -(g/L) sin(theta)

    This is a nonlinear system. For small angles, it approximates
    simple harmonic motion with period T = 2*pi*sqrt(L/g).

    Attributes:
        length: Pendulum length in meters
        g: Gravitational acceleration in m/s^2

    Example:
        from fmd.simulator import SimplePendulumJax, simulate
        from fmd.simulator.params import PENDULUM_1M

        pendulum = SimplePendulumJax(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])  # 0.5 rad initial angle
        result = simulate(pendulum, initial, dt=0.01, duration=10.0)

        # For gradient computation, use from_values to avoid params validation:
        pendulum = SimplePendulumJax.from_values(length=1.0, g=9.81)
    """

    length: float
    g: float = STANDARD_GRAVITY

    # Static metadata
    state_names: tuple[str, ...] = eqx.field(
        static=True, default=("theta", "theta_dot")
    )
    control_names: tuple[str, ...] = eqx.field(static=True, default=())

    def __init__(self, params: SimplePendulumParams):
        """Initialize simple pendulum from parameters.

        Args:
            params: SimplePendulumParams instance with validated parameters.

        Note:
            For JAX gradient computation, use from_values() instead to
            avoid non-JAX operations in params validation.
        """
        self.length = params.length
        self.g = params.g

    @classmethod
    def from_values(cls, length: float, g: float = STANDARD_GRAVITY) -> "SimplePendulumJax":
        """Create pendulum directly from values (JAX-traceable).

        Use this constructor when differentiating through pendulum
        parameters, as it avoids the attrs validation in SimplePendulumParams.

        Args:
            length: Pendulum length in meters
            g: Gravitational acceleration in m/s^2

        Returns:
            SimplePendulumJax instance

        Example:
            def loss(length):
                pendulum = SimplePendulumJax.from_values(length=length)
                result = simulate(pendulum, initial, dt=0.01, duration=1.0)
                return result.states[-1, 0] ** 2

            grad = jax.grad(loss)(1.0)  # Works!
        """
        # Create instance without going through __init__
        obj = object.__new__(cls)
        object.__setattr__(obj, "length", length)
        object.__setattr__(obj, "g", g)
        object.__setattr__(obj, "state_names", ("theta", "theta_dot"))
        object.__setattr__(obj, "control_names", ())
        return obj

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative.

        theta_ddot = -(g/L) * sin(theta)

        Args:
            state: Current state [theta, theta_dot]
            control: Control input (unused for pendulum)
            t: Current time (unused for pendulum)

        Returns:
            State derivative [theta_dot, theta_ddot]
        """
        theta = state[THETA]
        theta_dot = state[THETA_DOT]

        theta_ddot = -(self.g / self.length) * jnp.sin(theta)

        return jnp.array([theta_dot, theta_ddot])

    def default_state(self) -> Array:
        """Return default initial state (hanging at rest)."""
        return jnp.zeros(2)

    def default_control(self) -> Array:
        """Return empty control vector (no control inputs)."""
        return jnp.array([])

    def energy(self, state: Array) -> float:
        """Compute total mechanical energy.

        E = (1/2) * L^2 * theta_dot^2 + g * L * (1 - cos(theta))

        For a unit mass pendulum (m=1).
        This should be conserved (constant) during simulation.

        Args:
            state: Current state [theta, theta_dot]

        Returns:
            Total mechanical energy (kinetic + potential)
        """
        theta = state[THETA]
        theta_dot = state[THETA_DOT]

        # Kinetic energy (unit mass)
        T = 0.5 * self.length**2 * theta_dot**2

        # Potential energy (zero at bottom, unit mass)
        V = self.g * self.length * (1 - jnp.cos(theta))

        return T + V

    def period_small_angle(self) -> float:
        """Theoretical period for small angle oscillations.

        T = 2 * pi * sqrt(L / g)

        Returns:
            Period in seconds
        """
        return 2 * jnp.pi * jnp.sqrt(self.length / self.g)

    def cartesian_position(self, state: Array) -> tuple[float, float]:
        """Compute Cartesian coordinates of the pendulum bob.

        Origin is at the pivot, x is horizontal (right positive),
        y is vertical (up positive).

        Args:
            state: Current state [theta, theta_dot]

        Returns:
            Tuple of (x, y) coordinates
        """
        theta = state[THETA]
        x = self.length * jnp.sin(theta)
        y = -self.length * jnp.cos(theta)
        return x, y

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Position indices for symplectic integration.

        For the pendulum, theta (angle) is the position coordinate.
        """
        return (THETA,)

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Velocity indices for symplectic integration.

        For the pendulum, theta_dot (angular velocity) is the velocity coordinate.
        """
        return (THETA_DOT,)
