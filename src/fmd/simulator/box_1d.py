"""JAX 1D box dynamics with drag and friction.

Simple 1D models for testing control algorithms:
- Box1DJax: Linear drag only (viscous friction)
- Box1DFrictionJax: Linear drag + Coulomb (dry) friction
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from typing import Tuple

from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import Box1DParams, Box1DFrictionParams
from fmd.simulator.params.base import STANDARD_GRAVITY


# State vector indices
X = 0  # Position (m)
X_DOT = 1  # Velocity (m/s)

# Control vector indices
FORCE = 0  # Applied force (N)


class Box1DJax(JaxDynamicSystem):
    """1D box with linear drag - JAX implementation.

    State vector (2 elements):
        [0] x     - Position (m)
        [1] x_dot - Velocity (m/s)

    Control vector (1 element):
        [0] F - Applied force (N), positive = rightward

    Dynamics:
        x_ddot = (F - k * x_dot) / m

    where:
        m: mass (kg)
        k: drag coefficient (N·s/m)

    Attributes:
        mass: Box mass (kg)
        drag_coefficient: Linear drag coefficient k (N·s/m)

    Example:
        from fmd.simulator import Box1D, simulate, ConstantControl
        from fmd.simulator.params import BOX1D_DEFAULT
        import jax.numpy as jnp

        box = Box1D(BOX1D_DEFAULT)
        initial = jnp.array([0.0, 0.0])  # At rest
        control = ConstantControl(jnp.array([1.0]))  # 1N force
        result = simulate(box, initial, dt=0.01, duration=5.0, control=control)

        # For gradient computation, use from_values:
        box = Box1D.from_values(mass=1.0, drag_coefficient=0.1)
    """

    mass: float
    drag_coefficient: float

    # Static metadata
    state_names: Tuple[str, ...] = eqx.field(
        static=True, default=("x", "x_dot")
    )
    control_names: Tuple[str, ...] = eqx.field(static=True, default=("F",))

    def __init__(self, params: Box1DParams):
        """Initialize box model from parameters.

        Args:
            params: Box1DParams instance with validated model parameters.

        Note:
            For JAX gradient computation, use from_values() instead to
            avoid non-JAX operations in params validation.
        """
        self.mass = params.mass
        self.drag_coefficient = params.drag_coefficient

    @classmethod
    def from_values(
        cls,
        mass: float,
        drag_coefficient: float,
    ) -> "Box1DJax":
        """Create box directly from values (JAX-traceable).

        Use this constructor when differentiating through model
        parameters, as it avoids the attrs validation in Box1DParams.

        Args:
            mass: Box mass (kg)
            drag_coefficient: Linear drag coefficient (N·s/m)

        Returns:
            Box1DJax instance
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "mass", mass)
        object.__setattr__(obj, "drag_coefficient", drag_coefficient)
        object.__setattr__(obj, "state_names", ("x", "x_dot"))
        object.__setattr__(obj, "control_names", ("F",))
        return obj

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative.

        Args:
            state: Current state [x, x_dot]
            control: Control input [F]
            t: Current time (unused)

        Returns:
            State derivative [x_dot, x_ddot]
        """
        x_dot = state[X_DOT]

        # Handle empty or partial control
        F = jnp.where(control.shape[0] > FORCE, control[FORCE], 0.0)

        # Dynamics: x_ddot = (F - k * x_dot) / m
        x_ddot = (F - self.drag_coefficient * x_dot) / self.mass

        return jnp.array([x_dot, x_ddot])

    def default_state(self) -> Array:
        """Return default initial state (at rest at origin)."""
        return jnp.zeros(2)

    def default_control(self) -> Array:
        """Return default control (no force)."""
        return jnp.zeros(1)

    def equilibrium_velocity(self, force: float) -> float:
        """Compute steady-state velocity for a given constant force.

        At equilibrium: x_ddot = 0
        => F = k * x_dot
        => x_dot = F / k

        Args:
            force: Applied force (N)

        Returns:
            Equilibrium velocity (m/s), or inf if drag_coefficient == 0
        """
        if self.drag_coefficient == 0:
            return jnp.inf
        return force / self.drag_coefficient

    def time_constant(self) -> float:
        """Return the system time constant tau = m / k.

        The velocity approaches equilibrium as v(t) = v_eq * (1 - exp(-t/tau)).

        Returns:
            Time constant (s), or inf if drag_coefficient == 0
        """
        if self.drag_coefficient == 0:
            return jnp.inf
        return self.mass / self.drag_coefficient

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Position indices for symplectic integration.

        For Box1D: x (position).
        """
        return (X,)

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Velocity indices for symplectic integration.

        For Box1D: x_dot (velocity).
        """
        return (X_DOT,)


class Box1DFrictionJax(JaxDynamicSystem):
    """1D box with Coulomb friction and linear drag - JAX implementation.

    State vector (2 elements):
        [0] x     - Position (m)
        [1] x_dot - Velocity (m/s)

    Control vector (1 element):
        [0] F - Applied force (N), positive = rightward

    Dynamics:
        x_ddot = (F - sign(x_dot) * mu * m * g - k * x_dot) / m

    where:
        m: mass (kg)
        k: drag coefficient (N·s/m)
        mu: Coulomb friction coefficient (dimensionless)
        g: gravitational acceleration (m/s^2)

    Note:
        The Coulomb friction term uses jnp.sign(x_dot), which returns 0 when
        x_dot is exactly zero. This provides a simple approximation but does
        not capture static friction (stiction). For more realistic friction
        modeling, consider a smooth approximation like tanh(x_dot / epsilon).

    Attributes:
        mass: Box mass (kg)
        drag_coefficient: Linear drag coefficient k (N·s/m)
        friction_coefficient: Coulomb friction coefficient mu (dimensionless)
        g: Gravitational acceleration (m/s^2)

    Example:
        from fmd.simulator import Box1DFriction, simulate, ConstantControl
        from fmd.simulator.params import BOX1D_FRICTION_DEFAULT
        import jax.numpy as jnp

        box = Box1DFriction(BOX1D_FRICTION_DEFAULT)
        initial = jnp.array([0.0, 1.0])  # Moving at 1 m/s
        control = ConstantControl(jnp.array([0.0]))  # No force
        result = simulate(box, initial, dt=0.01, duration=5.0, control=control)

        # For gradient computation, use from_values:
        box = Box1DFriction.from_values(
            mass=1.0, drag_coefficient=0.1, friction_coefficient=0.3
        )
    """

    mass: float
    drag_coefficient: float
    friction_coefficient: float
    g: float

    # Static metadata
    state_names: Tuple[str, ...] = eqx.field(
        static=True, default=("x", "x_dot")
    )
    control_names: Tuple[str, ...] = eqx.field(static=True, default=("F",))

    def __init__(self, params: Box1DFrictionParams):
        """Initialize box model from parameters.

        Args:
            params: Box1DFrictionParams instance with validated model parameters.

        Note:
            For JAX gradient computation, use from_values() instead to
            avoid non-JAX operations in params validation.
        """
        self.mass = params.mass
        self.drag_coefficient = params.drag_coefficient
        self.friction_coefficient = params.friction_coefficient
        self.g = params.g

    @classmethod
    def from_values(
        cls,
        mass: float,
        drag_coefficient: float,
        friction_coefficient: float,
        g: float = STANDARD_GRAVITY,
    ) -> "Box1DFrictionJax":
        """Create box directly from values (JAX-traceable).

        Use this constructor when differentiating through model
        parameters, as it avoids the attrs validation in Box1DFrictionParams.

        Args:
            mass: Box mass (kg)
            drag_coefficient: Linear drag coefficient (N·s/m)
            friction_coefficient: Coulomb friction coefficient (dimensionless)
            g: Gravitational acceleration (m/s^2)

        Returns:
            Box1DFrictionJax instance
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "mass", mass)
        object.__setattr__(obj, "drag_coefficient", drag_coefficient)
        object.__setattr__(obj, "friction_coefficient", friction_coefficient)
        object.__setattr__(obj, "g", g)
        object.__setattr__(obj, "state_names", ("x", "x_dot"))
        object.__setattr__(obj, "control_names", ("F",))
        return obj

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative.

        Args:
            state: Current state [x, x_dot]
            control: Control input [F]
            t: Current time (unused)

        Returns:
            State derivative [x_dot, x_ddot]
        """
        x_dot = state[X_DOT]

        # Handle empty or partial control
        F = jnp.where(control.shape[0] > FORCE, control[FORCE], 0.0)

        # Coulomb friction force: -sign(x_dot) * mu * m * g
        friction_force = jnp.sign(x_dot) * self.friction_coefficient * self.mass * self.g

        # Linear drag force: k * x_dot
        drag_force = self.drag_coefficient * x_dot

        # Dynamics: x_ddot = (F - friction - drag) / m
        x_ddot = (F - friction_force - drag_force) / self.mass

        return jnp.array([x_dot, x_ddot])

    def default_state(self) -> Array:
        """Return default initial state (at rest at origin)."""
        return jnp.zeros(2)

    def default_control(self) -> Array:
        """Return default control (no force)."""
        return jnp.zeros(1)

    def friction_force(self) -> float:
        """Return the maximum Coulomb friction force magnitude.

        F_friction = mu * m * g

        Returns:
            Maximum friction force (N)
        """
        return self.friction_coefficient * self.mass * self.g

    def minimum_force_to_move(self) -> float:
        """Return the minimum force required to overcome static friction.

        This is the friction force magnitude mu * m * g. Any force larger
        than this (in magnitude) will cause the box to accelerate.

        Note: This model uses jnp.sign for Coulomb friction, which doesn't
        capture true stiction. The minimum force is the same as the kinetic
        friction force.

        Returns:
            Minimum force to overcome friction (N)
        """
        return self.friction_force()

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Position indices for symplectic integration.

        For Box1DFriction: x (position).
        """
        return (X,)

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Velocity indices for symplectic integration.

        For Box1DFriction: x_dot (velocity).
        """
        return (X_DOT,)
