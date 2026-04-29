"""Box1D parameter classes.

Immutable, validated parameters for Box1D dynamics models.
These simple 1D models are useful for testing and validating control algorithms.
"""

from __future__ import annotations

import attrs

from fmd.simulator.params.base import STANDARD_GRAVITY, is_finite, positive, non_negative


@attrs.define(frozen=True, slots=True)
class Box1DParams:
    """Immutable parameters for Box1D dynamics (linear drag only).

    A 1D box sliding with linear drag. This is one of the simplest
    dynamical systems, useful for testing integrators and controllers.

    Dynamics:
        x_ddot = (F - k * x_dot) / m

    where:
        x: position (m)
        x_dot: velocity (m/s)
        F: applied force (N)
        k: drag coefficient (N/(m/s))
        m: mass (kg)

    Attributes:
        mass: Box mass (kg). Must be positive.
        drag_coefficient: Linear drag coefficient (N/(m/s)). Must be non-negative.

    Example:
        >>> params = Box1DParams(mass=1.0, drag_coefficient=0.1)
        >>> from fmd.simulator.casadi import Box1DCasadiExact
        >>> model = Box1DCasadiExact(params)
    """

    mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Box mass"},
    )
    drag_coefficient: float = attrs.field(
        default=0.0,
        validator=[is_finite, non_negative],
        metadata={"unit": "N/(m/s)", "description": "Linear drag coefficient"},
    )

    def with_mass(self, mass: float) -> Box1DParams:
        """Return new params with updated mass.

        Args:
            mass: New mass (kg).

        Returns:
            New Box1DParams instance with updated mass.
        """
        return attrs.evolve(self, mass=mass)

    def with_drag_coefficient(self, drag_coefficient: float) -> Box1DParams:
        """Return new params with updated drag coefficient.

        Args:
            drag_coefficient: New drag coefficient (N/(m/s)).

        Returns:
            New Box1DParams instance with updated drag coefficient.
        """
        return attrs.evolve(self, drag_coefficient=drag_coefficient)


@attrs.define(frozen=True, slots=True)
class Box1DFrictionParams:
    """Immutable parameters for Box1D dynamics with Coulomb friction.

    A 1D box sliding with both linear drag and Coulomb (dry) friction.
    Useful for testing discontinuous dynamics and smooth approximations.

    Dynamics:
        x_ddot = (F - sign(x_dot) * mu * m * g - k * x_dot) / m

    where:
        x: position (m)
        x_dot: velocity (m/s)
        F: applied force (N)
        mu: Coulomb friction coefficient (dimensionless)
        k: linear drag coefficient (N/(m/s))
        m: mass (kg)
        g: gravitational acceleration (m/s^2)

    Attributes:
        mass: Box mass (kg). Must be positive.
        drag_coefficient: Linear drag coefficient (N/(m/s)). Must be non-negative.
        friction_coefficient: Coulomb friction coefficient (dimensionless). Must be non-negative.
        g: Gravitational acceleration (m/s^2). Must be positive.

    Example:
        >>> params = Box1DFrictionParams(
        ...     mass=1.0,
        ...     drag_coefficient=0.1,
        ...     friction_coefficient=0.3,
        ... )
        >>> from fmd.simulator.casadi import Box1DFrictionCasadiExact
        >>> model = Box1DFrictionCasadiExact(params)
    """

    mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Box mass"},
    )
    drag_coefficient: float = attrs.field(
        default=0.0,
        validator=[is_finite, non_negative],
        metadata={"unit": "N/(m/s)", "description": "Linear drag coefficient"},
    )
    friction_coefficient: float = attrs.field(
        default=0.0,
        validator=[is_finite, non_negative],
        metadata={"unit": "dimensionless", "description": "Coulomb friction coefficient"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )

    def with_mass(self, mass: float) -> Box1DFrictionParams:
        """Return new params with updated mass.

        Args:
            mass: New mass (kg).

        Returns:
            New Box1DFrictionParams instance with updated mass.
        """
        return attrs.evolve(self, mass=mass)

    def with_drag_coefficient(self, drag_coefficient: float) -> Box1DFrictionParams:
        """Return new params with updated drag coefficient.

        Args:
            drag_coefficient: New drag coefficient (N/(m/s)).

        Returns:
            New Box1DFrictionParams instance with updated drag coefficient.
        """
        return attrs.evolve(self, drag_coefficient=drag_coefficient)

    def with_friction_coefficient(self, friction_coefficient: float) -> Box1DFrictionParams:
        """Return new params with updated friction coefficient.

        Args:
            friction_coefficient: New Coulomb friction coefficient (dimensionless).

        Returns:
            New Box1DFrictionParams instance with updated friction coefficient.
        """
        return attrs.evolve(self, friction_coefficient=friction_coefficient)

    def with_gravity(self, g: float) -> Box1DFrictionParams:
        """Return new params with updated gravitational acceleration.

        Args:
            g: New gravitational acceleration (m/s^2).

        Returns:
            New Box1DFrictionParams instance with updated gravity.
        """
        return attrs.evolve(self, g=g)


# Default parameter presets
BOX1D_DEFAULT = Box1DParams(mass=1.0, drag_coefficient=0.1)
"""Default Box1D parameters for testing.

A 1 kg box with light drag (0.1 N/(m/s)).

Properties:
    - Time constant: 10 s
    - Steady-state velocity at 1 N force: 10 m/s
"""

BOX1D_FRICTION_DEFAULT = Box1DFrictionParams(
    mass=1.0, drag_coefficient=0.1, friction_coefficient=0.3
)
"""Default Box1D with friction parameters for testing.

A 1 kg box with light drag and moderate friction.

Properties:
    - Time constant (drag only): 10 s
    - Friction force: ~2.94 N (0.3 * 1.0 * 9.81)
    - Minimum force to overcome friction: ~2.94 N
"""
