"""SimplePendulum parameter class.

Immutable, validated parameters for the SimplePendulum dynamics model.
"""

from __future__ import annotations

import numpy as np
import attrs

from fmd.simulator.params.base import STANDARD_GRAVITY, is_finite, positive


@attrs.define(frozen=True, slots=True)
class SimplePendulumParams:
    """Immutable parameters for SimplePendulum dynamics.

    A simple pendulum consists of a point mass on a massless rod swinging
    under gravity. This is a classic 2D nonlinear system.

    Attributes:
        length: Pendulum length (m). Must be positive.
        g: Gravitational acceleration (m/s^2). Must be positive.
            Default is standard gravity (9.80665 m/s^2).

    Example:
        >>> params = SimplePendulumParams(length=1.0)
        >>> from fmd.simulator import SimplePendulum
        >>> pendulum = SimplePendulum(params)
    """

    length: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Pendulum length"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )

    @property
    def period_small_angle(self) -> float:
        """Theoretical period for small angle oscillations: T = 2*pi*sqrt(L/g) (s)."""
        return 2 * np.pi * np.sqrt(self.length / self.g)

    @property
    def natural_frequency(self) -> float:
        """Natural frequency for small angle oscillations: omega = sqrt(g/L) (rad/s)."""
        return np.sqrt(self.g / self.length)

    def with_length(self, length: float) -> SimplePendulumParams:
        """Return new params with updated length.

        Args:
            length: New pendulum length (m).

        Returns:
            New SimplePendulumParams instance with updated length.
        """
        return attrs.evolve(self, length=length)

    def with_gravity(self, g: float) -> SimplePendulumParams:
        """Return new params with updated gravitational acceleration.

        Args:
            g: New gravitational acceleration (m/s^2).

        Returns:
            New SimplePendulumParams instance with updated gravity.
        """
        return attrs.evolve(self, g=g)
