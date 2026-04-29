"""Cartpole (inverted pendulum) parameter class.

Immutable, validated parameters for the Cartpole dynamics model.
Based on Barto, Sutton, and Anderson (1983) and OpenAI Gym.
"""

from __future__ import annotations

import attrs
import numpy as np

from fmd.simulator.params.base import is_finite, positive, STANDARD_GRAVITY


@attrs.define(frozen=True, slots=True)
class CartpoleParams:
    """Immutable parameters for Cartpole (inverted pendulum) dynamics model.

    All values must be finite (no NaN/Inf) and positive.

    The Cartpole model consists of a cart that can move horizontally,
    with a pole attached at the pivot point. The goal is typically to
    balance the pole upright by applying horizontal forces to the cart.

    State vector: [x, x_dot, theta, theta_dot]
    - x: cart position (m)
    - x_dot: cart velocity (m/s)
    - theta: pole angle from vertical (rad), positive = clockwise
    - theta_dot: pole angular velocity (rad/s)

    Control: [F] - horizontal force on cart (N)

    Attributes:
        mass_cart: Cart mass (kg). Must be positive.
        mass_pole: Pole mass (kg). Must be positive.
        pole_length: Half-length of pole to center of mass (m). Must be positive.
        g: Gravitational acceleration (m/s^2). Must be positive.

    References:
        Barto, Sutton, Anderson (1983): "Neuronlike adaptive elements that can
        solve difficult learning control problems"

    Example:
        >>> params = CartpoleParams(
        ...     mass_cart=1.0,
        ...     mass_pole=0.1,
        ...     pole_length=0.5,
        ... )
        >>> from fmd.simulator import Cartpole
        >>> cartpole = Cartpole(params)
    """

    mass_cart: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Cart mass"},
    )
    mass_pole: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Pole mass"},
    )
    pole_length: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Half-length to pole center of mass"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )

    @property
    def total_mass(self) -> float:
        """Total system mass (cart + pole) in kg."""
        return self.mass_cart + self.mass_pole

    @property
    def natural_frequency(self) -> float:
        """Linearized natural frequency at unstable equilibrium (rad/s).

        omega = sqrt(g / L)

        This is the frequency of small oscillations about the upright position.
        """
        return np.sqrt(self.g / self.pole_length)

    @property
    def linearized_period(self) -> float:
        """Linearized period at unstable equilibrium (s).

        T = 2 * pi * sqrt(L / g)
        """
        return 2 * np.pi * np.sqrt(self.pole_length / self.g)

    @property
    def mass_ratio(self) -> float:
        """Pole mass to total mass ratio (dimensionless).

        This ratio appears in the equations of motion and affects
        the coupling between cart and pole dynamics.
        """
        return self.mass_pole / self.total_mass

    def with_mass_cart(self, mass_cart: float) -> CartpoleParams:
        """Return new params with updated cart mass.

        Args:
            mass_cart: New cart mass (kg).

        Returns:
            New CartpoleParams instance with updated cart mass.
        """
        return attrs.evolve(self, mass_cart=mass_cart)

    def with_mass_pole(self, mass_pole: float) -> CartpoleParams:
        """Return new params with updated pole mass.

        Args:
            mass_pole: New pole mass (kg).

        Returns:
            New CartpoleParams instance with updated pole mass.
        """
        return attrs.evolve(self, mass_pole=mass_pole)

    def with_pole_length(self, pole_length: float) -> CartpoleParams:
        """Return new params with updated pole length.

        Args:
            pole_length: New pole half-length (m).

        Returns:
            New CartpoleParams instance with updated pole length.
        """
        return attrs.evolve(self, pole_length=pole_length)

    def with_masses(
        self,
        cart: float | None = None,
        pole: float | None = None,
    ) -> CartpoleParams:
        """Return new params with updated masses.

        Only specified values are updated; others remain unchanged.

        Args:
            cart: New cart mass (kg), or None to keep current.
            pole: New pole mass (kg), or None to keep current.

        Returns:
            New CartpoleParams instance with updated mass values.
        """
        return attrs.evolve(
            self,
            mass_cart=cart if cart is not None else self.mass_cart,
            mass_pole=pole if pole is not None else self.mass_pole,
        )
