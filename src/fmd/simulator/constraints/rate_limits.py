"""Rate limit constraints for control inputs.

Rate limits constrain how fast control inputs can change: |du/dt| <= max_rate.
These constraints depend on the previous control value and are useful for
modeling actuator slew rate limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp

from fmd.simulator.constraints.base import (
    AbstractConstraint,
    Capability,
    ConstraintCategory,
)

if TYPE_CHECKING:
    from jax import Array


class ControlRateLimit(AbstractConstraint):
    """Rate limit on a single control channel.

    Constrains control rate of change, supporting asymmetric up/down limits.

    Constraint Formula:
        rate = (u[i] - u_prev[i]) / dt

        Returns 2-element vector:
            g = [rate - max_rate_up, -rate - max_rate_down]

        For the constraint to be satisfied (g <= 0):
            - g[0] <= 0 requires: rate <= max_rate_up  (increasing rate limit)
            - g[1] <= 0 requires: rate >= -max_rate_down  (decreasing rate limit)

        Combined: -max_rate_down <= rate <= max_rate_up

    Sign convention (matching all fomodynamics constraints):
        value <= 0: satisfied (rate within limits)
        value > 0: violated (rate exceeds limit)

    Example:
        # Symmetric 1.0 rad/s rate limit on control channel 0
        rate_limit = ControlRateLimit.symmetric(
            name="rudder_rate", index=0, max_rate=1.0, dt_default=0.02
        )

        # Evaluate with previous control
        u_prev = jnp.array([0.0])
        u = jnp.array([0.5])  # Trying to change by 0.5 in one step

        # With dt=0.02, rate = 0.5/0.02 = 25 rad/s, violates limit of 1.0
        value = rate_limit.value_with_prev(0.0, state, u, u_prev, dt=0.02)
        # value = [25 - 1, -25 - 1] = [24, -26], max > 0 means violated

    Attributes:
        index: Control channel index to constrain
        max_rate_up: Maximum rate for increasing control (positive)
        max_rate_down: Maximum rate for decreasing control (positive)
        dt_default: Optional default timestep if not provided at evaluation
    """

    index: int
    max_rate_up: float
    max_rate_down: float
    dt_default: Optional[float] = None

    def __init__(
        self,
        name: str,
        index: int,
        max_rate_up: float,
        max_rate_down: float,
        dt_default: Optional[float] = None,
    ):
        """Create a control rate limit constraint.

        Args:
            name: Unique identifier for this constraint
            index: Control channel index to constrain
            max_rate_up: Maximum rate for increasing control (must be positive)
            max_rate_down: Maximum rate for decreasing control (must be positive)
            dt_default: Optional default timestep

        Raises:
            ValueError: If max_rate_up or max_rate_down are not positive
        """
        if max_rate_up <= 0:
            raise ValueError(f"max_rate_up must be positive, got {max_rate_up}")
        if max_rate_down <= 0:
            raise ValueError(f"max_rate_down must be positive, got {max_rate_down}")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", ConstraintCategory.RATE_LIMIT)
        object.__setattr__(self, "capabilities", frozenset([Capability.HAS_RATE_LIMIT]))
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "max_rate_up", max_rate_up)
        object.__setattr__(self, "max_rate_down", max_rate_down)
        object.__setattr__(self, "dt_default", dt_default)

    @classmethod
    def symmetric(
        cls,
        name: str,
        index: int,
        max_rate: float,
        dt_default: Optional[float] = None,
    ) -> "ControlRateLimit":
        """Create symmetric rate limit (same up/down rate).

        Args:
            name: Unique identifier
            index: Control channel index
            max_rate: Maximum rate magnitude (same for increasing/decreasing)
            dt_default: Optional default timestep

        Returns:
            ControlRateLimit with equal up/down rates
        """
        return cls(
            name=name,
            index=index,
            max_rate_up=max_rate,
            max_rate_down=max_rate,
            dt_default=dt_default,
        )

    def value(self, t: float, x: Array, u: Array) -> Array:
        """Standard value() is not supported for rate limits.

        Rate limits require previous control, use value_with_prev() instead.

        Raises:
            NotImplementedError: Always, since rate limits need u_prev.
        """
        raise NotImplementedError(
            f"{self.name} is a rate limit constraint and requires u_prev. "
            "Use value_with_prev(t, x, u, u_prev, dt) instead."
        )

    def value_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float | None = None
    ) -> Array:
        """Evaluate rate limit constraint with previous control.

        Args:
            t: Time (unused, for interface consistency)
            x: State vector (unused, for interface consistency)
            u: Current control vector
            u_prev: Previous control vector
            dt: Timestep (uses dt_default if not provided)

        Returns:
            2-element vector: [rate - max_rate_up, -rate - max_rate_down]
            Both <= 0 when feasible.

        Raises:
            ValueError: If dt is None and dt_default is None
        """
        dt_actual = dt if dt is not None else self.dt_default
        if dt_actual is None:
            raise ValueError("dt must be provided or dt_default must be set")

        delta = u[self.index] - u_prev[self.index]
        rate = delta / dt_actual
        return jnp.array([rate - self.max_rate_up, -rate - self.max_rate_down])

    def clip_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float | None = None
    ) -> tuple[Array, Array]:
        """Clip control to respect rate limits.

        Args:
            t: Time (unused)
            x: State vector (returned unchanged)
            u: Current control vector
            u_prev: Previous control vector
            dt: Timestep (uses dt_default if not provided)

        Returns:
            Tuple (x, u_clipped) where u_clipped respects rate limits.

        Raises:
            ValueError: If dt is None and dt_default is None
        """
        dt_actual = dt if dt is not None else self.dt_default
        if dt_actual is None:
            raise ValueError("dt must be provided or dt_default must be set")

        u_min = u_prev[self.index] - self.max_rate_down * dt_actual
        u_max = u_prev[self.index] + self.max_rate_up * dt_actual
        u_clipped = u.at[self.index].set(jnp.clip(u[self.index], u_min, u_max))
        return x, u_clipped
