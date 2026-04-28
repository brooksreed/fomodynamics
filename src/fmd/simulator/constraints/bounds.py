"""Box and scalar bound constraints.

This module provides simple bound constraints on state or control elements:
- BoxConstraint: Two-sided bounds (lower <= val <= upper)
- ScalarBound: One-sided bounds (val <= bound or val >= bound)

Both support HARD_CLIP enforcement and have symbolic forms for CasADi.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from fmd.simulator.constraints.base import (
    AbstractConstraint,
    Capability,
    ConstraintCategory,
)

if TYPE_CHECKING:
    from jax import Array


class BoxConstraint(AbstractConstraint):
    """Two-sided bound on a single state or control element.

    Constraint: lower <= val <= upper

    Returns 2-element vector: [lower - val, val - upper]
    Both elements <= 0 when satisfied.

    Attributes:
        index: Which element of state/control to constrain
        lower: Lower bound
        upper: Upper bound
        on_state: True for state constraint, False for control

    Example:
        # Constrain state[0] (position) to [-10, 10]
        pos_bound = BoxConstraint("pos_x", index=0, lower=-10.0, upper=10.0)

        # Constrain control[1] (steering) to [-0.5, 0.5]
        steer_bound = BoxConstraint("steering", index=1, lower=-0.5, upper=0.5,
                                     on_state=False)
    """

    index: int
    lower: float
    upper: float
    on_state: bool = True

    def __init__(
        self,
        name: str,
        index: int,
        lower: float,
        upper: float,
        on_state: bool = True,
    ):
        """Create a box constraint.

        Args:
            name: Unique constraint name
            index: Index of element to constrain
            lower: Lower bound
            upper: Upper bound
            on_state: True for state, False for control

        Raises:
            ValueError: If lower > upper
        """
        if lower > upper:
            raise ValueError(f"lower ({lower}) must be <= upper ({upper})")

        cat = (
            ConstraintCategory.STATE_BOUND
            if on_state
            else ConstraintCategory.CONTROL_BOUND
        )
        caps = frozenset({Capability.HARD_CLIP, Capability.HAS_SYMBOLIC_FORM})

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", cat)
        object.__setattr__(self, "capabilities", caps)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "on_state", on_state)

    def value(self, t: float, x: Array, u: Array) -> Array:
        """Evaluate constraint: [lower - val, val - upper].

        Args:
            t: Time (unused, for interface consistency)
            x: State vector
            u: Control vector

        Returns:
            2-element array: both <= 0 when satisfied.
        """
        val = x[self.index] if self.on_state else u[self.index]
        return jnp.array([self.lower - val, val - self.upper])

    def clip(self, t: float, x: Array, u: Array) -> tuple[Array, Array]:
        """Clip the constrained element to [lower, upper].

        Args:
            t: Time (unused)
            x: State vector
            u: Control vector

        Returns:
            Tuple (x_clipped, u_clipped) with constraint enforced.
        """
        if self.on_state:
            x = x.at[self.index].set(jnp.clip(x[self.index], self.lower, self.upper))
        else:
            u = u.at[self.index].set(jnp.clip(u[self.index], self.lower, self.upper))
        return x, u


class ScalarBound(AbstractConstraint):
    """One-sided bound on a single state or control element.

    If is_upper=True:  constraint is val <= bound (returns val - bound)
    If is_upper=False: constraint is val >= bound (returns bound - val)

    Returns scalar: <= 0 when satisfied, > 0 when violated.

    Attributes:
        index: Which element of state/control to constrain
        bound: The bound value
        is_upper: True for upper bound (val <= bound), False for lower
        on_state: True for state constraint, False for control

    Example:
        # Maximum velocity constraint: v <= 10
        max_vel = ScalarBound("max_vel", index=3, bound=10.0, is_upper=True)

        # Minimum throttle: throttle >= 0
        min_throttle = ScalarBound("min_throttle", index=0, bound=0.0,
                                    is_upper=False, on_state=False)
    """

    index: int
    bound: float
    is_upper: bool = True
    on_state: bool = True

    def __init__(
        self,
        name: str,
        index: int,
        bound: float,
        is_upper: bool = True,
        on_state: bool = True,
    ):
        """Create a scalar bound constraint.

        Args:
            name: Unique constraint name
            index: Index of element to constrain
            bound: The bound value
            is_upper: True for val <= bound, False for val >= bound
            on_state: True for state, False for control
        """
        cat = (
            ConstraintCategory.STATE_BOUND
            if on_state
            else ConstraintCategory.CONTROL_BOUND
        )
        caps = frozenset({Capability.HARD_CLIP, Capability.HAS_SYMBOLIC_FORM})

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", cat)
        object.__setattr__(self, "capabilities", caps)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "bound", bound)
        object.__setattr__(self, "is_upper", is_upper)
        object.__setattr__(self, "on_state", on_state)

    def value(self, t: float, x: Array, u: Array) -> Array:
        """Evaluate constraint.

        Args:
            t: Time (unused, for interface consistency)
            x: State vector
            u: Control vector

        Returns:
            Scalar: <= 0 when satisfied.
            - Upper bound: val - bound (<= 0 when val <= bound)
            - Lower bound: bound - val (<= 0 when val >= bound)
        """
        val = x[self.index] if self.on_state else u[self.index]
        if self.is_upper:
            return val - self.bound  # <= 0 when val <= bound
        else:
            return self.bound - val  # <= 0 when val >= bound

    def clip(self, t: float, x: Array, u: Array) -> tuple[Array, Array]:
        """Clip the constrained element to satisfy the bound.

        Args:
            t: Time (unused)
            x: State vector
            u: Control vector

        Returns:
            Tuple (x_clipped, u_clipped) with constraint enforced.
        """
        if self.on_state:
            if self.is_upper:
                x = x.at[self.index].set(jnp.minimum(x[self.index], self.bound))
            else:
                x = x.at[self.index].set(jnp.maximum(x[self.index], self.bound))
        else:
            if self.is_upper:
                u = u.at[self.index].set(jnp.minimum(u[self.index], self.bound))
            else:
                u = u.at[self.index].set(jnp.maximum(u[self.index], self.bound))
        return x, u
