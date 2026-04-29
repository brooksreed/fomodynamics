"""Core constraint abstractions for fomodynamics.

This module defines the base interfaces for constraints used in
optimization and control. Constraints represent inequalities g(t,x,u) <= 0.

Constraints are NOT forces. For penalty forces and contact dynamics,
use JaxForceElement components in fmd.simulator.components.

Design Principles:
- Sign convention: value() <= 0 means satisfied, > 0 means violated
- Capabilities: typed enum for what enforcement methods a constraint supports
- Categories: semantic organization (not used for dispatch)
- Shape: value() returns () or (k,) array, each element is an inequality
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Sequence

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


class Capability(Enum):
    """Enforcement capabilities a constraint may support.

    These represent specific enforcement methods that require bespoke
    implementations. Generic operations like quadratic_penalty work on
    any constraint with value() and are NOT capabilities.

    Attributes:
        HARD_CLIP: Can enforce via jnp.clip (box constraints)
        PROJECTION: Can project state back to feasible set
        HAS_SYMBOLIC_FORM: Has CasADi symbolic implementation
        SYMBOLIC_EXACT: Symbolic form is exact (not an approximation)
        HAS_RATE_LIMIT: Requires u_prev for evaluation
    """

    HARD_CLIP = "hard_clip"
    PROJECTION = "projection"
    HAS_SYMBOLIC_FORM = "has_symbolic_form"
    SYMBOLIC_EXACT = "symbolic_exact"
    HAS_RATE_LIMIT = "has_rate_limit"


class ConstraintCategory(Enum):
    """Semantic categories for constraint organization.

    Used for filtering and documentation, NOT for dispatch.
    Dispatch should use Capability checks instead.

    Attributes:
        STATE_BOUND: Bounds on state elements (x_min <= x <= x_max)
        CONTROL_BOUND: Bounds on control inputs (u_min <= u <= u_max)
        RATE_LIMIT: Rate constraints (|du/dt| <= max_rate)
        PHYSICAL: Environmental constraints (surface, collision)
        KINEMATIC: Joint limits, workspace bounds
        SAFETY: Keep-out zones, stability margins
    """

    STATE_BOUND = "state_bound"
    CONTROL_BOUND = "control_bound"
    RATE_LIMIT = "rate_limit"
    PHYSICAL = "physical"
    KINEMATIC = "kinematic"
    SAFETY = "safety"


class AbstractConstraint(eqx.Module):
    """Base class for all constraints.

    Constraints represent inequalities of the form g(t, x, u) <= 0.
    They are used for optimization/control, NOT for generating forces.

    Subclasses must implement value(). Optional methods (clip, project)
    should be implemented if corresponding Capability is declared.

    Shape Semantics:
        - value() returns Array with shape () or (k,)
        - Each element g_i is an inequality: g_i <= 0 means satisfied
        - Violation is relu(value): max(value, 0)
        - Feasibility: all(value <= tol)

    Attributes:
        name: Unique identifier for this constraint
        category: Semantic category (for organization)
        capabilities: What enforcement methods this constraint supports
    """

    name: str
    category: ConstraintCategory = eqx.field(static=True)
    capabilities: frozenset[Capability] = eqx.field(static=True)

    @abstractmethod
    def value(self, t: float, x: Array, u: Array) -> Array:
        """Evaluate constraint inequality.

        Args:
            t: Time (for time-varying constraints)
            x: State vector
            u: Control vector

        Returns:
            Array of shape () or (k,) where each element g_i <= 0
            means satisfied, g_i > 0 means violated.
        """
        ...

    def clip(self, t: float, x: Array, u: Array) -> tuple[Array, Array]:
        """Clip state/control to feasible region.

        Requires Capability.HARD_CLIP.

        Args:
            t: Time
            x: State vector
            u: Control vector

        Returns:
            Tuple (x_clipped, u_clipped) satisfying the constraint.

        Raises:
            NotImplementedError: If constraint doesn't support HARD_CLIP.
        """
        raise NotImplementedError(f"{self.name} does not support clip")

    def project(self, t: float, x: Array, u: Array) -> tuple[Array, Array]:
        """Project state/control onto feasible set.

        Requires Capability.PROJECTION.

        Args:
            t: Time
            x: State vector
            u: Control vector

        Returns:
            Tuple (x_proj, u_proj) on the feasible set boundary or interior.

        Raises:
            NotImplementedError: If constraint doesn't support PROJECTION.
        """
        raise NotImplementedError(f"{self.name} does not support project")

    def has_capability(self, cap: Capability) -> bool:
        """Check if this constraint supports a capability.

        Args:
            cap: Capability to check

        Returns:
            True if the capability is supported.
        """
        return cap in self.capabilities

    def value_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float | None = None
    ) -> Array:
        """Evaluate constraint requiring previous control.

        Override this for rate limit constraints. Default raises NotImplementedError.

        Args:
            t: Time
            x: State vector
            u: Current control vector
            u_prev: Previous control vector
            dt: Timestep (optional, may be required by some implementations)

        Returns:
            Array of shape () or (k,) where each element g_i <= 0
            means satisfied, g_i > 0 means violated.

        Raises:
            NotImplementedError: If constraint doesn't support this method.
        """
        raise NotImplementedError(f"{self.name} does not support value_with_prev")

    def clip_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float | None = None
    ) -> tuple[Array, Array]:
        """Clip state/control requiring previous control.

        Override this for rate limit constraints. Default raises NotImplementedError.

        Args:
            t: Time
            x: State vector
            u: Current control vector
            u_prev: Previous control vector
            dt: Timestep (optional, may be required by some implementations)

        Returns:
            Tuple (x_clipped, u_clipped) satisfying the constraint.

        Raises:
            NotImplementedError: If constraint doesn't support this method.
        """
        raise NotImplementedError(f"{self.name} does not support clip_with_prev")


class ConstraintSet(eqx.Module):
    """Collection of constraints with query and evaluation methods.

    Provides utilities for filtering by category/capability, evaluating
    all constraints, and computing aggregate violations.

    Constraint names must be unique within a set (enforced at construction).

    Example:
        constraints = ConstraintSet([
            BoxConstraint("pos_x", index=0, lower=-10, upper=10),
            ScalarBound("vel_max", index=3, bound=5.0, is_upper=True),
        ])

        # Check feasibility
        if constraints.is_feasible(t, state, control):
            ...

        # Get all clippable constraints
        clippable = constraints.by_capability(Capability.HARD_CLIP)
    """

    constraints: tuple[AbstractConstraint, ...] = eqx.field(static=True)

    def __init__(self, constraints: Sequence[AbstractConstraint]):
        """Create a ConstraintSet from a sequence of constraints.

        Args:
            constraints: Sequence of constraints (names must be unique)

        Raises:
            ValueError: If constraint names are not unique.
        """
        names = [c.name for c in constraints]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate constraint names: {set(dupes)}")
        object.__setattr__(self, "constraints", tuple(constraints))

    def __len__(self) -> int:
        """Return number of constraints in the set."""
        return len(self.constraints)

    def __iter__(self):
        """Iterate over constraints."""
        return iter(self.constraints)

    def by_category(self, cat: ConstraintCategory) -> list[AbstractConstraint]:
        """Filter constraints by semantic category.

        Args:
            cat: Category to filter by

        Returns:
            List of constraints in that category.
        """
        return [c for c in self.constraints if c.category == cat]

    def by_capability(self, cap: Capability) -> list[AbstractConstraint]:
        """Filter constraints by enforcement capability.

        Args:
            cap: Capability to filter by

        Returns:
            List of constraints supporting that capability.
        """
        return [c for c in self.constraints if c.has_capability(cap)]

    def all_values(self, t: float, x: Array, u: Array) -> dict[str, Array]:
        """Evaluate all constraints.

        Args:
            t: Time
            x: State vector
            u: Control vector

        Returns:
            Dictionary mapping constraint names to their values.
        """
        return {c.name: c.value(t, x, u) for c in self.constraints}

    def max_violation(self, t: float, x: Array, u: Array) -> Array:
        """Compute maximum violation across all constraints.

        Violation is max(value, 0) for each constraint element.

        Args:
            t: Time
            x: State vector
            u: Control vector

        Returns:
            Scalar: maximum violation (0 if all satisfied).
        """
        if not self.constraints:
            return jnp.array(0.0)
        violations = []
        for c in self.constraints:
            val = c.value(t, x, u)
            # Flatten and take max of relu
            violations.append(jnp.max(jnp.maximum(val.ravel(), 0.0)))
        return jnp.max(jnp.array(violations))

    def is_feasible(self, t: float, x: Array, u: Array, tol: float = 0.0) -> Array:
        """Check if all constraints are satisfied within tolerance.

        Args:
            t: Time
            x: State vector
            u: Control vector
            tol: Tolerance for constraint satisfaction (default 0)

        Returns:
            Boolean array: True if max_violation <= tol.
        """
        return self.max_violation(t, x, u) <= tol

    def all_values_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float
    ) -> dict[str, Array]:
        """Evaluate all constraints including rate limits.

        For regular constraints, uses value(). For rate limit constraints
        (those with Capability.HAS_RATE_LIMIT), uses value_with_prev().

        Args:
            t: Time
            x: State vector
            u: Current control vector
            u_prev: Previous control vector (for rate limits)
            dt: Timestep (for rate limits)

        Returns:
            Dictionary mapping constraint names to their values.
        """
        result = {}
        for c in self.constraints:
            if c.has_capability(Capability.HAS_RATE_LIMIT):
                result[c.name] = c.value_with_prev(t, x, u, u_prev, dt)
            else:
                result[c.name] = c.value(t, x, u)
        return result

    def max_violation_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float
    ) -> Array:
        """Compute maximum violation including rate limit constraints.

        Violation is max(value, 0) for each constraint element.
        Uses value_with_prev() for rate limit constraints.

        Args:
            t: Time
            x: State vector
            u: Current control vector
            u_prev: Previous control vector (for rate limits)
            dt: Timestep (for rate limits)

        Returns:
            Scalar: maximum violation (0 if all satisfied).
        """
        if not self.constraints:
            return jnp.array(0.0)
        violations = []
        for c in self.constraints:
            if c.has_capability(Capability.HAS_RATE_LIMIT):
                val = c.value_with_prev(t, x, u, u_prev, dt)
            else:
                val = c.value(t, x, u)
            # Flatten and take max of relu
            violations.append(jnp.max(jnp.maximum(val.ravel(), 0.0)))
        return jnp.max(jnp.array(violations))

    def is_feasible_with_prev(
        self, t: float, x: Array, u: Array, u_prev: Array, dt: float, tol: float = 0.0
    ) -> Array:
        """Check if all constraints are satisfied including rate limits.

        Args:
            t: Time
            x: State vector
            u: Current control vector
            u_prev: Previous control vector (for rate limits)
            dt: Timestep (for rate limits)
            tol: Tolerance for constraint satisfaction (default 0)

        Returns:
            Boolean array: True if max_violation_with_prev <= tol.
        """
        return self.max_violation_with_prev(t, x, u, u_prev, dt) <= tol
