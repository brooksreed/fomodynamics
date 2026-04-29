"""Relaxation utilities for converting constraints to cost functions.

These wrap constraints into differentiable cost functions suitable for
optimization (iLQR, gradient descent, etc.).

All relaxations work with any constraint that implements value().
They are implemented as utility functions (not constraint methods) to
keep the constraint interface minimal and allow flexible composition.

Cost Function Properties:
    - quadratic_penalty: C1 continuous, zero when feasible, grows quadratically
    - log_barrier: Interior point method, infinite outside feasible region
    - smooth_relu_penalty: C-infinity, small cost even when feasible
    - exact_penalty: L1 penalty, non-smooth but exact for large weights
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from fmd.simulator.constraints.base import AbstractConstraint

if TYPE_CHECKING:
    from jax import Array


def quadratic_penalty(
    constraint: AbstractConstraint,
    weight: float = 1.0,
) -> Callable[[float, Array, Array], Array]:
    """Create quadratic penalty cost: weight * sum(relu(g)^2).

    The cost is zero when the constraint is satisfied (g <= 0) and
    grows quadratically with violation magnitude.

    Properties:
        - Zero when feasible
        - C1 continuous (derivative is continuous)
        - Gradient at boundary is zero (can slow convergence)

    Args:
        constraint: Constraint to penalize
        weight: Penalty weight (larger = harder constraint)

    Returns:
        Cost function (t, x, u) -> scalar cost

    Example:
        cost_fn = quadratic_penalty(pos_bound, weight=100.0)
        cost = cost_fn(t, state, control)
    """

    def cost(t: float, x: Array, u: Array) -> Array:
        g = constraint.value(t, x, u)
        violation = jnp.maximum(g, 0.0)
        return weight * jnp.sum(violation**2)

    return cost


def log_barrier(
    constraint: AbstractConstraint,
    scale: float = 1.0,
) -> Callable[[float, Array, Array], Array]:
    """Create log barrier cost: -scale * sum(log(-g)).

    Used in interior point methods. The cost goes to infinity as the
    constraint boundary is approached from the feasible side.

    Properties:
        - Only valid in strictly feasible region (g < 0)
        - Returns inf if any constraint is violated (g >= 0)
        - Smooth and convex in feasible region
        - Gradient becomes large near boundary

    Args:
        constraint: Constraint to create barrier for
        scale: Barrier scale (smaller = tighter approximation)

    Returns:
        Cost function (t, x, u) -> scalar cost

    Note:
        Start with large scale and decrease during optimization
        (barrier method / interior point method).
    """

    def cost(t: float, x: Array, u: Array) -> Array:
        g = constraint.value(t, x, u)
        is_feasible = jnp.all(g < 0)
        # Use jax.lax.cond for cleaner gradient handling.
        # When feasible, compute the actual barrier cost.
        # When infeasible, return inf (gradients undefined but consistent).
        barrier = jax.lax.cond(
            is_feasible,
            lambda: -scale * jnp.sum(jnp.log(-g)),
            lambda: jnp.inf,
        )
        return barrier

    return cost


def smooth_relu_penalty(
    constraint: AbstractConstraint,
    weight: float = 1.0,
    softness: float = 1.0,
) -> Callable[[float, Array, Array], Array]:
    """Create softplus-based smooth penalty.

    Uses softplus(g / softness) * softness for smooth gradients everywhere.
    Unlike quadratic_penalty, this is differentiable everywhere including
    at the constraint boundary.

    Properties:
        - Small but non-zero cost even when feasible
        - C-infinity (infinitely differentiable)
        - Non-zero gradient even in feasible region (helps optimization)
        - Approximates relu as softness -> 0

    Args:
        constraint: Constraint to penalize
        weight: Overall penalty weight
        softness: Controls smoothness (larger = smoother, more bias)

    Returns:
        Cost function (t, x, u) -> scalar cost

    Example:
        # For optimization, start with larger softness and anneal
        cost_fn = smooth_relu_penalty(pos_bound, weight=100.0, softness=0.1)
    """

    def cost(t: float, x: Array, u: Array) -> Array:
        g = constraint.value(t, x, u)
        # softplus(x) = log(1 + exp(x))
        # We use logaddexp for numerical stability
        penalty = softness * jnp.sum(jnp.logaddexp(0.0, g / softness))
        return weight * penalty

    return cost


def exact_penalty(
    constraint: AbstractConstraint,
    weight: float = 1.0,
) -> Callable[[float, Array, Array], Array]:
    """Create L1 exact penalty: weight * sum(relu(g)).

    The cost is proportional to the constraint violation magnitude.
    Called "exact" because for sufficiently large weight, the constrained
    and unconstrained optima coincide.

    Properties:
        - Zero when feasible
        - Linear growth with violation (not quadratic)
        - Non-smooth at boundary (subgradient methods needed)
        - Exact for large enough weight

    Args:
        constraint: Constraint to penalize
        weight: Penalty weight (must be large enough for exactness)

    Returns:
        Cost function (t, x, u) -> scalar cost

    Note:
        For convex problems, weight > max(Lagrange multiplier) suffices.
        In practice, start with moderate weight and increase if needed.
    """

    def cost(t: float, x: Array, u: Array) -> Array:
        g = constraint.value(t, x, u)
        violation = jnp.maximum(g, 0.0)
        return weight * jnp.sum(violation)

    return cost


def augmented_lagrangian(
    constraint: AbstractConstraint,
    weight: float = 1.0,
) -> Callable[[float, Array, Array, Array], tuple[Array, Array]]:
    """Create augmented Lagrangian cost and multiplier update.

    The augmented Lagrangian combines a Lagrangian term with a quadratic
    penalty. This is the basis for ALM (Augmented Lagrangian Method).

    Cost: lambda * g + (weight/2) * relu(lambda/weight + g)^2

    Args:
        constraint: Constraint to penalize
        weight: Penalty weight (rho in ALM literature)

    Returns:
        Tuple of:
        - cost_fn(t, x, u, lam) -> scalar cost
        - update_fn(t, x, u, lam) -> updated multiplier

    Example:
        cost_fn, update_fn = augmented_lagrangian(pos_bound, weight=10.0)
        lam = jnp.zeros(...)  # Initialize multiplier

        for iteration in range(max_iter):
            # Minimize augmented Lagrangian
            x_opt = minimize(lambda x: cost_fn(t, x, u, lam), x0)
            # Update multiplier
            lam = update_fn(t, x_opt, u, lam)
    """

    def cost(t: float, x: Array, u: Array, lam: Array) -> Array:
        g = constraint.value(t, x, u)
        # Augmented Lagrangian: lam * g + (rho/2) * relu(lam/rho + g)^2
        shifted = lam / weight + g
        penalty = jnp.maximum(shifted, 0.0)
        return jnp.sum(lam * g) + (weight / 2) * jnp.sum(penalty**2)

    def update(t: float, x: Array, u: Array, lam: Array) -> Array:
        g = constraint.value(t, x, u)
        # Update: lam_new = max(lam + rho * g, 0)
        return jnp.maximum(lam + weight * g, 0.0)

    return cost, update
