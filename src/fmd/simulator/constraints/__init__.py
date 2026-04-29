"""Constraint system for fomodynamics optimization and control.

This module provides constraint abstractions for:
- Optimization (iLQR, gradient descent) via relaxation functions
- MPC via constraint sets with symbolic forms
- Simulation enforcement via clip/project methods

Constraints represent inequalities g(t, x, u) <= 0. They are NOT forces.
For penalty forces and contact dynamics, use JaxForceElement components.

Quick Start:
    from fmd.simulator.constraints import BoxConstraint, ConstraintSet

    # Create bounds on control inputs
    throttle = BoxConstraint("throttle", index=0, lower=0.0, upper=1.0, on_state=False)
    steering = BoxConstraint("steering", index=1, lower=-0.5, upper=0.5, on_state=False)

    # Combine into a set
    constraints = ConstraintSet([throttle, steering])

    # Check feasibility
    is_ok = constraints.is_feasible(t=0.0, x=state, u=control)

    # Get maximum violation
    violation = constraints.max_violation(t=0.0, x=state, u=control)

For Optimization:
    from fmd.simulator.constraints import quadratic_penalty, smooth_relu_penalty

    # Create cost function from constraint
    cost_fn = quadratic_penalty(throttle, weight=100.0)
    cost = cost_fn(t, state, control)

See docs/public/constraints.md for full documentation.
"""

# Base abstractions
from fmd.simulator.constraints.base import (
    AbstractConstraint,
    Capability,
    ConstraintCategory,
    ConstraintSet,
)

# Bound constraints
from fmd.simulator.constraints.bounds import (
    BoxConstraint,
    ScalarBound,
)

# Physical/environmental constraints
from fmd.simulator.constraints.physical import (
    HalfSpaceConstraint,
    KeepOutZone,
)

# Rate limit constraints
from fmd.simulator.constraints.rate_limits import (
    ControlRateLimit,
)

# Relaxation utilities (constraint -> cost function)
from fmd.simulator.constraints.relaxations import (
    augmented_lagrangian,
    exact_penalty,
    log_barrier,
    quadratic_penalty,
    smooth_relu_penalty,
)

# Test utilities
from fmd.simulator.constraints.testing import (
    ConstraintTestHelper,
    _compare_jax_casadi as compare_jax_casadi,
)

__all__ = [
    # Base abstractions
    "AbstractConstraint",
    "Capability",
    "ConstraintCategory",
    "ConstraintSet",
    # Bound constraints
    "BoxConstraint",
    "ScalarBound",
    # Physical constraints
    "KeepOutZone",
    "HalfSpaceConstraint",
    # Rate limit constraints
    "ControlRateLimit",
    # Relaxations
    "quadratic_penalty",
    "log_barrier",
    "smooth_relu_penalty",
    "exact_penalty",
    "augmented_lagrangian",
    # Testing
    "ConstraintTestHelper",
    "compare_jax_casadi",
]
