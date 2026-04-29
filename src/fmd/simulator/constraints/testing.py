"""Test utilities for constraint implementations.

This module provides helper functions for testing constraints:
- Sign convention verification
- Clip enforcement validation
- JAX compatibility (JIT, vmap, autodiff)
- CasADi comparison (stub for future)

These are designed to be used in pytest tests for constraint implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from fmd.simulator.constraints.base import AbstractConstraint, Capability

if TYPE_CHECKING:
    from jax import Array


class ConstraintTestHelper:
    """Helper methods for testing constraint implementations.

    All methods are static and designed to be called from pytest tests.
    They raise AssertionError on failure with descriptive messages.

    Example:
        def test_my_constraint():
            c = MyConstraint(...)
            ConstraintTestHelper.check_jit_compatible(c)
            ConstraintTestHelper.check_vmap_compatible(c)
            ConstraintTestHelper.check_sign_convention(
                c,
                feasible_points=[(0.0, x_ok, u)],
                infeasible_points=[(0.0, x_bad, u)],
            )
    """

    @staticmethod
    def check_sign_convention(
        constraint: AbstractConstraint,
        feasible_points: Sequence[tuple[float, Array, Array]],
        infeasible_points: Sequence[tuple[float, Array, Array]],
    ) -> None:
        """Verify <= 0 for feasible points, > 0 for infeasible points.

        Args:
            constraint: Constraint to test
            feasible_points: List of (t, x, u) that should satisfy constraint
            infeasible_points: List of (t, x, u) that should violate constraint

        Raises:
            AssertionError: If sign convention is violated.
        """
        for i, (t, x, u) in enumerate(feasible_points):
            val = constraint.value(t, x, u)
            assert jnp.all(
                val <= 0
            ), f"Feasible point {i} has positive value: {val} (should be <= 0)"

        for i, (t, x, u) in enumerate(infeasible_points):
            val = constraint.value(t, x, u)
            assert jnp.any(
                val > 0
            ), f"Infeasible point {i} has non-positive value: {val} (should have > 0)"

    @staticmethod
    def check_clip_enforces(
        constraint: AbstractConstraint,
        test_points: Sequence[tuple[float, Array, Array]],
        tol: float = 1e-10,
    ) -> None:
        """Verify clip() produces feasible points.

        Args:
            constraint: Constraint to test (must support HARD_CLIP)
            test_points: List of (t, x, u) to clip and verify
            tol: Tolerance for feasibility check

        Raises:
            AssertionError: If clipped points are not feasible.

        Note:
            Skips if constraint doesn't support HARD_CLIP capability.
        """
        if not constraint.has_capability(Capability.HARD_CLIP):
            return

        for i, (t, x, u) in enumerate(test_points):
            x_clip, u_clip = constraint.clip(t, x, u)
            val = constraint.value(t, x_clip, u_clip)
            assert jnp.all(val <= tol), (
                f"Clipped point {i} still violated: {val} "
                f"(original x={x}, u={u}, clipped x={x_clip}, u={u_clip})"
            )

    @staticmethod
    def check_jit_compatible(
        constraint: AbstractConstraint,
        state_dim: int = 13,
        control_dim: int = 4,
    ) -> None:
        """Verify constraint.value can be JIT compiled.

        Uses eqx.filter_jit to properly handle equinox modules with
        array fields (which aren't hashable by jax.jit).

        Args:
            constraint: Constraint to test
            state_dim: Dimension of state vector for test
            control_dim: Dimension of control vector for test

        Raises:
            Exception: If JIT compilation fails.
        """
        x = jnp.zeros(state_dim)
        u = jnp.zeros(control_dim)
        # Use filter_jit for equinox modules with array fields
        jitted = eqx.filter_jit(constraint.value)
        _ = jitted(0.0, x, u)  # Should not raise

    @staticmethod
    def check_vmap_compatible(
        constraint: AbstractConstraint,
        batch_size: int = 10,
        state_dim: int = 13,
        control_dim: int = 4,
    ) -> None:
        """Verify constraint works under vmap (for MPC/batch rollouts).

        Args:
            constraint: Constraint to test
            batch_size: Number of points in batch
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector

        Raises:
            Exception: If vmap fails or output shape is wrong.
        """
        x_batch = jnp.zeros((batch_size, state_dim))
        u_batch = jnp.zeros((batch_size, control_dim))
        t_batch = jnp.zeros(batch_size)

        vmapped = jax.vmap(constraint.value)
        result = vmapped(t_batch, x_batch, u_batch)

        assert result.shape[0] == batch_size, (
            f"vmap result has wrong batch dimension: {result.shape[0]} != {batch_size}"
        )

    @staticmethod
    def check_differentiable(
        constraint: AbstractConstraint,
        t: float,
        x: Array,
        u: Array,
    ) -> tuple[Array, Array]:
        """Verify gradients exist and are finite.

        Args:
            constraint: Constraint to test
            t: Time
            x: State vector
            u: Control vector

        Returns:
            Tuple (grad_x, grad_u) of gradients.

        Raises:
            AssertionError: If gradients contain NaN.
        """

        def sum_value(x: Array, u: Array) -> Array:
            return jnp.sum(constraint.value(t, x, u))

        grad_x, grad_u = jax.grad(sum_value, argnums=(0, 1))(x, u)
        assert not jnp.any(jnp.isnan(grad_x)), f"NaN in grad_x: {grad_x}"
        assert not jnp.any(jnp.isnan(grad_u)), f"NaN in grad_u: {grad_u}"
        return grad_x, grad_u

    @staticmethod
    def check_value_shape(
        constraint: AbstractConstraint,
        t: float,
        x: Array,
        u: Array,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Verify value() returns correct shape.

        Args:
            constraint: Constraint to test
            t: Time
            x: State vector
            u: Control vector
            expected_shape: Expected output shape (None to skip shape check)

        Raises:
            AssertionError: If shape is wrong or not scalar/1D.
        """
        val = constraint.value(t, x, u)
        assert val.ndim <= 1, f"value() must return scalar or 1D, got shape {val.shape}"
        if expected_shape is not None:
            assert val.shape == expected_shape, (
                f"value() shape {val.shape} != expected {expected_shape}"
            )


def _compare_jax_casadi(
    jax_constraint: AbstractConstraint,
    casadi_func,  # CasADi function
    test_points: Sequence[tuple[float, Array, Array]],
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Compare JAX and CasADi constraint values.

    Compares numeric values at test points to verify JAX and CasADi
    implementations produce the same results.

    Args:
        jax_constraint: JAX constraint with value() method
        casadi_func: CasADi function (t, x, u) -> constraint value
        test_points: List of (t, x, u) tuples to test
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        NotImplementedError: CasADi integration not yet implemented.

    Note:
        This is a stub for future CasADi integration. When implemented,
        it will:
        1. Evaluate both implementations at each test point
        2. Compare values with numpy.allclose(rtol, atol)
        3. Optionally compare gradients via finite differences
    """
    # TODO: Implement when CasADi integration is added (Phase 4)
    raise NotImplementedError(
        "CasADi comparison not yet implemented. "
        "See docs/constraints.md for planned Phase 4 integration."
    )
