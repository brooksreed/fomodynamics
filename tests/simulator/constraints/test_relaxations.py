"""Tests for relaxation utilities (penalty functions)."""

import pytest
import jax
import jax.numpy as jnp

from fmd.simulator.constraints import (
    BoxConstraint,
    ScalarBound,
    quadratic_penalty,
    log_barrier,
    smooth_relu_penalty,
    exact_penalty,
    augmented_lagrangian,
)


class TestQuadraticPenalty:
    """Tests for quadratic_penalty relaxation."""

    def test_zero_when_feasible(self):
        """Should return zero when constraint is satisfied."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = quadratic_penalty(c, weight=1.0)
        x = jnp.array([0.0])
        cost = cost_fn(0.0, x, jnp.array([]))
        assert cost == 0.0

    def test_positive_when_violated(self):
        """Should return positive cost when constraint is violated."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = quadratic_penalty(c, weight=1.0)
        x = jnp.array([2.0])  # Violates upper bound by 1
        cost = cost_fn(0.0, x, jnp.array([]))
        assert cost > 0

    def test_quadratic_growth(self):
        """Cost should grow quadratically with violation."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = quadratic_penalty(c, weight=1.0)

        # Violations of 1, 2, 3
        cost_1 = cost_fn(0.0, jnp.array([1.0]), jnp.array([]))
        cost_2 = cost_fn(0.0, jnp.array([2.0]), jnp.array([]))
        cost_3 = cost_fn(0.0, jnp.array([3.0]), jnp.array([]))

        # Cost should be 1, 4, 9 (quadratic)
        assert jnp.abs(cost_1 - 1.0) < 1e-10
        assert jnp.abs(cost_2 - 4.0) < 1e-10
        assert jnp.abs(cost_3 - 9.0) < 1e-10

    def test_weight_scaling(self):
        """Weight should scale the cost."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn_1 = quadratic_penalty(c, weight=1.0)
        cost_fn_10 = quadratic_penalty(c, weight=10.0)

        x = jnp.array([1.0])
        cost_1 = cost_fn_1(0.0, x, jnp.array([]))
        cost_10 = cost_fn_10(0.0, x, jnp.array([]))

        assert jnp.abs(cost_10 - 10 * cost_1) < 1e-10

    def test_differentiable(self):
        """Should be differentiable."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = quadratic_penalty(c, weight=1.0)

        grad_fn = jax.grad(lambda x: cost_fn(0.0, x, jnp.array([])))
        x = jnp.array([1.5])
        grad = grad_fn(x)
        assert jnp.isfinite(grad[0])
        assert grad[0] != 0

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = quadratic_penalty(c, weight=1.0)
        jitted = jax.jit(cost_fn)
        cost = jitted(0.0, jnp.array([0.0]), jnp.array([]))
        assert jnp.isfinite(cost)


class TestLogBarrier:
    """Tests for log_barrier relaxation."""

    def test_finite_when_strictly_feasible(self):
        """Should return finite cost when strictly inside feasible region."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = log_barrier(c, scale=1.0)
        x = jnp.array([0.0])  # Well inside bounds
        cost = cost_fn(0.0, x, jnp.array([]))
        assert jnp.isfinite(cost)

    def test_inf_when_violated(self):
        """Should return inf when constraint is violated."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = log_barrier(c, scale=1.0)
        x = jnp.array([2.0])  # Outside bounds
        cost = cost_fn(0.0, x, jnp.array([]))
        assert cost == jnp.inf

    def test_inf_at_boundary(self):
        """Should return inf at boundary (g = 0)."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = log_barrier(c, scale=1.0)
        x = jnp.array([0.0])  # At boundary
        cost = cost_fn(0.0, x, jnp.array([]))
        assert cost == jnp.inf

    def test_increases_near_boundary(self):
        """Cost should increase as approaching boundary."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = log_barrier(c, scale=1.0)

        cost_far = cost_fn(0.0, jnp.array([-1.0]), jnp.array([]))
        cost_close = cost_fn(0.0, jnp.array([-0.1]), jnp.array([]))

        assert cost_close > cost_far

    def test_scale_parameter(self):
        """Scale should affect barrier steepness."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn_1 = log_barrier(c, scale=1.0)
        cost_fn_10 = log_barrier(c, scale=10.0)

        x = jnp.array([-0.5])
        cost_1 = cost_fn_1(0.0, x, jnp.array([]))
        cost_10 = cost_fn_10(0.0, x, jnp.array([]))

        # scale=10 should give 10x the cost
        assert jnp.abs(cost_10 - 10 * cost_1) < 1e-10

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = log_barrier(c, scale=1.0)
        jitted = jax.jit(cost_fn)
        cost = jitted(0.0, jnp.array([0.0]), jnp.array([]))
        assert jnp.isfinite(cost)


class TestSmoothReluPenalty:
    """Tests for smooth_relu_penalty relaxation."""

    def test_small_when_feasible(self):
        """Should return small (but non-zero) cost when feasible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = smooth_relu_penalty(c, weight=1.0, softness=0.1)
        x = jnp.array([0.0])
        cost = cost_fn(0.0, x, jnp.array([]))
        # softplus(g/s)*s for g << 0 is approximately s*exp(g/s) which is small
        assert cost < 0.5  # Should be small
        assert cost > 0  # But non-zero

    def test_positive_when_violated(self):
        """Should return larger cost when violated."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = smooth_relu_penalty(c, weight=1.0, softness=0.1)

        cost_ok = cost_fn(0.0, jnp.array([-1.0]), jnp.array([]))
        cost_bad = cost_fn(0.0, jnp.array([1.0]), jnp.array([]))

        assert cost_bad > cost_ok

    def test_differentiable_everywhere(self):
        """Should be differentiable everywhere including at boundary."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = smooth_relu_penalty(c, weight=1.0, softness=0.1)
        grad_fn = jax.grad(lambda x: cost_fn(0.0, x, jnp.array([])))

        # At boundary
        x_boundary = jnp.array([0.0])
        grad_boundary = grad_fn(x_boundary)
        assert jnp.isfinite(grad_boundary[0])

        # Inside feasible
        x_feasible = jnp.array([-1.0])
        grad_feasible = grad_fn(x_feasible)
        assert jnp.isfinite(grad_feasible[0])

        # Violated
        x_violated = jnp.array([1.0])
        grad_violated = grad_fn(x_violated)
        assert jnp.isfinite(grad_violated[0])

    def test_softness_parameter(self):
        """Larger softness should give smoother transition."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn_sharp = smooth_relu_penalty(c, weight=1.0, softness=0.01)
        cost_fn_soft = smooth_relu_penalty(c, weight=1.0, softness=1.0)

        # At feasible point, soft should give more cost (more "leaked" penalty)
        x = jnp.array([-0.5])
        cost_sharp = cost_fn_sharp(0.0, x, jnp.array([]))
        cost_soft = cost_fn_soft(0.0, x, jnp.array([]))

        assert cost_soft > cost_sharp

    def test_weight_scaling(self):
        """Weight should scale the cost."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn_1 = smooth_relu_penalty(c, weight=1.0, softness=0.1)
        cost_fn_10 = smooth_relu_penalty(c, weight=10.0, softness=0.1)

        x = jnp.array([1.0])
        cost_1 = cost_fn_1(0.0, x, jnp.array([]))
        cost_10 = cost_fn_10(0.0, x, jnp.array([]))

        assert jnp.abs(cost_10 - 10 * cost_1) < 1e-10

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = smooth_relu_penalty(c, weight=1.0, softness=0.1)
        jitted = jax.jit(cost_fn)
        cost = jitted(0.0, jnp.array([0.0]), jnp.array([]))
        assert jnp.isfinite(cost)


class TestExactPenalty:
    """Tests for exact_penalty (L1) relaxation."""

    def test_zero_when_feasible(self):
        """Should return zero when constraint is satisfied."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = exact_penalty(c, weight=1.0)
        x = jnp.array([0.0])
        cost = cost_fn(0.0, x, jnp.array([]))
        assert cost == 0.0

    def test_positive_when_violated(self):
        """Should return positive cost when violated."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = exact_penalty(c, weight=1.0)
        x = jnp.array([1.0])
        cost = cost_fn(0.0, x, jnp.array([]))
        assert cost > 0

    def test_linear_growth(self):
        """Cost should grow linearly with violation (L1)."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn = exact_penalty(c, weight=1.0)

        cost_1 = cost_fn(0.0, jnp.array([1.0]), jnp.array([]))
        cost_2 = cost_fn(0.0, jnp.array([2.0]), jnp.array([]))
        cost_3 = cost_fn(0.0, jnp.array([3.0]), jnp.array([]))

        # Cost should be 1, 2, 3 (linear)
        assert jnp.abs(cost_1 - 1.0) < 1e-10
        assert jnp.abs(cost_2 - 2.0) < 1e-10
        assert jnp.abs(cost_3 - 3.0) < 1e-10

    def test_weight_scaling(self):
        """Weight should scale the cost."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn_1 = exact_penalty(c, weight=1.0)
        cost_fn_10 = exact_penalty(c, weight=10.0)

        x = jnp.array([1.0])
        cost_1 = cost_fn_1(0.0, x, jnp.array([]))
        cost_10 = cost_fn_10(0.0, x, jnp.array([]))

        assert jnp.abs(cost_10 - 10 * cost_1) < 1e-10

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cost_fn = exact_penalty(c, weight=1.0)
        jitted = jax.jit(cost_fn)
        cost = jitted(0.0, jnp.array([0.0]), jnp.array([]))
        assert jnp.isfinite(cost)


class TestAugmentedLagrangian:
    """Tests for augmented_lagrangian relaxation."""

    def test_returns_cost_and_update(self):
        """Should return cost function and update function."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn, update_fn = augmented_lagrangian(c, weight=1.0)

        assert callable(cost_fn)
        assert callable(update_fn)

    def test_cost_feasible_zero_multiplier(self):
        """With zero multiplier and feasible point, cost should be small."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn, _ = augmented_lagrangian(c, weight=1.0)

        x = jnp.array([-1.0])  # Feasible
        lam = jnp.array(0.0)  # Zero multiplier
        cost = cost_fn(0.0, x, jnp.array([]), lam)

        # Cost = lam * g + (rho/2) * relu(lam/rho + g)^2
        # = 0 * (-1) + 0.5 * relu(0 + (-1))^2 = 0
        assert cost == 0.0

    def test_cost_violated(self):
        """With violated constraint, cost should be positive."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn, _ = augmented_lagrangian(c, weight=1.0)

        x = jnp.array([1.0])  # Violated by 1
        lam = jnp.array(0.0)
        cost = cost_fn(0.0, x, jnp.array([]), lam)

        assert cost > 0

    def test_multiplier_update_increases(self):
        """Multiplier should increase when constraint is violated."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        _, update_fn = augmented_lagrangian(c, weight=1.0)

        x = jnp.array([1.0])  # Violated by 1
        lam_0 = jnp.array(0.0)
        lam_1 = update_fn(0.0, x, jnp.array([]), lam_0)

        assert lam_1 > lam_0

    def test_multiplier_update_stays_nonnegative(self):
        """Multiplier should stay non-negative (for inequality constraints)."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        _, update_fn = augmented_lagrangian(c, weight=1.0)

        x = jnp.array([-1.0])  # Feasible
        lam_0 = jnp.array(0.5)
        lam_1 = update_fn(0.0, x, jnp.array([]), lam_0)

        # lam_new = max(lam + rho * g, 0) = max(0.5 + 1 * (-1), 0) = 0
        assert lam_1 >= 0

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=True)
        cost_fn, update_fn = augmented_lagrangian(c, weight=1.0)

        jitted_cost = jax.jit(cost_fn)
        jitted_update = jax.jit(update_fn)

        x = jnp.array([0.0])
        u = jnp.array([])
        lam = jnp.array(0.0)

        cost = jitted_cost(0.0, x, u, lam)
        lam_new = jitted_update(0.0, x, u, lam)

        assert jnp.isfinite(cost)
        assert jnp.isfinite(lam_new)
