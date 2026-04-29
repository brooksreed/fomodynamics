"""Tests for bound constraints (BoxConstraint, ScalarBound)."""

import pytest
import jax
import jax.numpy as jnp

from fmd.simulator.constraints import (
    BoxConstraint,
    ScalarBound,
    Capability,
    ConstraintCategory,
    ConstraintTestHelper,
)


class TestBoxConstraint:
    """Tests for BoxConstraint (two-sided bounds)."""

    def test_creation(self):
        """Valid BoxConstraint should be created."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        assert c.name == "test"
        assert c.index == 0
        assert c.lower == -1.0
        assert c.upper == 1.0
        assert c.on_state is True

    def test_creation_on_control(self):
        """BoxConstraint on control should have correct category."""
        c = BoxConstraint("test", index=0, lower=0.0, upper=1.0, on_state=False)
        assert c.category == ConstraintCategory.CONTROL_BOUND
        assert c.on_state is False

    def test_creation_on_state(self):
        """BoxConstraint on state should have correct category."""
        c = BoxConstraint("test", index=0, lower=0.0, upper=1.0, on_state=True)
        assert c.category == ConstraintCategory.STATE_BOUND

    def test_invalid_bounds_raises(self):
        """lower > upper should raise ValueError."""
        with pytest.raises(ValueError, match="must be <= upper"):
            BoxConstraint("bad", index=0, lower=1.0, upper=-1.0)

    def test_equal_bounds_ok(self):
        """lower == upper should be allowed (equality constraint)."""
        c = BoxConstraint("eq", index=0, lower=0.0, upper=0.0)
        assert c.lower == c.upper

    def test_capabilities(self):
        """Should have HARD_CLIP and HAS_SYMBOLIC_FORM."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        assert c.has_capability(Capability.HARD_CLIP)
        assert c.has_capability(Capability.HAS_SYMBOLIC_FORM)
        assert not c.has_capability(Capability.PROJECTION)

    def test_value_feasible_interior(self):
        """Value inside bounds should be <= 0."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val.shape == (2,)
        assert jnp.all(val <= 0)

    def test_value_at_lower_boundary(self):
        """Value at lower bound should have first element = 0."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([-1.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val[0] == 0.0  # lower - val = -1 - (-1) = 0
        assert val[1] < 0  # val - upper = -1 - 1 = -2

    def test_value_at_upper_boundary(self):
        """Value at upper bound should have second element = 0."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([1.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val[0] < 0  # lower - val = -1 - 1 = -2
        assert val[1] == 0.0  # val - upper = 1 - 1 = 0

    def test_value_violates_lower(self):
        """Value below lower bound should have positive first element."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([-2.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val[0] > 0  # lower - val = -1 - (-2) = 1

    def test_value_violates_upper(self):
        """Value above upper bound should have positive second element."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([2.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val[1] > 0  # val - upper = 2 - 1 = 1

    def test_clip_enforces_lower(self):
        """Clip should enforce lower bound."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([-5.0, 0.0, 0.0])
        u = jnp.array([])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert x_clip[0] == -1.0

    def test_clip_enforces_upper(self):
        """Clip should enforce upper bound."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([5.0, 0.0, 0.0])
        u = jnp.array([])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert x_clip[0] == 1.0

    def test_clip_preserves_feasible(self):
        """Clip should not change feasible values."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([0.5, 0.0, 0.0])
        u = jnp.array([])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert x_clip[0] == 0.5

    def test_clip_control(self):
        """Clip should work on control when on_state=False."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0, on_state=False)
        x = jnp.array([0.0])
        u = jnp.array([5.0])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert x_clip[0] == 0.0  # Unchanged
        assert u_clip[0] == 1.0  # Clipped

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        ConstraintTestHelper.check_jit_compatible(c)

    def test_vmap_compatible(self):
        """Should be vmap compatible."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        ConstraintTestHelper.check_vmap_compatible(c)

    def test_differentiable(self):
        """Should be differentiable."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        x = jnp.array([0.5, 0.0, 0.0])
        u = jnp.zeros(2)
        grad_x, grad_u = ConstraintTestHelper.check_differentiable(c, 0.0, x, u)
        # Gradient w.r.t. x[0] should be non-zero (sum of -1 and 1 = 0 actually)
        # The gradient of [lower - x[0], x[0] - upper] w.r.t x[0] is [-1, 1]
        # Sum is 0, so grad_x[0] = 0 is expected
        assert grad_x[0] == 0.0  # -1 + 1 = 0

    def test_sign_convention(self):
        """Should follow <= 0 satisfied, > 0 violated convention."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        u = jnp.array([])

        feasible_points = [
            (0.0, jnp.array([0.0, 0.0, 0.0]), u),
            (0.0, jnp.array([-1.0, 0.0, 0.0]), u),
            (0.0, jnp.array([1.0, 0.0, 0.0]), u),
        ]
        infeasible_points = [
            (0.0, jnp.array([-2.0, 0.0, 0.0]), u),
            (0.0, jnp.array([2.0, 0.0, 0.0]), u),
        ]

        ConstraintTestHelper.check_sign_convention(c, feasible_points, infeasible_points)


class TestScalarBound:
    """Tests for ScalarBound (one-sided bounds)."""

    def test_creation_upper(self):
        """Upper bound should be created correctly."""
        c = ScalarBound("test", index=0, bound=1.0, is_upper=True)
        assert c.name == "test"
        assert c.index == 0
        assert c.bound == 1.0
        assert c.is_upper is True

    def test_creation_lower(self):
        """Lower bound should be created correctly."""
        c = ScalarBound("test", index=0, bound=-1.0, is_upper=False)
        assert c.is_upper is False

    def test_capabilities(self):
        """Should have HARD_CLIP and HAS_SYMBOLIC_FORM."""
        c = ScalarBound("test", index=0, bound=1.0)
        assert c.has_capability(Capability.HARD_CLIP)
        assert c.has_capability(Capability.HAS_SYMBOLIC_FORM)

    def test_upper_bound_feasible(self):
        """Value below upper bound should be <= 0."""
        c = ScalarBound("test", index=0, bound=1.0, is_upper=True)
        x = jnp.array([0.5])
        val = c.value(0.0, x, jnp.array([]))
        assert val <= 0  # val - bound = 0.5 - 1.0 = -0.5

    def test_upper_bound_at_boundary(self):
        """Value at upper bound should be 0."""
        c = ScalarBound("test", index=0, bound=1.0, is_upper=True)
        x = jnp.array([1.0])
        val = c.value(0.0, x, jnp.array([]))
        assert val == 0.0

    def test_upper_bound_infeasible(self):
        """Value above upper bound should be > 0."""
        c = ScalarBound("test", index=0, bound=1.0, is_upper=True)
        x = jnp.array([1.5])
        val = c.value(0.0, x, jnp.array([]))
        assert val > 0  # val - bound = 1.5 - 1.0 = 0.5

    def test_lower_bound_feasible(self):
        """Value above lower bound should be <= 0."""
        c = ScalarBound("test", index=0, bound=-1.0, is_upper=False)
        x = jnp.array([0.0])
        val = c.value(0.0, x, jnp.array([]))
        assert val <= 0  # bound - val = -1.0 - 0.0 = -1.0

    def test_lower_bound_at_boundary(self):
        """Value at lower bound should be 0."""
        c = ScalarBound("test", index=0, bound=-1.0, is_upper=False)
        x = jnp.array([-1.0])
        val = c.value(0.0, x, jnp.array([]))
        assert val == 0.0

    def test_lower_bound_infeasible(self):
        """Value below lower bound should be > 0."""
        c = ScalarBound("test", index=0, bound=-1.0, is_upper=False)
        x = jnp.array([-1.5])
        val = c.value(0.0, x, jnp.array([]))
        assert val > 0  # bound - val = -1.0 - (-1.5) = 0.5

    def test_clip_upper_bound(self):
        """Clip should enforce upper bound."""
        c = ScalarBound("test", index=0, bound=1.0, is_upper=True)
        x = jnp.array([5.0])
        u = jnp.array([])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert x_clip[0] == 1.0

    def test_clip_lower_bound(self):
        """Clip should enforce lower bound."""
        c = ScalarBound("test", index=0, bound=-1.0, is_upper=False)
        x = jnp.array([-5.0])
        u = jnp.array([])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert x_clip[0] == -1.0

    def test_clip_control_upper(self):
        """Clip should work on control for upper bound."""
        c = ScalarBound("test", index=0, bound=1.0, is_upper=True, on_state=False)
        x = jnp.array([0.0])
        u = jnp.array([5.0])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert u_clip[0] == 1.0

    def test_clip_control_lower(self):
        """Clip should work on control for lower bound."""
        c = ScalarBound("test", index=0, bound=0.0, is_upper=False, on_state=False)
        x = jnp.array([0.0])
        u = jnp.array([-5.0])
        x_clip, u_clip = c.clip(0.0, x, u)
        assert u_clip[0] == 0.0

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = ScalarBound("test", index=0, bound=1.0)
        ConstraintTestHelper.check_jit_compatible(c, state_dim=3, control_dim=2)

    def test_vmap_compatible(self):
        """Should be vmap compatible."""
        c = ScalarBound("test", index=0, bound=1.0)
        ConstraintTestHelper.check_vmap_compatible(c, state_dim=3, control_dim=2)

    def test_differentiable(self):
        """Should be differentiable."""
        c = ScalarBound("test", index=0, bound=1.0)
        x = jnp.array([0.5, 0.0, 0.0])
        u = jnp.zeros(2)
        grad_x, grad_u = ConstraintTestHelper.check_differentiable(c, 0.0, x, u)
        # For upper bound: val - bound, gradient w.r.t x[0] is 1
        assert grad_x[0] == 1.0

    def test_value_shape_scalar(self):
        """ScalarBound should return scalar."""
        c = ScalarBound("test", index=0, bound=1.0)
        x = jnp.array([0.0])
        u = jnp.array([])
        ConstraintTestHelper.check_value_shape(c, 0.0, x, u, expected_shape=())

    def test_sign_convention(self):
        """Should follow <= 0 satisfied, > 0 violated convention."""
        c_upper = ScalarBound("upper", index=0, bound=1.0, is_upper=True)
        c_lower = ScalarBound("lower", index=0, bound=-1.0, is_upper=False)
        u = jnp.array([])

        # Upper bound test
        ConstraintTestHelper.check_sign_convention(
            c_upper,
            feasible_points=[
                (0.0, jnp.array([0.0]), u),
                (0.0, jnp.array([1.0]), u),
            ],
            infeasible_points=[
                (0.0, jnp.array([2.0]), u),
            ],
        )

        # Lower bound test
        ConstraintTestHelper.check_sign_convention(
            c_lower,
            feasible_points=[
                (0.0, jnp.array([0.0]), u),
                (0.0, jnp.array([-1.0]), u),
            ],
            infeasible_points=[
                (0.0, jnp.array([-2.0]), u),
            ],
        )
