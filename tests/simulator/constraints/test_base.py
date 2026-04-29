"""Tests for base constraint abstractions."""

import pytest
import jax
import jax.numpy as jnp

from fmd.simulator.constraints import (
    AbstractConstraint,
    Capability,
    ConstraintCategory,
    ConstraintSet,
    BoxConstraint,
    ScalarBound,
)


class TestCapabilityEnum:
    """Tests for Capability enum."""

    def test_values_exist(self):
        """All capability values should exist."""
        assert Capability.HARD_CLIP.value == "hard_clip"
        assert Capability.PROJECTION.value == "projection"
        assert Capability.HAS_SYMBOLIC_FORM.value == "has_symbolic_form"
        assert Capability.SYMBOLIC_EXACT.value == "symbolic_exact"
        assert Capability.HAS_RATE_LIMIT.value == "has_rate_limit"

    def test_enum_members(self):
        """Should have exactly 5 capabilities."""
        assert len(Capability) == 5


class TestConstraintCategoryEnum:
    """Tests for ConstraintCategory enum."""

    def test_values_exist(self):
        """All category values should exist."""
        assert ConstraintCategory.STATE_BOUND.value == "state_bound"
        assert ConstraintCategory.CONTROL_BOUND.value == "control_bound"
        assert ConstraintCategory.RATE_LIMIT.value == "rate_limit"
        assert ConstraintCategory.PHYSICAL.value == "physical"
        assert ConstraintCategory.KINEMATIC.value == "kinematic"
        assert ConstraintCategory.SAFETY.value == "safety"

    def test_enum_members(self):
        """Should have exactly 6 categories."""
        assert len(ConstraintCategory) == 6


class TestConstraintSet:
    """Tests for ConstraintSet collection."""

    def test_empty_set(self):
        """Empty constraint set should work."""
        cs = ConstraintSet([])
        assert len(cs) == 0

    def test_single_constraint(self, simple_state, simple_control):
        """Single constraint set should work."""
        c = BoxConstraint("test", index=0, lower=-1.0, upper=1.0)
        cs = ConstraintSet([c])
        assert len(cs) == 1

    def test_duplicate_names_rejected(self):
        """Duplicate constraint names should raise ValueError."""
        c1 = BoxConstraint("same_name", index=0, lower=0.0, upper=1.0)
        c2 = BoxConstraint("same_name", index=1, lower=0.0, upper=1.0)
        with pytest.raises(ValueError, match="Duplicate"):
            ConstraintSet([c1, c2])

    def test_unique_names_accepted(self):
        """Unique constraint names should be accepted."""
        c1 = BoxConstraint("name_1", index=0, lower=0.0, upper=1.0)
        c2 = BoxConstraint("name_2", index=1, lower=0.0, upper=1.0)
        cs = ConstraintSet([c1, c2])
        assert len(cs) == 2

    def test_by_category(self, simple_state, simple_control):
        """Filter by category should work."""
        c1 = BoxConstraint("state_c", index=0, lower=0.0, upper=1.0, on_state=True)
        c2 = BoxConstraint("ctrl_c", index=0, lower=0.0, upper=1.0, on_state=False)
        cs = ConstraintSet([c1, c2])

        state_bounds = cs.by_category(ConstraintCategory.STATE_BOUND)
        ctrl_bounds = cs.by_category(ConstraintCategory.CONTROL_BOUND)

        assert len(state_bounds) == 1
        assert len(ctrl_bounds) == 1
        assert state_bounds[0].name == "state_c"
        assert ctrl_bounds[0].name == "ctrl_c"

    def test_by_capability(self, simple_state, simple_control):
        """Filter by capability should work."""
        c1 = BoxConstraint("c1", index=0, lower=0.0, upper=1.0)
        cs = ConstraintSet([c1])

        clippable = cs.by_capability(Capability.HARD_CLIP)
        assert len(clippable) == 1

        projectable = cs.by_capability(Capability.PROJECTION)
        assert len(projectable) == 0

    def test_max_violation_empty(self, simple_state, simple_control):
        """Empty set should have zero max violation."""
        cs = ConstraintSet([])
        v = cs.max_violation(0.0, simple_state, simple_control)
        assert v == 0.0

    def test_max_violation_feasible(self, simple_state, simple_control):
        """Feasible constraints should have zero max violation."""
        c = BoxConstraint("c", index=0, lower=-10.0, upper=10.0)
        cs = ConstraintSet([c])
        v = cs.max_violation(0.0, simple_state, simple_control)
        assert v == 0.0

    def test_max_violation_infeasible(self):
        """Violated constraints should have positive max violation."""
        c = BoxConstraint("c", index=0, lower=5.0, upper=10.0)  # x[0]=0 violates lower
        cs = ConstraintSet([c])
        x = jnp.zeros(3)
        u = jnp.zeros(2)
        v = cs.max_violation(0.0, x, u)
        assert v == 5.0  # lower=5, value=0, violation=5

    def test_is_feasible_true(self, simple_state, simple_control):
        """Should return True when all constraints satisfied."""
        c = BoxConstraint("c", index=0, lower=-10.0, upper=10.0)
        cs = ConstraintSet([c])
        assert cs.is_feasible(0.0, simple_state, simple_control)

    def test_is_feasible_false(self):
        """Should return False when any constraint violated."""
        c = BoxConstraint("c", index=0, lower=5.0, upper=10.0)
        cs = ConstraintSet([c])
        x = jnp.zeros(3)
        u = jnp.zeros(2)
        assert not cs.is_feasible(0.0, x, u)

    def test_is_feasible_with_tolerance(self):
        """Should respect tolerance parameter."""
        c = BoxConstraint("c", index=0, lower=-0.5, upper=0.5)
        cs = ConstraintSet([c])
        x = jnp.array([0.6, 0.0, 0.0])  # Violates by 0.1
        u = jnp.zeros(2)

        assert not cs.is_feasible(0.0, x, u, tol=0.0)
        assert cs.is_feasible(0.0, x, u, tol=0.2)

    def test_all_values(self, simple_state, simple_control):
        """all_values should return dict of constraint values."""
        c1 = BoxConstraint("c1", index=0, lower=-1.0, upper=1.0)
        c2 = ScalarBound("c2", index=1, bound=5.0, is_upper=True)
        cs = ConstraintSet([c1, c2])

        values = cs.all_values(0.0, simple_state, simple_control)

        assert "c1" in values
        assert "c2" in values
        assert values["c1"].shape == (2,)  # BoxConstraint returns 2-element vector
        assert values["c2"].shape == ()  # ScalarBound returns scalar

    def test_iteration(self):
        """Should be iterable."""
        c1 = BoxConstraint("c1", index=0, lower=0.0, upper=1.0)
        c2 = BoxConstraint("c2", index=1, lower=0.0, upper=1.0)
        cs = ConstraintSet([c1, c2])

        names = [c.name for c in cs]
        assert names == ["c1", "c2"]

    def test_jit_compatible(self, simple_state, simple_control):
        """ConstraintSet methods should be JIT compatible."""
        c = BoxConstraint("c", index=0, lower=-1.0, upper=1.0)
        cs = ConstraintSet([c])

        # JIT the methods
        jitted_max_viol = jax.jit(cs.max_violation)
        jitted_is_feas = jax.jit(cs.is_feasible)

        # Should not raise
        v = jitted_max_viol(0.0, simple_state, simple_control)
        f = jitted_is_feas(0.0, simple_state, simple_control)

        assert jnp.isfinite(v)
        assert f


class TestAbstractConstraint:
    """Tests for AbstractConstraint base class behavior."""

    def test_has_capability_true(self):
        """has_capability should return True for supported capabilities."""
        c = BoxConstraint("test", index=0, lower=0.0, upper=1.0)
        assert c.has_capability(Capability.HARD_CLIP)
        assert c.has_capability(Capability.HAS_SYMBOLIC_FORM)

    def test_has_capability_false(self):
        """has_capability should return False for unsupported capabilities."""
        c = BoxConstraint("test", index=0, lower=0.0, upper=1.0)
        assert not c.has_capability(Capability.PROJECTION)
        assert not c.has_capability(Capability.SYMBOLIC_EXACT)

    def test_project_not_implemented(self, simple_state, simple_control):
        """project() should raise NotImplementedError if not supported."""
        c = BoxConstraint("test", index=0, lower=0.0, upper=1.0)
        with pytest.raises(NotImplementedError, match="does not support project"):
            c.project(0.0, simple_state, simple_control)


class TestConstraintSetRateLimitMethods:
    """Tests for rate-limit-aware ConstraintSet methods."""

    def test_all_values_with_prev_regular_constraint(self):
        """all_values_with_prev works for regular constraints."""
        c = BoxConstraint("pos", index=0, lower=-1.0, upper=1.0)
        cs = ConstraintSet([c])
        x = jnp.array([0.5, 0.0, 0.0])
        u = jnp.array([0.0, 0.0])
        u_prev = jnp.array([0.0, 0.0])

        values = cs.all_values_with_prev(0.0, x, u, u_prev, dt=0.02)
        assert "pos" in values
        assert values["pos"].shape == (2,)

    def test_all_values_with_prev_rate_limit_constraint(self):
        """all_values_with_prev uses value_with_prev for rate limits."""
        from fmd.simulator.constraints import ControlRateLimit

        rate_limit = ControlRateLimit.symmetric(
            name="ctrl_rate", index=0, max_rate=1.0, dt_default=0.02
        )
        cs = ConstraintSet([rate_limit])
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([0.5, 0.0])
        u_prev = jnp.array([0.0, 0.0])

        values = cs.all_values_with_prev(0.0, x, u, u_prev, dt=0.02)
        assert "ctrl_rate" in values
        assert values["ctrl_rate"].shape == (2,)

    def test_max_violation_with_prev_feasible(self):
        """max_violation_with_prev returns 0 for feasible rate limits."""
        from fmd.simulator.constraints import ControlRateLimit

        # Rate limit: max 50 units/s rate
        rate_limit = ControlRateLimit.symmetric(
            name="ctrl_rate", index=0, max_rate=50.0, dt_default=0.02
        )
        cs = ConstraintSet([rate_limit])
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([0.5, 0.0])  # Change of 0.5 in 0.02s = 25 units/s (within limit)
        u_prev = jnp.array([0.0, 0.0])

        violation = cs.max_violation_with_prev(0.0, x, u, u_prev, dt=0.02)
        assert violation == 0.0

    def test_max_violation_with_prev_violated(self):
        """max_violation_with_prev returns positive for violated rate limits."""
        from fmd.simulator.constraints import ControlRateLimit

        # Rate limit: max 1 unit/s rate
        rate_limit = ControlRateLimit.symmetric(
            name="ctrl_rate", index=0, max_rate=1.0, dt_default=0.02
        )
        cs = ConstraintSet([rate_limit])
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])  # Change of 1.0 in 0.02s = 50 units/s (exceeds limit)
        u_prev = jnp.array([0.0, 0.0])

        violation = cs.max_violation_with_prev(0.0, x, u, u_prev, dt=0.02)
        assert violation > 0.0

    def test_is_feasible_with_prev_true(self):
        """is_feasible_with_prev returns True for feasible constraints."""
        from fmd.simulator.constraints import ControlRateLimit

        rate_limit = ControlRateLimit.symmetric(
            name="ctrl_rate", index=0, max_rate=50.0, dt_default=0.02
        )
        cs = ConstraintSet([rate_limit])
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([0.5, 0.0])
        u_prev = jnp.array([0.0, 0.0])

        assert cs.is_feasible_with_prev(0.0, x, u, u_prev, dt=0.02)

    def test_is_feasible_with_prev_false(self):
        """is_feasible_with_prev returns False for violated constraints."""
        from fmd.simulator.constraints import ControlRateLimit

        rate_limit = ControlRateLimit.symmetric(
            name="ctrl_rate", index=0, max_rate=1.0, dt_default=0.02
        )
        cs = ConstraintSet([rate_limit])
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])
        u_prev = jnp.array([0.0, 0.0])

        assert not cs.is_feasible_with_prev(0.0, x, u, u_prev, dt=0.02)

    def test_mixed_constraints_with_prev(self):
        """Rate-limit methods work with mixed constraint sets."""
        from fmd.simulator.constraints import ControlRateLimit

        box = BoxConstraint("pos", index=0, lower=-10.0, upper=10.0)
        rate_limit = ControlRateLimit.symmetric(
            name="ctrl_rate", index=0, max_rate=50.0, dt_default=0.02
        )
        cs = ConstraintSet([box, rate_limit])
        x = jnp.array([5.0, 0.0, 0.0])  # Feasible for box constraint
        u = jnp.array([0.5, 0.0])
        u_prev = jnp.array([0.0, 0.0])

        # Both constraints should be feasible
        violation = cs.max_violation_with_prev(0.0, x, u, u_prev, dt=0.02)
        assert violation == 0.0
        assert cs.is_feasible_with_prev(0.0, x, u, u_prev, dt=0.02)
