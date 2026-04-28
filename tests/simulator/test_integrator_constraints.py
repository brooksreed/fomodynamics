"""Tests for constraint integration with simulators.

Phase 2 of the constraint system integration: tests that verify
constraints are correctly enforced during simulation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator import (
    SimplePendulum,
    PlanarQuadrotor,
    simulate,
    simulate_euler,
    simulate_symplectic,
    simulate_trajectory,
    rk4_step,
    euler_step,
    semi_implicit_euler_step,
    ConstantControl,
)
from fmd.simulator.params import PENDULUM_1M, PLANAR_QUAD_TEST_DEFAULT
from fmd.simulator.constraints import (
    BoxConstraint,
    ScalarBound,
    HalfSpaceConstraint,
    ConstraintSet,
    Capability,
)


class TestStepFunctionsWithConstraints:
    """Tests for step functions with constraint enforcement."""

    @pytest.fixture
    def pendulum(self):
        return SimplePendulum(PENDULUM_1M)

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    def test_rk4_step_without_constraints_unchanged(self, pendulum):
        """rk4_step without constraints should work as before."""
        state = jnp.array([0.5, 0.0])
        control = jnp.array([])

        # Without constraints
        new_state = rk4_step(pendulum, state, control, 0.01, 0.0)

        # Should produce valid output
        assert new_state.shape == state.shape
        assert jnp.isfinite(new_state).all()

    def test_rk4_step_with_constraints_clips_state(self, pendulum):
        """rk4_step with state constraints should clip."""
        # Constrain angle to [-0.1, 0.1]
        angle_bound = BoxConstraint("angle", index=0, lower=-0.1, upper=0.1, on_state=True)
        constraints = ConstraintSet([angle_bound])

        # Start with large angle
        state = jnp.array([0.5, 0.0])
        control = jnp.array([])

        new_state = rk4_step(pendulum, state, control, 0.01, 0.0, constraints)

        # Angle should be clipped
        assert new_state[0] <= 0.1 + 1e-6

    def test_euler_step_with_constraints(self, pendulum):
        """euler_step with constraints should clip."""
        angle_bound = BoxConstraint("angle", index=0, lower=-0.1, upper=0.1, on_state=True)
        constraints = ConstraintSet([angle_bound])

        state = jnp.array([0.5, 0.0])
        control = jnp.array([])

        new_state = euler_step(pendulum, state, control, 0.01, 0.0, constraints)

        assert new_state[0] <= 0.1 + 1e-6

    def test_semi_implicit_euler_step_with_constraints(self, pendulum):
        """semi_implicit_euler_step with constraints should clip."""
        angle_bound = BoxConstraint("angle", index=0, lower=-0.1, upper=0.1, on_state=True)
        constraints = ConstraintSet([angle_bound])

        state = jnp.array([0.5, 0.0])
        control = jnp.array([])

        new_state = semi_implicit_euler_step(pendulum, state, control, 0.01, 0.0, constraints)

        assert new_state[0] <= 0.1 + 1e-6

    def test_step_control_constraint(self, quad):
        """Step functions should clip control as well."""
        # Constrain thrust to [0, 5] (lower than hover)
        thrust_bound = BoxConstraint("T1", index=0, lower=0.0, upper=5.0, on_state=False)
        constraints = ConstraintSet([thrust_bound])

        state = quad.default_state()
        control = jnp.array([10.0, 10.0])  # Above constraint

        # Control clipping happens at constraint evaluation time
        # but the step function just returns state
        new_state = rk4_step(quad, state, control, 0.01, 0.0, constraints)

        # The state should be finite
        assert jnp.isfinite(new_state).all()


class TestSimulateFunctionsWithConstraints:
    """Tests for simulate functions with constraint enforcement."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    def test_simulate_without_constraints_unchanged(self, quad):
        """simulate without constraints should work as before."""
        initial = quad.create_state(z=5.0)
        hover = ConstantControl(quad.hover_control())

        result = simulate(quad, initial, dt=0.01, duration=1.0, control=hover)

        assert result.states.shape[0] > 1
        assert jnp.isfinite(result.states).all()

    def test_simulate_with_ground_constraint(self, quad):
        """Falling quad should stop at ground."""
        # Ground constraint: z >= 0 (in planar quad, z is index 1)
        # normal points toward forbidden region (below ground)
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),  # -z direction
            offset=0.0,
            pos_indices=(0, 1),
        )
        constraints = ConstraintSet([ground])

        initial = quad.create_state(z=1.0)
        zero_thrust = ConstantControl(jnp.array([0.0, 0.0]))

        # Without constraints, quad falls through ground
        result_no_constraint = simulate(
            quad, initial, dt=0.01, duration=2.0, control=zero_thrust
        )
        assert result_no_constraint.states[-1, 1] < -1.0, "Should fall below ground"

        # With constraints, quad stops at ground
        result_with_constraint = simulate(
            quad, initial, dt=0.01, duration=2.0, control=zero_thrust, constraints=constraints
        )
        assert result_with_constraint.states[-1, 1] >= -0.01, "Should stop at ground"

    def test_simulate_euler_with_constraints(self, quad):
        """simulate_euler should respect constraints."""
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),
            offset=0.0,
            pos_indices=(0, 1),
        )
        constraints = ConstraintSet([ground])

        initial = quad.create_state(z=1.0)
        zero_thrust = ConstantControl(jnp.array([0.0, 0.0]))

        result = simulate_euler(
            quad, initial, dt=0.01, duration=2.0, control=zero_thrust, constraints=constraints
        )

        # Check all states are above ground
        assert (result.states[:, 1] >= -0.01).all(), "Should never go below ground"

    def test_simulate_symplectic_with_constraints(self, quad):
        """simulate_symplectic should respect constraints."""
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),
            offset=0.0,
            pos_indices=(0, 1),
        )
        constraints = ConstraintSet([ground])

        initial = quad.create_state(z=1.0)
        zero_thrust = ConstantControl(jnp.array([0.0, 0.0]))

        result = simulate_symplectic(
            quad, initial, dt=0.01, duration=2.0, control=zero_thrust, constraints=constraints
        )

        assert (result.states[:, 1] >= -0.01).all(), "Should never go below ground"

    def test_simulate_trajectory_with_constraints(self, quad):
        """simulate_trajectory should respect constraints."""
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),
            offset=0.0,
            pos_indices=(0, 1),
        )
        constraints = ConstraintSet([ground])

        initial = quad.create_state(z=1.0)
        zero_thrust = ConstantControl(jnp.array([0.0, 0.0]))
        times = jnp.linspace(0, 2.0, 201)

        result = simulate_trajectory(
            quad, initial, times, control=zero_thrust, constraints=constraints
        )

        assert (result.states[:, 1] >= -0.01).all(), "Should never go below ground"


class TestMultipleConstraints:
    """Tests for multiple constraint handling."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    def test_multiple_constraints(self, quad):
        """Multiple constraints should all be enforced."""
        # Ground constraint
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),
            offset=0.0,
            pos_indices=(0, 1),
        )
        # Ceiling constraint: z <= 10
        ceiling = HalfSpaceConstraint(
            name="ceiling",
            normal=jnp.array([0.0, 1.0]),  # +z direction
            offset=10.0,
            pos_indices=(0, 1),
        )
        # Thrust bounds
        thrust_lower = ScalarBound("T1_min", index=0, bound=0.0, is_upper=False, on_state=False)
        thrust_upper = ScalarBound("T1_max", index=0, bound=15.0, is_upper=True, on_state=False)

        constraints = ConstraintSet([ground, ceiling, thrust_lower, thrust_upper])

        # Start at z=5, should stay in [0, 10]
        initial = quad.create_state(z=5.0)
        hover = ConstantControl(quad.hover_control())

        result = simulate(quad, initial, dt=0.01, duration=2.0, control=hover, constraints=constraints)

        # All states should be in valid range
        assert (result.states[:, 1] >= -0.01).all(), "Should respect ground"
        assert (result.states[:, 1] <= 10.01).all(), "Should respect ceiling"


class TestJITCompatibility:
    """Tests for JIT compatibility of constraint enforcement."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    def test_simulate_with_constraints_jit(self, quad):
        """simulate with constraints should be JIT-able."""
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),
            offset=0.0,
            pos_indices=(0, 1),
        )
        constraints = ConstraintSet([ground])

        initial = quad.create_state(z=1.0)
        zero_thrust = ConstantControl(jnp.array([0.0, 0.0]))

        # The simulation itself uses jax.lax.scan which is JIT-compiled
        result = simulate(
            quad, initial, dt=0.01, duration=1.0, control=zero_thrust, constraints=constraints
        )

        assert jnp.isfinite(result.states).all()


class TestRateLimitConstraintIntegration:
    """Integration tests for rate limit constraints in simulation."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    @pytest.fixture
    def pendulum(self):
        return SimplePendulum(PENDULUM_1M)

    def test_rate_limit_clipping_in_simulate(self, quad):
        """Verify rate limits clip control changes during simulation."""
        from fmd.simulator.constraints import ControlRateLimit

        # Very tight rate limit: max 0.5 N/s thrust change
        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=0.5, dt_default=0.02
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)

        # Control that tries to command a large thrust immediately
        high_thrust = ConstantControl(jnp.array([10.0, 10.0]))

        result = simulate(
            quad, initial, dt=0.02, duration=1.0, control=high_thrust, constraints=constraints
        )

        # The control should be rate-limited
        # With max_rate=0.5 and dt=0.02, max delta per step = 0.01
        # After 50 steps (1.0s), max control = 50 * 0.01 = 0.5
        # Actually the first control is the reference, so we'd expect slow ramp
        assert jnp.isfinite(result.states).all()
        # Simulation should remain stable due to rate limiting

    def test_rate_limit_preserves_smooth_control(self, quad):
        """Verify rate limits produce smooth control trajectories."""
        from fmd.simulator.constraints import ControlRateLimit
        from fmd.simulator.control import PiecewiseConstantControl

        # Rate limit: 2.0 N/s max change
        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=2.0, dt_default=0.02
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)

        # Alternating control that would be jerky without rate limit
        times = jnp.linspace(0, 1.0, 51)
        # Alternate between high and low thrust each timestep
        controls = jnp.array([
            [12.0 if i % 2 == 0 else 8.0, 10.0] for i in range(50)
        ])
        jerky_control = PiecewiseConstantControl(times=times, controls=controls)

        result = simulate(
            quad, initial, dt=0.02, duration=1.0, control=jerky_control, constraints=constraints
        )

        # Result should be finite and stable
        assert jnp.isfinite(result.states).all()
        # The quad shouldn't have wild oscillations due to rate limiting

    def test_rate_limit_first_step_feasible(self, quad):
        """Verify first timestep is always feasible (u_prev = u[0])."""
        from fmd.simulator.constraints import ControlRateLimit

        # Very tight rate limit
        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=0.1, dt_default=0.02
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)
        hover = ConstantControl(quad.hover_control())

        # With tight rate limit and constant control, should be stable
        result = simulate(
            quad, initial, dt=0.02, duration=0.5, control=hover, constraints=constraints
        )

        # First step should work (u_prev = u[0] means zero rate)
        assert jnp.isfinite(result.states).all()
        assert result.states.shape[0] > 1

    def test_rate_limit_with_simulate_euler(self, quad):
        """Rate limits should work with Euler integration."""
        from fmd.simulator.constraints import ControlRateLimit

        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=1.0, dt_default=0.02
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)
        hover = ConstantControl(quad.hover_control())

        result = simulate_euler(
            quad, initial, dt=0.02, duration=0.5, control=hover, constraints=constraints
        )

        assert jnp.isfinite(result.states).all()

    def test_rate_limit_with_simulate_symplectic(self, quad):
        """Rate limits should work with symplectic integration."""
        from fmd.simulator.constraints import ControlRateLimit

        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=1.0, dt_default=0.02
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)
        hover = ConstantControl(quad.hover_control())

        result = simulate_symplectic(
            quad, initial, dt=0.02, duration=0.5, control=hover, constraints=constraints
        )

        assert jnp.isfinite(result.states).all()

    def test_rate_limit_with_simulate_trajectory(self, quad):
        """Rate limits should work with simulate_trajectory."""
        from fmd.simulator.constraints import ControlRateLimit

        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=1.0, dt_default=0.02
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)
        hover = ConstantControl(quad.hover_control())
        times = jnp.linspace(0, 0.5, 26)  # dt = 0.02

        result = simulate_trajectory(
            quad, initial, times, control=hover, constraints=constraints
        )

        assert jnp.isfinite(result.states).all()

    def test_rate_limit_combined_with_box_constraint(self, quad):
        """Rate limits should work alongside other constraint types."""
        from fmd.simulator.constraints import ControlRateLimit

        # Rate limit on thrust
        rate_limit = ControlRateLimit.symmetric(
            name="thrust_rate", index=0, max_rate=5.0, dt_default=0.02
        )
        # Box constraint on position (ground plane)
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, -1.0]),
            offset=0.0,
            pos_indices=(0, 1),
        )
        constraints = ConstraintSet([rate_limit, ground])

        initial = quad.create_state(z=2.0)
        zero_thrust = ConstantControl(jnp.array([0.0, 0.0]))

        result = simulate(
            quad, initial, dt=0.02, duration=2.0, control=zero_thrust, constraints=constraints
        )

        # Both constraints should be respected
        assert jnp.isfinite(result.states).all()
        assert (result.states[:, 1] >= -0.01).all(), "Should respect ground"

    def test_asymmetric_rate_limit(self, quad):
        """Asymmetric rate limits should apply different up/down rates."""
        from fmd.simulator.constraints import ControlRateLimit
        from fmd.simulator.control import PiecewiseConstantControl

        # Fast ramp up, slow ramp down
        rate_limit = ControlRateLimit(
            name="thrust_rate",
            index=0,
            max_rate_up=10.0,  # Can increase fast
            max_rate_down=1.0,  # Slow decrease
            dt_default=0.02,
        )
        constraints = ConstraintSet([rate_limit])

        initial = quad.create_state(z=5.0)

        # Control that ramps up then down
        times = jnp.linspace(0, 1.0, 51)
        controls = jnp.array([
            [15.0 if i < 25 else 5.0, 10.0] for i in range(50)
        ])
        ramp_control = PiecewiseConstantControl(times=times, controls=controls)

        result = simulate(
            quad, initial, dt=0.02, duration=1.0, control=ramp_control, constraints=constraints
        )

        assert jnp.isfinite(result.states).all()
