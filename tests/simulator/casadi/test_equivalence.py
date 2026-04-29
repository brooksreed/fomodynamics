"""JAX/CasADi equivalence tests for dynamics models.

These tests verify that CasADi *Exact models produce numerically identical
results to their JAX counterparts at multiple levels:
- Level 0: forward_dynamics (continuous-time derivatives)
- Level 1: Jacobians (A, B matrices)
- Level 2: Trajectories (RK4 integration)

Covers the public dynamics models shipped with `fomodynamics`. Any
private vehicles live outside this repo and maintain their own
equivalence suites.
"""

import pytest
import numpy as np
import jax.numpy as jnp

casadi = pytest.importorskip("casadi")

from fmd.simulator.box_1d import Box1DJax, Box1DFrictionJax
from fmd.simulator.cartpole import CartpoleJax
from fmd.simulator.casadi import (
    Box1DCasadiExact,
    Box1DFrictionCasadiExact,
    CartpoleCasadiExact,
    rk4_step_function,
)
from fmd.simulator.params import BOX1D_DEFAULT, BOX1D_FRICTION_DEFAULT, CARTPOLE_CLASSIC
from fmd.simulator.integrator import rk4_step

from .conftest import DERIV_RTOL, DERIV_ATOL, JAC_RTOL, JAC_ATOL, TRAJ_RTOL, TRAJ_ATOL


class TestBox1DEquivalence:
    """Verify Box1DJax and Box1DCasadiExact produce identical results."""

    @pytest.fixture
    def jax_model(self):
        return Box1DJax(BOX1D_DEFAULT)

    @pytest.fixture
    def casadi_model(self):
        return Box1DCasadiExact(BOX1D_DEFAULT)

    def test_level0_derivative_random_sampling(self, jax_model, casadi_model, rng):
        """Level 0: Random states produce identical derivatives."""
        f = casadi_model.dynamics_function()

        for _ in range(100):
            x = rng.uniform([-10, -10], [10, 10])
            u = rng.uniform([-10], [10])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level0_derivative_at_boundaries(self, jax_model, casadi_model):
        """Level 0: Test at boundary conditions."""
        f = casadi_model.dynamics_function()

        cases = [
            (np.array([0, 0]), np.array([0.0]), "at rest"),
            (np.array([100, 0]), np.array([0.0]), "far from origin"),
            (np.array([0, 10]), np.array([0.0]), "high velocity"),
            (np.array([0, 0]), np.array([100.0]), "max force"),
        ]

        for state, control, desc in cases:
            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(state), jnp.array(control)))
            casadi_deriv = np.array(f(state, control)).flatten()
            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Mismatch at: {desc}"
            )

    def test_level1_jacobian_equivalence(self, jax_model, casadi_model, rng):
        """Level 1: Jacobians match."""
        AB = casadi_model.linearization_function()

        for _ in range(50):
            x = rng.uniform([-10, -10], [10, 10])
            u = rng.uniform([-10], [10])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
            np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level2_trajectory_equivalence(self, jax_model, casadi_model):
        """Level 2: Multi-step RK4 trajectories match."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        x0 = np.array([0.0, 0.0])
        u = np.array([1.0])

        # JAX trajectory
        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        # CasADi trajectory
        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)


class TestBox1DFrictionEquivalence:
    """Verify Box1DFrictionJax and Box1DFrictionCasadiExact match."""

    @pytest.fixture
    def jax_model(self):
        return Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)

    @pytest.fixture
    def casadi_model(self):
        return Box1DFrictionCasadiExact(BOX1D_FRICTION_DEFAULT)

    def test_level0_derivative_random_sampling(self, jax_model, casadi_model, rng):
        """Level 0: Random states produce identical derivatives."""
        f = casadi_model.dynamics_function()

        for _ in range(100):
            # Avoid exactly v=0 due to sign() discontinuity
            x = rng.uniform([-10, -10], [10, 10])
            if abs(x[1]) < 0.01:
                x[1] = 0.01 * np.sign(x[1]) if x[1] != 0 else 0.01
            u = rng.uniform([-10], [10])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level1_jacobian_equivalence(self, jax_model, casadi_model, rng):
        """Level 1: Jacobians match (away from v=0)."""
        AB = casadi_model.linearization_function()

        for _ in range(50):
            x = rng.uniform([-10, -10], [10, 10])
            if abs(x[1]) < 0.1:
                x[1] = 0.1 * np.sign(x[1]) if x[1] != 0 else 0.1
            u = rng.uniform([-10], [10])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
            np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level2_trajectory_equivalence(self, jax_model, casadi_model):
        """Level 2: Multi-step RK4 trajectories match."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        x0 = np.array([0.0, 1.0])  # Start with velocity to avoid sign() at 0
        u = np.array([0.5])

        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)


class TestCartpoleEquivalence:
    """Verify CartpoleJax and CartpoleCasadiExact produce identical results."""

    @pytest.fixture
    def jax_model(self):
        return CartpoleJax(CARTPOLE_CLASSIC)

    @pytest.fixture
    def casadi_model(self):
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_level0_derivative_golden_value(self, jax_model, casadi_model):
        """Level 0: Golden value test from plan."""
        f = casadi_model.dynamics_function()

        # state = [x, x_dot, theta, theta_dot] with theta=0.1 rad
        x = np.array([0.0, 0.0, 0.1, 0.0])
        u = np.array([0.0])

        jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
        casadi_deriv = np.array(f(x, u)).flatten()

        np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level0_derivative_random_sampling(self, jax_model, casadi_model, rng):
        """Level 0: 100 random states produce identical derivatives."""
        f = casadi_model.dynamics_function()

        for _ in range(100):
            # Random state: x in [-10, 10], x_dot in [-5, 5], theta in [-pi, pi], theta_dot in [-5, 5]
            x = np.array([
                rng.uniform(-10, 10),      # x
                rng.uniform(-5, 5),        # x_dot
                rng.uniform(-np.pi, np.pi),  # theta
                rng.uniform(-5, 5),        # theta_dot
            ])
            u = rng.uniform([-10], [10])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level0_derivative_at_boundaries(self, jax_model, casadi_model):
        """Level 0: Test at boundary/special conditions."""
        f = casadi_model.dynamics_function()

        cases = [
            # (state, control, description)
            (np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0]), "upright at rest"),
            (np.array([0.0, 0.0, np.pi, 0.0]), np.array([0.0]), "inverted (hanging) at rest"),
            (np.array([0.0, 0.0, np.pi / 2, 0.0]), np.array([0.0]), "horizontal right"),
            (np.array([0.0, 0.0, -np.pi / 2, 0.0]), np.array([0.0]), "horizontal left"),
            (np.array([0.0, 0.0, 0.0, 0.0]), np.array([10.0]), "upright with max force"),
            (np.array([100.0, 5.0, 0.1, 2.0]), np.array([-10.0]), "moving with negative force"),
        ]

        for state, control, desc in cases:
            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(state), jnp.array(control)))
            casadi_deriv = np.array(f(state, control)).flatten()
            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Mismatch at: {desc}"
            )

    def test_level1_jacobian_equivalence(self, jax_model, casadi_model, rng):
        """Level 1: Jacobians (A, B) match at 50+ random points."""
        AB = casadi_model.linearization_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(-10, 10),
                rng.uniform(-5, 5),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-5, 5),
            ])
            u = rng.uniform([-10], [10])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
            np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level1_jacobian_at_upright(self, jax_model, casadi_model):
        """Level 1: Jacobians at upright equilibrium (important linearization point)."""
        AB = casadi_model.linearization_function()

        x = np.array([0.0, 0.0, 0.0, 0.0])  # Upright
        u = np.array([0.0])

        jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
        jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

        casadi_A, casadi_B = AB(x, u)

        np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
        np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level2_trajectory_equivalence(self, jax_model, casadi_model):
        """Level 2: 100 RK4 steps with post_step enabled."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Start with small angle and some angular velocity (dynamic scenario)
        x0 = np.array([0.0, 0.0, 0.1, 0.5])
        u = np.array([1.0])  # Constant force

        # JAX trajectory
        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        # CasADi trajectory
        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)

    def test_level2_trajectory_swingup(self, jax_model, casadi_model):
        """Level 2: Trajectory during a swing-up maneuver (larger angles)."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Start inverted (hanging down) - pole will swing
        x0 = np.array([0.0, 0.0, np.pi - 0.1, 0.0])
        u = np.array([5.0])  # Push to start swinging

        # JAX trajectory
        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        # CasADi trajectory
        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)


class TestPlanarQuadrotorEquivalence:
    """Verify PlanarQuadrotorJax and PlanarQuadrotorCasadiExact produce identical results."""

    @pytest.fixture
    def jax_model(self):
        from fmd.simulator.planar_quadrotor import PlanarQuadrotorJax
        from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT
        return PlanarQuadrotorJax(PLANAR_QUAD_TEST_DEFAULT)

    @pytest.fixture
    def casadi_model(self):
        from fmd.simulator.casadi import PlanarQuadrotorCasadiExact
        from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT
        return PlanarQuadrotorCasadiExact(PLANAR_QUAD_TEST_DEFAULT)

    def test_level0_derivative_random_sampling(self, jax_model, casadi_model, rng):
        """Level 0: 100 random states produce identical derivatives."""
        f = casadi_model.dynamics_function()

        for _ in range(100):
            # Random state: x, z in [-10, 10], theta in [-pi, pi], velocities in [-5, 5]
            x = np.array([
                rng.uniform(-10, 10),       # x
                rng.uniform(-10, 10),       # z
                rng.uniform(-np.pi, np.pi), # theta
                rng.uniform(-5, 5),         # x_dot
                rng.uniform(-5, 5),         # z_dot
                rng.uniform(-5, 5),         # theta_dot
            ])
            # Control: thrusts in [0, 20] (must be non-negative for physical realism)
            u = rng.uniform([0, 0], [20, 20])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level0_derivative_at_hover(self, jax_model, casadi_model):
        """Level 0: Test at hover condition (equilibrium point)."""
        f = casadi_model.dynamics_function()

        # At hover: T1 = T2 = m*g/2
        hover_thrust = jax_model.hover_thrust_per_rotor()
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # At rest, upright
        u = np.array([hover_thrust, hover_thrust])

        jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
        casadi_deriv = np.array(f(x, u)).flatten()

        # At hover, z_ddot should be ~0 (gravity balanced by thrust)
        np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level0_derivative_at_boundaries(self, jax_model, casadi_model):
        """Level 0: Test at boundary/special conditions."""
        f = casadi_model.dynamics_function()

        cases = [
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0]), "at rest, no thrust"),
            (np.array([0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0]), np.array([5.0, 5.0]), "pitched 90 deg"),
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([10.0, 5.0]), "differential thrust"),
            (np.array([100.0, 50.0, 0.1, 5.0, -3.0, 1.0]), np.array([8.0, 8.0]), "general motion"),
        ]

        for state, control, desc in cases:
            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(state), jnp.array(control)))
            casadi_deriv = np.array(f(state, control)).flatten()
            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Mismatch at: {desc}"
            )

    def test_level1_jacobian_equivalence(self, jax_model, casadi_model, rng):
        """Level 1: Jacobians (A, B) match at 50+ random points."""
        AB = casadi_model.linearization_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(-10, 10),
                rng.uniform(-10, 10),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-5, 5),
                rng.uniform(-5, 5),
                rng.uniform(-5, 5),
            ])
            u = rng.uniform([0, 0], [20, 20])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
            np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level2_trajectory_equivalence(self, jax_model, casadi_model):
        """Level 2: 100 RK4 steps produce identical trajectories."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Start at hover with small disturbance
        hover_thrust = jax_model.hover_thrust_per_rotor()
        x0 = np.array([0.0, 1.0, 0.1, 0.5, 0.0, 0.2])
        u = np.array([hover_thrust + 0.5, hover_thrust - 0.5])  # Slight differential

        # JAX trajectory
        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        # CasADi trajectory
        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)


class TestBoat2DEquivalence:
    """Verify Boat2DJax and Boat2DCasadiExact produce identical results."""

    @pytest.fixture
    def jax_model(self):
        from fmd.simulator.boat_2d import Boat2DJax
        from fmd.simulator.params import BOAT2D_TEST_DEFAULT
        return Boat2DJax(BOAT2D_TEST_DEFAULT)

    @pytest.fixture
    def casadi_model(self):
        from fmd.simulator.casadi import Boat2DCasadiExact
        from fmd.simulator.params import BOAT2D_TEST_DEFAULT
        return Boat2DCasadiExact(BOAT2D_TEST_DEFAULT)

    def test_level0_derivative_random_sampling(self, jax_model, casadi_model, rng):
        """Level 0: 100 random states produce identical derivatives."""
        f = casadi_model.dynamics_function()

        for _ in range(100):
            x = np.array([
                rng.uniform(-100, 100),      # x (North position)
                rng.uniform(-100, 100),      # y (East position)
                rng.uniform(-np.pi, np.pi),  # psi (heading)
                rng.uniform(-5, 5),          # u (surge velocity)
                rng.uniform(-2, 2),          # v (sway velocity)
                rng.uniform(-1, 1),          # r (yaw rate)
            ])
            u = np.array([rng.uniform(-100, 100), rng.uniform(-50, 50)])  # thrust, yaw_moment

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_level0_derivative_at_boundaries(self, jax_model, casadi_model):
        """Level 0: Test at boundary/special conditions."""
        f = casadi_model.dynamics_function()

        cases = [
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0]), "at rest"),
            (np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]), np.array([0.0, 0.0]), "straight ahead"),
            (np.array([0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0]), np.array([50.0, 0.0]), "heading east, thrust"),
            (np.array([0.0, 0.0, 0.0, 3.0, 1.0, 0.5]), np.array([20.0, 10.0]), "turning with sideslip"),
            (np.array([0.0, 0.0, np.pi - 0.01, 2.0, 0.0, 0.1]), np.array([10.0, 5.0]), "near pi wrap"),
        ]

        for state, control, desc in cases:
            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(state), jnp.array(control)))
            casadi_deriv = np.array(f(state, control)).flatten()
            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Mismatch at: {desc}"
            )

    def test_level1_jacobian_equivalence(self, jax_model, casadi_model, rng):
        """Level 1: Jacobians (A, B) match at 50+ random points."""
        AB = casadi_model.linearization_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(-100, 100),
                rng.uniform(-100, 100),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-5, 5),
                rng.uniform(-2, 2),
                rng.uniform(-1, 1),
            ])
            u = np.array([rng.uniform(-100, 100), rng.uniform(-50, 50)])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
            np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level2_trajectory_equivalence(self, jax_model, casadi_model):
        """Level 2: 100 RK4 steps with post_step (heading wrap) enabled."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Start with some velocity and yaw rate
        x0 = np.array([0.0, 0.0, 0.0, 3.0, 0.5, 0.3])
        u = np.array([50.0, 10.0])  # thrust and yaw moment

        # JAX trajectory
        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        # CasADi trajectory
        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)

    def test_level2_trajectory_heading_wrap(self, jax_model, casadi_model):
        """Level 2: Trajectory that exercises heading wrap."""
        dt = 0.01
        steps = 100

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Start near pi with yaw rate - heading will wrap
        x0 = np.array([0.0, 0.0, 3.0, 2.0, 0.0, 0.5])  # psi ~= 3 rad, near pi
        u = np.array([20.0, 5.0])

        # JAX trajectory
        x_jax = jnp.array(x0)
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)

        # CasADi trajectory
        x_casadi = x0.copy()
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()

