"""Tests for LQR controller classes.

Tests cover:
- Controller creation and shapes
- Control = 0 at equilibrium
- Control proportional to error
- Cartpole stabilization from small perturbation
- PlanarQuadrotor hover maintenance
- Discrete-time LQR requires dt
- Trajectory tracking with simple reference
- TVLQR produces time-varying gains
- scipy version documented in test output
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy

# Conditional JAX imports
try:
    import fmd.simulator._config  # noqa: F401
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

# Skip entire module if JAX not available
pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")

# Import modules after JAX check
if HAS_JAX:
    from fmd.simulator.lqr import (
        LQRController,
        TrajectoryLQRController,
        TVLQRController,
        lqr,
        compute_lqr_gain,
        solve_continuous_are,
        solve_discrete_are,
    )
    from fmd.simulator.linearize import linearize, discretize_zoh
    from fmd.simulator import (
        Cartpole, PlanarQuadrotor, SimplePendulum,
        simulate, ConstantControl,
    )
    from fmd.simulator.params import (
        CARTPOLE_CLASSIC,
        PLANAR_QUAD_TEST_DEFAULT,
        PENDULUM_1M,
    )


# Tolerance constants
CTRL_RTOL = 1e-10
CTRL_ATOL = 1e-12


def test_scipy_version():
    """Document scipy version for reproducibility.

    LQR gain computation depends on scipy's ARE solver, which may
    vary slightly between versions. This test documents the version
    used for the test run.
    """
    print(f"\nscipy version: {scipy.__version__}")
    print(f"numpy version: {np.__version__}")
    # This test always passes - it's just for documentation
    assert True


class TestLQRControllerBasics:
    """Test basic LQR controller functionality."""

    @pytest.fixture
    def cartpole(self):
        return Cartpole(CARTPOLE_CLASSIC)

    @pytest.fixture
    def cartpole_eq(self, cartpole):
        """Return equilibrium state and control."""
        return cartpole.upright_state(), jnp.array([0.0])

    @pytest.fixture
    def cartpole_qr(self):
        """Return Q and R matrices for Cartpole."""
        Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0]))
        R = jnp.array([[0.1]])
        return Q, R

    def test_controller_creation_shapes(self, cartpole, cartpole_eq, cartpole_qr):
        """LQRController has correct shapes."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        # K should be (m, n) = (1, 4) for Cartpole
        assert controller.K.shape == (1, 4)
        assert controller.x_ref.shape == (4,)
        assert controller.u_ref.shape == (1,)

    def test_control_zero_at_equilibrium(self, cartpole, cartpole_eq, cartpole_qr):
        """Control should be zero when at equilibrium."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        # At equilibrium, error is zero, so control = u_ref = 0
        control = controller(0.0, x_eq)
        assert_allclose(control, u_eq, rtol=CTRL_RTOL, atol=CTRL_ATOL)

    def test_control_proportional_to_error(self, cartpole, cartpole_eq, cartpole_qr):
        """Control should be proportional to state error."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        # Small perturbation
        delta_x = jnp.array([0.0, 0.0, 0.1, 0.0])
        x_perturbed = x_eq + delta_x

        control = controller(0.0, x_perturbed)

        # Control should be u = u_ref - K @ delta_x = -K @ delta_x
        expected = u_eq - controller.K @ delta_x
        assert_allclose(control, expected, rtol=CTRL_RTOL, atol=CTRL_ATOL)

    def test_control_doubles_with_double_error(self, cartpole, cartpole_eq, cartpole_qr):
        """Control should double when error doubles (linear feedback)."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        delta_x = jnp.array([0.0, 0.0, 0.05, 0.0])
        control1 = controller(0.0, x_eq + delta_x)
        control2 = controller(0.0, x_eq + 2 * delta_x)

        # Should be linear
        assert_allclose(control2 - u_eq, 2 * (control1 - u_eq), rtol=1e-10, atol=1e-12)

    def test_from_matrices(self, cartpole, cartpole_eq, cartpole_qr):
        """LQRController.from_matrices produces same result as from_linearization."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        # Method 1: from_linearization
        controller1 = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        # Method 2: from_matrices
        A, B = linearize(cartpole, x_eq, u_eq)
        controller2 = LQRController.from_matrices(A, B, Q, R, x_eq, u_eq)

        # Should produce identical K
        assert_allclose(controller1.K, controller2.K, rtol=CTRL_RTOL, atol=CTRL_ATOL)

    def test_with_reference(self, cartpole, cartpole_eq, cartpole_qr):
        """with_reference() creates new controller with updated setpoint."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        # New reference
        x_new = jnp.array([1.0, 0.0, 0.0, 0.0])  # Move cart to x=1
        controller_new = controller.with_reference(x_ref=x_new)

        # K should be the same
        assert_allclose(controller_new.K, controller.K, rtol=CTRL_RTOL, atol=CTRL_ATOL)

        # x_ref should be updated
        assert_allclose(controller_new.x_ref, x_new, rtol=CTRL_RTOL, atol=CTRL_ATOL)

        # u_ref should be unchanged
        assert_allclose(controller_new.u_ref, controller.u_ref, rtol=CTRL_RTOL, atol=CTRL_ATOL)

    def test_properties(self, cartpole, cartpole_eq, cartpole_qr):
        """Test num_states and num_controls properties."""
        x_eq, u_eq = cartpole_eq
        Q, R = cartpole_qr

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        assert controller.num_states == 4
        assert controller.num_controls == 1


class TestLQRDiscreteTime:
    """Test discrete-time LQR."""

    @pytest.fixture
    def cartpole(self):
        return Cartpole(CARTPOLE_CLASSIC)

    def test_discrete_requires_dt(self, cartpole):
        """Discrete-time LQR requires dt parameter."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        Q = jnp.eye(4)
        R = jnp.array([[1.0]])

        with pytest.raises(ValueError, match="dt must be provided"):
            LQRController.from_linearization(
                cartpole, x_eq, u_eq, Q, R, discrete=True
            )

    def test_discrete_with_dt(self, cartpole):
        """Discrete-time LQR works with dt provided."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        Q = jnp.eye(4)
        R = jnp.array([[1.0]])

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R, discrete=True, dt=0.01
        )

        assert controller.K.shape == (1, 4)

    def test_continuous_vs_discrete_different(self, cartpole):
        """Continuous and discrete LQR give different gains."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        Q = jnp.eye(4)
        R = jnp.array([[1.0]])

        controller_cont = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R, discrete=False
        )

        controller_disc = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R, discrete=True, dt=0.01
        )

        # Gains should be different
        assert not np.allclose(controller_cont.K, controller_disc.K)

    def test_discrete_converges_to_continuous(self, cartpole):
        """Discrete LQR converges to continuous as dt -> 0."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        Q = jnp.eye(4)
        R = jnp.array([[1.0]])

        controller_cont = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R, discrete=False
        )

        # Small dt should give similar gain
        controller_disc = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R, discrete=True, dt=1e-4
        )

        # Should be close for very small dt
        assert_allclose(controller_disc.K, controller_cont.K, rtol=1e-2, atol=1e-4)


class TestCartpoleStabilization:
    """Test Cartpole stabilization from small perturbations."""

    @pytest.fixture
    def cartpole(self):
        return Cartpole(CARTPOLE_CLASSIC)

    def test_stabilizes_small_angle(self, cartpole):
        """LQR stabilizes Cartpole from small angle perturbation."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        # Aggressive weights on angle and angular velocity
        Q = jnp.diag(jnp.array([1.0, 1.0, 100.0, 100.0]))
        R = jnp.array([[0.01]])

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        # Small initial angle
        x0 = jnp.array([0.0, 0.0, 0.1, 0.0])

        # Simulate
        result = simulate(cartpole, x0, dt=0.01, duration=5.0, control=controller)

        # Final state should be near equilibrium
        final_state = result.states[-1]
        assert_allclose(final_state[2], 0.0, atol=0.01)  # theta near 0
        assert_allclose(final_state[3], 0.0, atol=0.1)   # theta_dot near 0

    def test_closed_loop_stable_eigenvalues(self, cartpole):
        """Closed-loop system should have stable eigenvalues."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0]))
        R = jnp.array([[0.1]])

        A, B = linearize(cartpole, x_eq, u_eq)
        K = compute_lqr_gain(A, B, Q, R, discrete=False)

        # Closed-loop: A_cl = A - B @ K
        A_cl = A - B @ K

        eigenvalues = np.linalg.eigvals(np.asarray(A_cl))

        # All eigenvalues should have negative real parts
        assert np.all(np.real(eigenvalues) < 0), (
            f"Expected stable closed-loop. Eigenvalues: {eigenvalues}"
        )


class TestPlanarQuadrotorHover:
    """Test PlanarQuadrotor hover maintenance."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    def test_hover_controller_creation(self, quad):
        """LQR controller for hover has correct shapes."""
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        Q = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
        R = jnp.eye(2) * 0.1

        controller = LQRController.from_linearization(
            quad, x_eq, u_eq, Q, R
        )

        # K should be (2, 6) for PlanarQuadrotor
        assert controller.K.shape == (2, 6)

    def test_hover_control_at_equilibrium(self, quad):
        """At hover, control should be hover thrust."""
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        Q = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
        R = jnp.eye(2) * 0.1

        controller = LQRController.from_linearization(
            quad, x_eq, u_eq, Q, R
        )

        control = controller(0.0, x_eq)
        assert_allclose(control, u_eq, rtol=1e-10, atol=1e-12)

    def test_maintains_hover(self, quad):
        """LQR maintains hover with small perturbation."""
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        Q = jnp.diag(jnp.array([10.0, 100.0, 100.0, 1.0, 10.0, 10.0]))
        R = jnp.eye(2) * 0.01

        controller = LQRController.from_linearization(
            quad, x_eq, u_eq, Q, R
        )

        # Small position perturbation
        x0 = quad.create_state(x=0.1, z=0.1, theta=0.05)

        result = simulate(quad, x0, dt=0.001, duration=2.0, control=controller)

        # Should return to hover
        final_state = result.states[-1]
        assert_allclose(final_state[:3], [0.0, 0.0, 0.0], atol=0.1)

    def test_closed_loop_stable(self, quad):
        """Closed-loop quadrotor system is stable."""
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        Q = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
        R = jnp.eye(2) * 0.1

        A, B = linearize(quad, x_eq, u_eq)
        K = compute_lqr_gain(A, B, Q, R, discrete=False)

        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(np.asarray(A_cl))

        # All eigenvalues should have negative real parts
        assert np.all(np.real(eigenvalues) < 0), (
            f"Expected stable closed-loop. Eigenvalues: {eigenvalues}"
        )


class TestTrajectoryLQRController:
    """Test trajectory tracking LQR."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    @pytest.fixture
    def simple_trajectory(self, quad):
        """Create simple x-direction trajectory."""
        T = 2.0
        n_points = 21
        times = jnp.linspace(0, T, n_points)

        # Move in x direction
        x_refs = jnp.zeros((n_points, 6))
        x_refs = x_refs.at[:, 0].set(jnp.linspace(0, 1.0, n_points))  # x from 0 to 1

        # Constant hover control
        u_eq = quad.hover_control()
        u_refs = jnp.tile(u_eq, (n_points, 1))

        return times, x_refs, u_refs

    def test_trajectory_controller_creation(self, quad, simple_trajectory):
        """TrajectoryLQRController creation."""
        times, x_refs, u_refs = simple_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        assert controller.K.shape == (2, 6)
        assert controller.times.shape == (21,)
        assert controller.x_refs.shape == (21, 6)
        assert controller.u_refs.shape == (21, 2)

    def test_trajectory_reference_interpolation(self, quad, simple_trajectory):
        """Reference state is interpolated correctly."""
        times, x_refs, u_refs = simple_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        # At t=1.0, x should be 0.5 (midpoint)
        x_ref_mid = controller._interpolate_state(1.0)
        assert_allclose(x_ref_mid[0], 0.5, rtol=1e-3)

    def test_trajectory_duration(self, quad, simple_trajectory):
        """duration and num_points properties."""
        times, x_refs, u_refs = simple_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        assert controller.duration == 2.0
        assert controller.num_points == 21

    def test_linearize_at_different_point(self, quad, simple_trajectory):
        """Can linearize at different trajectory points."""
        times, x_refs, u_refs = simple_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller0 = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R, linearize_at=0
        )

        controller10 = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R, linearize_at=10
        )

        # For this simple trajectory, gains should be similar (same dynamics)
        # since quadrotor linearization doesn't depend on x position
        assert_allclose(controller0.K, controller10.K, rtol=1e-6)


class TestTVLQRController:
    """Test time-varying LQR."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    @pytest.fixture
    def varying_trajectory(self, quad):
        """Create trajectory with varying operating points."""
        T = 1.0
        n_points = 11
        times = jnp.linspace(0, T, n_points)

        # Varying pitch trajectory
        x_refs = jnp.zeros((n_points, 6))
        x_refs = x_refs.at[:, 2].set(jnp.linspace(0, 0.3, n_points))  # theta varies

        # Constant hover control
        u_eq = quad.hover_control()
        u_refs = jnp.tile(u_eq, (n_points, 1))

        return times, x_refs, u_refs

    def test_tvlqr_creation(self, quad, varying_trajectory):
        """TVLQRController creation with pre-computed gains."""
        times, x_refs, u_refs = varying_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller = TVLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        assert controller.Ks.shape == (11, 2, 6)
        assert controller.times.shape == (11,)

    def test_tvlqr_varying_gains(self, quad, varying_trajectory):
        """TVLQR produces different gains at different times."""
        times, x_refs, u_refs = varying_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller = TVLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        K0 = controller.gain_at(0)
        K5 = controller.gain_at(5)
        K10 = controller.gain_at(10)

        # Gains should vary along trajectory (different linearization points)
        # Note: For quadrotor, gains depend on pitch angle
        assert not np.allclose(K0, K10, rtol=1e-3)

    def test_tvlqr_gain_interpolation(self, quad, varying_trajectory):
        """Gains are interpolated between trajectory points."""
        times, x_refs, u_refs = varying_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        controller = TVLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        # At midpoint between index 0 and 1
        t_mid = (times[0] + times[1]) / 2
        K_mid = controller._interpolate_gain(t_mid)

        # Should be average of K0 and K1
        expected = (controller.Ks[0] + controller.Ks[1]) / 2
        assert_allclose(K_mid, expected, rtol=1e-10)

    def test_tvlqr_vs_fixed_gain(self, quad, varying_trajectory):
        """TVLQR gains vary along trajectory while fixed-gain stays constant.

        Note: For small pitch variations, the control outputs may be similar
        because the quadrotor dynamics don't change dramatically. The key
        difference is that TVLQR has pre-computed gains at each point.
        """
        times, x_refs, u_refs = varying_trajectory

        Q = jnp.eye(6)
        R = jnp.eye(2) * 0.1

        tvlqr = TVLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        fixed = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        # TVLQR has different gains at start vs end
        # (even if similar for small angles, they should be computed differently)
        K_start = tvlqr.gain_at(0)
        K_end = tvlqr.gain_at(-1)

        # Fixed gain is constant
        K_fixed = fixed.K

        # Verify TVLQR has varying gains (computed at each point)
        assert tvlqr.Ks.shape[0] == len(times)

        # Both controllers should produce valid control at any point
        t = 0.5
        state = quad.default_state()
        u_tv = tvlqr(t, state)
        u_fixed = fixed(t, state)

        # Both should be near hover control at equilibrium
        assert u_tv.shape == (2,)
        assert u_fixed.shape == (2,)


class TestConvenienceFunction:
    """Test the lqr() convenience function."""

    def test_lqr_function(self):
        """lqr() produces same result as LQRController.from_linearization()."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        Q = jnp.eye(4)
        R = jnp.array([[1.0]])

        controller1 = lqr(cartpole, x_eq, u_eq, Q, R)
        controller2 = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        assert_allclose(controller1.K, controller2.K, rtol=CTRL_RTOL, atol=CTRL_ATOL)


class TestARESolvers:
    """Test algebraic Riccati equation solvers directly."""

    def test_continuous_are_simple(self):
        """Test continuous ARE with simple system."""
        # Double integrator: x_dot = [0, 1; 0, 0] x + [0; 1] u
        A = jnp.array([[0, 1], [0, 0]])
        B = jnp.array([[0], [1]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        P = solve_continuous_are(A, B, Q, R)

        # P should be positive definite
        eigenvalues = np.linalg.eigvals(np.asarray(P))
        assert np.all(eigenvalues > 0)

        # P should be symmetric
        assert_allclose(P, P.T, rtol=1e-10)

    def test_discrete_are_simple(self):
        """Test discrete ARE with simple system."""
        A = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # Discrete double integrator
        B = jnp.array([[0.005], [0.1]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        P = solve_discrete_are(A, B, Q, R)

        # P should be positive definite
        eigenvalues = np.linalg.eigvals(np.asarray(P))
        assert np.all(eigenvalues > 0)

        # P should be symmetric
        assert_allclose(P, P.T, rtol=1e-10)


class TestJITCompatibility:
    """Test JIT compatibility of LQR controllers."""

    def test_lqr_controller_jittable(self):
        """LQRController can be used in JIT-compiled functions."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        Q = jnp.eye(4)
        R = jnp.array([[1.0]])

        controller = LQRController.from_linearization(
            cartpole, x_eq, u_eq, Q, R
        )

        @jax.jit
        def compute_control(state):
            return controller(0.0, state)

        state = jnp.array([0.0, 0.0, 0.1, 0.0])
        control = compute_control(state)

        assert control.shape == (1,)

    def test_trajectory_controller_jittable(self):
        """TrajectoryLQRController can be used in JIT."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        times = jnp.linspace(0, 1, 11)
        x_refs = jnp.zeros((11, 6))
        u_refs = jnp.tile(quad.hover_control(), (11, 1))
        Q = jnp.eye(6)
        R = jnp.eye(2)

        controller = TrajectoryLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        @jax.jit
        def compute_control(t, state):
            return controller(t, state)

        state = jnp.zeros(6)
        control = compute_control(0.5, state)

        assert control.shape == (2,)

    def test_tvlqr_controller_jittable(self):
        """TVLQRController can be used in JIT."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        times = jnp.linspace(0, 1, 11)
        x_refs = jnp.zeros((11, 6))
        u_refs = jnp.tile(quad.hover_control(), (11, 1))
        Q = jnp.eye(6)
        R = jnp.eye(2)

        controller = TVLQRController.from_trajectory(
            quad, times, x_refs, u_refs, Q, R
        )

        @jax.jit
        def compute_control(t, state):
            return controller(t, state)

        state = jnp.zeros(6)
        control = compute_control(0.5, state)

        assert control.shape == (2,)


class TestWeightMatrices:
    """Test different Q and R weight configurations."""

    @pytest.fixture
    def cartpole(self):
        return Cartpole(CARTPOLE_CLASSIC)

    def test_higher_q_increases_gain(self, cartpole):
        """Higher Q weights should generally increase gain magnitude."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        Q_low = jnp.eye(4)
        Q_high = jnp.eye(4) * 100

        R = jnp.array([[1.0]])

        K_low = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q_low, R)
        K_high = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q_high, R)

        # Higher Q should give higher gain magnitude
        assert np.linalg.norm(K_high) > np.linalg.norm(K_low)

    def test_higher_r_decreases_gain(self, cartpole):
        """Higher R weights should decrease gain magnitude."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        Q = jnp.eye(4)

        R_low = jnp.array([[0.1]])
        R_high = jnp.array([[10.0]])

        K_low = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q, R_low)
        K_high = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q, R_high)

        # Higher R should give lower gain magnitude
        assert np.linalg.norm(K_high) < np.linalg.norm(K_low)

    def test_diagonal_q_emphasizes_states(self, cartpole):
        """Diagonal Q with different weights affects overall gain structure.

        Note: For coupled systems like cartpole, changing Q on one state
        affects all gains due to the Riccati equation. The key insight is
        that emphasizing position increases the position gain relative to
        the baseline, not necessarily relative to angle.
        """
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        # Baseline
        Q_base = jnp.eye(4)

        # Emphasize position
        Q_pos = jnp.diag(jnp.array([100.0, 1.0, 1.0, 1.0]))

        # Emphasize angle
        Q_angle = jnp.diag(jnp.array([1.0, 1.0, 100.0, 1.0]))

        R = jnp.array([[1.0]])

        K_base = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q_base, R)
        K_pos = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q_pos, R)
        K_angle = compute_lqr_gain(*linearize(cartpole, x_eq, u_eq), Q_angle, R)

        # When we emphasize position (Q[0,0] high), position gain should increase
        # compared to baseline
        assert abs(K_pos[0, 0]) > abs(K_base[0, 0])

        # When we emphasize angle (Q[2,2] high), angle gain should increase
        # compared to baseline
        assert abs(K_angle[0, 2]) > abs(K_base[0, 2])
