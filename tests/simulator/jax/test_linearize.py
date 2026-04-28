"""Tests for linearization utilities.

Tests cover:
- Shapes and structure of A, B matrices
- Cartpole linearization at upright (unstable eigenvalue)
- PlanarQuadrotor linearization at hover
- Discretization methods converge for small dt
- Controllability for stabilizable systems
- Regression guard: linearize() uses same dynamics as simulate()
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

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
    from fmd.simulator.linearize import (
        linearize,
        discretize_zoh,
        discretize_euler,
        controllability_matrix,
        is_controllable,
        observability_matrix,
        is_observable,
    )
    from fmd.simulator import Cartpole, PlanarQuadrotor, SimplePendulum
    from fmd.simulator.params import (
        CARTPOLE_CLASSIC,
        PLANAR_QUAD_TEST_DEFAULT,
        PENDULUM_1M,
    )
    from fmd.simulator import simulate, ConstantControl


# Tolerance constants (matching conftest.py)
DERIV_RTOL = 1e-12
DERIV_ATOL = 1e-14
LINEARIZE_RTOL = 1e-10
LINEARIZE_ATOL = 1e-12


class TestLinearizeBasics:
    """Test basic linearization functionality."""

    def test_cartpole_shapes(self):
        """A and B matrices have correct shapes for Cartpole."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        A, B = linearize(cartpole, x_eq, u_eq)

        # Cartpole: 4 states, 1 control
        assert A.shape == (4, 4), f"Expected A shape (4, 4), got {A.shape}"
        assert B.shape == (4, 1), f"Expected B shape (4, 1), got {B.shape}"

    def test_planar_quadrotor_shapes(self):
        """A and B matrices have correct shapes for PlanarQuadrotor."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        A, B = linearize(quad, x_eq, u_eq)

        # PlanarQuadrotor: 6 states, 2 controls
        assert A.shape == (6, 6), f"Expected A shape (6, 6), got {A.shape}"
        assert B.shape == (6, 2), f"Expected B shape (6, 2), got {B.shape}"

    def test_linearize_at_different_times(self):
        """Linearization at different times gives same result for time-invariant systems."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        A0, B0 = linearize(cartpole, x_eq, u_eq, t=0.0)
        A1, B1 = linearize(cartpole, x_eq, u_eq, t=1.0)
        A10, B10 = linearize(cartpole, x_eq, u_eq, t=10.0)

        # Time-invariant system should give identical results
        assert_allclose(A0, A1, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(A0, A10, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(B0, B1, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(B0, B10, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_linearize_accepts_numpy_arrays(self):
        """linearize() accepts both JAX and numpy arrays."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        # Numpy inputs
        x_eq_np = np.zeros(4)
        u_eq_np = np.zeros(1)

        A, B = linearize(cartpole, x_eq_np, u_eq_np)

        assert A.shape == (4, 4)
        assert B.shape == (4, 1)


class TestCartpoleLinearization:
    """Test Cartpole linearization at upright equilibrium."""

    @pytest.fixture
    def cartpole(self):
        return Cartpole(CARTPOLE_CLASSIC)

    @pytest.fixture
    def linearized_cartpole(self, cartpole):
        """Return A, B for Cartpole at upright equilibrium."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        return linearize(cartpole, x_eq, u_eq)

    def test_upright_has_unstable_eigenvalue(self, linearized_cartpole):
        """Cartpole at upright equilibrium should have an unstable eigenvalue.

        The inverted pendulum is unstable at the upright position, which means
        the A matrix should have at least one eigenvalue with positive real part.
        """
        A, B = linearized_cartpole

        eigenvalues = np.linalg.eigvals(np.asarray(A))
        real_parts = np.real(eigenvalues)

        # Should have at least one positive real eigenvalue (unstable)
        assert np.any(real_parts > 0), (
            f"Expected unstable eigenvalue at upright equilibrium. "
            f"Eigenvalues: {eigenvalues}"
        )

    def test_a_matrix_structure(self, linearized_cartpole):
        """A matrix has expected structure for Cartpole.

        State: [x, x_dot, theta, theta_dot]
        dx/dt = [x_dot, x_ddot, theta_dot, theta_ddot]

        A[0,1] = 1 (x_dot depends on x_dot in state)
        A[2,3] = 1 (theta_dot depends on theta_dot in state)
        """
        A, B = linearized_cartpole

        # Check kinematic relationships (exact)
        assert_allclose(A[0, 1], 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dx/dt = x_dot
        assert_allclose(A[2, 3], 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dtheta/dt = theta_dot

        # Positions don't affect velocities directly (at equilibrium)
        assert_allclose(A[0, 0], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dx/dt independent of x
        assert_allclose(A[2, 2], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dtheta/dt independent of theta

    def test_b_matrix_structure(self, linearized_cartpole):
        """B matrix has expected structure for Cartpole.

        Control: [F] (force on cart)
        Force affects x_ddot directly and theta_ddot indirectly.
        """
        A, B = linearized_cartpole

        # Force doesn't affect position derivatives directly
        assert_allclose(B[0, 0], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dx/dt independent of F
        assert_allclose(B[2, 0], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dtheta/dt independent of F

        # Force affects accelerations (exact values depend on physics)
        # x_ddot affected by F
        assert B[1, 0] != 0.0, "Force should affect cart acceleration"
        # theta_ddot affected by F
        assert B[3, 0] != 0.0, "Force should affect pole angular acceleration"

    def test_analytical_linearization(self, cartpole):
        """Compare linearization to analytical derivation at upright.

        For small angles around upright (theta=0):
        The linearized A matrix should match the analytical form.
        """
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        A, B = linearize(cartpole, x_eq, u_eq)

        # Extract parameters
        m_c = cartpole.mass_cart
        m_p = cartpole.mass_pole
        l = cartpole.pole_length
        g = cartpole.g

        # Analytical linearization at upright (theta=0, cos(theta)=1, sin(theta)=0)
        # From Barto, Sutton, Anderson equations:
        # At theta=0:
        #   theta_ddot = (g*theta + (-F)/(m_c + m_p)) / (l*(4/3 - m_p/(m_c + m_p)))
        #   x_ddot = (F + m_p*l*(-theta_ddot)) / (m_c + m_p)

        total_mass = m_c + m_p
        denom = l * (4.0/3.0 - m_p / total_mass)

        # A[3,2] = d(theta_ddot)/d(theta) at equilibrium
        expected_A32 = g / denom  # This should be positive (unstable)

        # A[3,0] = d(theta_ddot)/d(x) = 0 (no position dependence)
        assert_allclose(A[3, 0], 0.0, rtol=1e-8, atol=1e-10)

        # Check theta_ddot depends on theta (the instability)
        assert_allclose(A[3, 2], expected_A32, rtol=1e-6, atol=1e-8)

    def test_equilibrium_derivative_is_zero(self, cartpole):
        """State derivative should be zero at equilibrium."""
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        deriv = cartpole.forward_dynamics(x_eq, u_eq)

        assert_allclose(deriv, jnp.zeros(4), rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestPlanarQuadrotorLinearization:
    """Test PlanarQuadrotor linearization at hover."""

    @pytest.fixture
    def quad(self):
        return PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

    @pytest.fixture
    def linearized_quad(self, quad):
        """Return A, B for PlanarQuadrotor at hover."""
        x_eq = quad.default_state()
        u_eq = quad.hover_control()
        return linearize(quad, x_eq, u_eq)

    def test_hover_is_stable_uncontrolled(self, linearized_quad):
        """Quadrotor at hover has marginally stable eigenvalues (all real parts <= 0).

        Without control, the quadrotor in hover is like a pencil balanced on its tip:
        marginally stable (integrators for position) but not unstable like cartpole.
        """
        A, B = linearized_quad

        eigenvalues = np.linalg.eigvals(np.asarray(A))
        real_parts = np.real(eigenvalues)

        # All eigenvalues should have non-positive real parts (marginally stable)
        # Note: Due to double integrators in position, we expect zero eigenvalues
        assert np.all(real_parts <= 1e-10), (
            f"Expected marginally stable system at hover. "
            f"Eigenvalues: {eigenvalues}"
        )

    def test_a_matrix_structure(self, linearized_quad):
        """A matrix has expected structure for PlanarQuadrotor.

        State: [x, z, theta, x_dot, z_dot, theta_dot]
        dx/dt = [x_dot, z_dot, theta_dot, x_ddot, z_ddot, theta_ddot]
        """
        A, B = linearized_quad

        # Check kinematic relationships
        assert_allclose(A[0, 3], 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dx/dt = x_dot
        assert_allclose(A[1, 4], 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dz/dt = z_dot
        assert_allclose(A[2, 5], 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)  # dtheta/dt = theta_dot

        # Position/velocity don't affect each other (at hover, theta=0)
        assert_allclose(A[0, 0], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(A[1, 1], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(A[2, 2], 0.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_thrust_pitch_coupling(self, linearized_quad, quad):
        """Pitch angle affects horizontal acceleration at hover.

        At hover with theta=0:
        x_ddot = -(T_total/m) * sin(theta) ≈ -(T_total/m) * theta

        So A[3,2] = d(x_ddot)/d(theta) = -(T_total/m) * cos(0) = -T_total/m = -g
        """
        A, B = linearized_quad

        expected_coupling = -quad.g  # Negative because pitch up causes negative x acceleration
        assert_allclose(A[3, 2], expected_coupling, rtol=1e-8, atol=1e-10)

    def test_hover_derivative_is_zero(self, quad):
        """State derivative should be zero at hover equilibrium."""
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        deriv = quad.forward_dynamics(x_eq, u_eq)

        assert_allclose(deriv, jnp.zeros(6), rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestDiscretization:
    """Test discretization methods."""

    @pytest.fixture
    def simple_system(self):
        """A simple stable 2x2 system for testing."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [1.0]])
        return A, B

    @pytest.fixture
    def cartpole_system(self):
        """Cartpole linearization for discretization tests."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])
        return linearize(cartpole, x_eq, u_eq)

    def test_zoh_shapes(self, simple_system):
        """discretize_zoh returns correct shapes."""
        A, B = simple_system
        dt = 0.01

        Ad, Bd = discretize_zoh(A, B, dt)

        assert Ad.shape == A.shape
        assert Bd.shape == B.shape

    def test_euler_shapes(self, simple_system):
        """discretize_euler returns correct shapes."""
        A, B = simple_system
        dt = 0.01

        Ad, Bd = discretize_euler(A, B, dt)

        assert Ad.shape == A.shape
        assert Bd.shape == B.shape

    def test_euler_formula(self, simple_system):
        """discretize_euler follows Ad = I + A*dt, Bd = B*dt."""
        A, B = simple_system
        dt = 0.01

        Ad, Bd = discretize_euler(A, B, dt)

        expected_Ad = jnp.eye(2) + A * dt
        expected_Bd = B * dt

        assert_allclose(Ad, expected_Ad, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(Bd, expected_Bd, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_zoh_euler_converge_small_dt(self, simple_system):
        """ZOH and Euler discretization converge for very small dt.

        For small dt: exp(A*dt) ≈ I + A*dt + O(dt^2)
        """
        A, B = simple_system

        for dt in [1e-4, 1e-5, 1e-6]:
            Ad_zoh, Bd_zoh = discretize_zoh(A, B, dt)
            Ad_euler, Bd_euler = discretize_euler(A, B, dt)

            # Should be very close for small dt
            # Tolerance scales with dt (O(dt^2) error)
            tol = dt * 10  # Allow 10x dt error
            assert_allclose(Ad_zoh, Ad_euler, rtol=tol, atol=tol)
            assert_allclose(Bd_zoh, Bd_euler, rtol=tol, atol=tol)

    def test_zoh_preserves_stability_stable(self, simple_system):
        """ZOH discretization preserves stability for stable systems."""
        A, B = simple_system
        dt = 0.1

        Ad, Bd = discretize_zoh(A, B, dt)

        # Eigenvalues of discrete system
        eigenvalues = np.linalg.eigvals(np.asarray(Ad))

        # Stable continuous system -> stable discrete system
        # (eigenvalues inside unit circle)
        assert np.all(np.abs(eigenvalues) < 1.0)

    def test_zoh_unstable_system(self, cartpole_system):
        """ZOH discretization of unstable system has eigenvalue outside unit circle."""
        A, B = cartpole_system
        dt = 0.01

        Ad, Bd = discretize_zoh(A, B, dt)

        eigenvalues = np.linalg.eigvals(np.asarray(Ad))

        # Unstable continuous system -> unstable discrete system
        # (at least one eigenvalue outside unit circle)
        assert np.any(np.abs(eigenvalues) > 1.0), (
            f"Expected unstable discrete system. "
            f"Eigenvalues: {eigenvalues}, magnitudes: {np.abs(eigenvalues)}"
        )

    def test_zoh_identity_at_zero_dt(self, simple_system):
        """ZOH at dt=0 gives identity transition (approximately).

        Note: dt=0 exactly would be degenerate, so we test very small dt.
        """
        A, B = simple_system
        dt = 1e-10

        Ad, Bd = discretize_zoh(A, B, dt)

        assert_allclose(Ad, jnp.eye(2), rtol=1e-6, atol=1e-8)
        assert_allclose(Bd, jnp.zeros_like(B), rtol=1e-6, atol=1e-8)


class TestControllability:
    """Test controllability matrix and checking."""

    def test_controllability_matrix_shape(self):
        """Controllability matrix has correct shape."""
        A = jnp.array([[0, 1], [-1, -1]])
        B = jnp.array([[0], [1]])

        C = controllability_matrix(A, B)

        # Shape: (n, n*m) = (2, 2*1) = (2, 2)
        assert C.shape == (2, 2)

    def test_controllability_matrix_multi_input(self):
        """Controllability matrix for multi-input system."""
        A = jnp.array([[0, 1], [-1, -1]])
        B = jnp.array([[1, 0], [0, 1]])  # 2 inputs

        C = controllability_matrix(A, B)

        # Shape: (n, n*m) = (2, 2*2) = (2, 4)
        assert C.shape == (2, 4)

    def test_controllability_matrix_columns(self):
        """Controllability matrix columns are [B, AB, A^2B, ...]."""
        A = jnp.array([[0, 1], [-1, -1]])
        B = jnp.array([[0], [1]])

        C = controllability_matrix(A, B)

        # First column = B
        assert_allclose(C[:, 0:1], B, rtol=DERIV_RTOL, atol=DERIV_ATOL)

        # Second column = AB
        AB = A @ B
        assert_allclose(C[:, 1:2], AB, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_controllable_system(self):
        """A controllable system is detected as controllable."""
        # Simple harmonic oscillator with force input
        A = jnp.array([[0, 1], [-1, 0]])
        B = jnp.array([[0], [1]])

        assert is_controllable(A, B)

    def test_uncontrollable_system(self):
        """An uncontrollable system is detected as not controllable."""
        # Two decoupled states, only one has control
        A = jnp.array([[0, 0], [0, -1]])
        B = jnp.array([[0], [1]])

        # State x1 is uncontrollable (no path from control to it)
        assert not is_controllable(A, B)

    def test_cartpole_controllable(self):
        """Cartpole at upright is controllable."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        A, B = linearize(cartpole, x_eq, u_eq)

        assert is_controllable(A, B), "Cartpole should be controllable"

    def test_planar_quadrotor_controllable(self):
        """PlanarQuadrotor at hover is controllable."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        x_eq = quad.default_state()
        u_eq = quad.hover_control()

        A, B = linearize(quad, x_eq, u_eq)

        assert is_controllable(A, B), "PlanarQuadrotor should be controllable"


class TestObservability:
    """Test observability matrix and checking."""

    def test_observability_matrix_shape(self):
        """Observability matrix has correct shape."""
        A = jnp.array([[0, 1], [-1, -1]])
        C = jnp.array([[1, 0]])  # Measure position only

        O = observability_matrix(A, C)

        # Shape: (n*p, n) = (2*1, 2) = (2, 2)
        assert O.shape == (2, 2)

    def test_observability_matrix_rows(self):
        """Observability matrix rows are [C; CA; CA^2; ...]."""
        A = jnp.array([[0, 1], [-1, -1]])
        C = jnp.array([[1, 0]])

        O = observability_matrix(A, C)

        # First row = C
        assert_allclose(O[0:1, :], C, rtol=DERIV_RTOL, atol=DERIV_ATOL)

        # Second row = CA
        CA = C @ A
        assert_allclose(O[1:2, :], CA, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_observable_full_state(self):
        """Full state measurement is always observable."""
        A = jnp.array([[0, 1], [-1, -1]])
        C = jnp.eye(2)  # Full state measurement

        assert is_observable(A, C)

    def test_observable_partial_measurement(self):
        """Position-only measurement can still be observable."""
        A = jnp.array([[0, 1], [-1, -1]])
        C = jnp.array([[1, 0]])  # Position only

        # Observable if (A, C) is observable
        # O = [C; CA] = [[1, 0]; [0, 1]] -> rank 2
        assert is_observable(A, C)

    def test_unobservable_system(self):
        """An unobservable system is detected as not observable."""
        # Two decoupled states, only one measured
        A = jnp.array([[0, 0], [0, -1]])
        C = jnp.array([[0, 1]])  # Only measure second state

        # First state is unobservable
        assert not is_observable(A, C)


class TestLinearizeDynamicsConsistency:
    """Test that linearize uses the same dynamics as simulation.

    This is a critical regression guard: if someone changes the dynamics
    model, linearization should automatically use the new dynamics.
    """

    def test_linearized_prediction_matches_simulation(self):
        """Linear prediction matches nonlinear simulation for small perturbations.

        For small deviations from equilibrium, the linearized model should
        closely predict the behavior of the nonlinear simulation.
        """
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        # Get linearization
        A, B = linearize(cartpole, x_eq, u_eq)

        # Small perturbation
        delta_x = jnp.array([0.0, 0.0, 0.01, 0.0])  # Small angle
        x0 = x_eq + delta_x

        # Linear prediction: dx/dt ≈ A @ delta_x
        linear_deriv = A @ delta_x

        # Nonlinear derivative
        nonlinear_deriv = cartpole.forward_dynamics(x0, u_eq)

        # Should be close for small perturbations
        assert_allclose(linear_deriv, nonlinear_deriv, rtol=1e-2, atol=1e-4)

    def test_linearized_trajectory_matches_simulation(self):
        """Linear trajectory prediction matches simulation for short times.

        Integrate the linearized system and compare to full simulation.
        """
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        # Get linearization and discretize
        A, B = linearize(cartpole, x_eq, u_eq)
        dt = 0.001
        Ad, Bd = discretize_euler(A, B, dt)  # Use Euler for comparison

        # Small perturbation
        delta_x = jnp.array([0.0, 0.0, 0.005, 0.0])  # Very small angle
        x0 = x_eq + delta_x

        # Simulate nonlinear system for short duration
        duration = 0.01
        control = ConstantControl(u_eq)
        result = simulate(cartpole, x0, dt=dt, duration=duration, control=control)

        # Simulate linear system
        n_steps = int(duration / dt)
        x_linear = delta_x
        for _ in range(n_steps):
            x_linear = Ad @ x_linear + Bd @ (u_eq - u_eq)  # delta_u = 0

        # Final states (linear is relative to equilibrium)
        final_nonlinear = result.states[-1]
        final_linear_absolute = x_eq + x_linear

        # Should match closely for small perturbations and short times
        assert_allclose(final_linear_absolute, final_nonlinear, rtol=1e-2, atol=1e-4)

    def test_linearize_uses_forward_dynamics_directly(self):
        """Verify linearize uses system.forward_dynamics() method.

        This ensures any preprocessing/postprocessing in forward_dynamics
        is captured in the linearization.
        """
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        # Get linearization
        A, B = linearize(cartpole, x_eq, u_eq)

        # Manually compute Jacobians using finite differences
        eps = 1e-7
        A_fd = np.zeros((4, 4))
        B_fd = np.zeros((4, 1))

        f0 = cartpole.forward_dynamics(x_eq, u_eq)

        for i in range(4):
            x_pert = x_eq.at[i].set(x_eq[i] + eps)
            f_pert = cartpole.forward_dynamics(x_pert, u_eq)
            A_fd[:, i] = (np.asarray(f_pert) - np.asarray(f0)) / eps

        for i in range(1):
            u_pert = u_eq.at[i].set(u_eq[i] + eps)
            f_pert = cartpole.forward_dynamics(x_eq, u_pert)
            B_fd[:, i] = (np.asarray(f_pert) - np.asarray(f0)) / eps

        # JAX autodiff should match finite differences
        assert_allclose(A, A_fd, rtol=1e-5, atol=1e-7)
        assert_allclose(B, B_fd, rtol=1e-5, atol=1e-7)


class TestPendulumLinearization:
    """Test SimplePendulum linearization for comparison."""

    def test_pendulum_linearization_at_rest(self):
        """SimplePendulum linearization at bottom (stable equilibrium)."""
        pendulum = SimplePendulum(PENDULUM_1M)
        x_eq = jnp.array([0.0, 0.0])  # Bottom (theta=0)
        u_eq = jnp.array([])  # No control

        A, B = linearize(pendulum, x_eq, u_eq)

        # Shape: 2 states, 0 controls
        assert A.shape == (2, 2)
        assert B.shape == (2, 0)

        # A matrix structure:
        # [dtheta/dt]     [0, 1] [theta]
        # [dtheta_ddot/dt] = [-g/l, 0] [theta_dot]
        expected_A = jnp.array([
            [0.0, 1.0],
            [-pendulum.g / pendulum.length, 0.0]
        ])
        assert_allclose(A, expected_A, rtol=1e-10, atol=1e-12)

    def test_pendulum_stable_eigenvalues(self):
        """SimplePendulum at bottom has purely imaginary eigenvalues (stable)."""
        pendulum = SimplePendulum(PENDULUM_1M)
        x_eq = jnp.array([0.0, 0.0])
        u_eq = jnp.array([])

        A, B = linearize(pendulum, x_eq, u_eq)

        eigenvalues = np.linalg.eigvals(np.asarray(A))

        # Should be purely imaginary (center, marginally stable)
        # ±j*sqrt(g/l)
        real_parts = np.real(eigenvalues)
        assert_allclose(real_parts, [0.0, 0.0], atol=1e-10)

        # Imaginary parts should be ±sqrt(g/l)
        expected_freq = np.sqrt(pendulum.g / pendulum.length)
        imag_parts = np.abs(np.imag(eigenvalues))
        assert_allclose(sorted(imag_parts), [expected_freq, expected_freq], rtol=1e-10)


class TestJITCompatibility:
    """Test that linearization works with JIT compilation."""

    def test_linearize_jittable(self):
        """linearize() output can be used in JIT-compiled functions."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_eq = cartpole.upright_state()
        u_eq = jnp.array([0.0])

        # Get linearization (not JIT-able itself due to jacobian)
        A, B = linearize(cartpole, x_eq, u_eq)

        # But the results should work in JIT
        @jax.jit
        def linear_dynamics(delta_x):
            return A @ delta_x

        delta_x = jnp.array([0.0, 0.0, 0.1, 0.0])
        result = linear_dynamics(delta_x)

        assert result.shape == (4,)

    def test_discretize_zoh_jittable(self):
        """discretize_zoh() produces JIT-compatible outputs."""
        A = jnp.array([[0, 1], [-1, -1]])
        B = jnp.array([[0], [1]])

        Ad, Bd = discretize_zoh(A, B, 0.01)

        @jax.jit
        def discrete_dynamics(x, u):
            return Ad @ x + Bd @ u

        x = jnp.array([1.0, 0.0])
        u = jnp.array([0.5])
        result = discrete_dynamics(x, u)

        assert result.shape == (2,)

    def test_discretize_euler_fully_jittable(self):
        """discretize_euler() can be JIT-compiled entirely."""
        @jax.jit
        def jit_euler(A, B, dt):
            n = A.shape[0]
            I = jnp.eye(n)
            Ad = I + A * dt
            Bd = B * dt
            return Ad, Bd

        A = jnp.array([[0, 1], [-1, -1]])
        B = jnp.array([[0], [1]])
        dt = 0.01

        Ad_jit, Bd_jit = jit_euler(A, B, dt)
        Ad, Bd = discretize_euler(A, B, dt)

        assert_allclose(Ad_jit, Ad, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        assert_allclose(Bd_jit, Bd, rtol=DERIV_RTOL, atol=DERIV_ATOL)
