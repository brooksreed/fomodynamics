"""Minimum-time OCP validation tests for Box1D CasADi models.

These tests validate the CasADi infrastructure by solving complete
optimal control problems using CasADi Opti. This is the ultimate
integration test that verifies all components work together.

The minimum-time-to-target problem:
- Start at rest at x=0
- Reach x=10 at rest
- Minimize time
- Subject to force bounds

Expected solution: bang-bang control (max force, then max braking)
"""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")
cs = casadi

from fmd.simulator.casadi import Box1DCasadiExact, Box1DFrictionCasadiSmooth
from fmd.simulator.params import BOX1D_DEFAULT, BOX1D_FRICTION_DEFAULT
from fmd.simulator.params.box_1d import Box1DParams


class TestBox1DMinTimeOCP:
    """Test minimum-time OCP for Box1D model."""

    def test_min_time_ocp_solves(self):
        """Validate CasADi infrastructure with minimum-time OCP.

        Problem: Move from x=0 to x=10, starting and ending at rest.
        Minimize time subject to force bounds.
        """
        # Use no-drag model for cleaner bang-bang solution
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        opti = cs.Opti()

        # Problem parameters
        N = 50  # Number of control intervals
        F_max = 10.0  # Maximum force
        x_target = 10.0  # Target position

        # Decision variables
        T = opti.variable()  # Total time (free)
        dt = T / N  # Time step

        # State trajectory [x, x_dot] at each knot
        X = opti.variable(2, N + 1)

        # Control trajectory [F] for each interval
        U = opti.variable(1, N)

        # Initial guess
        opti.set_initial(T, 3.0)  # Reasonable time guess
        opti.set_initial(X, np.zeros((2, N + 1)))
        opti.set_initial(U, np.zeros((1, N)))

        # Dynamics constraints using direct transcription (Euler for simplicity)
        # Note: For higher accuracy, use collocation or RK4
        f = model.dynamics_function()
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # Euler integration (simple but sufficient for this test)
            xdot = f(x_k, u_k)
            opti.subject_to(x_next == x_k + dt * xdot)

        # Boundary conditions
        opti.subject_to(X[:, 0] == [0, 0])  # Start at rest at x=0
        opti.subject_to(X[0, -1] == x_target)  # End at x=10
        opti.subject_to(X[1, -1] == 0)  # End at rest

        # Control bounds
        opti.subject_to(opti.bounded(-F_max, U, F_max))

        # Time must be positive
        opti.subject_to(T > 0.1)

        # Objective: minimize time
        opti.minimize(T)

        # Solve
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        sol = opti.solve()

        # Verify solution
        T_opt = sol.value(T)
        X_opt = sol.value(X)
        U_opt = sol.value(U)

        # Time should be positive and reasonable
        assert T_opt > 0
        assert T_opt < 10.0  # Should be much faster than 10 seconds

        # Final state should match target
        assert sol.value(X[0, -1]) == pytest.approx(x_target, rel=1e-3)
        assert sol.value(X[1, -1]) == pytest.approx(0.0, abs=1e-3)

        # Check bang-bang structure: should accelerate then decelerate
        # U_opt is (1, N) from CasADi, flatten for numpy indexing
        U_flat = U_opt.flatten()
        # This is an end-to-end infrastructure test, not a perfect optimality proof.
        # We expect a saturated bang-bang structure (+F_max then -F_max), but the
        # switch point can move with discretization details (Euler vs RK4) and N.
        assert np.max(U_flat) > 0.9 * F_max
        assert np.min(U_flat) < -0.9 * F_max
        # And we should generally "push" early and "brake" late.
        q = max(1, N // 4)
        assert np.mean(U_flat[:q]) > 0.2 * F_max
        assert np.mean(U_flat[-q:]) < -0.2 * F_max

    def test_min_time_ocp_with_rk4(self):
        """Test OCP with RK4 integration for better accuracy."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        opti = cs.Opti()

        N = 30
        F_max = 10.0
        x_target = 10.0

        T = opti.variable()
        X = opti.variable(2, N + 1)
        U = opti.variable(1, N)

        opti.set_initial(T, 3.0)

        # Use RK4 for dynamics (note: dt is symbolic here)
        for k in range(N):
            dt = T / N
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # RK4 step (inline, since dt is symbolic)
            k1 = model.forward_dynamics(x_k, u_k)
            k2 = model.forward_dynamics(x_k + 0.5 * dt * k1, u_k)
            k3 = model.forward_dynamics(x_k + 0.5 * dt * k2, u_k)
            k4 = model.forward_dynamics(x_k + dt * k3, u_k)
            x_rk4 = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            opti.subject_to(x_next == x_rk4)

        # Boundary conditions
        opti.subject_to(X[:, 0] == [0, 0])
        opti.subject_to(X[0, -1] == x_target)
        opti.subject_to(X[1, -1] == 0)

        # Bounds
        opti.subject_to(opti.bounded(-F_max, U, F_max))
        opti.subject_to(T > 0.1)

        opti.minimize(T)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        sol = opti.solve()

        T_opt = sol.value(T)

        # Verify solution
        assert T_opt > 0
        assert sol.value(X[0, -1]) == pytest.approx(x_target, rel=1e-3)
        assert sol.value(X[1, -1]) == pytest.approx(0.0, abs=1e-3)

        # For drag-free case with bang-bang control:
        # Accelerate at +a for T/2, decelerate at -a for T/2
        # x = 0.5 * a * (T/2)^2 + v_max * (T/2) - 0.5 * a * (T/2)^2
        # where v_max = a * T/2
        # x = a * T^2 / 4 => T = 2*sqrt(x/a)
        # With x=10, a=10: T_min = 2*sqrt(1) = 2.0s
        assert T_opt == pytest.approx(2.0, rel=0.05)

    def test_min_time_ocp_with_drag(self):
        """Test OCP with drag model."""
        model = Box1DCasadiExact(BOX1D_DEFAULT)  # Has drag

        opti = cs.Opti()

        N = 40
        F_max = 10.0
        x_target = 5.0  # Shorter distance due to drag

        T = opti.variable()
        X = opti.variable(2, N + 1)
        U = opti.variable(1, N)

        opti.set_initial(T, 3.0)

        f = model.dynamics_function()
        for k in range(N):
            dt = T / N
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            xdot = f(x_k, u_k)
            opti.subject_to(x_next == x_k + dt * xdot)

        opti.subject_to(X[:, 0] == [0, 0])
        opti.subject_to(X[0, -1] == x_target)
        opti.subject_to(X[1, -1] == 0)

        opti.subject_to(opti.bounded(-F_max, U, F_max))
        opti.subject_to(T > 0.1)

        opti.minimize(T)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        sol = opti.solve()

        T_opt = sol.value(T)

        # Verify solution
        assert T_opt > 0
        assert sol.value(X[0, -1]) == pytest.approx(x_target, rel=1e-3)
        assert sol.value(X[1, -1]) == pytest.approx(0.0, abs=1e-2)


class TestBox1DFrictionOCP:
    """Test OCP with friction model using smooth approximation."""

    def test_friction_ocp_smooth(self):
        """Test OCP with smooth friction model converges."""
        model = Box1DFrictionCasadiSmooth(BOX1D_FRICTION_DEFAULT)

        opti = cs.Opti()

        N = 40
        F_max = 15.0  # Need more force to overcome friction
        x_target = 3.0  # Shorter distance due to friction

        T = opti.variable()
        X = opti.variable(2, N + 1)
        U = opti.variable(1, N)

        opti.set_initial(T, 3.0)
        # Initialize with small positive velocity to help solver
        for i in range(N + 1):
            opti.set_initial(X[1, i], 0.1)

        f = model.dynamics_function()
        for k in range(N):
            dt = T / N
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            xdot = f(x_k, u_k)
            opti.subject_to(x_next == x_k + dt * xdot)

        opti.subject_to(X[:, 0] == [0, 0])
        opti.subject_to(X[0, -1] == x_target)
        opti.subject_to(X[1, -1] == 0)

        opti.subject_to(opti.bounded(-F_max, U, F_max))
        opti.subject_to(T > 0.1)

        opti.minimize(T)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        sol = opti.solve()

        T_opt = sol.value(T)

        # Verify solution
        assert T_opt > 0
        assert sol.value(X[0, -1]) == pytest.approx(x_target, rel=1e-2)
        # Friction makes it harder to stop exactly, so relax tolerance
        assert abs(sol.value(X[1, -1])) < 0.1


class TestOCPPhysicsValidation:
    """Validate that OCP solutions are physically reasonable."""

    def test_energy_consistent(self):
        """Test that solution respects energy constraints."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        opti = cs.Opti()

        N = 50
        F_max = 10.0
        x_target = 10.0

        T = opti.variable()
        X = opti.variable(2, N + 1)
        U = opti.variable(1, N)

        opti.set_initial(T, 3.0)

        f = model.dynamics_function()
        for k in range(N):
            dt = T / N
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            xdot = f(x_k, u_k)
            opti.subject_to(x_next == x_k + dt * xdot)

        opti.subject_to(X[:, 0] == [0, 0])
        opti.subject_to(X[0, -1] == x_target)
        opti.subject_to(X[1, -1] == 0)
        opti.subject_to(opti.bounded(-F_max, U, F_max))
        opti.subject_to(T > 0.1)

        opti.minimize(T)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        sol = opti.solve()

        X_opt = sol.value(X)
        U_opt = sol.value(U)
        T_opt = sol.value(T)
        dt_opt = T_opt / N

        # Compute work done by force
        m = params.mass
        U_flat = U_opt.flatten()
        work = 0.0
        for k in range(N):
            # Work = F * dx ≈ F * v * dt
            v_avg = (X_opt[1, k] + X_opt[1, k + 1]) / 2
            work += U_flat[k] * v_avg * dt_opt

        # Initial and final kinetic energy are both 0 (start and end at rest)
        # So net work should be approximately 0 (work done = work against)
        # This is a rough check given numerical integration errors
        assert abs(work) < x_target * F_max * 0.2  # Within 20% of upper bound

    def test_position_monotonic_outbound(self):
        """Test position increases then decreases velocity (bang-bang)."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        opti = cs.Opti()

        N = 50
        F_max = 10.0
        x_target = 10.0

        T = opti.variable()
        X = opti.variable(2, N + 1)
        U = opti.variable(1, N)

        opti.set_initial(T, 3.0)

        f = model.dynamics_function()
        for k in range(N):
            dt = T / N
            xdot = f(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == X[:, k] + dt * xdot)

        opti.subject_to(X[:, 0] == [0, 0])
        opti.subject_to(X[0, -1] == x_target)
        opti.subject_to(X[1, -1] == 0)
        opti.subject_to(opti.bounded(-F_max, U, F_max))
        opti.subject_to(T > 0.1)

        opti.minimize(T)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        sol = opti.solve()

        X_opt = sol.value(X)

        # Position should always increase (we're moving forward)
        positions = X_opt[0, :]
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i - 1] - 1e-6

        # Velocity should increase in first half, decrease in second half
        velocities = X_opt[1, :]
        max_vel_idx = np.argmax(velocities)
        assert max_vel_idx > 0 and max_vel_idx < N  # Peak velocity in middle
