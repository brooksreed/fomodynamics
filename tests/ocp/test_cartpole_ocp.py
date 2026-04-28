"""Waiter's Task OCP tests for cartpole.

The Waiter's Task: Move the cartpole from [0,0,0,0] to [3,0,0,0] while keeping
the pole angle within ±5° (0.0873 rad) - like carrying a tray without spilling.

Test problem specification:
| Parameter        | Value           | Notes                    |
|-----------------|-----------------|--------------------------|
| Start state     | [0, 0, 0, 0]    | Upright at origin        |
| Target state    | [3, 0, 0, 0]    | Upright at x=3m, at rest |
| State constraint| |theta| ≤ 5°    | 0.0873 rad               |
| Control bound   | |F| ≤ 10N       | Hard constraint          |
| Time horizon    | T=4s (fixed)    | N=200 steps              |
| Terminal cost   | Qf = 1000 * I   | Drive to target          |
"""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")

from fmd.ocp import MultipleShootingOCP, OCPResult
from fmd.simulator.casadi import CartpoleCasadiExact
from fmd.simulator.params import CARTPOLE_CLASSIC


# Problem constants
THETA_MAX = 0.0873  # 5 degrees in radians
F_MAX = 10.0  # Maximum control force
X_TARGET = 3.0  # Target cart position
T_FIXED = 4.0  # Fixed time horizon
N_STEPS = 200  # Discretization steps


class TestFixedTimeWaitersTask:
    """Test fixed-time (T=4s) Waiter's Task OCP."""

    @pytest.fixture
    def model(self):
        """Create cartpole model with classic parameters."""
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_waiters_task_converges(self, model):
        """Test that the fixed-time Waiter's Task OCP converges."""
        ocp = MultipleShootingOCP(model, N=N_STEPS, T_fixed=T_FIXED)

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)  # theta
        ocp.set_terminal_cost(1000 * np.eye(4))

        result = ocp.solve()

        assert result.converged, f"Solver failed: {result.solver_stats}"

    def test_waiters_task_theta_constraint(self, model):
        """Test that theta stays within ±5° constraint."""
        ocp = MultipleShootingOCP(model, N=N_STEPS, T_fixed=T_FIXED)

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))

        result = ocp.solve()

        assert result.converged

        # Check theta constraint (with small numerical tolerance)
        theta_traj = result.states[:, 2]
        max_theta = np.max(np.abs(theta_traj))

        # Allow 0.2% tolerance for numerical solver
        assert max_theta < THETA_MAX * 1.002, (
            f"Theta constraint violated: max |theta| = {np.rad2deg(max_theta):.3f}° > 5°"
        )

    def test_waiters_task_control_bounds(self, model):
        """Test that control stays within ±10N bounds."""
        ocp = MultipleShootingOCP(model, N=N_STEPS, T_fixed=T_FIXED)

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))

        result = ocp.solve()

        assert result.converged

        # Control bounds should be exactly satisfied by IPOPT
        max_control = np.max(np.abs(result.controls))
        assert max_control <= F_MAX + 1e-6, (
            f"Control bound violated: max |F| = {max_control:.3f}N > {F_MAX}N"
        )

    def test_waiters_task_terminal_state(self, model):
        """Test that terminal state is close to target."""
        ocp = MultipleShootingOCP(model, N=N_STEPS, T_fixed=T_FIXED)

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))

        result = ocp.solve()

        assert result.converged

        # Check terminal state
        x_final = result.states[-1]
        target = np.array([X_TARGET, 0, 0, 0])

        # Position accuracy
        assert abs(x_final[0] - X_TARGET) < 0.05, (
            f"Final position error: {abs(x_final[0] - X_TARGET):.4f}m"
        )
        # Velocity should be small
        assert abs(x_final[1]) < 0.1, f"Final velocity: {x_final[1]:.4f} m/s"
        # Angle should be nearly zero
        assert abs(x_final[2]) < np.deg2rad(1), f"Final angle: {np.rad2deg(x_final[2]):.4f}°"
        # Angular velocity should be small
        assert abs(x_final[3]) < np.deg2rad(5), f"Final omega: {np.rad2deg(x_final[3]):.4f}°/s"

    def test_waiters_task_physical_reasonableness(self, model):
        """Test that solution is physically reasonable."""
        ocp = MultipleShootingOCP(model, N=N_STEPS, T_fixed=T_FIXED)

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))

        result = ocp.solve()

        assert result.converged

        # Cart should move monotonically towards target (roughly)
        x_traj = result.states[:, 0]
        # At least 80% of the trajectory should be increasing
        increases = np.sum(np.diff(x_traj) > -0.01)  # Small tolerance for noise
        assert increases > 0.8 * N_STEPS, "Cart position not monotonically increasing"

        # Cart should start at 0 and end near X_TARGET
        assert x_traj[0] == pytest.approx(0, abs=1e-6)
        assert x_traj[-1] == pytest.approx(X_TARGET, abs=0.1)


class TestMinTimeWaitersTask:
    """Test free-time (minimum-time) Waiter's Task OCP."""

    @pytest.fixture
    def model(self):
        """Create cartpole model with classic parameters."""
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_mintime_waiters_task_converges(self, model):
        """Test that the minimum-time Waiter's Task converges."""
        ocp = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged, f"Solver failed: {result.solver_stats}"

    def test_mintime_waiters_task_time_in_range(self, model):
        """Test that optimal time is in reasonable range."""
        ocp = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged

        # Expect T to be in reasonable range based on physics
        # Without theta constraint, a simple move would be faster
        # With constraint, expect roughly 2-6 seconds
        assert 2.0 < result.T < 6.0, (
            f"Optimal time T={result.T:.2f}s outside expected range [2, 6]"
        )

    def test_mintime_waiters_task_constraints_satisfied(self, model):
        """Test that all constraints are satisfied in minimum-time solution."""
        ocp = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )

        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged

        # Theta constraint
        max_theta = np.max(np.abs(result.states[:, 2]))
        assert max_theta < THETA_MAX * 1.002

        # Control bounds
        max_control = np.max(np.abs(result.controls))
        assert max_control <= F_MAX + 1e-6

        # Terminal state
        x_final = result.states[-1]
        assert abs(x_final[0] - X_TARGET) < 0.05
        assert abs(x_final[1]) < 0.1


class TestUnconstrainedComparison:
    """Compare constrained vs unconstrained solutions."""

    @pytest.fixture
    def model(self):
        """Create cartpole model with classic parameters."""
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_unconstrained_faster_than_constrained(self, model):
        """Test that removing theta constraint gives faster solution."""
        # Constrained problem
        ocp_constrained = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )
        ocp_constrained.set_initial_state([0, 0, 0, 0])
        ocp_constrained.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp_constrained.set_control_bounds(-F_MAX, F_MAX)
        ocp_constrained.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp_constrained.set_terminal_cost(1000 * np.eye(4))
        ocp_constrained.set_time_cost(weight=1.0)

        result_constrained = ocp_constrained.solve()

        # Unconstrained problem (no theta bounds)
        ocp_unconstrained = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )
        ocp_unconstrained.set_initial_state([0, 0, 0, 0])
        ocp_unconstrained.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp_unconstrained.set_control_bounds(-F_MAX, F_MAX)
        ocp_unconstrained.set_terminal_cost(1000 * np.eye(4))
        ocp_unconstrained.set_time_cost(weight=1.0)

        result_unconstrained = ocp_unconstrained.solve()

        assert result_constrained.converged
        assert result_unconstrained.converged

        # Unconstrained should be faster (or equal)
        assert result_unconstrained.T <= result_constrained.T + 0.1, (
            f"Unconstrained T={result_unconstrained.T:.2f}s should be <= "
            f"constrained T={result_constrained.T:.2f}s"
        )

    def test_unconstrained_uses_larger_angles(self, model):
        """Test that unconstrained solution uses larger pole angles."""
        # Unconstrained problem
        ocp_unconstrained = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )
        ocp_unconstrained.set_initial_state([0, 0, 0, 0])
        ocp_unconstrained.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp_unconstrained.set_control_bounds(-F_MAX, F_MAX)
        ocp_unconstrained.set_terminal_cost(1000 * np.eye(4))
        ocp_unconstrained.set_time_cost(weight=1.0)

        result = ocp_unconstrained.solve()

        assert result.converged

        # Max theta should exceed the constraint value
        max_theta = np.max(np.abs(result.states[:, 2]))
        assert max_theta > THETA_MAX, (
            f"Unconstrained max |theta|={np.rad2deg(max_theta):.2f}° should exceed 5°"
        )
