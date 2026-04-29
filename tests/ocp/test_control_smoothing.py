"""Tests for control smoothing features in OCP.

Tests control rate penalty (set_control_rate_cost), control rate bounds
(set_control_rate_bounds), and chattering detection (compute_control_smoothness).
"""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")

from fmd.ocp import MultipleShootingOCP, OCPResult, compute_control_smoothness
from fmd.simulator.casadi import CartpoleCasadiExact
from fmd.simulator.params import CARTPOLE_CLASSIC


# Problem constants (same as test_cartpole_ocp.py)
THETA_MAX = 0.0873  # 5 degrees in radians
F_MAX = 10.0
X_TARGET = 3.0
T_FIXED = 4.0
N_STEPS = 200


class TestControlRateCost:
    """Test control rate cost (set_control_rate_cost)."""

    @pytest.fixture
    def model(self):
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_set_control_rate_cost_scalar(self, model):
        """Test that scalar Rd is accepted and broadcast correctly."""
        ocp = MultipleShootingOCP(model, N=50, T_fixed=2.0)
        result = ocp.set_control_rate_cost(1.0)

        # Should return self for chaining
        assert result is ocp

        # Internal attribute should be a matrix
        assert ocp._Rd is not None
        assert ocp._Rd.shape == (1, 1)  # Cartpole has 1 control
        assert ocp._Rd[0, 0] == 1.0

    def test_set_control_rate_cost_array(self, model):
        """Test that 1D array Rd is accepted as diagonal."""
        ocp = MultipleShootingOCP(model, N=50, T_fixed=2.0)
        ocp.set_control_rate_cost([2.0])

        assert ocp._Rd is not None
        assert ocp._Rd.shape == (1, 1)
        assert ocp._Rd[0, 0] == 2.0

    def test_control_rate_cost_reduces_chattering(self, model):
        """Test that adding rate cost reduces control chattering."""
        # Baseline: minimum-time without rate cost
        ocp_baseline = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )
        ocp_baseline.set_initial_state([0, 0, 0, 0])
        ocp_baseline.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp_baseline.set_control_bounds(-F_MAX, F_MAX)
        ocp_baseline.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp_baseline.set_terminal_cost(1000 * np.eye(4))
        ocp_baseline.set_time_cost(weight=1.0)

        result_baseline = ocp_baseline.solve()
        assert result_baseline.converged

        smoothness_baseline = compute_control_smoothness(result_baseline)

        # With rate cost
        ocp_smooth = MultipleShootingOCP(
            model, N=N_STEPS, T_bounds=(1.0, 10.0), T_init=T_FIXED
        )
        ocp_smooth.set_initial_state([0, 0, 0, 0])
        ocp_smooth.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp_smooth.set_control_bounds(-F_MAX, F_MAX)
        ocp_smooth.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp_smooth.set_terminal_cost(1000 * np.eye(4))
        ocp_smooth.set_time_cost(weight=1.0)
        ocp_smooth.set_control_rate_cost(0.1)  # Add rate penalty

        result_smooth = ocp_smooth.solve()
        assert result_smooth.converged

        smoothness_smooth = compute_control_smoothness(result_smooth)

        # Rate cost should reduce sign change ratio
        assert smoothness_smooth["sign_change_ratio"] < smoothness_baseline["sign_change_ratio"], (
            f"Rate cost should reduce chattering: "
            f"baseline={smoothness_baseline['sign_change_ratio']:.2f}, "
            f"with_cost={smoothness_smooth['sign_change_ratio']:.2f}"
        )

        # Rate cost may increase time slightly (expected trade-off)
        # But should still be in reasonable range
        assert result_smooth.T < 10.0, f"Time too long: {result_smooth.T:.2f}s"


class TestControlRateBounds:
    """Test control rate bounds (set_control_rate_bounds)."""

    @pytest.fixture
    def model(self):
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_set_control_rate_bounds_scalar(self, model):
        """Test that scalar bounds are accepted."""
        ocp = MultipleShootingOCP(model, N=50, T_fixed=2.0)
        result = ocp.set_control_rate_bounds(-100.0, 100.0)

        # Should return self for chaining
        assert result is ocp

        assert ocp._du_bounds is not None
        assert ocp._du_bounds[0].shape == (1,)
        assert ocp._du_bounds[1].shape == (1,)
        assert ocp._du_bounds[0][0] == -100.0
        assert ocp._du_bounds[1][0] == 100.0

    def test_control_rate_bounds_satisfied(self, model):
        """Test that rate bounds are respected in solution."""
        # Use moderate rate limit
        rate_limit = 50.0  # N/s

        ocp = MultipleShootingOCP(model, N=N_STEPS, T_fixed=T_FIXED)
        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))
        ocp.set_control_rate_bounds(-rate_limit, rate_limit)

        result = ocp.solve()
        assert result.converged, f"Solver failed: {result.solver_stats}"

        # Check rate constraint satisfaction
        dt = result.T / len(result.controls)
        du = np.diff(result.controls[:, 0])
        du_dt = du / dt

        max_rate = np.max(np.abs(du_dt))
        # Allow small numerical tolerance
        assert max_rate <= rate_limit * 1.01, (
            f"Rate bound violated: max |du/dt| = {max_rate:.2f} > {rate_limit}"
        )


class TestControlSmoothnessAnalysis:
    """Test compute_control_smoothness function."""

    def test_smooth_signal_detection(self):
        """Test that a smooth signal is classified correctly."""
        # Create a mock OCPResult with smooth control
        N = 100
        times = np.linspace(0, 2.0, N + 1)
        states = np.zeros((N + 1, 4))
        # Smooth sinusoidal control
        controls = np.sin(np.linspace(0, 2 * np.pi, N)).reshape(-1, 1)

        result = OCPResult(
            states=states,
            controls=controls,
            times=times,
            T=2.0,
            cost=0.0,
            converged=True,
            solver_stats={},
        )

        metrics = compute_control_smoothness(result)

        # Smooth signal should have low sign change ratio (only crosses zero twice)
        assert metrics["sign_change_ratio"] < 0.1
        assert not metrics["is_chattering"]
        assert metrics["chattering_severity"] == "none"

    def test_chattering_signal_detection(self):
        """Test that a chattering signal is classified correctly."""
        # Create a mock OCPResult with chattering control
        N = 100
        times = np.linspace(0, 2.0, N + 1)
        states = np.zeros((N + 1, 4))
        # Alternating control (severe chattering)
        controls = np.array([10.0 if i % 2 == 0 else -10.0 for i in range(N)]).reshape(-1, 1)

        result = OCPResult(
            states=states,
            controls=controls,
            times=times,
            T=2.0,
            cost=0.0,
            converged=True,
            solver_stats={},
        )

        metrics = compute_control_smoothness(result)

        # Alternating signal should have very high sign change ratio
        assert metrics["sign_change_ratio"] > 0.9
        assert metrics["is_chattering"]
        assert metrics["chattering_severity"] == "severe"

    def test_severity_thresholds(self):
        """Test chattering severity classification thresholds."""
        N = 100
        times = np.linspace(0, 2.0, N + 1)
        states = np.zeros((N + 1, 4))

        def make_result(sign_change_frac):
            """Create result with approximately the given sign change fraction."""
            # Start with all positive
            controls = np.ones(N)
            # Flip signs for desired fraction
            num_changes = int(sign_change_frac * (N - 1))
            for i in range(1, num_changes + 1):
                controls[i] = -controls[i - 1]
            return OCPResult(
                states=states,
                controls=controls.reshape(-1, 1),
                times=times,
                T=2.0,
                cost=0.0,
                converged=True,
                solver_stats={},
            )

        # Test thresholds
        result_none = make_result(0.05)
        assert compute_control_smoothness(result_none)["chattering_severity"] == "none"

        result_mild = make_result(0.2)
        assert compute_control_smoothness(result_mild)["chattering_severity"] == "mild"

        result_moderate = make_result(0.4)
        assert compute_control_smoothness(result_moderate)["chattering_severity"] == "moderate"

        result_severe = make_result(0.7)
        assert compute_control_smoothness(result_severe)["chattering_severity"] == "severe"


class TestIntegration:
    """Integration tests for control smoothing with real OCP problems."""

    @pytest.fixture
    def model(self):
        return CartpoleCasadiExact(CARTPOLE_CLASSIC)

    def test_mintime_baseline_has_chattering(self, model):
        """Verify baseline min-time problem exhibits chattering."""
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

        metrics = compute_control_smoothness(result)

        # Without smoothing, this problem should exhibit significant chattering
        assert metrics["is_chattering"], (
            f"Expected chattering in baseline, got severity={metrics['chattering_severity']}"
        )

    def test_smoothing_options_work_together(self, model):
        """Test that rate cost and bounds can be used together."""
        ocp = MultipleShootingOCP(model, N=100, T_fixed=T_FIXED)
        ocp.set_initial_state([0, 0, 0, 0])
        ocp.set_terminal_state([X_TARGET, 0, 0, 0])
        ocp.set_control_bounds(-F_MAX, F_MAX)
        ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)
        ocp.set_terminal_cost(1000 * np.eye(4))
        ocp.set_control_rate_cost(0.01)
        ocp.set_control_rate_bounds(-100.0, 100.0)

        result = ocp.solve()

        # Should converge with both options
        assert result.converged, f"Solver failed: {result.solver_stats}"

        # Terminal state should be reached
        assert abs(result.states[-1, 0] - X_TARGET) < 0.1
