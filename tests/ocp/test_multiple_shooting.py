"""Tests for MultipleShootingOCP solver.

These tests validate the OCP infrastructure using Box1D models
as a regression test against Phase 1 results.
"""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")

from fmd.ocp import MultipleShootingOCP, OCPResult
from fmd.simulator.casadi import Box1DCasadiExact, Box1DFrictionCasadiSmooth
from fmd.simulator.params import BOX1D_DEFAULT, BOX1D_FRICTION_DEFAULT
from fmd.simulator.params.box_1d import Box1DParams


class TestMultipleShootingOCPBasic:
    """Basic functionality tests for MultipleShootingOCP."""

    def test_init_fixed_time(self):
        """Test initialization with fixed time."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)
        ocp = MultipleShootingOCP(model, N=50, T_fixed=2.0)

        assert ocp.N == 50
        assert ocp.T_fixed == 2.0
        assert ocp.T_bounds is None
        assert ocp.nx == 2
        assert ocp.nu == 1

    def test_init_free_time(self):
        """Test initialization with free time."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)
        ocp = MultipleShootingOCP(model, N=50, T_bounds=(0.5, 5.0), T_init=2.0)

        assert ocp.T_fixed is None
        assert ocp.T_bounds == (0.5, 5.0)
        assert ocp.T_init == 2.0

    def test_init_requires_time_specification(self):
        """Test that either T_fixed or T_bounds must be specified."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        with pytest.raises(ValueError, match="Must specify either"):
            MultipleShootingOCP(model, N=50)

    def test_init_cannot_specify_both(self):
        """Test that both T_fixed and T_bounds cannot be specified."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        with pytest.raises(ValueError, match="Cannot specify both"):
            MultipleShootingOCP(model, N=50, T_fixed=2.0, T_bounds=(0.5, 5.0))

    @pytest.mark.parametrize("bad_m", [0, -1, 1.5])
    def test_init_rejects_invalid_substeps(self, bad_m):
        """M must be a positive integer."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        with pytest.raises(ValueError, match="M must be a positive integer"):
            MultipleShootingOCP(model, N=50, T_fixed=2.0, M=bad_m)

    def test_method_chaining(self):
        """Test that set methods return self for chaining."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        ocp = (
            MultipleShootingOCP(model, N=50, T_fixed=2.0)
            .set_initial_state([0, 0])
            .set_terminal_state([1, 0])
            .set_control_bounds(-10, 10)
            .set_state_bounds(0, -5, 5)
            .set_running_cost(np.eye(2), np.eye(1))
            .set_terminal_cost(np.eye(2))
        )

        assert isinstance(ocp, MultipleShootingOCP)


class TestBox1DMinTimeOCP:
    """Test minimum-time OCP for Box1D model (regression vs Phase 1)."""

    def test_min_time_free_time(self):
        """Validate free-time OCP converges to expected minimum time.

        For drag-free box with bang-bang control:
        T_min = 2 * sqrt(x_target / a_max) = 2 * sqrt(10/10) = 2.0s
        """
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        ocp = MultipleShootingOCP(model, N=30, T_bounds=(0.1, 10.0), T_init=3.0)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([10, 0])
        ocp.set_control_bounds(-10.0, 10.0)
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged
        assert result.T == pytest.approx(2.0, rel=0.05)
        assert result.states[-1, 0] == pytest.approx(10.0, rel=0.01)
        assert result.states[-1, 1] == pytest.approx(0.0, abs=0.1)

    def test_fixed_time_regulation(self):
        """Test fixed-time regulation to target state."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        ocp = MultipleShootingOCP(model, N=50, T_fixed=3.0)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([5, 0])
        ocp.set_control_bounds(-10.0, 10.0)
        # Add small control cost to regularize
        ocp.set_running_cost(np.zeros((2, 2)), 0.01 * np.eye(1))
        ocp.set_terminal_cost(1000 * np.eye(2))

        result = ocp.solve()

        assert result.converged
        assert result.T == 3.0
        assert result.states[-1, 0] == pytest.approx(5.0, abs=0.1)
        assert result.states[-1, 1] == pytest.approx(0.0, abs=0.1)

    def test_bang_bang_structure(self):
        """Test that minimum-time solution has bang-bang control."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        ocp = MultipleShootingOCP(model, N=50, T_bounds=(0.1, 10.0), T_init=3.0)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([10, 0])
        ocp.set_control_bounds(-10.0, 10.0)
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged

        # Controls should hit bounds (bang-bang)
        F_max = 10.0
        assert np.max(result.controls) > 0.9 * F_max
        assert np.min(result.controls) < -0.9 * F_max

        # First half should be positive (accelerate), second half negative (brake)
        mid = len(result.controls) // 2
        assert np.mean(result.controls[:mid]) > 0.5 * F_max
        assert np.mean(result.controls[mid:]) < -0.5 * F_max


class TestBox1DWithDrag:
    """Test OCP with drag model."""

    def test_ocp_with_drag(self):
        """Test OCP with drag model converges."""
        model = Box1DCasadiExact(BOX1D_DEFAULT)

        ocp = MultipleShootingOCP(model, N=40, T_bounds=(0.1, 10.0), T_init=3.0)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([5, 0])
        ocp.set_control_bounds(-10.0, 10.0)
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged
        assert result.T > 0
        assert result.states[-1, 0] == pytest.approx(5.0, rel=0.01)
        # With drag, harder to stop exactly
        assert abs(result.states[-1, 1]) < 0.1


class TestBox1DWithFriction:
    """Test OCP with friction model."""

    def test_ocp_with_smooth_friction(self):
        """Test OCP with smooth friction model converges."""
        model = Box1DFrictionCasadiSmooth(BOX1D_FRICTION_DEFAULT)

        ocp = MultipleShootingOCP(model, N=40, T_bounds=(0.1, 10.0), T_init=3.0)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([3, 0])
        ocp.set_control_bounds(-15.0, 15.0)  # More force needed for friction
        ocp.set_time_cost(weight=1.0)

        result = ocp.solve()

        assert result.converged
        assert result.T > 0
        assert result.states[-1, 0] == pytest.approx(3.0, rel=0.02)


class TestOCPResultStructure:
    """Test OCPResult data structure."""

    def test_result_shapes(self):
        """Test that result arrays have correct shapes."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        N = 30
        ocp = MultipleShootingOCP(model, N=N, T_fixed=2.0)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([5, 0])
        ocp.set_control_bounds(-10.0, 10.0)

        result = ocp.solve()

        assert result.states.shape == (N + 1, 2)
        assert result.controls.shape == (N, 1)
        assert result.times.shape == (N + 1,)
        assert len(result.times) == N + 1

    def test_times_vector(self):
        """Test that times vector is correct."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        N = 30
        T = 3.0
        ocp = MultipleShootingOCP(model, N=N, T_fixed=T)
        ocp.set_initial_state([0, 0])
        ocp.set_terminal_state([5, 0])
        ocp.set_control_bounds(-10.0, 10.0)

        result = ocp.solve()

        assert result.times[0] == 0.0
        assert result.times[-1] == pytest.approx(T)
        assert len(result.times) == N + 1
        # Check uniform spacing
        dt = T / N
        np.testing.assert_allclose(np.diff(result.times), dt)

    def test_initial_state_preserved(self):
        """Test that initial state constraint is satisfied."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.0)
        model = Box1DCasadiExact(params)

        x0 = [1.0, 0.5]
        ocp = MultipleShootingOCP(model, N=30, T_fixed=2.0)
        ocp.set_initial_state(x0)
        ocp.set_terminal_state([5, 0])
        ocp.set_control_bounds(-10.0, 10.0)

        result = ocp.solve()

        np.testing.assert_allclose(result.states[0], x0, atol=1e-6)
