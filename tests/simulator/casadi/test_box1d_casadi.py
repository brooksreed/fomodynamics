"""Unit tests for Box1D CasADi models."""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")
cs = casadi

from fmd.simulator.casadi import (
    Box1DCasadiExact,
    Box1DFrictionCasadiExact,
    Box1DFrictionCasadiSmooth,
)
from fmd.simulator.params import BOX1D_DEFAULT, BOX1D_FRICTION_DEFAULT


class TestBox1DCasadiExact:
    """Tests for Box1DCasadiExact model."""

    def test_construction(self):
        """Test model construction."""
        model = Box1DCasadiExact(BOX1D_DEFAULT)
        assert model.num_states == 2
        assert model.num_controls == 1

    def test_dynamics_function_shape(self):
        """Test dynamics function has correct shapes."""
        model = Box1DCasadiExact(BOX1D_DEFAULT)
        f = model.dynamics_function()
        assert f.n_in() == 2
        assert f.n_out() == 1
        assert f.size1_in(0) == 2  # x has 2 states
        assert f.size1_in(1) == 1  # u has 1 control
        assert f.size1_out(0) == 2  # xdot has 2 states

    def test_numerical_evaluation(self):
        """Test numerical evaluation of dynamics."""
        model = Box1DCasadiExact(BOX1D_DEFAULT)
        f = model.dynamics_function()
        x = np.array([0.0, 0.0])
        u = np.array([1.0])
        xdot = np.array(f(x, u)).flatten()
        # At rest with F=1, m=1: acc = 1
        np.testing.assert_allclose(xdot, [0.0, 1.0], atol=1e-14)

    def test_jacobian_shapes(self):
        """Test Jacobian function has correct shapes."""
        model = Box1DCasadiExact(BOX1D_DEFAULT)
        AB = model.linearization_function()
        x = np.array([0.0, 1.0])
        u = np.array([1.0])
        A, B = AB(x, u)
        assert np.array(A).shape == (2, 2)
        assert np.array(B).shape == (2, 1)


class TestBox1DFrictionCasadiExact:
    """Tests for Box1DFrictionCasadiExact model."""

    def test_friction_direction(self):
        """Test friction opposes motion."""
        model = Box1DFrictionCasadiExact(BOX1D_FRICTION_DEFAULT)
        f = model.dynamics_function()

        # Moving right -> deceleration
        xdot_right = np.array(f([0.0, 1.0], [0.0])).flatten()
        assert xdot_right[1] < 0

        # Moving left -> acceleration towards zero
        xdot_left = np.array(f([0.0, -1.0], [0.0])).flatten()
        assert xdot_left[1] > 0


class TestBox1DFrictionCasadiSmooth:
    """Tests for Box1DFrictionCasadiSmooth model."""

    def test_smooth_near_zero(self):
        """Test smooth approximation near zero velocity."""
        model_exact = Box1DFrictionCasadiExact(BOX1D_FRICTION_DEFAULT)
        model_smooth = Box1DFrictionCasadiSmooth(BOX1D_FRICTION_DEFAULT)

        f_exact = model_exact.dynamics_function()
        f_smooth = model_smooth.dynamics_function()

        # Away from zero, should be similar
        x = np.array([0.0, 1.0])
        u = np.array([0.0])
        xdot_exact = np.array(f_exact(x, u)).flatten()
        xdot_smooth = np.array(f_smooth(x, u)).flatten()
        np.testing.assert_allclose(xdot_smooth, xdot_exact, rtol=0.05)

    def test_differentiable_at_zero(self):
        """Test model is differentiable at zero velocity."""
        model = Box1DFrictionCasadiSmooth(BOX1D_FRICTION_DEFAULT)
        AB = model.linearization_function()

        # Should not raise at v=0
        x = np.array([0.0, 0.0])
        u = np.array([0.0])
        A, B = AB(x, u)
        assert not np.any(np.isnan(A))
        assert not np.any(np.isnan(B))
