"""RK4 integrator equivalence tests between JAX and CasADi.

These tests verify that the CasADi RK4 integrator produces identical
results to the JAX RK4 integrator. This is critical for ensuring that
MPC solutions using CasADi will match simulations using JAX.

Tests are performed at multiple levels:
- Single step comparison
- Multi-step trajectory comparison
- With and without post_step
"""

import pytest
import numpy as np
import jax.numpy as jnp

casadi = pytest.importorskip("casadi")

from fmd.simulator.box_1d import Box1DJax, Box1DFrictionJax
from fmd.simulator.casadi import (
    Box1DCasadiExact,
    Box1DFrictionCasadiExact,
    rk4_step_casadi,
    rk4_step_with_post_step_casadi,
    rk4_step_function,
)
from fmd.simulator.params import BOX1D_DEFAULT, BOX1D_FRICTION_DEFAULT
from fmd.simulator.integrator import rk4_step

from .conftest import TRAJ_RTOL, TRAJ_ATOL


class TestRK4SingleStep:
    """Test single RK4 step equivalence."""

    def test_box1d_single_step_no_post_step(self):
        """Single RK4 step without post_step matches JAX."""
        jax_model = Box1DJax(BOX1D_DEFAULT)
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        x0 = np.array([1.0, 2.0])
        u0 = np.array([0.5])

        # JAX rk4_step applies system.post_step(). For Box1D, post_step is the
        # identity map, so this is still a fair comparison to CasADi's
        # include_post_step=False step.
        x_jax = rk4_step(jax_model, jnp.array(x0), jnp.array(u0), dt)

        # CasADi step without post_step
        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=False)
        x_casadi = np.array(rk4_func(x0, u0)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)

    def test_box1d_single_step_with_post_step(self):
        """Single RK4 step with post_step matches JAX simulation."""
        jax_model = Box1DJax(BOX1D_DEFAULT)
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        x0 = np.array([1.0, 2.0])
        u0 = np.array([0.5])

        # JAX step (includes post_step)
        x_jax = rk4_step(jax_model, jnp.array(x0), jnp.array(u0), dt)

        # CasADi step with post_step
        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)
        x_casadi = np.array(rk4_func(x0, u0)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)

    def test_box1d_friction_single_step(self):
        """Single RK4 step for friction model."""
        jax_model = Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)
        casadi_model = Box1DFrictionCasadiExact(BOX1D_FRICTION_DEFAULT)

        dt = 0.01
        x0 = np.array([0.0, 1.0])  # Moving with positive velocity
        u0 = np.array([0.5])

        x_jax = rk4_step(jax_model, jnp.array(x0), jnp.array(u0), dt)
        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)
        x_casadi = np.array(rk4_func(x0, u0)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)


class TestRK4MultiStep:
    """Test multi-step RK4 trajectory equivalence."""

    def test_box1d_trajectory_constant_control(self):
        """Multi-step trajectory with constant control."""
        jax_model = Box1DJax(BOX1D_DEFAULT)
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        steps = 200
        x0 = np.array([0.0, 0.0])
        u = np.array([1.0])

        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Simulate both
        x_jax = jnp.array(x0)
        x_casadi = x0.copy()

        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)
            x_casadi = np.array(rk4_func(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)

    def test_box1d_trajectory_varying_control(self, rng):
        """Multi-step trajectory with random varying control."""
        jax_model = Box1DJax(BOX1D_DEFAULT)
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        steps = 100
        x0 = np.array([0.0, 0.0])

        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Generate random control sequence
        controls = rng.uniform(-5, 5, size=(steps, 1))

        # Simulate both
        x_jax = jnp.array(x0)
        x_casadi = x0.copy()

        for i in range(steps):
            u = controls[i]
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)
            x_casadi = np.array(rk4_func(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)

    def test_box1d_friction_trajectory(self):
        """Multi-step trajectory for friction model."""
        jax_model = Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)
        casadi_model = Box1DFrictionCasadiExact(BOX1D_FRICTION_DEFAULT)

        dt = 0.01
        steps = 150
        x0 = np.array([0.0, 2.0])  # Start with velocity to avoid sign(0) issues
        u = np.array([0.0])  # No external force, just friction deceleration

        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)

        x_jax = jnp.array(x0)
        x_casadi = x0.copy()

        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)
            x_casadi = np.array(rk4_func(x_casadi, u)).flatten()

        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL)


class TestRK4SymbolicEvaluation:
    """Test symbolic RK4 can be evaluated numerically."""

    def test_symbolic_rk4_construction(self):
        """Test that symbolic RK4 expressions can be constructed."""
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        x = casadi.SX.sym("x", 2)
        u = casadi.SX.sym("u", 1)
        dt = 0.01

        # Should not raise
        x_next = rk4_step_casadi(casadi_model, x, u, dt)
        assert x_next.shape == (2, 1)

    def test_symbolic_rk4_with_post_step(self):
        """Test that symbolic RK4 with post_step can be constructed."""
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        x = casadi.SX.sym("x", 2)
        u = casadi.SX.sym("u", 1)
        dt = 0.01

        # Should not raise
        x_next = rk4_step_with_post_step_casadi(casadi_model, x, u, dt)
        assert x_next.shape == (2, 1)

    def test_rk4_function_caching(self):
        """Test that RK4 functions are properly cached."""
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        f1 = casadi_model.rk4_step_function(dt)
        f2 = casadi_model.rk4_step_function(dt)

        # Should return the same cached function
        assert f1 is f2

        # Different dt should return different function
        f3 = casadi_model.rk4_step_function(dt=0.02)
        assert f1 is not f3

    def test_rk4_function_caching_with_post_step(self):
        """Test caching differentiates include_post_step."""
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        f_no_post = casadi_model.rk4_step_function(dt, include_post_step=False)
        f_with_post = casadi_model.rk4_step_function(dt, include_post_step=True)

        # Should be different functions
        assert f_no_post is not f_with_post


class TestRK4TimeStepping:
    """Test RK4 with different time steps."""

    def test_different_dt_values(self):
        """Test RK4 equivalence holds for different dt values."""
        jax_model = Box1DJax(BOX1D_DEFAULT)
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        x0 = np.array([0.0, 1.0])
        u = np.array([0.5])

        for dt in [0.001, 0.01, 0.05, 0.1]:
            rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)

            x_jax = rk4_step(jax_model, jnp.array(x0), jnp.array(u), dt)
            x_casadi = np.array(rk4_func(x0, u)).flatten()

            np.testing.assert_allclose(
                x_casadi, np.array(x_jax), rtol=TRAJ_RTOL, atol=TRAJ_ATOL,
                err_msg=f"Mismatch at dt={dt}"
            )

    def test_accumulation_stability(self):
        """Test that numerical errors don't accumulate excessively."""
        jax_model = Box1DJax(BOX1D_DEFAULT)
        casadi_model = Box1DCasadiExact(BOX1D_DEFAULT)

        dt = 0.01
        steps = 1000  # Long trajectory
        x0 = np.array([0.0, 0.0])
        u = np.array([0.1])

        rk4_func = rk4_step_function(casadi_model, dt, include_post_step=True)

        x_jax = jnp.array(x0)
        x_casadi = x0.copy()

        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)
            x_casadi = np.array(rk4_func(x_casadi, u)).flatten()

        # Use slightly relaxed tolerances for long trajectory
        np.testing.assert_allclose(x_casadi, np.array(x_jax), rtol=1e-8, atol=1e-10)
