"""Tests for Box1D JAX models."""

import pytest
import jax.numpy as jnp
import numpy as np
from jax import jit

from fmd.simulator.box_1d import Box1DJax, Box1DFrictionJax, X, X_DOT, FORCE
from fmd.simulator.params import (
    Box1DParams, Box1DFrictionParams,
    BOX1D_DEFAULT, BOX1D_FRICTION_DEFAULT,
)
from fmd.simulator import simulate, ConstantControl


class TestBox1DJax:
    """Tests for Box1DJax model."""

    def test_construction(self):
        """Test model construction from params."""
        model = Box1DJax(BOX1D_DEFAULT)
        assert model.num_states == 2
        assert model.num_controls == 1
        assert model.state_names == ("x", "x_dot")
        assert model.control_names == ("F",)

    def test_from_values(self):
        """Test JAX-traceable construction."""
        model = Box1DJax.from_values(mass=2.0, drag_coefficient=0.5)
        assert model.mass == 2.0
        assert model.drag_coefficient == 0.5

    def test_forward_dynamics_at_rest(self):
        """Test dynamics at rest with no force."""
        model = Box1DJax(BOX1D_DEFAULT)
        x = jnp.array([0.0, 0.0])
        u = jnp.array([0.0])
        xdot = model.forward_dynamics(x, u)
        np.testing.assert_allclose(xdot, [0.0, 0.0], atol=1e-14)

    def test_forward_dynamics_with_force(self):
        """Test dynamics with applied force."""
        params = Box1DParams(mass=2.0, drag_coefficient=0.0)
        model = Box1DJax(params)
        x = jnp.array([0.0, 0.0])
        u = jnp.array([4.0])
        xdot = model.forward_dynamics(x, u)
        # F=4, m=2 -> a=2
        np.testing.assert_allclose(xdot, [0.0, 2.0], atol=1e-14)

    def test_forward_dynamics_with_drag(self):
        """Test dynamics with velocity and drag."""
        params = Box1DParams(mass=1.0, drag_coefficient=0.5)
        model = Box1DJax(params)
        x = jnp.array([1.0, 2.0])  # velocity = 2
        u = jnp.array([0.0])  # no force
        xdot = model.forward_dynamics(x, u)
        # drag = 0.5 * 2 = 1, acc = -1
        np.testing.assert_allclose(xdot, [2.0, -1.0], atol=1e-14)

    def test_simulation(self):
        """Test simulation produces reasonable trajectory."""
        model = Box1DJax(BOX1D_DEFAULT)
        x0 = jnp.array([0.0, 0.0])
        control = ConstantControl(jnp.array([1.0]))
        result = simulate(model, x0, dt=0.01, duration=1.0, control=control)
        # Should have moved forward
        assert result.states[-1, X] > 0
        assert result.states[-1, X_DOT] > 0

    def test_jit_compilation(self):
        """Test model can be JIT compiled."""
        model = Box1DJax(BOX1D_DEFAULT)

        @jit
        def step(x, u):
            return model.forward_dynamics(x, u)

        x = jnp.array([0.0, 1.0])
        u = jnp.array([1.0])
        xdot = step(x, u)
        assert xdot.shape == (2,)

    def test_jacobians(self):
        """Test Jacobian computation."""
        model = Box1DJax(BOX1D_DEFAULT)
        x = jnp.array([0.0, 1.0])
        u = jnp.array([1.0])

        A = model.get_state_jacobian(x, u)
        B = model.get_control_jacobian(x, u)

        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        # For Box1D: A = [[0, 1], [0, -k/m]]
        # B = [[0], [1/m]]


class TestBox1DFrictionJax:
    """Tests for Box1DFrictionJax model."""

    def test_construction(self):
        """Test model construction from params."""
        model = Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)
        assert model.num_states == 2
        assert model.num_controls == 1

    def test_friction_opposing_motion(self):
        """Test friction opposes motion direction."""
        params = Box1DFrictionParams(
            mass=1.0, drag_coefficient=0.0, friction_coefficient=0.5
        )
        model = Box1DFrictionJax(params)

        # Moving right -> friction acts left
        x_right = jnp.array([0.0, 1.0])
        xdot_right = model.forward_dynamics(x_right, jnp.array([0.0]))
        assert xdot_right[1] < 0  # Deceleration

        # Moving left -> friction acts right
        x_left = jnp.array([0.0, -1.0])
        xdot_left = model.forward_dynamics(x_left, jnp.array([0.0]))
        assert xdot_left[1] > 0  # Deceleration (towards zero)

    def test_friction_magnitude(self):
        """Test friction force magnitude."""
        params = Box1DFrictionParams(
            mass=2.0, drag_coefficient=0.0, friction_coefficient=0.5, g=10.0
        )
        model = Box1DFrictionJax(params)
        x = jnp.array([0.0, 1.0])  # Moving right
        u = jnp.array([0.0])
        xdot = model.forward_dynamics(x, u)
        # friction = mu * m * g = 0.5 * 2 * 10 = 10
        # acc = -10 / 2 = -5
        np.testing.assert_allclose(xdot[1], -5.0, atol=1e-10)


class TestBox1DSymplecticSupport:
    """Tests for symplectic integration support on Box1D models."""

    def test_box1d_position_indices(self):
        """Box1D correctly identifies position index."""
        model = Box1DJax(BOX1D_DEFAULT)
        assert model.position_indices == (X,)
        assert model.position_indices == (0,)

    def test_box1d_velocity_indices(self):
        """Box1D correctly identifies velocity index."""
        model = Box1DJax(BOX1D_DEFAULT)
        assert model.velocity_indices == (X_DOT,)
        assert model.velocity_indices == (1,)

    def test_box1d_supports_symplectic(self):
        """Box1D supports symplectic integration."""
        model = Box1DJax(BOX1D_DEFAULT)
        assert model.supports_symplectic is True

    def test_box1d_friction_position_indices(self):
        """Box1DFriction correctly identifies position index."""
        model = Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)
        assert model.position_indices == (0,)

    def test_box1d_friction_velocity_indices(self):
        """Box1DFriction correctly identifies velocity index."""
        model = Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)
        assert model.velocity_indices == (1,)

    def test_box1d_friction_supports_symplectic(self):
        """Box1DFriction supports symplectic integration."""
        model = Box1DFrictionJax(BOX1D_FRICTION_DEFAULT)
        assert model.supports_symplectic is True

    def test_symplectic_indices_partition_state(self):
        """Position and velocity indices together cover all states."""
        model = Box1DJax(BOX1D_DEFAULT)
        all_indices = set(model.position_indices) | set(model.velocity_indices)
        expected = set(range(model.num_states))
        assert all_indices == expected, "Indices should partition the state vector"

    def test_symplectic_indices_disjoint(self):
        """Position and velocity indices are disjoint."""
        model = Box1DJax(BOX1D_DEFAULT)
        pos_set = set(model.position_indices)
        vel_set = set(model.velocity_indices)
        assert pos_set.isdisjoint(vel_set), "Position and velocity indices must not overlap"
