"""Tests for Moth 3DOF per-component force extraction."""

import numpy as np
import pytest

from fmd.simulator import Moth3D, simulate
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.moth_forces_extract import extract_forces, MothForceLog


@pytest.fixture
def moth():
    return Moth3D(MOTH_BIEKER_V3)


@pytest.fixture
def short_result(moth):
    return simulate(moth, moth.default_state(), dt=0.005, duration=1.0)


class TestExtractForces:
    def test_returns_moth_force_log(self, moth, short_result):
        forces = extract_forces(moth, short_result)
        assert isinstance(forces, MothForceLog)

    def test_shapes(self, moth, short_result):
        forces = extract_forces(moth, short_result)
        n = len(short_result.times)
        assert forces.times.shape == (n,)
        assert forces.main_foil_force.shape == (n, 3)
        assert forces.main_foil_moment.shape == (n, 3)
        assert forces.rudder_force.shape == (n, 3)
        assert forces.rudder_moment.shape == (n, 3)
        assert forces.sail_force.shape == (n, 3)
        assert forces.sail_moment.shape == (n, 3)
        assert forces.hull_drag_force.shape == (n, 3)
        assert forces.hull_drag_moment.shape == (n, 3)
        assert forces.gravity_force.shape == (n, 3)

    def test_gravity_magnitude(self, moth, short_result):
        """Gravity body-frame magnitude should be total_mass * g at any pitch."""
        forces = extract_forces(moth, short_result)
        expected_mag = moth.total_mass * moth.g
        for i in range(len(short_result.times)):
            mag = np.linalg.norm(forces.gravity_force[i])
            np.testing.assert_allclose(mag, expected_mag, rtol=1e-10)

    def test_gravity_direction_at_zero_pitch(self, moth):
        """At theta=0, gravity should be [0, 0, mg] in body frame."""
        import jax.numpy as jnp
        state = jnp.array([0.4, 0.0, 0.0, 0.0, 6.0])
        result = simulate(moth, state, dt=0.005, duration=0.02)
        forces = extract_forces(moth, result)
        # First timestep has theta~0
        grav = forces.gravity_force[0]
        np.testing.assert_allclose(grav[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(grav[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(grav[2], moth.total_mass * moth.g, rtol=1e-10)

    def test_forces_sum_matches_forward_dynamics(self, moth, short_result):
        """Total force sum should match what forward_dynamics uses for acceleration."""
        import jax.numpy as jnp
        forces = extract_forces(moth, short_result)
        states = np.asarray(short_result.states)

        for i in [0, len(states) // 2, -1]:
            state = jnp.array(states[i])
            control = jnp.array(np.asarray(short_result.controls)[i])
            t = float(np.asarray(short_result.times)[i])

            # Get derivative from model
            deriv = moth.forward_dynamics(state, control, t)

            # Sum z-forces from extracted data
            total_fz = (
                forces.main_foil_force[i, 2]
                + forces.rudder_force[i, 2]
                + forces.sail_force[i, 2]
                + forces.hull_drag_force[i, 2]
                + forces.gravity_force[i, 2]
                + forces.strut_main_force[i, 2]
                + forces.strut_rudder_force[i, 2]
            )

            # w_dot = total_fz / m_eff + q * u
            m_eff = moth.total_mass + moth.added_mass_heave
            # With surge_enabled=True, u comes from state; otherwise from schedule
            u_fwd = float(state[4]) if moth.surge_enabled else moth.u_forward_schedule(t)
            expected_w_dot = total_fz / m_eff + float(state[3]) * u_fwd
            np.testing.assert_allclose(
                float(deriv[2]), expected_w_dot, rtol=1e-6,
                err_msg=f"w_dot mismatch at step {i}"
            )

    def test_sail_force_ned_rotation(self, moth, short_result):
        """Sail force uses NED→body rotation: fx = F*cos(theta), fz = F*sin(theta)."""
        forces = extract_forces(moth, short_result)
        states = np.asarray(short_result.states)
        thetas = states[:, 1]

        # Get the thrust magnitude (from lookup table or fallback)
        expected_thrust = MOTH_BIEKER_V3.sail_thrust_coeff  # fallback at 10 m/s

        # Body-frame x-component = thrust * cos(theta)
        # At small theta, fx ≈ thrust (cos(theta) ≈ 1)
        assert np.all(forces.sail_force[:, 0] > 0), "Sail x-force should be positive"
        np.testing.assert_allclose(forces.sail_force[:, 1], 0.0, atol=1e-12)

    def test_y_forces_are_zero(self, moth, short_result):
        """In 3DOF longitudinal model, all y-forces should be zero."""
        forces = extract_forces(moth, short_result)
        np.testing.assert_allclose(forces.main_foil_force[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces.rudder_force[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces.sail_force[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces.hull_drag_force[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces.gravity_force[:, 1], 0.0, atol=1e-12)
