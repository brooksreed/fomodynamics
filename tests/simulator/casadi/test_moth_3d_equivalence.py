"""JAX/CasADi equivalence tests for Moth3D dynamics.

These tests verify that Moth3DCasadiExact produces numerically identical
results to Moth3D (JAX) at multiple levels:
- Level 0: forward_dynamics (continuous-time derivatives)
- Level 1: Jacobians (A, B matrices)
- Level 2: Trajectories (RK4 integration)
"""

import pytest
import numpy as np
import jax.numpy as jnp

casadi = pytest.importorskip("casadi")


from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
from fmd.simulator.casadi.moth_3d import Moth3DCasadiExact
from fmd.simulator.casadi import rk4_step_function
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.integrator import rk4_step

from .conftest import DERIV_RTOL, DERIV_ATOL, JAC_RTOL, JAC_ATOL, TRAJ_RTOL, TRAJ_ATOL


# Default test configuration
HEEL_ANGLE = np.deg2rad(30.0)
U_FORWARD = 10.0
VENTILATION_MODE = "smooth"
VENTILATION_THRESHOLD = 0.30


class TestMoth3DEquivalence:
    """Verify Moth3D (JAX) and Moth3DCasadiExact produce identical results."""

    @pytest.fixture
    def jax_model(self):
        """JAX model with constant u_forward and default sailor position."""
        return Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
        )

    @pytest.fixture
    def casadi_model(self):
        """CasADi model with matching configuration."""
        return Moth3DCasadiExact(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
            u_forward=U_FORWARD,
        )

    def test_level0_derivative_random_sampling(self, jax_model, casadi_model, rng):
        """Level 0: 100+ random states produce identical derivatives."""
        f = casadi_model.dynamics_function()

        for _ in range(120):
            x = np.array([
                rng.uniform(-0.5, 1.0),     # pos_d (includes foiling above water)
                rng.uniform(-0.3, 0.3),     # theta
                rng.uniform(-2, 2),         # w
                rng.uniform(-2, 2),         # q
                rng.uniform(3, 12),         # u
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),   # main_flap_angle
                rng.uniform(-0.05, 0.10),   # rudder_elevator_angle
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"State: {x}, Control: {u}",
            )

    def test_level0_derivative_at_trim(self, jax_model, casadi_model):
        """Level 0: Match at approximate trim point."""
        f = casadi_model.dynamics_function()

        # Approximate trim: CG above water at -0.25m, 10 m/s
        x = np.array([-0.25, 0.0, 0.0, 0.0, 10.0])
        u = np.array([0.0, 0.0])

        jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
        casadi_deriv = np.array(f(x, u)).flatten()

        np.testing.assert_allclose(
            casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
            err_msg="Mismatch at trim point",
        )

    def test_level0_ventilation_region(self, jax_model, casadi_model, rng):
        """Level 0: States near the surface where ventilation is active."""
        f = casadi_model.dynamics_function()

        for _ in range(50):
            # pos_d near 0 and slightly negative (above waterline)
            x = np.array([
                rng.uniform(-0.1, 0.15),    # pos_d near/above surface
                rng.uniform(-0.2, 0.2),     # theta
                rng.uniform(-1, 1),         # w
                rng.uniform(-1, 1),         # q
                rng.uniform(4, 8),          # u
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Ventilation region - State: {x}",
            )

    def test_level1_jacobian_equivalence(self, jax_model, casadi_model, rng):
        """Level 1: A, B matrices match at 50+ random points."""
        AB = casadi_model.linearization_function()

        for _ in range(60):
            x = np.array([
                rng.uniform(0.05, 1.0),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
                rng.uniform(3, 12),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(
                np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"A matrix mismatch at state: {x}",
            )
            np.testing.assert_allclose(
                np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"B matrix mismatch at state: {x}",
            )

    def test_level2_trajectory_equivalence(self, jax_model, casadi_model, artifact_saver):
        """Level 2: 100+ RK4 steps produce identical trajectories."""
        dt = 0.005
        steps = 240

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        # Start at moderate depth with small pitch disturbance
        x0 = np.array([-0.25, 0.05, 0.1, 0.1, 10.0])
        u = np.array([0.05, 0.02])  # Small flap and elevator

        # JAX trajectory
        x_jax = jnp.array(x0)
        jax_hist = [np.array(x_jax)]
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)
            jax_hist.append(np.array(x_jax))

        # CasADi trajectory
        x_casadi = x0.copy()
        casadi_hist = [x_casadi.copy()]
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()
            casadi_hist.append(x_casadi.copy())

        artifact_saver.save("test_level2_trajectory_equivalence", {
            "steps": np.arange(steps + 1),
            "jax_states": np.stack(jax_hist),
            "casadi_states": np.stack(casadi_hist),
        })

        # Check ALL intermediate states, not just final
        jax_all = np.stack(jax_hist)
        casadi_all = np.stack(casadi_hist)
        np.testing.assert_allclose(
            casadi_all, jax_all, rtol=TRAJ_RTOL, atol=TRAJ_ATOL,
            err_msg="Trajectory diverged at some intermediate step",
        )


class TestMoth3DSurgeEquivalence:
    """Verify equivalence with surge_enabled=True."""

    @pytest.fixture
    def jax_model(self):
        """JAX model with surge enabled."""
        return Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=True,
        )

    @pytest.fixture
    def casadi_model(self):
        """CasADi model with surge enabled."""
        return Moth3DCasadiExact(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=True,
            u_forward=U_FORWARD,
        )

    def test_level0_surge_enabled(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match with surge dynamics enabled."""
        f = casadi_model.dynamics_function()

        for _ in range(120):
            x = np.array([
                rng.uniform(0.05, 1.0),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
                rng.uniform(3, 12),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Surge enabled - State: {x}, Control: {u}",
            )

    def test_level1_jacobian_surge_enabled(self, jax_model, casadi_model, rng):
        """Level 1: A, B matrices match with surge enabled."""
        AB = casadi_model.linearization_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(0.05, 1.0),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
                rng.uniform(3, 12),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL)
            np.testing.assert_allclose(np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL)

    def test_level2_trajectory_surge_enabled(self, jax_model, casadi_model):
        """Level 2: Trajectory with surge dynamics enabled — check all steps."""
        dt = 0.005
        steps = 240

        rk4_casadi = rk4_step_function(casadi_model, dt, include_post_step=True)

        x0 = np.array([-0.25, 0.05, 0.1, 0.1, 10.0])
        u = np.array([0.05, 0.02])

        # JAX trajectory
        x_jax = jnp.array(x0)
        jax_hist = [np.array(x_jax)]
        for _ in range(steps):
            x_jax = rk4_step(jax_model, x_jax, jnp.array(u), dt)
            jax_hist.append(np.array(x_jax))

        # CasADi trajectory
        x_casadi = x0.copy()
        casadi_hist = [x_casadi.copy()]
        for _ in range(steps):
            x_casadi = np.array(rk4_casadi(x_casadi, u)).flatten()
            casadi_hist.append(x_casadi.copy())

        # Check all intermediate states
        np.testing.assert_allclose(
            np.stack(casadi_hist), np.stack(jax_hist),
            rtol=TRAJ_RTOL, atol=TRAJ_ATOL,
            err_msg="Surge trajectory diverged at some intermediate step",
        )


class TestMoth3DLowSpeedEquivalence:
    """Verify equivalence at low speeds near the max(u, 0.1) clamp (F9)."""

    @pytest.fixture
    def jax_model(self):
        """JAX model with surge enabled (speed can decay toward zero)."""
        return Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=True,
        )

    @pytest.fixture
    def casadi_model(self):
        """CasADi model with surge enabled."""
        return Moth3DCasadiExact(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=True,
            u_forward=U_FORWARD,
        )

    def test_level0_low_speed_derivatives(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match at very low speeds (u in [0.05, 0.5])."""
        f = casadi_model.dynamics_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(0.1, 0.8),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(0.05, 0.5),     # Very low speed
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Low speed - State: {x}, Control: {u}",
            )

    def test_level1_low_speed_jacobians(self, jax_model, casadi_model, rng):
        """Level 1: A, B matrices match at low speeds near max(u, 0.1) clamp."""
        AB = casadi_model.linearization_function()

        for _ in range(30):
            x = np.array([
                rng.uniform(0.1, 0.8),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(0.05, 0.5),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(
                np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"Low speed A mismatch at state: {x}",
            )
            np.testing.assert_allclose(
                np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"Low speed B mismatch at state: {x}",
            )


class TestMoth3DBinaryVentilationEquivalence:
    """Verify Moth3D JAX/CasADi equivalence with ventilation_mode='binary'.

    Tests Level 0 (derivative) equivalence with states near the ventilation
    boundary where the binary mode switches between 0 and 1.
    """

    @pytest.fixture
    def jax_model(self):
        """JAX model with binary ventilation."""
        return Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode="binary",
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
        )

    @pytest.fixture
    def casadi_model(self):
        """CasADi model with binary ventilation."""
        return Moth3DCasadiExact(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE,
            ventilation_mode="binary",
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
            u_forward=U_FORWARD,
        )

    def test_level0_binary_deeply_submerged(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match when foil is deeply submerged (no ventilation)."""
        f = casadi_model.dynamics_function()

        for _ in range(30):
            x = np.array([
                rng.uniform(0.3, 1.0),     # pos_d: well below surface
                rng.uniform(-0.2, 0.2),     # theta
                rng.uniform(-1, 1),         # w
                rng.uniform(-1, 1),         # q
                rng.uniform(4, 8),          # u
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Binary submerged - State: {x}, Control: {u}",
            )

    def test_level0_binary_above_surface(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match when foil is above surface (fully ventilated)."""
        f = casadi_model.dynamics_function()

        for _ in range(30):
            x = np.array([
                rng.uniform(-0.3, -0.05),   # pos_d: above surface
                rng.uniform(-0.2, 0.2),      # theta
                rng.uniform(-1, 1),          # w
                rng.uniform(-1, 1),          # q
                rng.uniform(4, 8),           # u
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Binary above surface - State: {x}, Control: {u}",
            )

    def test_level0_binary_near_boundary(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match near the ventilation boundary.

        States just above and just below the ventilation transition (pos_d ~ 0).
        Avoids exact boundary where binary switch is discontinuous.
        """
        f = casadi_model.dynamics_function()

        for _ in range(30):
            # Slightly above boundary (ventilated)
            x_above = np.array([
                rng.uniform(-0.05, -0.01),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(5, 7),
            ])
            u = np.array([
                rng.uniform(-0.10, 0.15),
                rng.uniform(-0.03, 0.05),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x_above), jnp.array(u)))
            casadi_deriv = np.array(f(x_above, u)).flatten()
            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Binary near boundary (above) - State: {x_above}",
            )

            # Slightly below boundary (not ventilated)
            x_below = np.array([
                rng.uniform(0.01, 0.05),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(5, 7),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x_below), jnp.array(u)))
            casadi_deriv = np.array(f(x_below, u)).flatten()
            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Binary near boundary (below) - State: {x_below}",
            )

    def test_level1_binary_jacobian_submerged(self, jax_model, casadi_model, rng):
        """Level 1: A, B matrices match for binary ventilation when deeply submerged (F9)."""
        AB = casadi_model.linearization_function()

        for _ in range(30):
            x = np.array([
                rng.uniform(0.3, 1.0),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(4, 8),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(
                np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"Binary Jacobian (submerged) at state: {x}",
            )
            np.testing.assert_allclose(
                np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"Binary B Jacobian (submerged) at state: {x}",
            )

    def test_level1_binary_jacobian_above_surface(self, jax_model, casadi_model, rng):
        """Level 1: A, B matrices match for binary ventilation above surface (F9)."""
        AB = casadi_model.linearization_function()

        for _ in range(30):
            x = np.array([
                rng.uniform(-0.3, -0.05),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(4, 8),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))

            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(
                np.array(casadi_A), jax_A, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"Binary Jacobian (above surface) at state: {x}",
            )
            np.testing.assert_allclose(
                np.array(casadi_B), jax_B, rtol=JAC_RTOL, atol=JAC_ATOL,
                err_msg=f"Binary B Jacobian (above surface) at state: {x}",
            )


class TestMoth3DExtendedRegimeEquivalence:
    """Verify equivalence in extended state regimes (F18).

    Tests large theta (capsizing), theta near pi (post_step wrapping),
    and negative pos_d (boat flying high above water).
    """

    @pytest.fixture
    def jax_model(self):
        return Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
        )

    @pytest.fixture
    def casadi_model(self):
        return Moth3DCasadiExact(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
            u_forward=U_FORWARD,
        )

    def test_level0_large_theta(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match at large pitch angles (capsizing regime)."""
        f = casadi_model.dynamics_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(0.05, 0.8),
                rng.uniform(-1.2, 1.2),    # Large theta (up to ~70 deg)
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
                rng.uniform(3, 10),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Large theta - State: {x}",
            )

    def test_level0_negative_pos_d(self, jax_model, casadi_model, rng):
        """Level 0: Derivatives match with negative pos_d (boat above water)."""
        f = casadi_model.dynamics_function()

        for _ in range(50):
            x = np.array([
                rng.uniform(-0.5, -0.01),   # Negative pos_d: above water
                rng.uniform(-0.3, 0.3),
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
                rng.uniform(4, 10),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Negative pos_d - State: {x}",
            )

    def test_level1_large_theta_jacobians(self, jax_model, casadi_model, rng):
        """Level 1: Jacobians match at large theta.

        Slightly relaxed tolerances vs standard Jacobian tests because
        autodiff through trigonometric functions at large angles accumulates
        more numerical differences between JAX and CasADi AD backends.
        """
        AB = casadi_model.linearization_function()

        # Relaxed tolerances for extended regime
        ext_jac_rtol = 1e-7
        ext_jac_atol = 1e-9

        for _ in range(30):
            x = np.array([
                rng.uniform(0.05, 0.8),
                rng.uniform(-1.2, 1.2),
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
                rng.uniform(3, 10),
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_A = np.array(jax_model.get_state_jacobian(jnp.array(x), jnp.array(u)))
            jax_B = np.array(jax_model.get_control_jacobian(jnp.array(x), jnp.array(u)))
            casadi_A, casadi_B = AB(x, u)

            np.testing.assert_allclose(
                np.array(casadi_A), jax_A, rtol=ext_jac_rtol, atol=ext_jac_atol,
                err_msg=f"Large theta Jacobian at state: {x}",
            )
            np.testing.assert_allclose(
                np.array(casadi_B), jax_B, rtol=ext_jac_rtol, atol=ext_jac_atol,
            )
