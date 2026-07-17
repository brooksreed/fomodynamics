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


class TestMoth3DFreeSurfaceLiftEquivalence:
    """C1.F: parity for the FSL knobs (enable_free_surface_lift,
    ventilation_sharpness) in both settings.

    The default-ON case is covered by every test above; these lock the
    non-default configurations so the mirror cannot silently drift.
    """

    def _pair(self, enable_fsl: bool, sharpness: float):
        jax_model = Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
            ventilation_sharpness=sharpness,
            enable_free_surface_lift=enable_fsl,
        )
        casadi_model = Moth3DCasadiExact(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=False,
            u_forward=U_FORWARD,
            ventilation_sharpness=sharpness,
            enable_free_surface_lift=enable_fsl,
        )
        return jax_model, casadi_model

    @pytest.mark.parametrize(
        "enable_fsl,sharpness",
        [(False, 6.0), (True, 3.0), (False, 3.0)],
        ids=["fsl-off", "sharpness-3", "fsl-off-sharpness-3"],
    )
    def test_level0_derivative_near_surface(self, enable_fsl, sharpness, rng):
        """Derivative parity across the approach-to-surface band where the
        FSL factor varies most (foil center depth ~ 0.1-0.5 m)."""
        jax_model, casadi_model = self._pair(enable_fsl, sharpness)
        f = casadi_model.dynamics_function()

        for _ in range(60):
            x = np.array([
                rng.uniform(-1.55, -1.0),   # pos_d: near-surface band
                rng.uniform(-0.15, 0.15),   # theta
                rng.uniform(-1, 1),         # w
                rng.uniform(-1, 1),         # q
                rng.uniform(5, 12),         # u
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"FSL parity mismatch at state: {x}",
            )

    def test_fsl_changes_dynamics_near_surface(self):
        """Sanity: the flag is live — near the surface, FSL on vs off must
        produce different heave accelerations (guards against a silently
        dead flag passing parity trivially)."""
        jax_on, _ = self._pair(True, 6.0)
        jax_off, _ = self._pair(False, 6.0)
        x = jnp.array([-1.30, 0.0, 0.0, 0.0, 10.0])
        u = jnp.array([0.05, 0.0])
        d_on = np.array(jax_on.forward_dynamics(x, u))
        d_off = np.array(jax_off.forward_dynamics(x, u))
        assert abs(d_on[2] - d_off[2]) > 1e-3


class TestMoth3DSpeedGovernorEquivalence:
    """C2.C0: parity for the affine speed-governor sail (empty thrust table,
    thrust_slope = -Kp), including the F_sail >= 0 clamp.

    The default MOTH_BIEKER_V3 uses the calibrated thrust *table*, so the
    affine branch — and its new max(.,0) clamp — is never exercised by the
    tests above. This locks the JAX and CasADi affine branches in lockstep,
    driving u past the governor's zero-thrust kink so the clamp actually
    binds in both mirrors.
    """

    # Governor: F_sail = T0 + Kp*(u_target - u); here T0=75.5 N at u_target=10,
    # Kp=40 => zero-thrust kink at u = 10 + 75.5/40 ~= 11.9 m/s.
    KP = 40.0
    U_TARGET = 10.0
    T0 = 75.5

    def _governor_params(self):
        import attrs
        coeff = self.T0 + self.KP * self.U_TARGET   # thrust_coeff = T0 + Kp*u_target
        return attrs.evolve(
            MOTH_BIEKER_V3,
            sail_thrust_speeds=(),      # empty table -> affine branch
            sail_thrust_values=(),
            sail_thrust_coeff=coeff,
            sail_thrust_slope=-self.KP,
        )

    def _pair(self):
        params = self._governor_params()
        jax_model = Moth3D(
            params,
            u_forward=ConstantSchedule(U_FORWARD),
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=True,
        )
        casadi_model = Moth3DCasadiExact(
            params,
            heel_angle=HEEL_ANGLE,
            ventilation_mode=VENTILATION_MODE,
            ventilation_threshold=VENTILATION_THRESHOLD,
            surge_enabled=True,
            u_forward=U_FORWARD,
        )
        return jax_model, casadi_model

    def test_level0_governor_including_clamp(self, rng):
        """Derivative parity across a u band that straddles the zero-thrust
        kink (~11.9 m/s), so the clamp binds for the higher-u samples in
        both mirrors identically."""
        jax_model, casadi_model = self._pair()
        f = casadi_model.dynamics_function()

        for _ in range(120):
            x = np.array([
                rng.uniform(-1.5, -0.5),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(8, 16),         # straddles the 11.9 m/s kink
            ])
            u = np.array([
                rng.uniform(-0.17, 0.26),
                rng.uniform(-0.05, 0.10),
            ])

            jax_deriv = np.array(jax_model.forward_dynamics(jnp.array(x), jnp.array(u)))
            casadi_deriv = np.array(f(x, u)).flatten()

            np.testing.assert_allclose(
                casadi_deriv, jax_deriv, rtol=DERIV_RTOL, atol=DERIV_ATOL,
                err_msg=f"Governor parity mismatch at state: {x}",
            )

    def test_clamp_binds_above_kink(self):
        """Sanity: the clamp is live — above the kink the sail force is
        exactly zero (not negative) in both mirrors, and positive below."""
        from fmd.simulator.components.moth_forces import MothSailForce

        sail = MothSailForce(
            thrust_coeff=self.T0 + self.KP * self.U_TARGET,
            thrust_slope=-self.KP,
        )
        state = jnp.array([-1.3, 0.0, 0.0, 0.0, 10.0])
        control = jnp.array([0.0, 0.0])
        # Below the kink: positive thrust equal to the affine value.
        f_below, _ = sail.compute_moth(state, control, u_forward=10.0)
        assert float(f_below[0]) == pytest.approx(self.T0, abs=1e-6)  # T0 at u_target
        # Above the kink (u=15 > 11.9): affine value negative -> clamped to 0.
        f_above, _ = sail.compute_moth(state, control, u_forward=15.0)
        assert float(f_above[0]) == pytest.approx(0.0, abs=1e-9)
