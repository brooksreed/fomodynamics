"""Tests for LQRController and MechanicalWandController."""

from fmd.simulator import _config  # noqa: F401

import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator.controllers import LQRController, MechanicalWandController
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.moth_3d import MAIN_FLAP_MIN, MAIN_FLAP_MAX
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.components.moth_wand import create_wand_linkage


@pytest.fixture(scope="module")
def controller_setup():
    """Shared LQR design for controller tests."""
    lqr_result = design_moth_lqr(u_forward=10.0)
    trim = lqr_result.trim

    u_min = jnp.array([MAIN_FLAP_MIN, MOTH_BIEKER_V3.rudder_elevator_min])
    u_max = jnp.array([MAIN_FLAP_MAX, MOTH_BIEKER_V3.rudder_elevator_max])

    controller = LQRController(
        K=jnp.array(lqr_result.K),
        x_trim=jnp.array(trim.state),
        u_trim=jnp.array(trim.control),
        u_min=u_min,
        u_max=u_max,
    )
    return controller, lqr_result


class TestLQRController:
    """Tests for LQR controller output and bounds."""

    def test_at_trim_returns_trim_control(self, controller_setup):
        """At trim state, controller returns trim control."""
        controller, lqr_result = controller_setup
        trim = lqr_result.trim

        u = controller.control(jnp.array(trim.state), 0.0)
        np.testing.assert_allclose(u, np.array(trim.control), atol=1e-10)

    def test_perturbation_produces_nonzero_correction(self, controller_setup):
        """Perturbed state produces different control from trim."""
        controller, lqr_result = controller_setup
        trim = lqr_result.trim

        x_pert = jnp.array(trim.state).at[0].set(trim.state[0] + 0.05)
        u = controller.control(x_pert, 0.0)

        # Should differ from trim control
        assert not np.allclose(u, np.array(trim.control), atol=1e-6)

    def test_control_clipped_to_bounds(self, controller_setup):
        """Large perturbation produces saturated control within bounds."""
        controller, lqr_result = controller_setup
        trim = lqr_result.trim

        # Large perturbation to force saturation
        x_extreme = jnp.array(trim.state).at[0].set(trim.state[0] + 1.0)
        u = controller.control(x_extreme, 0.0)

        assert np.all(u >= np.array(controller.u_min) - 1e-10), (
            f"Control below lower bound: {u}"
        )
        assert np.all(u <= np.array(controller.u_max) + 1e-10), (
            f"Control above upper bound: {u}"
        )

    def test_matches_manual_lqr_computation(self, controller_setup):
        """Controller output matches manual K@(x-x_trim) computation."""
        controller, lqr_result = controller_setup
        trim = lqr_result.trim

        x = jnp.array(trim.state).at[0].set(trim.state[0] + 0.01)
        u = controller.control(x, 0.0)

        # Manual computation
        K = jnp.array(lqr_result.K)
        x_trim = jnp.array(trim.state)
        u_trim = jnp.array(trim.control)
        u_expected = u_trim - K @ (x[:K.shape[1]] - x_trim[:K.shape[1]])
        u_expected = jnp.clip(u_expected, controller.u_min, controller.u_max)

        np.testing.assert_allclose(u, u_expected, atol=1e-12)


class TestMechanicalWandController:
    """Tests for MechanicalWandController: linkage output, bounds, elevator trim."""

    @pytest.fixture()
    def wand_controller(self):
        """Create a MechanicalWandController with default linkage."""
        linkage = create_wand_linkage()
        u_min = jnp.array([MAIN_FLAP_MIN, MOTH_BIEKER_V3.rudder_elevator_min])
        u_max = jnp.array([MAIN_FLAP_MAX, MOTH_BIEKER_V3.rudder_elevator_max])
        elevator_trim = 0.05
        controller = MechanicalWandController(
            linkage=linkage,
            elevator_trim=elevator_trim,
            u_min=u_min,
            u_max=u_max,
        )
        return controller, linkage

    def test_linkage_output_matches_direct(self, wand_controller):
        """Flap from controller matches direct linkage computation."""
        controller, linkage = wand_controller

        wand_angle = 0.5  # ~29 degrees
        x_est = jnp.zeros(5).at[0].set(wand_angle)
        u = controller.control(x_est, 0.0)

        expected_flap = linkage.compute(jnp.array(wand_angle))
        np.testing.assert_allclose(u[0], expected_flap, atol=1e-10)

    def test_elevator_trim_passthrough(self, wand_controller):
        """Elevator output is always the fixed trim value."""
        controller, _ = wand_controller

        for wand_angle in [0.0, 0.3, 0.7, 1.2]:
            x_est = jnp.zeros(5).at[0].set(wand_angle)
            u = controller.control(x_est, 0.0)
            # Elevator should be clipped version of elevator_trim
            expected_elev = jnp.clip(
                jnp.array(controller.elevator_trim),
                controller.u_min[1],
                controller.u_max[1],
            )
            np.testing.assert_allclose(u[1], expected_elev, atol=1e-10)

    def test_control_clipped_to_bounds(self, wand_controller):
        """Control output is always within bounds."""
        controller, _ = wand_controller

        # Test across range of wand angles
        for wand_angle in [0.0, 0.3, 0.5, 0.8, 1.2, jnp.pi / 2]:
            x_est = jnp.zeros(5).at[0].set(wand_angle)
            u = controller.control(x_est, float(0.0))
            assert np.all(u >= np.array(controller.u_min) - 1e-10), (
                f"Control below lower bound at wand_angle={wand_angle}: {u}"
            )
            assert np.all(u <= np.array(controller.u_max) + 1e-10), (
                f"Control above upper bound at wand_angle={wand_angle}: {u}"
            )

    def test_wand_angle_zero_vertical(self, wand_controller):
        """Wand vertical (angle=0) produces expected flap direction (with clipping)."""
        controller, linkage = wand_controller
        # Wand at 0 (vertical) = boat high -> flap should be negative (less lift)
        x_est = jnp.zeros(5).at[0].set(0.0)
        u = controller.control(x_est, 0.0)
        raw_flap = linkage.compute(jnp.array(0.0))
        expected_flap = jnp.clip(raw_flap, controller.u_min[0], controller.u_max[0])
        np.testing.assert_allclose(u[0], expected_flap, atol=1e-10)

    def test_reads_slot_0_only(self, wand_controller):
        """Controller uses only x_est[0], ignoring other slots."""
        controller, _ = wand_controller

        wand_angle = 0.5
        x1 = jnp.array([wand_angle, 0.0, 0.0, 0.0, 0.0])
        x2 = jnp.array([wand_angle, 99.0, -5.0, 10.0, 100.0])

        u1 = controller.control(x1, 0.0)
        u2 = controller.control(x2, 0.0)

        np.testing.assert_allclose(u1, u2, atol=1e-10)

    def test_monotonic_flap_response(self, wand_controller):
        """Larger wand angle (boat lower) produces more positive flap (more lift).

        Uses angles in the unsaturated range to avoid clipping at bounds.
        """
        controller, _ = wand_controller

        # Fastpoint is ~45 deg; use angles around it for linear region
        angles = [0.6, 0.7, 0.8, 0.9]
        flaps = []
        for angle in angles:
            x_est = jnp.zeros(5).at[0].set(angle)
            u = controller.control(x_est, 0.0)
            flaps.append(float(u[0]))

        # Flap should increase monotonically with wand angle
        for i in range(len(flaps) - 1):
            assert flaps[i + 1] > flaps[i], (
                f"Flap not monotonically increasing: {flaps}"
            )
