"""Tests for LQRController, MechanicalWandController, and PIDController."""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator.controllers import (
    LQRController,
    MechanicalWandController,
    PIDController,
    _pid_pos_d_estimate,
)
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.moth_3d import MAIN_FLAP_MIN, MAIN_FLAP_MAX
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.components.moth_wand import (
    DEFAULT_WAND_LENGTH,
    create_wand_linkage,
    wand_angle_from_state,
)


def _replace_gains(
    controller: PIDController,
    *,
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
    Kb: float | None = None,
) -> PIDController:
    """Return a copy of ``controller`` with the given PID gains overridden.

    Used by tests that want to isolate behaviour from the default-gain
    choice (e.g., force ``Kd=0`` so a proportional-sign assertion does
    not depend on the derivative contribution).
    """
    return PIDController(
        Kp=jnp.array(controller.Kp) if Kp is None else jnp.array(Kp),
        Ki=jnp.array(controller.Ki) if Ki is None else jnp.array(Ki),
        Kd=jnp.array(controller.Kd) if Kd is None else jnp.array(Kd),
        Kb=jnp.array(controller.Kb) if Kb is None else jnp.array(Kb),
        dt=controller.dt,
        pos_d_target=controller.pos_d_target,
        flap_trim=controller.flap_trim,
        elevator_trim=controller.elevator_trim,
        wand_length=controller.wand_length,
        wand_pivot_z_body=controller.wand_pivot_z_body,
        heel_angle=controller.heel_angle,
        wand_angle_offset=controller.wand_angle_offset,
        u_min=controller.u_min,
        u_max=controller.u_max,
    )


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


# =============================================================================
# PIDController
# =============================================================================


@pytest.fixture(scope="module")
def pid_setup():
    """Build a PIDController calibrated to the MOTH_BIEKER_V3 trim point."""
    from fmd.simulator.moth_scenarios import create_pid_wand_config

    lqr = design_moth_lqr(u_forward=10.0)
    sensor, estimator, controller = create_pid_wand_config(
        lqr, params=MOTH_BIEKER_V3, heel_angle=np.deg2rad(30.0)
    )
    return sensor, estimator, controller, lqr


class TestPIDController:
    """Tests for PIDController: inversion identity, gain signs, saturation."""

    def test_closed_form_inversion_round_trip(self, pid_setup):
        """Inversion evaluated at trim wand angle reproduces ``pos_d_target``."""
        _, _, controller, lqr = pid_setup
        wand_pivot = MOTH_BIEKER_V3.wand_pivot_position
        trim_wand_angle = wand_angle_from_state(
            pos_d=jnp.array(lqr.trim.state[0]),
            theta=jnp.array(lqr.trim.state[1]),
            wand_pivot_position=jnp.asarray(wand_pivot),
            wand_length=DEFAULT_WAND_LENGTH,
            heel_angle=np.deg2rad(30.0),
        )
        pos_d_est = controller.estimate_pos_d(trim_wand_angle)
        np.testing.assert_allclose(
            float(pos_d_est), float(controller.pos_d_target), atol=1e-9
        )

    def test_zero_gains_returns_flap_trim(self, pid_setup):
        """With Kp=Ki=Kd=0 the controller returns the trim flap command."""
        _, _, controller, _ = pid_setup
        zeroed = _replace_gains(controller, Kp=0.0, Ki=0.0, Kd=0.0)
        # Use an arbitrary wand angle far from trim
        x_est = jnp.zeros(5).at[0].set(0.4)
        u, _ = zeroed.control(x_est, 0.0, zeroed.init_controller_state())
        np.testing.assert_allclose(
            float(u[0]), float(zeroed.flap_trim), atol=1e-9
        )

    def test_proportional_sign(self, pid_setup):
        """Boat below trim (positive height_err) → flap moves up (more lift).

        Constructs a Kd=0 controller explicitly so the assertion does not
        depend on the default-gain choice (a future non-zero Kd could
        silently change the test's meaning).
        """
        _, _, controller, lqr = pid_setup
        controller = _replace_gains(controller, Kd=0.0)
        wand_pivot = MOTH_BIEKER_V3.wand_pivot_position
        # Wand angle that corresponds to the boat being lower than trim:
        # boat lower → pivot deeper → ratio = h/L smaller → arccos larger.
        trim_wand_angle = float(
            wand_angle_from_state(
                pos_d=jnp.array(lqr.trim.state[0]),
                theta=jnp.array(lqr.trim.state[1]),
                wand_pivot_position=jnp.asarray(wand_pivot),
                wand_length=DEFAULT_WAND_LENGTH,
                heel_angle=np.deg2rad(30.0),
            )
        )
        wand_angle_low = trim_wand_angle + 0.05  # boat sunk a bit
        x_est = jnp.zeros(5).at[0].set(wand_angle_low)
        u, _ = controller.control(x_est, 0.0, controller.init_controller_state())
        # With Kd=0 the assertion isolates the proportional contribution.
        assert float(u[0]) > float(controller.flap_trim), (
            f"Expected flap > flap_trim for boat-below-trim, got u[0]={float(u[0])}"
        )

    def test_elevator_held_at_trim(self, pid_setup):
        """Elevator output is the (clipped) trim value regardless of wand angle."""
        _, _, controller, _ = pid_setup
        for wand_angle in [0.3, 0.5, 0.7, 0.9]:
            x_est = jnp.zeros(5).at[0].set(wand_angle)
            u, _ = controller.control(x_est, 0.0, controller.init_controller_state())
            expected = float(
                jnp.clip(controller.elevator_trim, controller.u_min[1], controller.u_max[1])
            )
            np.testing.assert_allclose(float(u[1]), expected, atol=1e-10)

    def test_flap_saturation_enforced(self, pid_setup):
        """Extreme wand angle clips flap to bounds."""
        _, _, controller, _ = pid_setup
        # Very large wand angle → boat far below trim → large positive err
        x_est = jnp.zeros(5).at[0].set(1.5)
        u, _ = controller.control(x_est, 0.0, controller.init_controller_state())
        assert float(u[0]) <= float(controller.u_max[0]) + 1e-10
        # Very small wand angle → boat far above trim → large negative err
        x_est = jnp.zeros(5).at[0].set(0.0)
        u, _ = controller.control(x_est, 0.0, controller.init_controller_state())
        assert float(u[0]) >= float(controller.u_min[0]) - 1e-10

    def test_vmap_batched(self, pid_setup):
        """vmap over a batch of wand-angle measurements works."""
        _, _, controller, _ = pid_setup

        wand_angles = jnp.array([0.4, 0.5, 0.6, 0.7, 0.8])
        x_est_batch = jnp.zeros((5, 5)).at[:, 0].set(wand_angles)
        # All seeds share the same fresh controller state
        ctrl_state = controller.init_controller_state()

        def _single(x_est):
            u, _ = controller.control(x_est, 0.0, ctrl_state)
            return u

        u_batch = jax.vmap(_single)(x_est_batch)
        assert u_batch.shape == (5, 2)
        # Each output should be within bounds
        assert np.all(np.asarray(u_batch[:, 0]) <= float(controller.u_max[0]) + 1e-9)
        assert np.all(np.asarray(u_batch[:, 0]) >= float(controller.u_min[0]) - 1e-9)

    def test_integrator_accumulates(self, pid_setup):
        """Outside saturation, the integrator update equals ``err * dt`` per step.

        Asserts the actual update law (not just monotonicity), so a future
        change to the integration scheme (or to anti-windup behaviour
        below the saturation limit) gets caught.
        """
        _, _, controller, _ = pid_setup
        # Choose a wand angle that produces a small enough error to keep
        # the flap unsaturated for the whole 5 steps. Trim wand angle
        # is ~0.7-0.8 rad; small positive offset gives a modest err.
        x_est = jnp.zeros(5).at[0].set(float(controller.wand_angle_offset * 0.0 + 0.78))
        # Solve for the resulting height_err so we can predict the integrator.
        pos_d_est = float(controller.estimate_pos_d(jnp.array(x_est[0])))
        height_err = pos_d_est - float(controller.pos_d_target)
        dt = float(controller.dt)

        ctrl_state = controller.init_controller_state()
        integrators = [float(ctrl_state.integrator)]
        for _ in range(5):
            u, ctrl_state = controller.control(x_est, 0.0, ctrl_state)
            # Sanity: flap must not saturate during this test.
            assert (
                float(controller.u_min[0]) + 1e-6
                < float(u[0])
                < float(controller.u_max[0]) - 1e-6
            ), "Test precondition violated: flap saturated."
            integrators.append(float(ctrl_state.integrator))
        diffs = np.diff(integrators)
        # Below saturation, each step should add exactly ``err * dt``.
        np.testing.assert_allclose(
            diffs, np.full_like(diffs, height_err * dt), atol=1e-9
        )

    def test_anti_windup_holds_during_saturation(self, pid_setup):
        """Under sustained saturation, anti-windup keeps the integrator bounded.

        With a high ``Kp`` and a large constant height-error signal the
        unsaturated flap command exceeds ``u_max`` from step 1; the
        classical update would let the integrator grow without bound.
        The Aström back-calculation update used here clamps it to a
        finite asymptote.
        """
        _, _, controller, _ = pid_setup
        # Force the flap deep into saturation: large Kp, modest Ki (so Kb
        # = 1 / Ki is finite and effective).
        controller = _replace_gains(controller, Kp=50.0, Ki=1.0, Kd=0.0)
        # Wand angle far from trim -> large positive height_err and a
        # positive flap command well past u_max.
        x_est = jnp.zeros(5).at[0].set(1.4)
        ctrl_state = controller.init_controller_state()
        integrators = []
        for _ in range(200):
            u, ctrl_state = controller.control(x_est, 0.0, ctrl_state)
            integrators.append(float(ctrl_state.integrator))
            # Confirm we are indeed saturated for the whole run.
            assert float(u[0]) >= float(controller.u_max[0]) - 1e-9

        integrators = np.asarray(integrators)
        # Anti-windup invariant: the integrator must STOP changing once
        # the system reaches a steady (saturated) state. Classical update
        # would keep adding ``err * dt`` indefinitely; back-calculation
        # winds the integrator to the value that makes
        # ``u_unsat = u_sat``, then holds it. The asymptotic value can be
        # large in absolute terms (it cancels Kp*err to land at u_max)
        # but the step-to-step change near the end must be tiny.
        dt = float(controller.dt)
        pos_d_est = float(
            controller.estimate_pos_d(jnp.array(x_est[0]))
        )
        height_err = pos_d_est - float(controller.pos_d_target)
        classical_per_step = abs(height_err * dt)

        last_diffs = np.abs(np.diff(integrators[-10:]))
        # Anti-windup should drive the step change to ~1e-3 of the
        # classical update at steady state (the residual is the
        # tiny ``err * dt`` that the back-calculation does not cancel).
        assert np.max(last_diffs) < 0.05 * classical_per_step, (
            f"Anti-windup did not stabilise; max step-change in last 10 "
            f"= {np.max(last_diffs)}, classical per-step = {classical_per_step}"
        )

        # Disabling anti-windup (Kb=0) should make the integrator grow
        # linearly without bound.
        no_aw = _replace_gains(controller, Kb=0.0)
        ctrl_state2 = no_aw.init_controller_state()
        ints2 = []
        for _ in range(200):
            _, ctrl_state2 = no_aw.control(x_est, 0.0, ctrl_state2)
            ints2.append(float(ctrl_state2.integrator))
        ints2 = np.asarray(ints2)
        # Without anti-windup, last 10 steps should each add ~err * dt
        # (no clamping).
        last_diffs_no_aw = np.abs(np.diff(ints2[-10:]))
        np.testing.assert_allclose(
            last_diffs_no_aw,
            np.full_like(last_diffs_no_aw, classical_per_step),
            rtol=1e-6,
        )
        # And the integrator should keep growing in magnitude (last >
        # first by a lot).
        assert abs(ints2[-1]) > 10.0 * abs(ints2[0]), (
            f"Without anti-windup, integrator should grow significantly: "
            f"no_aw_first={ints2[0]}, no_aw_final={ints2[-1]}"
        )


def test_pid_wand_config_inversion_exact_at_arbitrary_pos_d_with_trim_theta():
    """With heel baked in and offset calibrated at trim, the inversion is
    pos_d-agnostic under theta = trim_theta.

    Sweep pos_d over a wide range, forward-map to wand angle at trim_theta
    and constant heel, then inversion-map back. The result must match the
    starting pos_d to within numerical tolerance.

    This tests the mathematical guarantee of the cos(heel) formula:
    estimate_pos_d(wand_angle_from_state(pos_d, trim_theta, heel)) == pos_d
    for ANY pos_d, not just at the natural trim.
    """
    from fmd.simulator.moth_scenarios import create_pid_wand_config

    lqr = design_moth_lqr(u_forward=10.0)
    heel = np.deg2rad(30.0)
    sensor, estimator, controller = create_pid_wand_config(
        lqr, params=MOTH_BIEKER_V3, heel_angle=heel
    )

    trim_theta = float(lqr.trim.state[1])
    pivot = jnp.asarray(MOTH_BIEKER_V3.wand_pivot_position)
    natural_pos_d = float(lqr.trim.state[0])

    # The wand-angle arccos is valid only where the wand doesn't bottom out
    # (angle > 0). For MOTH_BIEKER_V3 at 30° heel and trim theta, the physical
    # limit is ~15 cm below natural trim. We sweep within the valid range:
    # 10 cm deeper than trim to 35 cm shallower (higher boat, larger wand angle).
    for pos_d in np.linspace(natural_pos_d - 0.10, natural_pos_d + 0.35, 9):
        wand_angle = float(wand_angle_from_state(
            pos_d=pos_d,
            theta=trim_theta,
            wand_pivot_position=pivot,
            wand_length=DEFAULT_WAND_LENGTH,
            heel_angle=heel,
        ))
        # Call the same math the PID uses at runtime — exercises the
        # exact formula, not a re-derivation.
        pos_d_est = _pid_pos_d_estimate(
            wand_angle,
            wand_pivot_z=float(pivot[2]),
            wand_length=DEFAULT_WAND_LENGTH,
            heel_angle=heel,
            wand_angle_offset=float(controller.wand_angle_offset),
        )
        assert abs(pos_d_est - pos_d) < 1e-6, (
            f"Inversion not exact at pos_d={pos_d:.4f}: "
            f"estimated {pos_d_est:.6f}, error {pos_d_est - pos_d:.2e}"
        )
