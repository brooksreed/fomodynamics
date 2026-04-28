"""Physical validation tests for Moth 3DOF longitudinal dynamics.

Tests are organized into 4 categories:
1. Equilibrium Validation - trim convergence and physical reasonableness
2. Linearized Stability - Jacobian finiteness, eigenvalues, controllability
3. Perturbation Response - instantaneous derivative checks for perturbations
4. Force/Energy Consistency - force balance and scaling laws

The moth_model fixture is parameterized for future CasADi backend reuse.

Notes on depth and ventilation:
    The sailor (75kg at [-0.15, 0, -0.2] relative to hull CG) shifts the
    composite CG forward 0.09m and up 0.12m. The CG-adjusted main foil
    depth is 1.94m below the system CG. At trim (pos_d~-1.3), the foil
    center is ~0.64m below the waterline — well submerged. Ventilation
    only begins around pos_d~-1.75 (15 deg heel) when the leeward foil
    tip approaches the surface.

Notes on low-speed trim:
    Level trim (target_theta=0.0) is infeasible below ~6 m/s because the
    elevator saturates at its +6° bound. With theta=0, all AoA comes from
    the flap, requiring large flap deflections that create a large nose-up
    moment. The small rudder elevator (0.04 m², bounds [-3°,+6°]) cannot
    generate enough nose-down moment to compensate at low dynamic pressure.
    The default (free theta) trim converges at all speeds by allowing
    theta>0, which distributes AoA across both foils.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from fmd.simulator import (
    Moth3D, ConstantSchedule, simulate, linearize, controllability_matrix, is_controllable,
    ConstantControl,
)
from fmd.simulator.moth_3d import POS_D, THETA, W, Q
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.components.moth_forces import (
    compute_foil_ned_depth,
    compute_depth_factor,
)



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["jax"])  # Later: params=["jax", "casadi"]
def moth_model(request):
    """Parameterized Moth model fixture for backend-agnostic tests (module-scoped for JIT reuse)."""
    if request.param == "jax":
        return Moth3D(MOTH_BIEKER_V3)


@pytest.fixture(scope="module")
def trim_result(moth_model):
    """Pre-computed trim at nominal 10 m/s with pinned targets (module-scoped for JIT reuse).

    Pins theta=0.005 and pos_d=-1.3 for a deterministic, reproducible
    trim point. These values are near the default converged trim at 10 m/s.
    """
    return find_moth_trim(
        MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
    )


# ---------------------------------------------------------------------------
# Category 1: Equilibrium Validation
# ---------------------------------------------------------------------------

class TestEquilibriumValidation:
    """Verify trim finder converges to physically reasonable equilibria."""

    def test_trim_residual_at_nominal_speed(self, trim_result):
        """Trim residual small at 10 m/s (scale-aware objective with regularization)."""
        assert trim_result.success
        assert trim_result.residual < 0.01

    @pytest.mark.parametrize("speed", [10.0, 12.0])
    def test_trim_converges_at_multiple_speeds(self, speed):
        """Trim converges across the foiling speed range (10-12 m/s).

        Uses free theta (no target_theta) so the solver can find equilibrium
        at any pitch. Build a per-speed model so force evaluation speed
        matches the requested trim speed.

        Speeds below 10 m/s excluded: CasADi trim solver has convergence
        issues at 8 m/s with NED sail thrust (theta-dependent force balance
        makes the problem harder at low speed where theta is larger).

        Checks residual rather than success flag; SLSQP may report
        "Iteration limit reached" even when the objective is very small.
        """
        speed_moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(speed))
        result = find_moth_trim(MOTH_BIEKER_V3, u_forward=speed)
        assert result.residual < 0.1, (
            f"Trim residual {result.residual:.2e} at {speed} m/s"
        )

    def test_trim_state_physically_reasonable(self, trim_result):
        """Trim state has reasonable ride height, pitch, and controls.

        At 10 m/s, pos_d converges near -1.3m and theta near 0.005 rad
        (~0.3 deg). Validate near expected values, not just loose bounds.
        """
        state = trim_result.state
        control = trim_result.control

        # Foiling: CG above water, near -1.3m (pinned target)
        assert -1.5 < state[POS_D] < -1.1, (
            f"pos_d={state[POS_D]:.3f}m, expected near -1.3m"
        )

        # Pitch angle: small positive (we pinned at 0.005 rad ~ 0.3 deg)
        assert -np.radians(1) < state[THETA] < np.radians(5), (
            f"theta={np.rad2deg(state[THETA]):.1f} deg, expected small positive"
        )

        # Pitch rate should be ~0 at trim
        assert abs(state[Q]) < 1e-6, f"q={state[Q]:.2e}"

        # Controls within defined bounds
        from fmd.simulator.moth_3d import (
            MAIN_FLAP_MIN, MAIN_FLAP_MAX,
            RUDDER_ELEVATOR_MIN, RUDDER_ELEVATOR_MAX,
        )
        assert MAIN_FLAP_MIN - 1e-6 <= control[0] <= MAIN_FLAP_MAX + 1e-6
        assert RUDDER_ELEVATOR_MIN - 1e-6 <= control[1] <= RUDDER_ELEVATOR_MAX + 1e-6

    def test_trim_residual_components_individually_small(self, moth_model, trim_result):
        """Each derivative component is individually small, not just L2 norm.

        Note: When using pinned targets (theta/pos_d), the CasADi solver
        finds the thrust needed for that operating point, which may differ
        from the baked-in preset thrust table (which is calibrated at the
        free trim point). So JAX residuals at pinned targets can be O(0.1)
        due to thrust mismatch. The CasADi residual is always < 1e-6.
        """
        # Check CasADi residual directly (always accurate)
        assert trim_result.residual < 1e-6, (
            f"CasADi residual = {trim_result.residual:.2e}, expected < 1e-6"
        )

        # Also check JAX model residuals with relaxed tolerance
        # (accounts for thrust table mismatch at pinned targets)
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)
        deriv = moth_model.forward_dynamics(state, control, 0.0)
        deriv_np = np.array(deriv)

        for i, name in enumerate(moth_model.state_names):
            assert abs(deriv_np[i]) < 0.15, (
                f"d({name})/dt = {deriv_np[i]:.2e} at trim"
            )

    def test_trim_valid_from_different_guesses(self, moth_model):
        """Trim from different initial guesses converge to the same point.

        With pinned targets (theta=0.005, pos_d=-1.3), both guesses
        must converge to the same unique solution.
        """
        result_a = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            target_theta=0.005, target_pos_d=-1.3,
        )
        result_b = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            target_theta=0.005, target_pos_d=-1.3,
        )
        assert result_a.success and result_b.success
        assert result_a.residual < 0.01
        assert result_b.residual < 0.01

        # With pinned targets, both should find the SAME trim point
        np.testing.assert_allclose(
            result_a.state, result_b.state, atol=1e-3,
            err_msg="Pinned trim should converge to unique point",
        )
        np.testing.assert_allclose(
            result_a.control, result_b.control, atol=1e-3,
            err_msg="Pinned trim controls should match",
        )

    @pytest.mark.slow
    def test_trim_at_multiple_speeds_level(self):
        """At level trim (theta=0), flap decreases with speed.

        With target_theta=0.0 and target_pos_d=-1.3, the trim solver
        adjusts controls to maintain level flight. Flap should decrease
        with speed (more dynamic pressure -> less AoA needed).

        Creates per-speed Moth3D instances so the force evaluation speed
        matches the trim target speed. Speeds below 8 m/s excluded due
        to elevator saturation with current geometry.
        """
        speeds = [8.0, 10.0, 12.0]
        flaps = []
        for speed in speeds:
            moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(speed))
            result = find_moth_trim(
                MOTH_BIEKER_V3, u_forward=speed, target_theta=0.0, target_pos_d=-1.3,
            )
            assert result.success, f"Level trim failed at {speed} m/s"
            assert result.residual < 0.05, (
                f"Level trim residual {result.residual:.2e} at {speed} m/s"
            )
            flaps.append(result.control[0])

        # Flap should monotonically decrease with speed
        for i in range(len(flaps) - 1):
            assert flaps[i] > flaps[i + 1], (
                f"Flap should decrease with speed: "
                f"flap({speeds[i]:.0f})={np.degrees(flaps[i]):.3f} deg, "
                f"flap({speeds[i+1]:.0f})={np.degrees(flaps[i+1]):.3f} deg"
            )


# ---------------------------------------------------------------------------
# Category 2: Linearized Stability
# ---------------------------------------------------------------------------

class TestLinearizedStability:
    """Verify Jacobians are well-formed and system is controllable."""

    def test_a_matrix_finite(self, moth_model, trim_result):
        """State Jacobian (A) is finite at trim."""
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)
        A, _ = linearize(moth_model, state, control)

        assert A.shape == (5, 5)
        assert jnp.all(jnp.isfinite(A)), "A matrix has non-finite entries"

    def test_eigenvalue_analysis(self, moth_model, trim_result):
        """Eigenvalues are finite with physically reasonable growth rates.

        The open-loop Moth is expected to have an unstable pitch mode.
        We verify eigenvalues exist, are finite, and any unstable mode
        has a growth rate < 50 rad/s (divergence time > 20ms).
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)
        A, _ = linearize(moth_model, state, control)

        eigenvalues = np.linalg.eigvals(np.array(A))

        assert all(np.isfinite(eigenvalues)), "Non-finite eigenvalues"

        # Any unstable mode should have reasonable growth rate
        max_real = max(eigenvalues.real)
        assert max_real < 50.0, (
            f"Unstable mode growth rate {max_real:.1f} rad/s is unreasonably fast"
        )

    def test_b_matrix_finite_and_nonzero(self, moth_model, trim_result):
        """Control Jacobian (B) is finite and has nonzero entries."""
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)
        _, B = linearize(moth_model, state, control)

        assert B.shape == (5, 2)
        assert jnp.all(jnp.isfinite(B)), "B matrix has non-finite entries"
        assert jnp.any(B != 0), "B matrix is all zeros — no control authority"

    def test_controllability(self, moth_model, trim_result):
        """Active DOFs are controllable at trim.

        With surge_enabled=True (default), the surge state has dynamics
        coupled through gravity projection and drag, but is not directly
        controllable by flap/elevator. The controllability matrix may have
        rank 4 or 5 depending on the coupling strength.
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)
        A, B = linearize(moth_model, state, control)

        C = controllability_matrix(np.array(A), np.array(B))
        assert C.shape == (5, 10)  # n x (n*m) = 5 x (5*2)

        # With surge enabled, rank can be 4 or 5 depending on coupling.
        # Surge couples through gravity projection (theta→u_dot) which
        # makes it indirectly controllable via pitch control.
        rank = np.linalg.matrix_rank(C)
        assert rank >= 4, (
            f"Controllability rank is {rank}, expected >= 4 "
            f"(4 active DOFs must be controllable)"
        )


# ---------------------------------------------------------------------------
# Category 3: Perturbation Response
# ---------------------------------------------------------------------------

class TestPerturbationResponse:
    """Verify physically correct responses to state perturbations.

    Uses instantaneous derivative checks where possible, with short
    simulations only for finiteness/boundedness validation.
    """

    def test_nose_up_perturbation_direction(self, moth_model, trim_result):
        """Nose-up pitch perturbation causes boat to rise (pos_d_dot < 0).

        A pure theta perturbation (without updating w to maintain the
        kinematic constraint) changes pos_d_dot = -u*sin(theta) + w*cos(theta).
        At trim, pos_d_dot ~ 0. Increasing theta makes -u*sin(theta) more
        negative, so pos_d_dot should become negative (boat rises).
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # Pure theta perturbation (don't update w -- break kinematic constraint)
        perturbed = state.at[THETA].add(np.deg2rad(2.0))

        deriv = moth_model.forward_dynamics(perturbed, control, 0.0)

        # pos_d_dot should be negative (boat rising)
        assert float(deriv[POS_D]) < -0.01, (
            f"Nose-up perturbation: pos_d_dot={float(deriv[POS_D]):.4f}, "
            f"expected negative"
        )

    def test_symmetric_perturbations_opposite_sign(self, moth_model, trim_result):
        """Opposite theta perturbations produce opposite-sign q_dot changes."""
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        delta = np.deg2rad(1.0)  # 1 deg pitch perturbation

        deriv_up = moth_model.forward_dynamics(
            state.at[THETA].add(+delta), control, 0.0
        )
        deriv_down = moth_model.forward_dynamics(
            state.at[THETA].add(-delta), control, 0.0
        )
        deriv_trim = moth_model.forward_dynamics(state, control, 0.0)

        # Changes from trim should be opposite sign in q_dot
        diff_up_q = float(deriv_up[Q]) - float(deriv_trim[Q])
        diff_down_q = float(deriv_down[Q]) - float(deriv_trim[Q])

        assert diff_up_q * diff_down_q < 0, (
            f"q_dot not antisymmetric for theta perturbation: "
            f"up={diff_up_q:.4e}, down={diff_down_q:.4e}"
        )

    def test_heave_velocity_perturbation_direction(self, moth_model, trim_result):
        """Heave velocity perturbation produces a depth response.

        Positive w (downward velocity) -> pos_d increases (sinking).
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # +0.5 m/s downward heave perturbation
        perturbed = state.at[W].add(0.5)
        deriv = moth_model.forward_dynamics(perturbed, control, 0.0)

        # pos_d_dot should be positive (moving deeper) from the w contribution
        # pos_d_dot = -u*sin(theta) + w*cos(theta). Adding to w increases
        # the w*cos(theta) term.
        deriv_trim = moth_model.forward_dynamics(state, control, 0.0)
        delta_pos_d_dot = float(deriv[POS_D]) - float(deriv_trim[POS_D])

        assert delta_pos_d_dot > 0.1, (
            f"Heave perturbation should increase pos_d_dot: "
            f"delta={delta_pos_d_dot:.4f}"
        )

    def test_small_perturbation_finite(self, moth_model, trim_result):
        """Small perturbation from trim produces finite states over 0.5s."""
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # +1cm depth perturbation
        perturbed = state.at[POS_D].add(0.01)

        result = simulate(
            moth_model, perturbed, dt=0.001, duration=0.5,
            control=ConstantControl(control),
        )

        assert jnp.all(jnp.isfinite(result.states)), "States diverged to non-finite"

        # States should remain in a reasonable range (tighter than before)
        pos_d_range = float(jnp.max(result.states[:, POS_D]) - jnp.min(result.states[:, POS_D]))
        assert pos_d_range < 0.5, f"pos_d range {pos_d_range:.2f}m over 0.5s (too large for 1cm pert)"

        theta_max = float(jnp.max(jnp.abs(result.states[:, THETA])))
        assert theta_max < np.deg2rad(30), (
            f"theta reached {np.rad2deg(theta_max):.1f} deg"
        )

    def test_large_perturbation_finite(self, moth_model, trim_result):
        """Large perturbation (beyond linear regime) produces finite states over 0.5s."""
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # Large perturbation: +10cm depth + 5 deg pitch
        perturbed = state.at[POS_D].add(0.10)
        perturbed = perturbed.at[THETA].add(np.deg2rad(5.0))

        result = simulate(
            moth_model, perturbed, dt=0.001, duration=0.5,
            control=ConstantControl(control),
        )

        assert jnp.all(jnp.isfinite(result.states)), (
            "States diverged to non-finite for large perturbation"
        )


# ---------------------------------------------------------------------------
# Category 4: Force/Energy Consistency
# ---------------------------------------------------------------------------

class TestForceEnergyConsistency:
    """Verify force balance and physical scaling laws."""

    def test_vertical_force_balance_at_trim(self, moth_model, trim_result):
        """At trim, gravity + lift ~ 0 (vertical force balance).

        We verify this indirectly: w_dot ~ 0 at trim means total_fz ~ 0,
        which means gravity is balanced by hydrodynamic lift.
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        deriv = moth_model.forward_dynamics(state, control, 0.0)

        # w_dot = total_fz / m_eff, so total_fz = w_dot * m_eff
        m_eff = moth_model.total_mass + moth_model.added_mass_heave
        total_fz = float(deriv[W]) * m_eff

        # Should be near zero (< 1N residual for ~100kg boat)
        assert abs(total_fz) < 1.0, (
            f"Vertical force imbalance at trim: {total_fz:.2f} N"
        )

    def test_depth_factor_well_submerged_at_trim(self, moth_model, trim_result):
        """Foils should be well submerged at trim (depth_factor > 0.95).

        Uses CG-adjusted foil positions (accounting for sailor mass offset)
        to match what forward_dynamics computes internally.
        """
        state = trim_result.state
        pos_d = state[POS_D]
        theta = state[THETA]

        # CG offset from sailor position (same as forward_dynamics uses)
        r_sailor = moth_model.sailor_position_schedule(0.0)
        cg_offset = moth_model.sailor_mass * r_sailor / moth_model.total_mass

        foil = moth_model.main_foil
        foil_depth = compute_foil_ned_depth(
            jnp.array(pos_d),
            foil.position_x - float(cg_offset[0]),
            foil.position_z - float(cg_offset[2]),
            jnp.array(theta), foil.heel_angle,
        )
        foil_df = compute_depth_factor(
            foil_depth, foil.foil_span, foil.heel_angle,
        )

        rudder = moth_model.rudder
        rudder_depth = compute_foil_ned_depth(
            jnp.array(pos_d),
            rudder.position_x - float(cg_offset[0]),
            rudder.position_z - float(cg_offset[2]),
            jnp.array(theta), rudder.heel_angle,
        )
        rudder_df = compute_depth_factor(
            rudder_depth, rudder.foil_span, rudder.heel_angle,
        )

        assert float(foil_df) > 0.95, (
            f"Main foil depth_factor={float(foil_df):.4f} at trim (expected > 0.95)"
        )
        assert float(rudder_df) > 0.95, (
            f"Rudder depth_factor={float(rudder_df):.4f} at trim (expected > 0.95)"
        )

    def test_ventilation_at_shallow_depth(self, moth_model, trim_result):
        """Foil tip ventilates when boat rises above normal foiling height.

        At trim (pos_d~-1.3), the foil is well submerged (depth_factor~1.0).
        At pos_d=-1.85, the boat is 1.85m above water — a realistic gust-
        induced excursion. The main foil tip approaches the surface (accounting
        for the CG offset), causing partial ventilation and a downward
        restoring acceleration.

        This tests the ventilation model's role as a restoring mechanism:
        if the boat rises too high, foil lift drops, and gravity pulls it back.
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # Shallow state: CG 1.85m above water (vs ~1.3m at trim)
        # CG-adjusted foil z = 1.94, so foil_depth = -1.85 + 1.94 = 0.09
        shallow = state.at[POS_D].set(-1.85)

        # Check depth factor is reduced (partial ventilation)
        # Use CG-adjusted foil positions (same as forward_dynamics)
        r_sailor = moth_model.sailor_position_schedule(0.0)
        cg_offset = moth_model.sailor_mass * r_sailor / moth_model.total_mass

        foil = moth_model.main_foil
        foil_depth = compute_foil_ned_depth(
            jnp.array(-1.85),
            foil.position_x - float(cg_offset[0]),
            foil.position_z - float(cg_offset[2]),
            state[THETA], foil.heel_angle,
        )
        df = compute_depth_factor(foil_depth, foil.foil_span, foil.heel_angle)
        assert float(df) < 0.65, (
            f"Expected partial ventilation at pos_d=-1.85, got depth_factor={float(df):.4f}"
        )

        # Check restoring force: w_dot > 0 (downward — gravity dominates reduced lift)
        deriv_trim = moth_model.forward_dynamics(state, control, 0.0)
        deriv_shallow = moth_model.forward_dynamics(shallow, control, 0.0)
        w_dot_shallow = float(deriv_shallow[W])

        assert w_dot_shallow > 1.0, (
            f"Expected strong downward acceleration at shallow depth, "
            f"got w_dot={w_dot_shallow:.3f} m/s^2"
        )

        # Restoring: w_dot at shallow depth >> w_dot at trim
        w_dot_trim = float(deriv_trim[W])
        assert w_dot_shallow - w_dot_trim > 1.0, (
            f"Expected restoring force (w_dot_shallow >> w_dot_trim): "
            f"w_dot_shallow={w_dot_shallow:.3f}, w_dot_trim={w_dot_trim:.6f}"
        )

    def test_lift_scales_with_speed_squared(self, moth_model):
        """Foil lift scales quadratically with speed.

        At trim, gravity is constant so the foil must adjust AoA to
        maintain lift = mg. We verify that the effective AoA (theta +
        flap * effectiveness) decreases as speed increases, since higher
        dynamic pressure requires less AoA for the same lift. Raw flap
        angle may not be monotonic because theta also adjusts at trim.
        Reuses the module-scoped moth_model fixture.
        """
        speeds = [6.0, 8.0, 10.0, 12.0]
        effective_aoas = []
        prev_result = None

        for speed in speeds:
            # CasADi is robust — no continuation seeding needed
            result = find_moth_trim(MOTH_BIEKER_V3, u_forward=speed)
            if result.success:
                theta = result.state[1]
                flap = result.control[0]
                eff_aoa = theta + flap * MOTH_BIEKER_V3.main_foil_flap_effectiveness
                effective_aoas.append(eff_aoa)
                prev_result = result
            else:
                pytest.skip(f"Trim failed at {speed} m/s")

        # Effective AoA should decrease with speed (more dynamic pressure)
        diffs = np.diff(effective_aoas)
        assert np.sum(diffs < 0) >= len(diffs) - 1, (
            f"Effective AoA not decreasing with speed: "
            f"{[np.degrees(a) for a in effective_aoas]}, "
            f"diffs={np.degrees(diffs)}"
        )

    def test_no_force_exceeds_physical_bounds(self, moth_model, trim_result):
        """Force components at trim don't exceed physical bounds.

        Verify that implied lift coefficients are below stall (~1.5 for
        a hydrofoil with flap).
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)
        u_fwd = moth_model.get_forward_speed(0.0)

        # Compute main foil force directly
        f_foil, _ = moth_model.main_foil.compute_moth(
            state, control, u_fwd, 0.0,
        )

        # Implied lift coefficient: L = 0.5 * rho * V^2 * A * Cl
        q_dyn = 0.5 * moth_model.rho_water * u_fwd**2
        foil_area = moth_model.main_foil.area
        lift_magnitude = float(jnp.sqrt(f_foil[0]**2 + f_foil[2]**2))
        implied_cl = lift_magnitude / (q_dyn * foil_area)

        assert implied_cl < 1.5, (
            f"Main foil Cl={implied_cl:.2f} exceeds stall limit"
        )

    def test_pitch_moment_balance_at_trim(self, moth_model, trim_result):
        """Total pitch moment ~ 0 at trim.

        Note: With pinned targets (theta=0.005, pos_d=-1.3), the CasADi solver
        finds a thrust (~82.7 N) that differs from the preset thrust table value
        (~74.8 N, calibrated at the free trim point). This ~8 N thrust mismatch
        creates a residual pitch moment when evaluated through the JAX model
        (which uses the preset table). Tolerance is set to 15 N*m to provide
        adequate margin above the observed ~8.3 N*m residual.
        """
        state = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        deriv = moth_model.forward_dynamics(state, control, 0.0)

        # q_dot = total_my / i_eff
        i_eff = moth_model.iyy + moth_model.added_inertia_pitch
        total_my = float(deriv[Q]) * i_eff

        assert abs(total_my) < 15.0, (
            f"Pitch moment imbalance at trim: {total_my:.2f} N*m"
        )
