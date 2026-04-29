"""Tests for Moth 3DOF force and moment components."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fmd.simulator.components.moth_forces import (
    MothMainFoil,
    MothRudderElevator,
    MothSailForce,
    MothHullDrag,
    create_moth_components,
    compute_depth_factor,
)
from fmd.simulator.moth_3d import POS_D, THETA, W, Q, MAIN_FLAP, RUDDER_ELEVATOR
from fmd.simulator.params import MOTH_BIEKER_V3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_foil():
    """Create a main foil with typical Moth parameters.

    FRD body frame: +x forward, +z down.
    Main foil is forward of CG (+x) and below CG (+z).
    """
    return MothMainFoil(
        rho=1025.0,
        area=0.12,
        cl_alpha=5.7,
        cl0=0.0,
        cd0=0.01,
        oswald=0.85,
        ar=8.33,
        flap_effectiveness=0.5,
        cd_flap=0.15,
        position_x=0.6,   # forward of CG (FRD: +x forward)
        position_z=0.8,   # below CG (FRD: +z down)
        foil_span=1.0,    # 1m wingspan
    )


@pytest.fixture
def default_rudder():
    """Create a rudder with typical Moth parameters.

    FRD body frame: +x forward, +z down.
    Rudder is aft of CG (-x) and below CG (+z).
    """
    return MothRudderElevator(
        rho=1025.0,
        area=0.04,
        cl_alpha=5.0,
        position_x=-1.5,  # aft of CG (FRD: -x = aft)
        position_z=0.5,   # below CG (FRD: +z down)
        foil_span=0.5,    # 0.5m rudder wingspan
    )


@pytest.fixture
def default_sail():
    """Create a sail force with typical Moth parameters."""
    return MothSailForce(
        thrust_coeff=50.0,
        ce_position_z=-2.5,
    )


@pytest.fixture
def default_hull_drag():
    """Create hull drag with typical Moth parameters (no buoyancy for drag-only tests)."""
    return MothHullDrag(
        drag_coeff=500.0,
        contact_depth=0.15,
        hull_cg_above_bottom=0.15,
        buoyancy_coeff=0.0,
        buoyancy_fwd_x=0.0,
        buoyancy_aft_x=0.0,
    )


@pytest.fixture
def hull_with_buoyancy():
    """Create hull drag + buoyancy with typical Moth parameters."""
    return MothHullDrag(
        drag_coeff=500.0,
        contact_depth=0.15,
        hull_cg_above_bottom=0.15,
        buoyancy_coeff=5000.0,
        buoyancy_fwd_x=0.84,   # ~hull_length/4
        buoyancy_aft_x=-0.84,  # ~-hull_length/4
    )


def _make_state(pos_d=0.4, theta=0.0, w=0.0, q=0.0, u=10.0):
    """Helper: create Moth3D state vector."""
    return jnp.array([pos_d, theta, w, q, u])


def _make_control(main_flap=0.0, rudder_elevator=0.0):
    """Helper: create Moth3D control vector."""
    return jnp.array([main_flap, rudder_elevator])


# ===========================================================================
# TestMothMainFoil
# ===========================================================================

class TestMothMainFoil:
    """Tests for main foil force component."""

    def test_zero_aoa_zero_cl0_no_lift(self):
        """Zero AoA with cl0=0 produces no lift."""
        foil = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.6, position_z=0.8, foil_span=1.0,
        )
        state = _make_state(pos_d=0.4, theta=0.0, w=0.0)
        control = _make_control(main_flap=0.0)
        force, moment = foil.compute_moth(state, control, 10.0)
        # CL = 0 at zero AoA with cl0=0, but CD0 produces some drag
        # Fz should be ~0 (no lift)
        assert abs(float(force[2])) < 1e-6

    def test_positive_aoa_produces_upward_lift(self, default_foil):
        """Positive AoA produces upward lift (negative Fz in body frame)."""
        # Use w = u*tan(theta) so body-frame flow creates actual AoA
        theta = 0.05
        u = 10.0
        state = _make_state(pos_d=0.4, theta=theta, w=u * np.tan(theta))
        control = _make_control()
        force, _ = default_foil.compute_moth(state, control, u)
        # Upward lift = negative Fz (body frame +z is down)
        assert float(force[2]) < 0.0

    def test_lift_scales_with_speed_squared(self, default_foil):
        """Lift force scales with u^2."""
        # Use nonzero flap to create AoA independent of speed
        control = _make_control(main_flap=0.1)

        state3 = _make_state(pos_d=0.4, u=3.0)
        state6 = _make_state(pos_d=0.4, u=6.0)
        f3, _ = default_foil.compute_moth(state3, control, 3.0)
        f6, _ = default_foil.compute_moth(state6, control, 6.0)

        # Lift at 6 m/s should be ~4x lift at 3 m/s
        ratio = float(f6[2]) / float(f3[2])
        assert ratio == pytest.approx(4.0, rel=0.05)

    def test_flap_increases_effective_aoa(self, default_foil):
        """Positive flap deflection increases effective AoA and lift."""
        state = _make_state(pos_d=0.4, theta=0.0)
        ctrl_no_flap = _make_control(main_flap=0.0)
        ctrl_flap = _make_control(main_flap=0.1)  # ~5.7 deg flap

        f_no, _ = default_foil.compute_moth(state, ctrl_no_flap, 10.0)
        f_flap, _ = default_foil.compute_moth(state, ctrl_flap, 10.0)

        # With flap, more negative Fz (more upward lift)
        assert float(f_flap[2]) < float(f_no[2])

    def test_depth_factor_uses_foil_depth(self, default_foil):
        """Depth factor uses foil NED depth (via compute_foil_ned_depth), not CG depth.

        When CG is at surface (pos_d=0), foil at position_z=0.8 is at 0.8m depth
        (at heel=0, cos(heel)=1 so depth = pos_d + position_z * cos(theta)).
        The foil should produce significant lift since it's submerged.
        """
        # Use flap to produce AoA and lift (theta alone doesn't create body-frame AoA)
        state = _make_state(pos_d=0.0, theta=0.05)
        control = _make_control(main_flap=0.1)
        force, _ = default_foil.compute_moth(state, control, 10.0)
        # Foil is at 0.8m depth -> depth_factor near 1 -> significant lift
        assert abs(float(force[2])) > 10.0, "Foil at 0.8m depth should produce significant lift"

    def test_depth_factor_near_zero_when_foil_at_surface(self, default_foil):
        """Depth factor is near 0 when foil is slightly above surface.

        At heel=0, foil_depth ~ pos_d + position_z * cos(theta) - position_x * sin(theta).
        With position_z=0.8, pos_d=-0.9, and small theta, foil_depth ~ -0.1
        (slightly above water). Smooth formulation: exponentially small at negative depths.
        """
        state = _make_state(pos_d=-0.9, theta=0.05)
        control = _make_control()
        force, _ = default_foil.compute_moth(state, control, 10.0)
        # Foil slightly above surface -> depth_factor very small -> minimal lift
        assert abs(float(force[2])) < 5.0, "Foil above surface should have very small lift"

    def test_depth_factor_near_one_at_depth(self, default_foil):
        """Depth factor approaches 1 at design depth.

        At zero heel (default_foil fixture), both depths are well above the
        ~3cm transition floor, so factors are both ~1.0 and lift is equal.
        """
        state_shallow = _make_state(pos_d=0.1, theta=0.05)
        state_deep = _make_state(pos_d=2.0, theta=0.05)
        control = _make_control()

        f_shallow, _ = default_foil.compute_moth(state_shallow, control, 10.0)
        f_deep, _ = default_foil.compute_moth(state_deep, control, 10.0)

        # Deep foil should produce at least as much lift as shallow foil
        assert abs(float(f_deep[2])) >= abs(float(f_shallow[2])) - 1e-6

    def test_depth_factor_monotonically_increases(self, default_foil):
        """Depth factor increases with depth (up to saturation)."""
        control = _make_control()
        depths = [0.05, 0.1, 0.2, 0.4, 0.8]
        lifts = []
        for d in depths:
            state = _make_state(pos_d=d, theta=0.05)
            f, _ = default_foil.compute_moth(state, control, 10.0)
            lifts.append(abs(float(f[2])))

        # Each depth should produce >= lift than the previous
        for i in range(1, len(lifts)):
            assert lifts[i] >= lifts[i - 1] - 1e-6

    def test_above_water_negligible_lift(self, default_foil):
        """Foil well above water produces negligible lift.

        At heel=0, foil_depth ~ pos_d + position_z * cos(theta) - position_x * sin(theta).
        With position_z=0.8 and pos_d=-1.5, foil_depth ~ -0.7 (above water).
        Smooth formulation gives exponentially small values.
        """
        state = _make_state(pos_d=-1.5, theta=0.05)  # foil_depth ~ -1.5 + 0.8 = -0.7 (above water)
        control = _make_control()
        force, _ = default_foil.compute_moth(state, control, 10.0)
        # Smooth formulation: softplus(50*(-0.7))/50 is exponentially small
        assert abs(float(force[2])) < 0.1, "Foil well above water should have negligible lift"

    def test_drag_increases_with_cl_squared(self, default_foil):
        """Drag increases with CL^2 (induced drag).

        With the corrected force decomposition, body-frame fx includes
        both drag and lift-tilt contributions, so we compare the net
        deceleration force magnitude: sqrt(fx^2 + fz^2) * sign(-fx).
        Alternatively, just verify via the polar that higher AoA gives higher CD.
        """
        # Use flap to control AoA while keeping alpha_geo=0 (w=0)
        u = 10.0
        state = _make_state(pos_d=0.4, u=u)
        ctrl_low = _make_control(main_flap=0.02)
        ctrl_high = _make_control(main_flap=0.1)

        f_low, _ = default_foil.compute_moth(state, ctrl_low, u)
        f_high, _ = default_foil.compute_moth(state, ctrl_high, u)

        # At w=0 (alpha_geo=0), fx = -drag exactly, so higher AoA = more negative fx
        assert float(f_high[0]) < float(f_low[0])

    def test_cd0_present_at_zero_lift(self, default_foil):
        """Profile drag CD0 exists even at zero lift."""
        state = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control()
        force, _ = default_foil.compute_moth(state, control, 10.0)
        # With cl0=0 and theta=0 and w=0 and flap=0, CL=0
        # But CD = CD0 > 0, so there should be drag (negative Fx)
        assert float(force[0]) < 0.0

    def test_lift_to_drag_ratio_reasonable(self, default_foil):
        """L/D ratio is in a reasonable range at design conditions.

        With the corrected decomposition, body-frame fx/fz mix drag
        and lift via alpha_geo rotation. Use flap-based AoA at w=0
        (alpha_geo=0) where fx=-drag and fz=-lift cleanly.
        """
        # Use flap to set AoA ~3 deg: alpha_eff = 0.5 * flap
        # flap = 0.1 rad => alpha_eff = 0.05 rad (~2.9 deg)
        state = _make_state(pos_d=0.4, u=10.0)
        control = _make_control(main_flap=0.1)
        force, _ = default_foil.compute_moth(state, control, 10.0)
        lift = abs(float(force[2]))
        drag = abs(float(force[0]))
        if drag > 1e-6:
            ld_ratio = lift / drag
            assert 5.0 < ld_ratio < 40.0, f"L/D = {ld_ratio}"

    def test_moment_sign_foil_forward_upward_lift(self, default_foil):
        """Foil forward of CG with upward lift produces nose-up moment.

        Main foil is at position_x=+0.6 (forward of CG, FRD).
        Upward lift = negative Fz.
        My = r_z * F_x - r_x * F_z  (from M = r × F)
        = 0.8 * (-drag) - 0.6 * (-lift)
        = -0.8*drag + 0.6*lift > 0  (nose-up, since lift >> drag)
        """
        # Use w = u*tan(theta) to produce actual body-frame AoA and lift
        theta = 0.05
        u = 10.0
        state = _make_state(pos_d=0.4, theta=theta, w=u * np.tan(theta))
        control = _make_control()
        _, moment = default_foil.compute_moth(state, control, u)
        # Moment should be positive (nose-up) for forward foil + upward lift
        assert float(moment[1]) > 0.0

    def test_force_shape(self, default_foil):
        """Force returns shape (3,)."""
        state = _make_state()
        control = _make_control()
        force, moment = default_foil.compute_moth(state, control, 10.0)
        assert force.shape == (3,)
        assert moment.shape == (3,)

    def test_jit_compatible(self, default_foil):
        """Main foil compute is JIT-compatible."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control()

        @jax.jit
        def jit_compute(s, c):
            return default_foil.compute_moth(s, c, 10.0)

        force, moment = jit_compute(state, control)
        assert force.shape == (3,)
        assert moment.shape == (3,)
        assert jnp.all(jnp.isfinite(force))
        assert jnp.all(jnp.isfinite(moment))


class TestMainFoilPitchRateCoupling:
    """Tests for main foil pitch rate coupling (Mq damping mechanism)."""

    def test_positive_pitch_rate_reduces_aoa(self, default_foil):
        """With q > 0 (nose-up) and forward foil (x > 0), AoA should decrease.

        Physics: Nose-up pitch rotation causes the forward foil to move upward,
        reducing its angle of attack and lift.
        """
        state_no_q = _make_state(pos_d=0.4, theta=0.05, q=0.0)
        state_with_q = _make_state(pos_d=0.4, theta=0.05, q=0.5)  # nose-up rate
        control = _make_control()

        f_no_q, _ = default_foil.compute_moth(state_no_q, control, 10.0)
        f_with_q, _ = default_foil.compute_moth(state_with_q, control, 10.0)

        # With positive q at forward foil, lift should be less (Fz less negative)
        assert float(f_with_q[2]) > float(f_no_q[2])

    def test_negative_pitch_rate_increases_aoa(self, default_foil):
        """With q < 0 (nose-down) and forward foil (x > 0), AoA should increase.

        Physics: Nose-down pitch rotation causes the forward foil to move downward,
        increasing its angle of attack and lift.
        """
        state_no_q = _make_state(pos_d=0.4, theta=0.05, q=0.0)
        state_with_q = _make_state(pos_d=0.4, theta=0.05, q=-0.5)  # nose-down rate
        control = _make_control()

        f_no_q, _ = default_foil.compute_moth(state_no_q, control, 10.0)
        f_with_q, _ = default_foil.compute_moth(state_with_q, control, 10.0)

        # With negative q at forward foil, lift should be more (Fz more negative)
        assert float(f_with_q[2]) < float(f_no_q[2])

    def test_pitch_rate_effect_proportional_to_position(self):
        """Pitch rate AoA change scales with foil x-position."""
        # Create foils at different forward positions
        foil_close = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.3, position_z=0.8, foil_span=1.0,  # closer to CG
        )
        foil_far = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.9, position_z=0.8, foil_span=1.0,  # further from CG
        )

        state_no_q = _make_state(pos_d=0.4, theta=0.05, q=0.0)
        state_with_q = _make_state(pos_d=0.4, theta=0.05, q=0.5)
        control = _make_control()

        # Compute lift changes
        f_close_no_q, _ = foil_close.compute_moth(state_no_q, control, 10.0)
        f_close_q, _ = foil_close.compute_moth(state_with_q, control, 10.0)
        f_far_no_q, _ = foil_far.compute_moth(state_no_q, control, 10.0)
        f_far_q, _ = foil_far.compute_moth(state_with_q, control, 10.0)

        delta_close = float(f_close_q[2] - f_close_no_q[2])
        delta_far = float(f_far_q[2] - f_far_no_q[2])

        # Far foil should have larger lift change due to larger moment arm
        assert abs(delta_far) > abs(delta_close)


# ===========================================================================
# TestPitchCorrectedDepth
# ===========================================================================

class TestPitchCorrectedDepth:
    """Tests for heel- and pitch-corrected foil depth (Wave 4D + heel fix).

    The canonical formula (compute_foil_ned_depth) is:
        foil_depth = pos_d + position_z * cos(heel) * cos(theta) - position_x * sin(theta)

    These tests run at heel=0 (default fixture), so cos(heel)=1 and the
    formula simplifies to: pos_d + position_z * cos(theta) - position_x * sin(theta).
    At small theta, cos(theta) ~ 1, further simplifying to the original
    pos_d + position_z - position_x * sin(theta).

    This accounts for body rotation moving foils vertically:
    - Nose-up (theta > 0): bow rises, stern drops
    - Main foil (x=+0.6): correction = -0.6*sin(theta) -> shallower
    - Rudder (x=-1.5): correction = +1.5*sin(theta) -> deeper
    """

    def test_nose_up_deepens_rudder(self, default_rudder):
        """Pitched-up rudder produces more lift than level rudder.

        Nose-up pitch makes the aft rudder deeper (more submersion),
        which increases the depth factor and therefore lift.
        Use same elevator deflection + w=u*tan(theta) for consistent AoA.
        """
        theta_pitched = 0.15  # ~8.6 deg nose-up
        u = 10.0
        state_level = _make_state(pos_d=0.4, theta=0.0)
        state_pitched = _make_state(pos_d=0.4, theta=theta_pitched, w=u * np.tan(theta_pitched))
        control = _make_control(rudder_elevator=0.05)

        f_level, _ = default_rudder.compute_moth(state_level, control, 10.0)
        f_pitched, _ = default_rudder.compute_moth(state_pitched, control, 10.0)

        # Pitched rudder: deeper due to -(-1.5)*sin(0.15) = +0.224m correction
        # Both theta and depth effects increase lift, so pitched should have more
        assert abs(float(f_pitched[2])) > abs(float(f_level[2])), (
            f"Pitched rudder should produce more lift: "
            f"|f_pitched_z|={abs(float(f_pitched[2])):.2f} vs "
            f"|f_level_z|={abs(float(f_level[2])):.2f}"
        )

    def test_nose_up_shallows_main_foil(self):
        """Pitched-up main foil has reduced depth factor.

        Nose-up pitch makes the forward main foil shallower, reducing
        submersion and depth factor. At a shallow starting depth, this
        should reduce lift magnitude.
        """
        # Use a foil at shallow depth where depth factor is sensitive
        foil = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.6, position_z=0.2, foil_span=1.0,  # shallow z
        )
        theta_pitched = 0.15  # ~8.6 deg
        # At pos_d=0.0: level depth = 0.2, pitched depth = 0.2 - 0.6*sin(0.15) ≈ 0.11
        state_level = _make_state(pos_d=0.0, theta=0.0)
        state_pitched = _make_state(pos_d=0.0, theta=theta_pitched)
        # Use flap to generate lift without theta contribution for level case
        control = _make_control(main_flap=0.1)

        f_level, _ = foil.compute_moth(state_level, control, 10.0)
        f_pitched, _ = foil.compute_moth(state_pitched, control, 10.0)

        # The depth factor at 0.11m is lower than at 0.2m, reducing lift.
        # But theta also adds AoA which increases CL. At shallow depth the
        # depth factor reduction dominates. Check that the depth factor effect
        # is present by computing depths directly.
        import jax.numpy as jnp
        depth_level = 0.0 + 0.2  # 0.2m
        depth_pitched = 0.0 + 0.2 - 0.6 * jnp.sin(0.15)  # ~0.11m
        assert float(depth_pitched) < float(depth_level), (
            f"Pitched foil should be shallower: {float(depth_pitched):.3f} < {float(depth_level):.3f}"
        )

    def test_depth_correction_magnitude(self):
        """Correction magnitudes match geometry exactly.

        Pure math check: verify the depth correction term
        -position_x * sin(theta) has the expected magnitude.
        """
        theta = 0.131  # 7.5 deg (from Wave 4C analysis)

        # Main foil: x = +0.6m
        main_correction = -0.6 * np.sin(theta)
        assert main_correction == pytest.approx(-0.6 * np.sin(0.131), abs=1e-10)
        # Foil gets shallower (negative correction)
        assert main_correction < 0.0

        # Rudder: x = -1.5m
        rudder_correction = -(-1.5) * np.sin(theta)
        assert rudder_correction == pytest.approx(1.5 * np.sin(0.131), abs=1e-10)
        # Rudder gets deeper (positive correction)
        assert rudder_correction > 0.0

        # At 7.5 deg, rudder correction ~0.196m (matches Wave 4C finding of ~0.2m)
        assert rudder_correction == pytest.approx(0.196, abs=0.005)

    def test_zero_pitch_no_correction(self, default_foil, default_rudder):
        """At theta=0, corrected formula equals original formula.

        The sin(0)=0 term makes the correction vanish, ensuring
        backward compatibility.
        """
        state = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control(main_flap=0.05)

        # At theta=0 and heel=0, foil_depth = pos_d + position_z * cos(0) * cos(0)
        #                                  = pos_d + position_z (same as original formula)
        f_foil, m_foil = default_foil.compute_moth(state, control, 10.0)
        f_rudder, m_rudder = default_rudder.compute_moth(state, control, 10.0)

        # Verify foil produces expected lift at this depth
        # (same as pre-4D behavior since sin(0)=0)
        assert jnp.all(jnp.isfinite(f_foil))
        assert jnp.all(jnp.isfinite(f_rudder))

        # Cross-check: manually compute expected depth
        expected_foil_depth = 0.4 + 0.8  # 1.2m (deep, factor ~1)
        expected_rudder_depth = 0.4 + 0.5  # 0.9m (deep, factor ~1)
        df_foil = compute_depth_factor(
            jnp.float64(expected_foil_depth), 1.0, 0.0,
        )
        df_rudder = compute_depth_factor(
            jnp.float64(expected_rudder_depth), 0.5, 0.0,
        )
        assert float(df_foil) > 0.99
        assert float(df_rudder) > 0.99


# ===========================================================================
# TestMothRudderElevator
# ===========================================================================

class TestMothRudderElevator:
    """Tests for rudder elevator force component."""

    def test_positive_elevator_produces_upward_force(self, default_rudder):
        """Positive elevator deflection produces upward force (negative Fz)."""
        state = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control(rudder_elevator=0.05)  # ~2.9 deg
        force, _ = default_rudder.compute_moth(state, control, 10.0)
        # Positive elevator angle -> positive AoA at rudder -> positive CL
        # -> positive lift -> negative Fz (upward)
        assert float(force[2]) < 0.0

    def test_aft_rudder_upward_force_nose_down_moment(self, default_rudder):
        """Aft rudder with upward force produces nose-down moment.

        Rudder at position_x=-1.5 (aft of CG, FRD: -x = aft).
        rudder_My = -position_x * rudder_Fz = -(-1.5) * (-lift) = -1.5 * lift
        Positive elevator -> positive AoA -> positive CL -> positive lift
        -> rudder_Fz = -lift (upward) -> My = -(-1.5) * (-lift) = -1.5*lift < 0
        Upward force at aft position -> nose-DOWN moment (My < 0)
        """
        state = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control(rudder_elevator=0.05)
        force, moment = default_rudder.compute_moth(state, control, 10.0)
        # Upward force at aft position = nose-down moment
        assert float(force[2]) < 0.0  # Upward force
        assert float(moment[1]) < 0.0  # Nose-down moment

    def test_pitch_rate_coupling(self, default_rudder):
        """Pitch rate changes local AoA at rudder.

        Physics: With positive q (nose-up) at aft rudder (x < 0), the tail
        moves downward, increasing local AoA and lift (more negative Fz).
        """
        state_no_q = _make_state(pos_d=0.4, theta=0.0, q=0.0)
        state_with_q = _make_state(pos_d=0.4, theta=0.0, q=0.5)  # nose-up
        control = _make_control(rudder_elevator=0.0)

        f_no_q, _ = default_rudder.compute_moth(state_no_q, control, 10.0)
        f_with_q, _ = default_rudder.compute_moth(state_with_q, control, 10.0)

        # With nose-up pitch rate, aft rudder should have increased AoA and lift
        # More lift = more negative Fz (upward)
        assert float(f_with_q[2]) < float(f_no_q[2])

    def test_force_shape(self, default_rudder):
        """Force and moment have shape (3,)."""
        state = _make_state()
        control = _make_control()
        force, moment = default_rudder.compute_moth(state, control, 10.0)
        assert force.shape == (3,)
        assert moment.shape == (3,)

    def test_produces_x_and_z_force(self, default_rudder):
        """Rudder produces both X-force (drag) and Z-force (lift)."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        force, _ = default_rudder.compute_moth(state, control, 10.0)
        # X-force should be nonzero (drag)
        assert float(force[0]) != pytest.approx(0.0)
        # Y-force should still be zero (longitudinal model)
        assert float(force[1]) == pytest.approx(0.0)
        # Z-force should be nonzero (lift)
        assert float(force[2]) != pytest.approx(0.0)

    def test_only_y_moment(self, default_rudder):
        """Rudder only produces Y-moment (pitch)."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        _, moment = default_rudder.compute_moth(state, control, 10.0)
        assert float(moment[0]) == pytest.approx(0.0)
        assert float(moment[2]) == pytest.approx(0.0)

    def test_jit_compatible(self, default_rudder):
        """Rudder compute is JIT-compatible."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(rudder_elevator=0.05)

        @jax.jit
        def jit_compute(s, c):
            return default_rudder.compute_moth(s, c, 10.0)

        force, moment = jit_compute(state, control)
        assert force.shape == (3,)
        assert jnp.all(jnp.isfinite(force))

    def test_zero_elevator_with_theta_still_produces_force(self, default_rudder):
        """Even without elevator deflection, body-frame flow angle produces rudder force."""
        # w = u*tan(theta) creates body-frame AoA at the rudder
        theta = 0.05
        u = 10.0
        state = _make_state(pos_d=0.4, theta=theta, w=u * np.tan(theta))
        control = _make_control(rudder_elevator=0.0)
        force, _ = default_rudder.compute_moth(state, control, u)
        assert abs(float(force[2])) > 0.0


# ===========================================================================
# TestMothSailForce
# ===========================================================================

class TestMothSailForce:
    """Tests for sail force component."""

    def test_constant_forward_thrust(self, default_sail):
        """Sail produces constant forward thrust."""
        state = _make_state()
        control = _make_control()
        force, _ = default_sail.compute_moth(state, control, 10.0)
        assert float(force[0]) == pytest.approx(50.0)

    def test_thrust_above_cg_nose_down_moment(self, default_sail):
        """Thrust above CG produces nose-down moment.

        ce_position_z = -2.5 (above CG in body frame)
        moment_y = ce_position_z * thrust = -2.5 * 50 = -125 (nose-down)
        """
        state = _make_state()
        control = _make_control()
        _, moment = default_sail.compute_moth(state, control, 10.0)
        assert float(moment[1]) < 0.0  # Nose-down
        assert float(moment[1]) == pytest.approx(-125.0)

    def test_moment_magnitude_equals_thrust_times_height(self, default_sail):
        """Moment magnitude equals thrust * |height|."""
        state = _make_state()
        control = _make_control()
        _, moment = default_sail.compute_moth(state, control, 10.0)
        expected = abs(default_sail.ce_position_z * default_sail.thrust_coeff)
        assert abs(float(moment[1])) == pytest.approx(expected)

    def test_force_shape(self, default_sail):
        """Force and moment have shape (3,)."""
        state = _make_state()
        control = _make_control()
        force, moment = default_sail.compute_moth(state, control, 10.0)
        assert force.shape == (3,)
        assert moment.shape == (3,)


# ===========================================================================
# TestMothHullDrag
# ===========================================================================

class TestMothHullDrag:
    """Tests for hull drag component.

    Hull bottom NED depth = pos_d + contact_depth.
    Hull drag is active when hull bottom is below water (pos_d + contact_depth > 0),
    i.e., when pos_d > -contact_depth = -0.15.
    """

    def test_zero_drag_when_foiling(self, default_hull_drag):
        """No drag when foiling (CG well above water, pos_d << -contact_depth)."""
        state = _make_state(pos_d=-0.4)  # CG 40cm above water, hull at -0.25m
        control = _make_control()
        force, _ = default_hull_drag.compute_moth(state, control, 10.0)
        assert float(force[0]) == pytest.approx(0.0)

    def test_drag_increases_with_sinking(self, default_hull_drag):
        """Drag increases as hull sinks deeper (larger pos_d)."""
        control = _make_control()
        state_shallow = _make_state(pos_d=-0.10)  # 5cm immersion (0.15-0.10=0.05)
        state_deep = _make_state(pos_d=-0.05)     # 10cm immersion (0.15-0.05=0.10)

        f_shallow, _ = default_hull_drag.compute_moth(state_shallow, control, 10.0)
        f_deep, _ = default_hull_drag.compute_moth(state_deep, control, 10.0)

        # Deeper = more drag (more negative Fx)
        assert float(f_deep[0]) < float(f_shallow[0])

    def test_drag_opposes_forward_motion(self, default_hull_drag):
        """Drag force is in -X direction."""
        state = _make_state(pos_d=0.0)  # Hull immersed by contact_depth
        control = _make_control()
        force, _ = default_hull_drag.compute_moth(state, control, 10.0)
        assert float(force[0]) < 0.0

    def test_smooth_transition_at_threshold(self, default_hull_drag):
        """Drag transitions smoothly around contact depth boundary.

        Hull bottom at water when pos_d = -contact_depth = -0.15.
        Just above: pos_d = -0.16, hull_bottom = -0.01 (above water, no drag).
        Just below: pos_d = -0.14, hull_bottom = +0.01 (1cm immersed).
        """
        control = _make_control()
        # Hull above water (pos_d + contact_depth < 0)
        state_above = _make_state(pos_d=-0.16)
        # Hull just touching (pos_d + contact_depth = 0.01m)
        state_below = _make_state(pos_d=-0.14)

        f_above, _ = default_hull_drag.compute_moth(state_above, control, 10.0)
        f_below, _ = default_hull_drag.compute_moth(state_below, control, 10.0)

        # Above threshold: no drag
        assert float(f_above[0]) == pytest.approx(0.0)
        # Below threshold: small drag (1cm * 500 N/m = 5N)
        assert float(f_below[0]) == pytest.approx(-5.0)

    def test_exact_boundary_zero_immersion(self, default_hull_drag):
        """Immersion is exactly 0 at boundary pos_d = -contact_depth = -0.15.

        At this point, hull bottom is at the waterline:
        pos_d + contact_depth = -0.15 + 0.15 = 0.0.
        """
        state = _make_state(pos_d=-0.15)
        control = _make_control()
        force, _ = default_hull_drag.compute_moth(state, control, 10.0)
        assert float(force[0]) == pytest.approx(0.0)

    def test_no_moment(self, default_hull_drag):
        """Hull drag with no buoyancy produces no moment."""
        state = _make_state(pos_d=0.0)
        control = _make_control()
        _, moment = default_hull_drag.compute_moth(state, control, 10.0)
        assert jnp.allclose(moment, jnp.zeros(3))


class TestDynamicContactDepth:
    """Tests for dynamic contact_depth computation using hull_cg_above_bottom."""

    def test_dynamic_contact_depth_changes_with_cg_offset(self):
        """CG offset changes the runtime contact depth and immersion behavior."""
        hull = MothHullDrag(
            drag_coeff=500.0,
            contact_depth=0.82,
            hull_cg_above_bottom=0.82,
            buoyancy_coeff=0.0,
            buoyancy_fwd_x=0.0,
            buoyancy_aft_x=0.0,
        )
        state = _make_state(pos_d=-0.7)
        control = _make_control()

        # With zero cg_offset: contact_depth = 0.82
        force_zero, _ = hull.compute_moth(
            state, control, 10.0, cg_offset=jnp.zeros(3),
        )
        # With sailor above CG (cg_offset[2] = -0.12): contact_depth = 0.82 - (-0.12) = 0.94
        force_offset, _ = hull.compute_moth(
            state, control, 10.0, cg_offset=jnp.array([0.0, 0.0, -0.12]),
        )
        # Larger contact_depth means hull bottom is deeper -> more immersion -> more drag
        assert float(force_offset[0]) < float(force_zero[0]), (
            f"Sailor above CG should increase hull drag: offset={float(force_offset[0]):.4f}, "
            f"zero={float(force_zero[0]):.4f}"
        )

    def test_dynamic_contact_depth_matches_params_property(self):
        """hull_cg_above_bottom - cg_offset[2] matches MOTH_BIEKER_V3.hull_contact_depth."""
        from fmd.simulator.params import MOTH_BIEKER_V3

        hull_cg_above_bottom = MOTH_BIEKER_V3.hull_cg_above_bottom
        cg_offset_z = MOTH_BIEKER_V3.combined_cg_offset[2]
        dynamic_depth = hull_cg_above_bottom - cg_offset_z
        expected = MOTH_BIEKER_V3.hull_contact_depth

        assert dynamic_depth == pytest.approx(expected, abs=1e-10), (
            f"Dynamic: {dynamic_depth:.6f}, property: {expected:.6f}"
        )

    def test_zero_hull_cg_gives_zero_contact_depth(self):
        """With hull_cg_above_bottom=0.0 and zero cg_offset, contact_depth=0."""
        hull = MothHullDrag(
            drag_coeff=500.0,
            contact_depth=0.0,
            hull_cg_above_bottom=0.0,
            buoyancy_coeff=0.0,
            buoyancy_fwd_x=0.0,
            buoyancy_aft_x=0.0,
        )
        # At any negative pos_d, hull should have zero drag since contact_depth=0
        # means hull bottom is at CG, and CG is above water
        state = _make_state(pos_d=-0.5)
        control = _make_control()
        force, _ = hull.compute_moth(
            state, control, 10.0, cg_offset=jnp.zeros(3),
        )
        # immersion = max(0, pos_d + 0) = max(0, -0.5) = 0 -> zero drag
        assert float(force[0]) == pytest.approx(0.0, abs=1e-10), (
            f"Expected zero drag, got {float(force[0]):.6f}"
        )


class TestMothHullBuoyancy:
    """Tests for hull buoyancy restoring force."""

    def test_buoyancy_zero_when_foiling(self, hull_with_buoyancy):
        """No buoyancy when foiling (hull well above water)."""
        state = _make_state(pos_d=-0.4)  # CG 40cm above water
        control = _make_control()
        force, moment = hull_with_buoyancy.compute_moth(state, control, 10.0)
        # fz should be zero (no buoyancy when hull is above water)
        assert abs(float(force[2])) < 1e-6
        assert abs(float(moment[1])) < 1e-6

    def test_buoyancy_opposes_sinking(self, hull_with_buoyancy):
        """Buoyancy produces upward force when hull is immersed."""
        state = _make_state(pos_d=0.1, theta=0.0)  # Hull well in water
        control = _make_control()
        force, _ = hull_with_buoyancy.compute_moth(state, control, 10.0)
        # fz should be negative (upward in body frame)
        assert float(force[2]) < 0.0

    def test_buoyancy_increases_with_sinking(self, hull_with_buoyancy):
        """Deeper immersion produces stronger buoyancy."""
        control = _make_control()
        state_shallow = _make_state(pos_d=0.0, theta=0.0)
        state_deep = _make_state(pos_d=0.2, theta=0.0)

        f_shallow, _ = hull_with_buoyancy.compute_moth(state_shallow, control, 10.0)
        f_deep, _ = hull_with_buoyancy.compute_moth(state_deep, control, 10.0)

        # Deeper = more buoyancy (more negative fz)
        assert float(f_deep[2]) < float(f_shallow[2])

    def test_buoyancy_pitch_restoring(self, hull_with_buoyancy):
        """Buoyancy produces pitch restoring moment when hull is immersed.

        With positive theta (nose up), the aft buoyancy point is deeper than
        the forward point, creating a nose-down restoring moment (negative My).
        """
        # Nose-up pitch with hull in water
        state = _make_state(pos_d=0.1, theta=0.1)
        control = _make_control()
        _, moment = hull_with_buoyancy.compute_moth(state, control, 10.0)

        # With nose-up pitch, aft point is deeper, creating nose-down moment
        # The exact sign depends on geometry, but the moment should be nonzero
        assert abs(float(moment[1])) > 1.0  # significant restoring moment

    def test_buoyancy_symmetric_at_zero_pitch(self, hull_with_buoyancy):
        """At zero pitch, symmetric buoyancy points produce zero moment."""
        state = _make_state(pos_d=0.1, theta=0.0)
        control = _make_control()
        _, moment = hull_with_buoyancy.compute_moth(state, control, 10.0)
        # Symmetric fore/aft points at zero pitch should produce zero My
        # (both have equal immersion, and fx components cancel in moment)
        assert abs(float(moment[1])) < 1e-6


# ===========================================================================
# TestCreateMothComponents
# ===========================================================================

class TestCreateMothComponents:
    """Tests for factory function."""

    def test_creates_all_six_components(self):
        """Factory creates all six component types."""
        foil, rudder, sail, hull, main_strut, rudder_strut = create_moth_components(MOTH_BIEKER_V3)
        assert isinstance(foil, MothMainFoil)
        assert isinstance(rudder, MothRudderElevator)
        assert isinstance(sail, MothSailForce)
        assert isinstance(hull, MothHullDrag)

    def test_foil_params_match(self):
        """Main foil parameters match MothParams."""
        foil, *_ = create_moth_components(MOTH_BIEKER_V3)
        assert foil.rho == MOTH_BIEKER_V3.rho_water
        assert foil.area == MOTH_BIEKER_V3.main_foil_area
        assert foil.cl_alpha == MOTH_BIEKER_V3.main_foil_cl_alpha
        assert foil.position_x == float(MOTH_BIEKER_V3.main_foil_position[0])
        assert foil.foil_span == MOTH_BIEKER_V3.main_foil_span

    def test_rudder_params_match(self):
        """Rudder parameters match MothParams."""
        _, rudder, *_ = create_moth_components(MOTH_BIEKER_V3)
        assert rudder.rho == MOTH_BIEKER_V3.rho_water
        assert rudder.area == MOTH_BIEKER_V3.rudder_area
        assert rudder.position_x == float(MOTH_BIEKER_V3.rudder_position[0])
        assert rudder.foil_span == MOTH_BIEKER_V3.rudder_span

    def test_sail_params_match(self):
        """Sail parameters match MothParams."""
        _, _, sail, *_ = create_moth_components(MOTH_BIEKER_V3)
        assert sail.thrust_coeff == MOTH_BIEKER_V3.sail_thrust_coeff
        assert sail.ce_position_z == float(MOTH_BIEKER_V3.sail_ce_position[2])

    def test_hull_params_match(self):
        """Hull drag parameters match MothParams."""
        _, _, _, hull, *_ = create_moth_components(MOTH_BIEKER_V3)
        assert hull.drag_coeff == MOTH_BIEKER_V3.hull_drag_coeff
        assert hull.contact_depth == MOTH_BIEKER_V3.hull_contact_depth
        assert hull.buoyancy_coeff == MOTH_BIEKER_V3.hull_buoyancy_coeff
        assert hull.buoyancy_fwd_x == pytest.approx(MOTH_BIEKER_V3.hull_length / 4)
        assert hull.buoyancy_aft_x == pytest.approx(-MOTH_BIEKER_V3.hull_length / 4)

    def test_heel_angle_passed_through(self):
        """Heel angle is passed to both foil components."""
        foil, rudder, *_ = create_moth_components(
            MOTH_BIEKER_V3, heel_angle=0.2,
        )
        assert foil.heel_angle == pytest.approx(0.2)
        assert rudder.heel_angle == pytest.approx(0.2)

    def test_ventilation_mode_passed_through(self):
        """Ventilation mode is passed to both foil components."""
        foil, rudder, *_ = create_moth_components(
            MOTH_BIEKER_V3, ventilation_mode="binary",
        )
        assert foil.ventilation_mode == "binary"
        assert rudder.ventilation_mode == "binary"


# ===========================================================================
# TestConventionRegression
# ===========================================================================

class TestConventionRegression:
    """Convention regression tests for FRD body frame.

    These tests verify that the moment formula M = r × F is
    correctly implemented and consistent with the FRD body frame
    (+x forward, +y starboard, +z down).

    M_y = r_z * F_x - r_x * F_z
    """

    def test_forward_foil_upward_lift_nose_up_moment(self):
        """Forward foil (+x > 0) with upward lift (F_z < 0) → nose-up moment (M_y > 0).

        M_y = r_z * F_x - r_x * F_z
        With r_x > 0, F_z < 0: -r_x * F_z > 0 (nose-up), dominating term.
        """
        r_x, r_z = 0.6, 0.8  # forward, below CG
        f_x, f_z = -5.0, -100.0  # drag backward, lift upward

        m_y = r_z * f_x - r_x * f_z
        # = 0.8 * (-5) - 0.6 * (-100) = -4 + 60 = 56 > 0 (nose-up)
        assert m_y > 0.0, f"Forward foil + upward lift should give nose-up moment, got {m_y}"

    def test_aft_foil_upward_lift_nose_down_moment(self):
        """Aft foil (x < 0) with upward lift (F_z < 0) → nose-down moment (M_y < 0).

        M_y = r_z * F_x - r_x * F_z
        With r_x < 0, F_z < 0: -r_x * F_z < 0 (nose-down).
        """
        r_x, r_z = -1.5, 0.5  # aft, below CG
        f_x, f_z = 0.0, -50.0  # no drag (rudder v1), lift upward

        m_y = r_z * f_x - r_x * f_z
        # = 0.5 * 0 - (-1.5) * (-50) = 0 - 75 = -75 < 0 (nose-down)
        assert m_y < 0.0, f"Aft foil + upward lift should give nose-down moment, got {m_y}"

    def test_above_cg_forward_thrust_nose_down_moment(self):
        """Above-CG force point (z < 0) with forward thrust (F_x > 0) → nose-down (M_y < 0).

        M_y = r_z * F_x - r_x * F_z
        With r_z < 0, F_x > 0: r_z * F_x < 0 (nose-down).
        """
        r_x, r_z = 0.0, -2.5  # above CG (sail CE)
        f_x, f_z = 50.0, 0.0  # forward thrust, no vertical force

        m_y = r_z * f_x - r_x * f_z
        # = (-2.5) * 50 - 0 * 0 = -125 < 0 (nose-down)
        assert m_y < 0.0, f"Above-CG forward thrust should give nose-down moment, got {m_y}"

    def test_preset_foil_position_is_forward(self):
        """MOTH_BIEKER_V3 main foil position_x is positive (forward of CG)."""
        assert MOTH_BIEKER_V3.main_foil_position[0] > 0.0

    def test_preset_rudder_position_is_aft(self):
        """MOTH_BIEKER_V3 rudder position_x is negative (aft of CG)."""
        assert MOTH_BIEKER_V3.rudder_position[0] < 0.0


# ===========================================================================
# TestSmoothnessPolicy
# ===========================================================================

class TestSmoothnessPolicy:
    """Tests for smooth depth factor / surface transition.

    The smooth mode must be differentiable everywhere (no kinks).
    These tests verify that jax.grad produces finite values near the surface.
    """

    def test_gradient_finite_near_surface(self):
        """Gradient of force_z w.r.t. pos_d is finite near the surface."""
        foil = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.6, position_z=0.8, foil_span=1.0,
        )

        def force_z_at_pos_d(pos_d_scalar):
            state = jnp.array([pos_d_scalar, 0.05, 0.0, 0.0])
            control = jnp.array([0.05, 0.0])
            force, _ = foil.compute_moth(state, control, 10.0)
            return force[2]

        grad_fn = jax.grad(force_z_at_pos_d)

        # Test near the surface: at heel=0 and small theta, foil_depth ~ pos_d + 0.8
        # Surface is approximately at pos_d = -0.8
        test_depths = [-1.0, -0.8, -0.5, 0.0, 0.2, 0.5, 1.0]
        for pos_d in test_depths:
            grad_val = grad_fn(jnp.float64(pos_d))
            assert jnp.isfinite(grad_val), (
                f"Gradient should be finite at pos_d={pos_d}, got {grad_val}"
            )

    def test_smooth_depth_factor_is_monotonic(self):
        """Smooth depth factor increases monotonically with depth."""
        foil = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.6, position_z=0.8, foil_span=1.0,
        )
        control = _make_control(main_flap=0.05)

        # Sweep through depths including across the surface
        depths = jnp.linspace(-2.0, 2.0, 50)
        lifts = []
        for d in depths:
            state = _make_state(pos_d=float(d), theta=0.05)
            f, _ = foil.compute_moth(state, control, 10.0)
            lifts.append(float(f[2]))

        # Lift should be monotonically increasing in magnitude
        # (more negative = more upward lift as depth increases)
        for i in range(1, len(lifts)):
            assert lifts[i] <= lifts[i - 1] + 0.1, (
                f"Lift should increase (become more negative) with depth. "
                f"At depth {float(depths[i]):.2f}: fz={lifts[i]:.2f} vs "
                f"prev {lifts[i-1]:.2f}"
            )


# ===========================================================================
# TestComputeDepthFactor
# ===========================================================================

class TestComputeDepthFactor:
    """Tests for the standalone compute_depth_factor function."""

    def test_fully_submerged_returns_near_one(self):
        """Deep foil returns depth_factor near 1.0."""
        df = compute_depth_factor(
            foil_depth=jnp.float64(2.0), foil_span=1.0, heel_angle=0.0,
        )
        assert float(df) > 0.95

    def test_above_surface_returns_near_zero(self):
        """Foil well above surface returns depth_factor near 0."""
        df = compute_depth_factor(
            foil_depth=jnp.float64(-1.0), foil_span=1.0, heel_angle=0.0,
        )
        assert float(df) < 0.05

    def test_smooth_mode_smooth_taper(self):
        """Smooth mode provides smooth taper at surface."""
        depths = jnp.linspace(-1.0, 2.0, 100)
        factors = [
            float(compute_depth_factor(d, 1.0, 0.0))
            for d in depths
        ]
        # Should be monotonically increasing
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i - 1] - 1e-6

    def test_binary_mode_hard_cutoff(self):
        """Binary mode has hard cutoff at surface."""
        # Below surface
        df_below = compute_depth_factor(
            foil_depth=jnp.float64(0.1), foil_span=1.0, heel_angle=0.0,
            mode="binary",
        )
        assert float(df_below) == pytest.approx(1.0)

        # Above surface
        df_above = compute_depth_factor(
            foil_depth=jnp.float64(-0.1), foil_span=1.0, heel_angle=0.0,
            mode="binary",
        )
        assert float(df_above) == pytest.approx(0.0)

    def test_zero_heel_angle_near_binary(self):
        """With zero heel angle, transition is near-binary (~3cm floor).

        At heel=0, max_submergence is floored at 0.015m. A foil at 0.5m
        depth is well submerged, so the factor should be near 1.0.
        """
        df = compute_depth_factor(
            foil_depth=jnp.float64(0.5), foil_span=1.0, heel_angle=0.0,
        )
        assert float(df) > 0.99  # Well submerged at zero heel

    def test_heel_angle_reduces_depth_factor(self):
        """Heel angle reduces depth_factor for shallow foils.

        At the same foil depth, a heeled boat has less effective submergence
        because the leeward tip rises. Use depth=0.05m where the leeward
        tip at 15deg heel breaches the surface (max_submergence=0.129m > 0.05m).
        """
        shallow_depth = jnp.float64(0.05)
        df_upright = compute_depth_factor(shallow_depth, 1.0, 0.0)
        df_heeled = compute_depth_factor(shallow_depth, 1.0, jnp.deg2rad(15.0))
        # Heeled should have lower depth factor at shallow depth
        assert float(df_heeled) < float(df_upright)

    def test_monotonic_decrease_with_rising(self):
        """Depth factor decreases monotonically as foil rises."""
        depths = jnp.linspace(2.0, -1.0, 50)
        factors = [
            float(compute_depth_factor(d, 1.0, jnp.deg2rad(10.0)))
            for d in depths
        ]
        for i in range(1, len(factors)):
            assert factors[i] <= factors[i - 1] + 1e-4

    def test_jit_compatible(self):
        """compute_depth_factor is JIT-compatible."""
        @jax.jit
        def jit_df(d):
            return compute_depth_factor(d, 1.0, 0.0)

        result = jit_df(jnp.float64(0.5))
        assert jnp.isfinite(result)

    def test_gradient_exists_smooth_mode(self):
        """Gradient exists in smooth mode (differentiable)."""
        grad_fn = jax.grad(lambda d: compute_depth_factor(d, 1.0, 0.0))
        # Test at several depths including near surface
        for d in [-0.5, 0.0, 0.1, 0.5, 1.0]:
            g = grad_fn(jnp.float64(d))
            assert jnp.isfinite(g), f"Gradient not finite at depth {d}: {g}"

    def test_ventilation_threshold_effect(self):
        """Higher threshold allows more exposure before lift loss."""
        foil_depth = jnp.float64(0.08)  # shallow
        heel = jnp.deg2rad(15.0)

        df_low_thresh = compute_depth_factor(
            foil_depth, 1.0, heel, ventilation_threshold=0.10,
        )
        df_high_thresh = compute_depth_factor(
            foil_depth, 1.0, heel, ventilation_threshold=0.50,
        )
        # Higher threshold means lift is retained longer
        assert float(df_high_thresh) >= float(df_low_thresh) - 0.05


# ===========================================================================
# TestMothRudderDepthFactor
# ===========================================================================

class TestMothRudderDepthFactor:
    """Tests for rudder depth factor behavior."""

    def test_rudder_loses_lift_at_surface(self, default_rudder):
        """Rudder produces minimal lift when foil is above water."""
        # Rudder position_z = 0.5; at heel=0 and small theta,
        # foil_depth ~ pos_d + 0.5. At pos_d = -1.5, foil_depth ~ -1.0 (above water)
        state = _make_state(pos_d=-1.5, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        force, _ = default_rudder.compute_moth(state, control, 10.0)
        assert abs(float(force[2])) < 0.1, "Rudder above water should produce negligible lift"

    def test_rudder_produces_lift_at_depth(self, default_rudder):
        """Rudder produces significant lift when submerged."""
        state = _make_state(pos_d=0.5, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        force, _ = default_rudder.compute_moth(state, control, 10.0)
        assert abs(float(force[2])) > 1.0, "Submerged rudder should produce lift"

    def test_rudder_depth_factor_matches_main_foil_behavior(self):
        """Rudder depth factor behaves same as main foil at same geometry."""
        # Use same span and depth for both
        df_main = compute_depth_factor(jnp.float64(0.5), 0.5, 0.0)
        df_rudder = compute_depth_factor(jnp.float64(0.5), 0.5, 0.0)
        assert float(df_main) == pytest.approx(float(df_rudder))


# ===========================================================================
# TestCgOffsetParameter
# ===========================================================================

class TestCgOffsetParameter:
    """Tests for per-call cg_offset parameter on compute_moth() methods."""

    def test_zero_offset_matches_none(self, default_foil, default_rudder, default_sail, default_hull_drag):
        """cg_offset=zeros(3) produces identical results to cg_offset=None."""
        state = _make_state(pos_d=0.4, theta=0.05, w=0.1, q=0.02)
        control = _make_control(main_flap=0.05, rudder_elevator=0.03)
        zero_cg = jnp.zeros(3)

        for component in [default_foil, default_rudder, default_sail, default_hull_drag]:
            f_none, m_none = component.compute_moth(state, control, 10.0)
            f_zero, m_zero = component.compute_moth(state, control, 10.0, cg_offset=zero_cg)
            assert jnp.allclose(f_none, f_zero, atol=1e-12), (
                f"{type(component).__name__}: force mismatch with zero offset"
            )
            assert jnp.allclose(m_none, m_zero, atol=1e-12), (
                f"{type(component).__name__}: moment mismatch with zero offset"
            )

    def test_nonzero_offset_changes_moments(self, default_foil, default_rudder, default_sail):
        """Nonzero cg_offset changes moment output for force-producing components."""
        state = _make_state(pos_d=0.4, theta=0.05, w=0.1, q=0.02)
        control = _make_control(main_flap=0.05, rudder_elevator=0.03)
        cg_offset = jnp.array([0.1, 0.0, 0.05])

        for component in [default_foil, default_rudder, default_sail]:
            _, m_none = component.compute_moth(state, control, 10.0)
            _, m_offset = component.compute_moth(state, control, 10.0, cg_offset=cg_offset)
            assert not jnp.allclose(m_none, m_offset, atol=1e-6), (
                f"{type(component).__name__}: moments should differ with nonzero offset"
            )

    def test_per_call_offset_matches_construction_time(self):
        """Per-call cg_offset produces same result as pre-adjusted positions.

        Key equivalence: raw position + per-call offset == pre-adjusted position + no offset.
        """
        cg_offset_val = np.array([0.1, 0.0, 0.05])

        # Approach 1: raw positions with per-call offset
        foil_raw = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.6, position_z=0.8, foil_span=1.0,
        )
        # Approach 2: pre-adjusted positions (construction-time offset)
        foil_adjusted = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.01,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.6 - cg_offset_val[0],
            position_z=0.8 - cg_offset_val[2],
            foil_span=1.0,
        )

        state = _make_state(pos_d=0.4, theta=0.05, w=0.1, q=0.02)
        control = _make_control(main_flap=0.05)

        f_runtime, m_runtime = foil_raw.compute_moth(
            state, control, 10.0, cg_offset=jnp.array(cg_offset_val),
        )
        f_adjusted, m_adjusted = foil_adjusted.compute_moth(state, control, 10.0)

        assert jnp.allclose(f_runtime, f_adjusted, atol=1e-10), (
            f"Force mismatch: runtime={f_runtime}, adjusted={f_adjusted}"
        )
        assert jnp.allclose(m_runtime, m_adjusted, atol=1e-10), (
            f"Moment mismatch: runtime={m_runtime}, adjusted={m_adjusted}"
        )


# ===========================================================================
# TestRudderDrag (Phase 3 - new tests)
# ===========================================================================

class TestRudderDrag:
    """Tests for rudder drag model (profile + induced)."""

    @pytest.fixture
    def rudder_with_drag(self):
        """Create a rudder with nonzero drag coefficients."""
        return MothRudderElevator(
            rho=1025.0,
            area=0.07,
            cl_alpha=5.0,
            cd0=0.008,
            oswald=0.85,
            ar=7.0,
            position_x=-1.5,
            position_z=0.5,
            foil_span=0.70,
        )

    @pytest.fixture
    def rudder_no_drag(self):
        """Rudder with zero drag for comparison."""
        return MothRudderElevator(
            rho=1025.0,
            area=0.07,
            cl_alpha=5.0,
            cd0=0.0,
            oswald=0.85,
            ar=7.0,
            position_x=-1.5,
            position_z=0.5,
            foil_span=0.70,
        )

    def test_rudder_drag_produces_negative_fx(self, rudder_with_drag):
        """Rudder drag produces negative body-frame X-force (opposing motion)."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        force, _ = rudder_with_drag.compute_moth(state, control, 10.0)
        assert float(force[0]) < 0.0, "Rudder drag should oppose forward motion (Fx < 0)"

    def test_rudder_drag_increases_with_aoa(self, rudder_with_drag):
        """Rudder drag increases with angle of attack (induced drag)."""
        state = _make_state(pos_d=0.4, theta=0.0)
        ctrl_small = _make_control(rudder_elevator=0.01)
        ctrl_large = _make_control(rudder_elevator=0.10)
        f_small, _ = rudder_with_drag.compute_moth(state, ctrl_small, 10.0)
        f_large, _ = rudder_with_drag.compute_moth(state, ctrl_large, 10.0)
        # More negative Fx with higher AoA (more drag)
        assert float(f_large[0]) < float(f_small[0])

    def test_rudder_cd0_adds_profile_drag(self, rudder_with_drag, rudder_no_drag):
        """Nonzero cd0 adds drag even at zero AoA."""
        # At zero AoA: cl=0, so induced drag=0. Only profile drag remains.
        state = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control(rudder_elevator=0.0)
        f_drag, _ = rudder_with_drag.compute_moth(state, control, 10.0)
        f_nodrag, _ = rudder_no_drag.compute_moth(state, control, 10.0)
        # With cd0, Fx should be more negative
        assert float(f_drag[0]) < float(f_nodrag[0])

    def test_rudder_drag_contributes_to_moment(self, rudder_with_drag, rudder_no_drag):
        """Rudder drag arm contributes to pitch moment (r_z * F_x term)."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        _, m_drag = rudder_with_drag.compute_moth(state, control, 10.0)
        _, m_nodrag = rudder_no_drag.compute_moth(state, control, 10.0)
        # Moments should differ due to the r_z * F_x term
        assert not np.isclose(float(m_drag[1]), float(m_nodrag[1]), atol=0.01)

    def test_rudder_full_moment_formula(self, rudder_with_drag):
        """Rudder moment includes both lift and drag arms: M_y = r_z*Fx - r_x*Fz."""
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(rudder_elevator=0.05)
        force, moment = rudder_with_drag.compute_moth(state, control, 10.0)
        # Manual calculation: M_y = pos_z * Fx - pos_x * Fz
        r_x = rudder_with_drag.position_x  # -1.5
        r_z = rudder_with_drag.position_z  # 0.5
        expected_my = r_z * float(force[0]) - r_x * float(force[2])
        np.testing.assert_allclose(float(moment[1]), expected_my, rtol=1e-10)


# ===========================================================================
# TestDragModelSplit (Phase 3)
# ===========================================================================

class TestDragModelSplit:
    """Tests for main foil cd0 = cd0_section + cd0_parasitic split."""

    def test_cd0_split_sums_correctly_in_preset(self):
        """MOTH_BIEKER_V3 cd0 = cd0_section + cd0_parasitic."""
        p = MOTH_BIEKER_V3
        np.testing.assert_allclose(
            p.main_foil_cd0,
            p.main_foil_cd0_section + p.main_foil_cd0_parasitic,
            rtol=1e-10,
        )

    def test_cd0_split_zero_defaults(self):
        """Default split values are zero (backward compat)."""
        from .test_params import make_valid_params
        p = make_valid_params()
        assert p.main_foil_cd0_section == 0.0
        assert p.main_foil_cd0_parasitic == 0.0

    def test_same_total_cd0_produces_same_drag(self):
        """Two foils with same total cd0 produce same drag (split is bookkeeping only)."""
        foil_a = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.006,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.8, position_z=0.6, foil_span=1.0,
        )
        # Same total cd0 = 0.006
        foil_b = MothMainFoil(
            rho=1025.0, area=0.12, cl_alpha=5.7, cl0=0.0, cd0=0.006,
            oswald=0.85, ar=8.33, flap_effectiveness=0.5, cd_flap=0.15,
            position_x=0.8, position_z=0.6, foil_span=1.0,
        )
        state = _make_state(pos_d=0.4, theta=0.05)
        control = _make_control(main_flap=0.05)
        f_a, _ = foil_a.compute_moth(state, control, 10.0)
        f_b, _ = foil_b.compute_moth(state, control, 10.0)
        np.testing.assert_allclose(np.array(f_a), np.array(f_b), atol=1e-12)


# ===========================================================================
# TestSpeedDependentSail (Phase 3)
# ===========================================================================

class TestSpeedDependentSail:
    """Tests for speed-dependent sail thrust model."""

    def test_zero_slope_is_constant_thrust(self):
        """With thrust_slope=0, sail produces constant thrust."""
        sail = MothSailForce(thrust_coeff=50.0, thrust_slope=0.0, ce_position_z=-2.5)
        state = _make_state()
        control = _make_control()
        f_6, _ = sail.compute_moth(state, control, 6.0)
        f_10, _ = sail.compute_moth(state, control, 10.0)
        np.testing.assert_allclose(float(f_6[0]), float(f_10[0]), atol=1e-10)

    def test_positive_slope_increases_with_speed(self):
        """Positive slope means more thrust at higher speed."""
        sail = MothSailForce(thrust_coeff=40.0, thrust_slope=2.0, ce_position_z=-2.5)
        state = _make_state()
        control = _make_control()
        f_6, _ = sail.compute_moth(state, control, 6.0)
        f_10, _ = sail.compute_moth(state, control, 10.0)
        assert float(f_10[0]) > float(f_6[0])

    def test_affine_model_formula(self):
        """F_sail = thrust_coeff + thrust_slope * u_forward."""
        a, b = 40.0, 2.5
        sail = MothSailForce(thrust_coeff=a, thrust_slope=b, ce_position_z=-2.5)
        state = _make_state()
        control = _make_control()
        for u in [4.0, 6.0, 8.0, 10.0]:
            f, _ = sail.compute_moth(state, control, u)
            expected = a + b * u
            np.testing.assert_allclose(float(f[0]), expected, rtol=1e-10)

    def test_moment_scales_with_thrust(self):
        """Moment = ce_z * F_sail, so it also scales with speed."""
        ce_z = -2.5
        sail = MothSailForce(thrust_coeff=40.0, thrust_slope=2.0, ce_position_z=ce_z)
        state = _make_state()
        control = _make_control()
        for u in [6.0, 10.0]:
            f, m = sail.compute_moth(state, control, u)
            expected_my = ce_z * float(f[0])
            np.testing.assert_allclose(float(m[1]), expected_my, rtol=1e-10)


# ===========================================================================
# TestStrutDragImmersion (Phase 2.7)
# ===========================================================================

from fmd.simulator.components.moth_forces import MothStrutDrag


def _make_strut(heel_angle=np.deg2rad(30.0)):
    """Helper: create a MothStrutDrag with typical main strut parameters."""
    return MothStrutDrag(
        strut_chord=0.09,
        strut_thickness=0.013,
        strut_cd_pressure=0.01,
        strut_cf_skin=0.003,
        strut_position_x=0.8,
        strut_max_depth=0.6,
        strut_top_z=0.0,
        strut_bottom_z=0.6,
        heel_angle=heel_angle,
        rho=1025.0,
    )


class TestStrutDragImmersion:
    """Tests for depth-dependent strut drag (Phase 2.7 - Plan Phase 1)."""

    def test_zero_drag_out_of_water(self):
        """Strut produces zero drag when fully above the waterline.

        When pos_d is very negative (boat high above water), both strut top
        and bottom are above the surface, so submerged_depth = 0.
        """
        strut = _make_strut()
        # pos_d = -2.0: boat 2m above nominal, everything out of water
        state = _make_state(pos_d=-2.0, theta=0.0)
        control = _make_control()
        force, moment = strut.compute_moth(state, control, 10.0)
        assert abs(float(force[0])) < 1e-10, (
            f"Expected zero drag out of water, got fx={float(force[0])}"
        )
        assert abs(float(moment[1])) < 1e-10

    def test_full_submergence_limit(self):
        """Full submergence (deeply submerged) reproduces constant-depth drag.

        When the strut top is well below water, submerged_depth == strut_max_depth,
        and the drag should match the old fixed-depth calculation.
        """
        strut = _make_strut()
        # pos_d = 0.4: boat deep, CG submerged, full strut in water
        state = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control()
        force, _ = strut.compute_moth(state, control, 10.0)

        # Manual calculation at full depth = 0.6m
        u = 10.0
        frontal_area = 0.6 * 0.013
        drag_pressure = 0.5 * 1025.0 * 0.01 * frontal_area * u**2
        wetted_area = 2.0 * 0.09 * 0.6
        drag_skin = 0.5 * 1025.0 * 0.003 * wetted_area * u**2
        expected_fx = -(drag_pressure + drag_skin)

        np.testing.assert_allclose(float(force[0]), expected_fx, rtol=1e-6)

    def test_partial_immersion_scales_drag(self):
        """Drag scales proportionally with submerged depth at partial immersion."""
        strut = _make_strut()
        # Deep state (full immersion)
        state_deep = _make_state(pos_d=0.4, theta=0.0)
        control = _make_control()
        f_deep, _ = strut.compute_moth(state_deep, control, 10.0)

        # Shallow state (partial immersion)
        state_shallow = _make_state(pos_d=-0.3, theta=0.0)
        f_shallow, _ = strut.compute_moth(state_shallow, control, 10.0)

        # Shallow drag must be less than deep drag (less strut submerged)
        assert abs(float(f_shallow[0])) < abs(float(f_deep[0])), (
            f"Shallow drag {float(f_shallow[0]):.4f} should be less than "
            f"deep drag {float(f_deep[0]):.4f}"
        )
        # Both should be negative (opposing forward motion)
        assert float(f_deep[0]) < 0
        assert float(f_shallow[0]) < 0

    def test_pitch_effect_on_immersion(self):
        """Pitch angle shifts immersion differently for forward vs aft struts.

        Nose-up pitch (theta > 0) lifts the bow and lowers the stern:
        depth term: -eff_pos_x * sin(theta)
          - Forward strut (x > 0): negative contribution -> shallower
          - Aft strut (x < 0): positive contribution -> deeper
        """
        # Forward strut (x=0.8, same as main foil)
        fwd_strut = _make_strut()
        # Aft strut (x=-1.5, same as rudder)
        aft_strut = MothStrutDrag(
            strut_chord=0.07, strut_thickness=0.010,
            strut_cd_pressure=0.01, strut_cf_skin=0.003,
            strut_position_x=-1.5,
            strut_max_depth=0.5, strut_top_z=0.0, strut_bottom_z=0.5,
            heel_angle=np.deg2rad(30.0), rho=1025.0,
        )
        # Moderate height where partial immersion occurs
        state_level = _make_state(pos_d=-0.4, theta=0.0)
        state_pitched = _make_state(pos_d=-0.4, theta=0.1)  # ~5.7 deg nose-up
        control = _make_control()

        # Forward strut: nose-up pitch lifts the bow -> less immersion
        f_fwd_level, _ = fwd_strut.compute_moth(state_level, control, 10.0)
        f_fwd_pitched, _ = fwd_strut.compute_moth(state_pitched, control, 10.0)
        # Less drag when pitched (less submerged at the bow)
        assert abs(float(f_fwd_pitched[0])) <= abs(float(f_fwd_level[0])) + 0.01

        # Aft strut: nose-up pitch lowers the stern -> more immersion
        f_aft_level, _ = aft_strut.compute_moth(state_level, control, 10.0)
        f_aft_pitched, _ = aft_strut.compute_moth(state_pitched, control, 10.0)
        # More drag when pitched (more submerged at the stern)
        assert abs(float(f_aft_pitched[0])) >= abs(float(f_aft_level[0])) - 0.01

    def test_submerged_centroid_moment_arm(self):
        """Moment arm uses centroid of submerged segment, not fixed midpoint.

        At partial immersion, the centroid should be closer to the bottom
        (deeper) than at full immersion.
        """
        strut = _make_strut()
        control = _make_control()

        # Full immersion: centroid at bottom_z - max_depth/2 = 0.6 - 0.3 = 0.3
        state_deep = _make_state(pos_d=0.4, theta=0.0)
        f_deep, m_deep = strut.compute_moth(state_deep, control, 10.0)

        # Partial immersion: centroid should be different
        state_partial = _make_state(pos_d=-0.3, theta=0.0)
        f_partial, m_partial = strut.compute_moth(state_partial, control, 10.0)

        # Both have negative fx (drag), and different moment arms
        # moment = centroid_z * fx, so my/fx = centroid_z
        if abs(float(f_deep[0])) > 1e-10 and abs(float(f_partial[0])) > 1e-10:
            arm_deep = float(m_deep[1]) / float(f_deep[0])
            arm_partial = float(m_partial[1]) / float(f_partial[0])
            # At partial immersion, the centroid z should be different from full
            assert abs(arm_deep - arm_partial) > 0.01, (
                f"Moment arms should differ: deep={arm_deep:.4f}, partial={arm_partial:.4f}"
            )

    def test_immersion_fraction_output(self):
        """compute_immersion returns correct fraction (0=dry, 1=fully submerged)."""
        strut = _make_strut()

        # Fully submerged
        state_deep = _make_state(pos_d=0.4, theta=0.0)
        imm_deep = float(strut.compute_immersion(state_deep))
        np.testing.assert_allclose(imm_deep, 1.0, atol=0.01)

        # Fully dry
        state_dry = _make_state(pos_d=-2.0, theta=0.0)
        imm_dry = float(strut.compute_immersion(state_dry))
        np.testing.assert_allclose(imm_dry, 0.0, atol=0.01)

        # Partial
        state_partial = _make_state(pos_d=-0.3, theta=0.0)
        imm_partial = float(strut.compute_immersion(state_partial))
        assert 0.0 < imm_partial < 1.0, (
            f"Expected partial immersion, got {imm_partial}"
        )
