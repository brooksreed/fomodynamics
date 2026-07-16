"""C1.F: Free-surface lift correction sigma(h/c) tests.

Locks the FSL model form approved 2026-07-15:
- sigma(h/c) = (1 + 16*(h/c)^2) / (2 + 16*(h/c)^2), the classic high-Froude
  image-vortex result: -> 1 deep, 0.5 at the surface, clamped for h <= 0.
- 3-point spanwise sampling at strip centers y = {-s/3, 0, +s/3} so a heeled
  foil's shallow leeward sections reduce lift before the tip breaches.
- sigma multiplies total circulatory lift (cl0 + cl_alpha*alpha); induced
  drag is divided by sigma (image downwash raises drag per unit lift).
- Clean separation from ventilation: sigma models gradual pre-breach
  circulation loss (bounded at 0.5); compute_depth_factor keeps modeling
  the post-breach cliff. The two must not double-count.

Also locks the exposed ventilation_sharpness parameter (previously a
hardcoded tanh gain of 6.0).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fmd.simulator.components.moth_forces import (
    MothMainFoil,
    MothRudderElevator,
    compute_depth_factor,
    compute_free_surface_factor,
    create_moth_components,
)
from fmd.simulator.params import MOTH_BIEKER_V3

CHORD = MOTH_BIEKER_V3.main_foil_chord      # 0.089 m
SPAN = MOTH_BIEKER_V3.main_foil_span        # 0.95 m
HEEL_30 = np.deg2rad(30.0)


def _sigma_2d(h_over_c: float) -> float:
    """Reference 2D image formula, no sampling, no clamping."""
    hc2 = 16.0 * h_over_c**2
    return (1.0 + hc2) / (2.0 + hc2)


class TestSigmaFunction:
    """Pure-function properties of compute_free_surface_factor."""

    def test_limit_deep(self):
        """sigma -> 1 when the foil is many chords deep (both heels)."""
        for heel in (0.0, HEEL_30):
            sigma = float(compute_free_surface_factor(5.0, CHORD, SPAN, heel))
            assert sigma > 0.999

    def test_limit_surface_upright(self):
        """sigma = 0.5 exactly at the surface with zero heel (2D limit)."""
        sigma = float(compute_free_surface_factor(0.0, CHORD, SPAN, 0.0))
        assert sigma == pytest.approx(0.5, abs=1e-12)

    def test_clamped_above_surface(self):
        """For h <= 0 sigma stays at 0.5 — breach behavior belongs to the
        ventilation model, not FSL (no double-count)."""
        for depth in (0.0, -0.05, -1.0):
            sigma = float(compute_free_surface_factor(depth, CHORD, SPAN, 0.0))
            assert sigma == pytest.approx(0.5, abs=1e-12)

    def test_bounds(self):
        """sigma in [0.5, 1] everywhere, including partially-emerged states."""
        for depth in np.linspace(-0.5, 2.0, 101):
            for heel in (0.0, HEEL_30):
                sigma = float(compute_free_surface_factor(
                    float(depth), CHORD, SPAN, heel))
                assert 0.5 - 1e-12 <= sigma <= 1.0

    def test_monotonic_in_depth(self):
        """Deeper foil never has less lift retention."""
        for heel in (0.0, HEEL_30):
            depths = np.linspace(-0.1, 1.0, 200)
            sigmas = [float(compute_free_surface_factor(
                float(d), CHORD, SPAN, heel)) for d in depths]
            assert np.all(np.diff(sigmas) >= -1e-12)

    def test_zero_heel_degenerates_to_center_formula(self):
        """At zero heel all 3 stations coincide: sigma_eff == sigma_2d(h/c)."""
        for depth in (0.05, 0.15, 0.4):
            sigma = float(compute_free_surface_factor(depth, CHORD, SPAN, 0.0))
            assert sigma == pytest.approx(_sigma_2d(depth / CHORD), rel=1e-12)

    def test_heel_reduces_sigma_in_study_band(self):
        """At the same center depth, a 30-deg-heeled foil has shallow leeward
        sections and loses more lift than an upright one (tip +2 cm case)."""
        depth = 0.2575  # leeward tip +2 cm below surface at 30 deg heel
        sigma_heeled = float(compute_free_surface_factor(
            depth, CHORD, SPAN, HEEL_30))
        sigma_upright = float(compute_free_surface_factor(
            depth, CHORD, SPAN, 0.0))
        assert sigma_heeled < sigma_upright - 0.005

    def test_c1_continuity_at_surface(self):
        """The h <= 0 clamp is C^1: d(sigma)/d(depth) -> 0 approaching the
        surface from either side (d(sigma_2d)/dh = 0 at h = 0)."""
        grad = jax.grad(
            lambda d: compute_free_surface_factor(d, CHORD, SPAN, 0.0))
        assert float(grad(-1e-8)) == pytest.approx(0.0, abs=1e-9)
        assert float(grad(1e-8)) == pytest.approx(0.0, abs=1e-4)
        # gradient is finite and positive slightly deeper
        assert 0.0 < float(grad(0.05)) < np.inf


def _main_foil(enable_fsl: bool, heel: float = HEEL_30) -> MothMainFoil:
    components = create_moth_components(
        MOTH_BIEKER_V3, heel_angle=heel,
        enable_free_surface_lift=enable_fsl)
    return components[0]


def _rudder(enable_fsl: bool, heel: float = HEEL_30) -> MothRudderElevator:
    components = create_moth_components(
        MOTH_BIEKER_V3, heel_angle=heel,
        enable_free_surface_lift=enable_fsl)
    return components[1]


def _lift_drag(foil, pos_d: float, flap: float = 0.05, u: float = 10.0):
    """Vertical (lift-ish, -fz) and streamwise (drag-ish, -fx) body force."""
    state = jnp.array([pos_d, 0.0, 0.0, 0.0, u])
    control = jnp.array([flap, 0.0])
    force, _ = foil.compute_moth(state, control, u)
    return -float(force[2]), -float(force[0])


# pos_d values chosen so the leeward tip stays submerged (depth_factor ~ 1,
# pre-breach) while the foil center moves through the approach-to-surface
# band at 30 deg heel. With cg_offset=None the foil center NED depth is
# pos_d + 1.82*cos(30 deg) ~ pos_d + 1.576, and the leeward tip rides
# (span/2)*sin(30 deg) ~ 0.2375 shallower than the center.
POS_D_VERY_DEEP = -0.5   # center ~ 1.08 m (h/c ~ 12): sigma ~ 1
POS_D_DEEP = -0.9        # center ~ 0.68 m: deep reference inside the polar
POS_D_HIGH = -1.30       # center ~ 0.28 m, tip ~ +4 cm: pre-breach, df ~ 1


class TestMainFoilFSL:
    """Component-level directional behavior."""

    def test_lift_falls_approaching_surface_pre_breach(self):
        """With FSL on, lift has a real depth gradient BEFORE the tip
        breaches — the heave stiffness the pre-FSL model lacked."""
        foil = _main_foil(enable_fsl=True)
        lift_deep, _ = _lift_drag(foil, POS_D_DEEP)
        lift_high, _ = _lift_drag(foil, POS_D_HIGH)
        assert lift_high < lift_deep
        # meaningful, not epsilon: at least 0.5% loss over the band
        # (measured ~0.9% at tip +4 cm vs deep, 30 deg heel, theta=0)
        assert lift_high < 0.995 * lift_deep

    def test_pre_fsl_lift_is_depth_flat_pre_breach(self):
        """With FSL off, pre-breach lift is essentially flat with depth —
        the historical trim null space this chunk closes."""
        foil = _main_foil(enable_fsl=False)
        lift_deep, _ = _lift_drag(foil, POS_D_DEEP)
        lift_high, _ = _lift_drag(foil, POS_D_HIGH)
        assert lift_high == pytest.approx(lift_deep, rel=5e-3)

    def test_disable_flag_recovers_deep_water_limit(self):
        """FSL on and off agree when the foil is deep (sigma -> 1)."""
        lift_on, drag_on = _lift_drag(_main_foil(True), POS_D_VERY_DEEP)
        lift_off, drag_off = _lift_drag(_main_foil(False), POS_D_VERY_DEEP)
        assert lift_on == pytest.approx(lift_off, rel=1e-3)
        assert drag_on == pytest.approx(drag_off, rel=1e-3)

    def test_drag_at_fixed_lift_rises_near_surface(self):
        """The racing gradient: to hold the SAME lift near the surface the
        foil needs more alpha and pays more induced drag (cd_i ~ 1/sigma)."""
        foil = _main_foil(enable_fsl=True)
        target_lift, drag_deep = _lift_drag(foil, POS_D_DEEP, flap=0.05)

        # bisect flap at the high ride height to match the deep lift
        lo, hi = 0.0, 0.3
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lift_mid, _ = _lift_drag(foil, POS_D_HIGH, flap=mid)
            if lift_mid < target_lift:
                lo = mid
            else:
                hi = mid
        flap_high = 0.5 * (lo + hi)
        lift_high, drag_high = _lift_drag(foil, POS_D_HIGH, flap=flap_high)
        assert lift_high == pytest.approx(target_lift, rel=1e-4)
        assert drag_high > drag_deep

    def test_rudder_fsl_applies(self):
        """Rudder inherits the same correction."""
        rudder_on = _rudder(True)
        rudder_off = _rudder(False)
        state = jnp.array([POS_D_HIGH, 0.0, 0.0, 0.0, 10.0])
        control = jnp.array([0.0, 0.08])
        f_on, _ = rudder_on.compute_moth(state, control, 10.0)
        f_off, _ = rudder_off.compute_moth(state, control, 10.0)
        assert abs(float(f_on[2])) < abs(float(f_off[2]))


class TestVentilationSharpness:
    """The previously hardcoded tanh gain 6.0 is now a named parameter."""

    def test_default_reproduces_legacy_cliff(self):
        """Default sharpness (6.0) matches the historical hardcoded value."""
        df_default = float(compute_depth_factor(0.20, SPAN, HEEL_30))
        df_explicit = float(compute_depth_factor(
            0.20, SPAN, HEEL_30, ventilation_sharpness=6.0))
        assert df_default == pytest.approx(df_explicit, abs=1e-15)

    def test_lower_sharpness_softens_cliff(self):
        """Smaller gain spreads the ventilation transition over more depth."""
        # transition width measured between df=0.25 and df=0.75 crossings
        def width(sharpness: float) -> float:
            depths = np.linspace(0.05, 0.35, 601)
            dfs = np.array([float(compute_depth_factor(
                float(d), SPAN, HEEL_30,
                ventilation_sharpness=sharpness)) for d in depths])
            above = depths[dfs > 0.75]
            below = depths[dfs < 0.25]
            return float(above.min() - below.max())

        assert width(3.0) > width(6.0) > width(12.0)

    def test_factory_plumbs_sharpness_and_fsl_flag(self):
        components = create_moth_components(
            MOTH_BIEKER_V3, heel_angle=HEEL_30,
            ventilation_sharpness=4.0, enable_free_surface_lift=False)
        main_foil, rudder = components[0], components[1]
        for comp in (main_foil, rudder):
            assert comp.ventilation_sharpness == 4.0
            assert comp.enable_free_surface_lift is False
        assert main_foil.chord == MOTH_BIEKER_V3.main_foil_chord
        assert rudder.chord == MOTH_BIEKER_V3.rudder_chord
