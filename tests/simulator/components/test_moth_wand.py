"""Tests for Moth wand linkage geometry module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fmd.simulator.components.moth_wand import (
    WandLinkage,
    WandLinkageState,
    _safe_arccos,
    wand_angle_from_state,
    wand_angle_from_state_waves,
    wand_world_angle_from_height,
    create_wand_linkage,
    gearing_ratio_from_rod,
    DEFAULT_WAND_LENGTH,
    DEFAULT_FASTPOINT,
    DEFAULT_BELLCRANK_ANGLE,
)
from fmd.simulator.waves import WaveField
from fmd.simulator.params.wave import WaveParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linkage():
    """Default wand linkage with plan parameters."""
    return create_wand_linkage()


@pytest.fixture
def wand_pivot():
    """Typical wand pivot position (bowsprit, forward of CG)."""
    return jnp.array([1.6, 0.0, 0.1])


# ---------------------------------------------------------------------------
# TestSafeArccos
# ---------------------------------------------------------------------------

class TestSafeArccos:
    """Tests for _safe_arccos gradient-safe boundary handling."""

    def test_above_upper_bound_returns_zero(self):
        """Input above 1 clamps to arccos(1) = 0."""
        result = _safe_arccos(jnp.float64(1.5))
        np.testing.assert_allclose(float(result), 0.0, atol=1e-10)

    def test_below_lower_bound_returns_pi_over_2(self):
        """Input below 0 clamps to arccos(0) = pi/2."""
        result = _safe_arccos(jnp.float64(-0.5))
        np.testing.assert_allclose(float(result), np.pi / 2, atol=1e-10)

    def test_grad_at_upper_boundary_finite(self):
        """Gradient at x=1.0 is finite (no 0*inf NaN)."""
        grad_val = jax.grad(_safe_arccos)(jnp.float64(1.0))
        assert jnp.isfinite(grad_val), f"Gradient at x=1.0 is not finite: {grad_val}"

    def test_grad_at_zero_finite(self):
        """Gradient at x=0.0 is finite."""
        grad_val = jax.grad(_safe_arccos)(jnp.float64(0.0))
        assert jnp.isfinite(grad_val), f"Gradient at x=0.0 is not finite: {grad_val}"

    def test_grad_at_interior_matches_analytic(self):
        """Gradient at x=0.5 matches d(arccos)/dx = -1/sqrt(1-x^2)."""
        x = jnp.float64(0.5)
        grad_val = jax.grad(_safe_arccos)(x)
        expected = -1.0 / jnp.sqrt(1.0 - x**2)
        np.testing.assert_allclose(float(grad_val), float(expected), rtol=1e-5)


# ---------------------------------------------------------------------------
# TestWandAngleFromState
# ---------------------------------------------------------------------------

class TestWandAngleFromState:
    """Tests for wand_angle_from_state."""

    def test_pivot_at_surface(self, wand_pivot):
        """Pivot at water surface -> wand horizontal (90 deg)."""
        # pos_d=0, theta=0, pivot_z=0.1 -> NED depth = 0.1
        # h = -0.1 (below surface) -> clip to 0 -> arccos(0) = pi/2
        # Actually with pivot at z=0.1 and pos_d such that depth ~ 0:
        # depth = pos_d + pivot_z * cos(0) - pivot_x * sin(0) = pos_d + 0.1
        # For depth=0: pos_d = -0.1, h=0 -> arccos(0) = pi/2
        pos_d = jnp.float64(-0.1)
        theta = jnp.float64(0.0)
        angle = wand_angle_from_state(pos_d, theta, wand_pivot)
        np.testing.assert_allclose(float(angle), np.pi / 2, atol=1e-10)

    def test_pivot_above_water(self, wand_pivot):
        """Pivot above water -> wand angle between 0 and 90 deg."""
        # pos_d = -0.5 -> depth = -0.5 + 0.1 = -0.4
        # h = 0.4, ratio = 0.4/1.175 -> arccos gives angle in (0, pi/2)
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.0)
        angle = wand_angle_from_state(pos_d, theta, wand_pivot)
        assert 0.0 < float(angle) < np.pi / 2

    def test_pivot_at_max_height(self, wand_pivot):
        """Pivot at exactly wand_length above water -> wand vertical (0 deg)."""
        # depth = pos_d + 0.1; h = -depth = -(pos_d + 0.1)
        # For h = wand_length = 1.175: pos_d = -1.275
        pos_d = jnp.float64(-1.275)
        theta = jnp.float64(0.0)
        angle = wand_angle_from_state(pos_d, theta, wand_pivot)
        np.testing.assert_allclose(float(angle), 0.0, atol=1e-6)

    def test_clamps_at_limits(self, wand_pivot):
        """Clamps when pivot is well above wand_length or below surface."""
        # Way above: h >> wand_length -> clip to 1 -> arccos(1) = 0
        pos_d = jnp.float64(-5.0)
        theta = jnp.float64(0.0)
        angle_high = wand_angle_from_state(pos_d, theta, wand_pivot)
        np.testing.assert_allclose(float(angle_high), 0.0, atol=1e-10)

        # Way below: h << 0 -> clip to 0 -> arccos(0) = pi/2
        pos_d = jnp.float64(2.0)
        angle_low = wand_angle_from_state(pos_d, theta, wand_pivot)
        np.testing.assert_allclose(float(angle_low), np.pi / 2, atol=1e-10)

    def test_pitch_affects_pivot_depth(self, wand_pivot):
        """Pitch angle changes effective pivot depth."""
        pos_d = jnp.float64(-0.5)
        theta_zero = jnp.float64(0.0)
        theta_pos = jnp.float64(0.1)  # nose up

        angle_zero = wand_angle_from_state(pos_d, theta_zero, wand_pivot)
        angle_pitched = wand_angle_from_state(pos_d, theta_pos, wand_pivot)

        # Nose-up pitch raises the forward pivot -> smaller wand angle
        assert float(angle_pitched) < float(angle_zero)

    def test_jit_compatible(self, wand_pivot):
        """Function can be JIT compiled."""
        jitted = jax.jit(wand_angle_from_state, static_argnums=(3, 4))
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.0)
        result = jitted(pos_d, theta, wand_pivot)
        assert jnp.isfinite(result)

    def test_grad_compatible(self, wand_pivot):
        """Differentiable w.r.t. pos_d."""
        def f(pos_d):
            return wand_angle_from_state(
                pos_d, jnp.float64(0.0), wand_pivot
            )

        grad_fn = jax.grad(f)
        pos_d = jnp.float64(-0.5)
        grad_val = grad_fn(pos_d)
        assert jnp.isfinite(grad_val)
        # Wand angle should increase as pos_d increases (boat sinks)
        assert float(grad_val) > 0


# ---------------------------------------------------------------------------
# TestWandLinkage
# ---------------------------------------------------------------------------

class TestWandLinkage:
    """Tests for WandLinkage kinematic computation."""

    def test_fastpoint_zero_offset_gives_zero_flap(self, linkage):
        """At fastpoint with zero offset, flap = 0."""
        flap = linkage.compute(jnp.float64(DEFAULT_FASTPOINT))
        np.testing.assert_allclose(float(flap), 0.0, atol=1e-10)

    def test_wand_at_zero_gives_negative_flap(self, linkage):
        """Wand at 0 deg (boat high) -> flap negative (up)."""
        flap = linkage.compute(jnp.float64(0.0))
        assert float(flap) < 0.0

    def test_wand_at_80_gives_positive_flap(self, linkage):
        """Wand at 80 deg (boat low) -> flap positive (down)."""
        flap = linkage.compute(jnp.float64(jnp.radians(80.0)))
        assert float(flap) > 0.0

    def test_monotonic_response(self, linkage):
        """Response is monotonically increasing over [0, 80] degrees."""
        angles = jnp.linspace(0.0, jnp.radians(80.0), 50)
        flaps = jax.vmap(linkage.compute)(angles)
        diffs = jnp.diff(flaps)
        assert jnp.all(diffs > 0), "Response should be monotonically increasing"

    def test_compute_detailed_matches_compute(self, linkage):
        """compute_detailed returns matching flap angle."""
        wand_angle = jnp.float64(jnp.radians(45.0))
        scalar_flap = linkage.compute(wand_angle)
        detailed = linkage.compute_detailed(wand_angle)

        np.testing.assert_allclose(
            float(detailed.flap_angle), float(scalar_flap), atol=1e-14
        )

    def test_compute_detailed_returns_all_fields(self, linkage):
        """compute_detailed returns a WandLinkageState with all fields."""
        wand_angle = jnp.float64(jnp.radians(45.0))
        state = linkage.compute_detailed(wand_angle)

        assert isinstance(state, WandLinkageState)
        assert len(state) == 6
        for field_val in state:
            assert jnp.isfinite(field_val)

    def test_pullrod_offset_shifts_curve(self, linkage):
        """Pullrod offset shifts the entire response curve."""
        wand_angle = jnp.float64(DEFAULT_FASTPOINT)

        # With positive offset, flap at fastpoint should be positive (not zero)
        shifted = create_wand_linkage(pullrod_offset=0.005)
        flap_shifted = shifted.compute(wand_angle)
        assert float(flap_shifted) > 0.0

        # With negative offset, flap at fastpoint should be negative
        shifted_neg = create_wand_linkage(pullrod_offset=-0.005)
        flap_shifted_neg = shifted_neg.compute(wand_angle)
        assert float(flap_shifted_neg) < 0.0

    def test_gearing_ratio_scales_displacement(self, linkage):
        """Gearing ratio scales the pullrod displacement."""
        wand_angle = jnp.float64(jnp.radians(50.0))

        linkage_1x = create_wand_linkage(gearing_ratio=1.0)
        linkage_2x = create_wand_linkage(gearing_ratio=2.0)

        state_1x = linkage_1x.compute_detailed(wand_angle)
        state_2x = linkage_2x.compute_detailed(wand_angle)

        # Aft pullrod displacement should scale with gearing ratio
        np.testing.assert_allclose(
            float(state_2x.aft_pullrod_dx),
            2.0 * float(state_1x.aft_pullrod_dx),
            rtol=1e-12,
        )

    def test_gain_peaks_at_fastpoint(self, linkage):
        """Differential gain peaks near fastpoint."""
        angles = jnp.linspace(jnp.radians(5.0), jnp.radians(75.0), 50)
        gains = jax.vmap(linkage.gain)(angles)

        peak_idx = int(jnp.argmax(gains))
        peak_angle_deg = float(jnp.degrees(angles[peak_idx]))
        fp_deg = float(np.degrees(DEFAULT_FASTPOINT))

        # Peak should be near the fastpoint
        assert abs(peak_angle_deg - fp_deg) < 5.0, (
            f"Gain peak at {peak_angle_deg:.1f} deg, expected ~{fp_deg:.0f} deg"
        )

    def test_jit_compatible(self, linkage):
        """WandLinkage.compute works under JIT."""
        jitted = jax.jit(linkage.compute)
        result = jitted(jnp.float64(jnp.radians(30.0)))
        assert jnp.isfinite(result)

    def test_grad_compatible(self, linkage):
        """WandLinkage.compute is differentiable."""
        grad_fn = jax.grad(linkage.compute)
        grad_val = grad_fn(jnp.float64(jnp.radians(30.0)))
        assert jnp.isfinite(grad_val)

    def test_bellcrank_angle_affects_gain(self):
        """Non-90-degree bellcrank angle reduces gain by sin(alpha)."""
        wand_angle = jnp.float64(DEFAULT_FASTPOINT)

        linkage_90 = create_wand_linkage()  # alpha = pi/2
        linkage_60 = create_wand_linkage(bellcrank_angle=np.radians(60.0))

        gain_90 = float(linkage_90.gain(wand_angle))
        gain_60 = float(linkage_60.gain(wand_angle))

        # At fastpoint (phi=0), gain scales as sin(alpha)
        np.testing.assert_allclose(
            gain_60 / gain_90, np.sin(np.radians(60.0)), rtol=1e-10
        )

    def test_bellcrank_angle_nonlinear_at_large_displacement(self):
        """Non-90-degree bellcrank introduces nonlinearities at large angles."""
        linkage_90 = create_wand_linkage()
        linkage_60 = create_wand_linkage(bellcrank_angle=np.radians(60.0))

        # At fastpoint, the ratio is exactly sin(60)/sin(90) = sin(60)
        fp_ratio = float(linkage_60.gain(jnp.float64(DEFAULT_FASTPOINT))) / \
                   float(linkage_90.gain(jnp.float64(DEFAULT_FASTPOINT)))

        # At a different angle, the ratio should differ (nonlinearity)
        off_ratio = float(linkage_60.gain(jnp.float64(jnp.radians(60.0)))) / \
                    float(linkage_90.gain(jnp.float64(jnp.radians(60.0))))

        assert abs(fp_ratio - off_ratio) > 0.001, (
            f"Bellcrank angle should introduce nonlinearity: "
            f"fastpoint ratio={fp_ratio:.4f}, off-fastpoint ratio={off_ratio:.4f}"
        )

    def test_arcsin_clipping_no_nan(self):
        """Extreme inputs don't produce NaN due to arcsin clipping."""
        # Large wand lever + gearing could push arcsin argument > 1
        extreme = create_wand_linkage(
            wand_lever=0.050,  # 50mm - way bigger than default
            gearing_ratio=3.0,
        )
        # At extremes, the clipping should prevent NaN
        flap = extreme.compute(jnp.float64(jnp.radians(80.0)))
        assert jnp.isfinite(flap)

        flap_zero = extreme.compute(jnp.float64(0.0))
        assert jnp.isfinite(flap_zero)


# ---------------------------------------------------------------------------
# TestWandLinkagePhysics
# ---------------------------------------------------------------------------

class TestWandLinkagePhysics:
    """Physics validation tests for expected behavior."""

    def test_expected_flap_range(self, linkage):
        """Flap range approximately matches targets."""
        flap_0 = float(jnp.degrees(linkage.compute(jnp.float64(0.0))))
        flap_80 = float(jnp.degrees(
            linkage.compute(jnp.float64(jnp.radians(80.0)))
        ))

        # With measured params (L_w=20mm, gearing≈0.765, L_p=L_v=30mm, L_f=30mm, fp=45°)
        assert -25.0 < flap_0 < -17.0, f"Flap at 0 deg: {flap_0:.1f}, expected ~-21"
        assert 13.0 < flap_80 < 22.0, f"Flap at 80 deg: {flap_80:.1f}, expected ~+17"

    def test_regression_values(self, linkage):
        """Exact regression values to catch parameter drift."""
        flap_0 = float(jnp.degrees(linkage.compute(jnp.float64(0.0))))
        flap_fp = float(jnp.degrees(
            linkage.compute(jnp.float64(DEFAULT_FASTPOINT))
        ))
        flap_80 = float(jnp.degrees(
            linkage.compute(jnp.float64(jnp.radians(80.0)))
        ))
        gain_at_fp = float(linkage.gain(jnp.float64(DEFAULT_FASTPOINT)))

        np.testing.assert_allclose(flap_0, -21.130, atol=0.1)
        np.testing.assert_allclose(flap_fp, 0.0, atol=0.001)
        np.testing.assert_allclose(flap_80, 17.002, atol=0.1)
        np.testing.assert_allclose(gain_at_fp, 0.5098, atol=0.0001)

    def test_gain_curve_shape(self, linkage):
        """Gain is higher near fastpoint than at extremes."""
        gain_at_5 = float(linkage.gain(jnp.float64(jnp.radians(5.0))))
        gain_at_fp = float(linkage.gain(jnp.float64(DEFAULT_FASTPOINT)))
        gain_at_75 = float(linkage.gain(jnp.float64(jnp.radians(75.0))))

        assert gain_at_fp > gain_at_5, "Gain at fastpoint should exceed gain at 5 deg"
        assert gain_at_fp > gain_at_75, "Gain at fastpoint should exceed gain at 75 deg"


# ---------------------------------------------------------------------------
# TestWandEndToEnd
# ---------------------------------------------------------------------------

class TestWandEndToEnd:
    """End-to-end tests: pos_d -> wand_angle -> flap_angle."""

    def test_boat_high_gives_negative_flap(self):
        """Boat high (pos_d << 0) produces negative (up) flap."""
        pivot = jnp.array([1.6, 0.0, 0.1])
        linkage = create_wand_linkage()
        pos_d = jnp.float64(-1.2)  # high enough for wand angle < fastpoint
        theta = jnp.float64(0.0)

        wand_angle = wand_angle_from_state(pos_d, theta, pivot)
        flap = linkage.compute(wand_angle)

        assert float(flap) < 0.0, "Boat high should give negative (up) flap"

    def test_boat_mid_gives_positive_flap(self):
        """Boat mid (pos_d closer to 0) produces positive (down) flap."""
        pivot = jnp.array([1.6, 0.0, 0.1])
        linkage = create_wand_linkage()
        pos_d = jnp.float64(-0.3)
        theta = jnp.float64(0.0)

        wand_angle = wand_angle_from_state(pos_d, theta, pivot)
        flap = linkage.compute(wand_angle)

        assert float(flap) > 0.0, "Boat mid should give positive (down) flap"

    def test_monotonic_pos_d_to_flap(self):
        """Lower pos_d (higher boat) produces more negative flap."""
        pivot = jnp.array([1.6, 0.0, 0.1])
        linkage = create_wand_linkage()
        theta = jnp.float64(0.0)

        pos_d_values = [-0.8, -0.6, -0.4, -0.3]
        flaps = []
        for pd in pos_d_values:
            wa = wand_angle_from_state(jnp.float64(pd), theta, pivot)
            flaps.append(float(linkage.compute(wa)))

        # As pos_d increases (boat sinks), flap should increase (more positive)
        for i in range(len(flaps) - 1):
            assert flaps[i] < flaps[i + 1], (
                f"Flap not monotonic: pos_d={pos_d_values[i]:.1f} -> "
                f"flap={flaps[i]:.4f}, pos_d={pos_d_values[i+1]:.1f} -> "
                f"flap={flaps[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# TestGearingRodHelper
# ---------------------------------------------------------------------------

class TestGearingRodHelper:
    """Tests for gearing_ratio_from_rod helper."""

    def test_default_rod_dimensions(self):
        """Default rod gives expected ratio."""
        ratio = gearing_ratio_from_rod(0.170, 0.130)
        np.testing.assert_allclose(ratio, 130.0 / 170.0, rtol=1e-12)

    def test_tap_at_full_length(self):
        """Tap at full length gives ratio = 1."""
        ratio = gearing_ratio_from_rod(0.170, 0.170)
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-12)

    def test_adjustment_range(self):
        """Typical adjustment range gives ratios < 1."""
        ratio_low = gearing_ratio_from_rod(0.170, 0.100)
        ratio_high = gearing_ratio_from_rod(0.170, 0.150)
        assert 0.5 < ratio_low < 0.7
        assert 0.8 < ratio_high < 1.0
        assert ratio_low < ratio_high


# ---------------------------------------------------------------------------
# TestWandAngleWaveAware
# ---------------------------------------------------------------------------

class TestWandAngleWaveAware:
    """Tests for wave-aware wand angle computation."""

    @pytest.fixture
    def wand_pivot(self):
        """Typical wand pivot position (bowsprit, forward of CG)."""
        return jnp.array([1.6, 0.0, 0.1])

    @pytest.fixture
    def regular_waves_small(self):
        """Small regular waves (0.1m amplitude, 6s period)."""
        return WaveField.from_params(WaveParams.regular(0.05, 6.0))

    @pytest.fixture
    def regular_waves_large(self):
        """Larger regular waves (0.25m amplitude, 10s period)."""
        return WaveField.from_params(WaveParams.regular(0.25, 10.0))

    @pytest.fixture
    def short_steep_waves(self):
        """Short steep waves (0.15m amplitude, 3s period)."""
        return WaveField.from_params(WaveParams.regular(0.15, 3.0))

    @pytest.fixture
    def jonswap_moderate(self):
        """Moderate JONSWAP spectrum."""
        return WaveField.from_params(WaveParams(
            significant_wave_height=0.3, peak_period=6.0,
            spectrum_type="jonswap", gamma=3.3, num_components=30, seed=42,
        ))

    @pytest.fixture
    def jonswap_steep(self):
        """Steep JONSWAP spectrum."""
        return WaveField.from_params(WaveParams(
            significant_wave_height=0.5, peak_period=4.0,
            spectrum_type="jonswap", gamma=5.0, num_components=30, seed=42,
        ))

    # --- Calm-water equivalence ---

    def test_calm_water_none_matches_direct(self, wand_pivot):
        """wave_field=None exactly matches wand_angle_from_state."""
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.05)
        fwd_speed = jnp.float64(5.0)

        direct = wand_angle_from_state(pos_d, theta, wand_pivot)
        waves = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, None, wand_pivot, DEFAULT_WAND_LENGTH,
        )
        # Should be bit-for-bit identical (delegates directly)
        assert float(waves) == float(direct)

    def test_calm_water_multiple_states(self, wand_pivot):
        """Calm-water equivalence holds across a range of states."""
        for pos_d_val in [-0.8, -0.5, -0.3, -0.1]:
            for theta_val in [-0.05, 0.0, 0.05, 0.1]:
                pos_d = jnp.float64(pos_d_val)
                theta = jnp.float64(theta_val)
                direct = wand_angle_from_state(pos_d, theta, wand_pivot)
                waves = wand_angle_from_state_waves(
                    pos_d, theta, jnp.float64(5.0), 2.0, None,
                    wand_pivot, DEFAULT_WAND_LENGTH,
                )
                assert float(waves) == float(direct), (
                    f"Mismatch at pos_d={pos_d_val}, theta={theta_val}"
                )

    # --- Convergence validation ---

    def test_convergence_regular_waves(self, wand_pivot, regular_waves_small):
        """Default n_iterations achieves <1e-6 rad for regular waves."""
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.0)
        fwd_speed = jnp.float64(5.0)

        # Reference with many iterations
        ref = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, regular_waves_small,
            wand_pivot, DEFAULT_WAND_LENGTH, n_iterations=20,
        )
        # Default (5 iterations)
        result = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, regular_waves_small,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        assert abs(float(result) - float(ref)) < 1e-6

    def test_convergence_short_steep_waves(self, wand_pivot, short_steep_waves):
        """Default n_iterations achieves <1e-6 rad for short steep waves."""
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.0)
        fwd_speed = jnp.float64(5.0)

        ref = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, short_steep_waves,
            wand_pivot, DEFAULT_WAND_LENGTH, n_iterations=20,
        )
        result = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, short_steep_waves,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        assert abs(float(result) - float(ref)) < 1e-6

    def test_convergence_jonswap_steep(self, wand_pivot, jonswap_steep):
        """Default n_iterations achieves <1e-6 rad for steep JONSWAP."""
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.0)
        fwd_speed = jnp.float64(5.0)

        ref = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, jonswap_steep,
            wand_pivot, DEFAULT_WAND_LENGTH, n_iterations=20,
        )
        result = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, jonswap_steep,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        assert abs(float(result) - float(ref)) < 1e-6

    def test_convergence_sweep_typical_conditions(self, wand_pivot, jonswap_moderate):
        """Sweep states and times, all within <1e-6 rad of reference."""
        fwd_speed = jnp.float64(5.0)

        for pos_d_val in [-0.8, -0.5, -0.3]:
            for t in [0.5, 1.0, 2.0, 5.0]:
                pos_d = jnp.float64(pos_d_val)
                theta = jnp.float64(0.0)

                ref = wand_angle_from_state_waves(
                    pos_d, theta, fwd_speed, t, jonswap_moderate,
                    wand_pivot, DEFAULT_WAND_LENGTH, n_iterations=20,
                )
                result = wand_angle_from_state_waves(
                    pos_d, theta, fwd_speed, t, jonswap_moderate,
                    wand_pivot, DEFAULT_WAND_LENGTH,
                )
                err = abs(float(result) - float(ref))
                assert err < 1e-6, (
                    f"Error {err:.2e} at pos_d={pos_d_val}, t={t}"
                )

    # --- Wave effect on angle ---

    def test_waves_change_angle(self, wand_pivot, regular_waves_large):
        """Waves produce a different angle than calm water."""
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.0)
        fwd_speed = jnp.float64(5.0)

        calm = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, None,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        wavy = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, regular_waves_large,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        assert float(calm) != float(wavy)

    # --- Edge cases ---

    def test_wand_saturation_at_surface(self, wand_pivot, regular_waves_small):
        """Wand saturates at pi/2 when pivot is near/below water surface."""
        # Pivot at surface: pos_d=-0.1, pivot_z=0.1 => depth ~ 0
        pos_d = jnp.float64(-0.1)
        theta = jnp.float64(0.0)
        fwd_speed = jnp.float64(5.0)

        result = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, regular_waves_small,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        # Wand should be at or very near horizontal
        np.testing.assert_allclose(float(result), np.pi / 2, atol=0.1)
        assert jnp.isfinite(result)

    def test_wand_saturation_pivot_underwater(self, wand_pivot, regular_waves_large):
        """Wand saturates at pi/2 when pivot is well below surface."""
        pos_d = jnp.float64(0.5)  # very low boat, pivot submerged
        theta = jnp.float64(0.0)
        fwd_speed = jnp.float64(5.0)

        result = wand_angle_from_state_waves(
            pos_d, theta, fwd_speed, 1.0, regular_waves_large,
            wand_pivot, DEFAULT_WAND_LENGTH,
        )
        np.testing.assert_allclose(float(result), np.pi / 2, atol=1e-10)
        assert jnp.isfinite(result)

    def test_no_nan_steep_waves_all_states(self, wand_pivot, short_steep_waves):
        """No NaN produced across range of states with steep waves."""
        fwd_speed = jnp.float64(5.0)
        for pos_d_val in [-1.0, -0.5, -0.1, 0.0, 0.5]:
            for theta_val in [-0.1, 0.0, 0.1]:
                for t in [0.0, 0.5, 1.0, 3.0]:
                    result = wand_angle_from_state_waves(
                        jnp.float64(pos_d_val), jnp.float64(theta_val),
                        fwd_speed, t, short_steep_waves,
                        wand_pivot, DEFAULT_WAND_LENGTH,
                    )
                    assert jnp.isfinite(result), (
                        f"NaN at pos_d={pos_d_val}, theta={theta_val}, t={t}"
                    )
                    # Hull-relative angle: world part in [0, pi/2], minus theta
                    # (§4.6), so the valid range shifts by -theta.
                    assert (
                        -theta_val - 1e-10
                        <= float(result)
                        <= np.pi / 2 - theta_val + 1e-10
                    )

    # --- JIT and grad compatibility ---

    def test_jit_compatible(self, wand_pivot, regular_waves_small):
        """Function works under JIT compilation."""
        @jax.jit
        def f(pos_d, theta, fwd_speed, t):
            return wand_angle_from_state_waves(
                pos_d, theta, fwd_speed, t, regular_waves_small,
                wand_pivot, DEFAULT_WAND_LENGTH,
            )

        result = f(jnp.float64(-0.5), jnp.float64(0.0), jnp.float64(5.0), 1.0)
        assert jnp.isfinite(result)

    def test_grad_compatible(self, wand_pivot, regular_waves_small):
        """Differentiable w.r.t. pos_d with wave field."""
        def f(pos_d):
            return wand_angle_from_state_waves(
                pos_d, jnp.float64(0.0), jnp.float64(5.0), 1.0,
                regular_waves_small, wand_pivot, DEFAULT_WAND_LENGTH,
            )

        grad_val = jax.grad(f)(jnp.float64(-0.5))
        assert jnp.isfinite(grad_val)
        # Wand angle increases as boat sinks (positive gradient)
        assert float(grad_val) > 0

    def test_wand_tip_near_waterline(self, wand_pivot, regular_waves_small):
        """Wand near waterline: angle close to pi/2 but still finite."""
        # Find pos_d where pivot is about wand_length above water
        # (wand nearly vertical, tip just touching surface)
        # depth = pos_d + 0.1 (pivot_z at theta=0)
        # h = -depth, wand tip at surface when h = L => pos_d = -(L + 0.1)
        pos_d_high = jnp.float64(-(DEFAULT_WAND_LENGTH + 0.1))
        result_high = wand_angle_from_state_waves(
            pos_d_high, jnp.float64(0.0), jnp.float64(5.0), 1.0,
            regular_waves_small, wand_pivot, DEFAULT_WAND_LENGTH,
        )
        # Should be near 0 (vertical) in calm water, but waves may shift slightly
        assert jnp.isfinite(result_high)
        assert 0.0 <= float(result_high) < np.pi / 4  # should be small angle

        # Now check wand tip near waterline from low side
        pos_d_low = jnp.float64(-0.15)
        result_low = wand_angle_from_state_waves(
            pos_d_low, jnp.float64(0.0), jnp.float64(5.0), 1.0,
            regular_waves_small, wand_pivot, DEFAULT_WAND_LENGTH,
        )
        assert jnp.isfinite(result_low)
        # Should be near pi/2 (horizontal)
        assert float(result_low) > np.pi / 3


# ---------------------------------------------------------------------------
# TestWandHullFrame (§4.6 hull-fixed wand angle)
# ---------------------------------------------------------------------------

class TestWandHullFrame:
    """Wand angle is hull-relative: theta_w_hull = theta_w_world - theta.

    The mechanical linkage housing is bolted to the hull, so at pitch theta the
    angle that drives the flap differs from the world-vertical geometry by
    exactly theta (longitudinal plane). This restores the pitch-feedback path
    (flap error ~ gain*theta) the world-vertical convention deleted (§4.6).
    """

    @pytest.fixture
    def origin_pivot(self):
        """Pivot at CG origin: pivot depth == pos_d, independent of pitch.

        Isolates the -theta frame term from the pivot-height geometry (which
        otherwise also varies with pitch through the forward moment arm).
        """
        return jnp.array([0.0, 0.0, 0.0])

    @pytest.fixture
    def fwd_pivot(self):
        """Realistic forward wand pivot (bowsprit, forward of CG)."""
        return jnp.array([1.6, 0.0, 0.1])

    def test_zero_pitch_equals_world_primitive(self, origin_pivot):
        """At theta=0 the hull angle equals the world-vertical geometry."""
        for pos_d_val in [-0.8, -0.5, -0.2]:
            pos_d = jnp.float64(pos_d_val)
            angle = wand_angle_from_state(
                pos_d, jnp.float64(0.0), origin_pivot, DEFAULT_WAND_LENGTH
            )
            world = wand_world_angle_from_height(
                -pos_d, DEFAULT_WAND_LENGTH  # h = -depth, depth = pos_d here
            )
            assert abs(float(angle) - float(world)) < 1e-9

    def test_frame_term_is_exactly_minus_pitch(self, origin_pivot):
        """With pivot depth pitch-independent, wand(theta) - wand(0) == -theta.

        Pinpoints the hull-frame term in isolation: a sign flip here (the exact
        §4.6 bug) makes this fail immediately, independent of the linkage.
        """
        pos_d = jnp.float64(-0.5)  # h = 0.5 m, comfortably interior
        base = wand_angle_from_state(
            pos_d, jnp.float64(0.0), origin_pivot, DEFAULT_WAND_LENGTH
        )
        for theta_val in [-0.10, -0.03, 0.03, 0.10]:
            theta = jnp.float64(theta_val)
            angle = wand_angle_from_state(
                pos_d, theta, origin_pivot, DEFAULT_WAND_LENGTH
            )
            assert abs(float(angle - base) - (-theta_val)) < 1e-6, (
                f"frame term wrong at theta={theta_val}"
            )

    def test_flap_differs_by_gain_times_pitch(self, origin_pivot):
        """§6.7: flap at (pos_d, theta) vs (same height, 0) differs by ~gain*theta.

        Same pivot height (origin pivot ⇒ world angle fixed), so the only
        difference between the two states is the -theta hull-frame term, which
        the linkage converts to a flap change of gain * (-theta) to first order.
        """
        linkage = create_wand_linkage()
        pos_d = jnp.float64(-0.5)
        theta = jnp.float64(0.02)  # ~1.1 deg, small enough for linearization

        wand_level = wand_angle_from_state(
            pos_d, jnp.float64(0.0), origin_pivot, DEFAULT_WAND_LENGTH
        )
        wand_pitched = wand_angle_from_state(
            pos_d, theta, origin_pivot, DEFAULT_WAND_LENGTH
        )
        flap_level = linkage.compute(wand_level)
        flap_pitched = linkage.compute(wand_pitched)

        gain = linkage.gain(wand_level)
        expected = float(gain) * (-float(theta))  # = gain * (wand_pitched - wand_level)
        actual = float(flap_pitched - flap_level)
        # First-order prediction; tolerance covers linkage curvature over ~1 deg
        assert abs(actual - expected) < 0.02 * abs(expected) + 1e-6
        # And the effect is real (non-negligible): comparable to gain*theta
        assert abs(actual) > 0.5 * abs(float(gain) * float(theta))

    def test_fixed_depth_wand_responds_to_pitch_via_frame_term(self, fwd_pivot):
        """Gate: at fixed depth, the wand angle carries the -theta path.

        On a realistic forward pivot both effects are present: the world
        geometry (pivot-height moment arm) AND the hull-frame -theta term. Lock
        that the total decomposes exactly as world_angle(theta) - theta, so the
        restored feedback path cannot silently regress.
        """
        from fmd.simulator.components.moth_forces import compute_foil_ned_depth

        pos_d = jnp.float64(-0.5)
        for theta_val in [-0.06, 0.04, 0.09]:
            theta = jnp.float64(theta_val)
            total = wand_angle_from_state(
                pos_d, theta, fwd_pivot, DEFAULT_WAND_LENGTH
            )
            # Reconstruct the world-vertical part from the pitched pivot height
            depth = compute_foil_ned_depth(
                pos_d, float(fwd_pivot[0]), float(fwd_pivot[2]), theta, 0.0
            )
            world = wand_world_angle_from_height(-depth, DEFAULT_WAND_LENGTH)
            assert abs(float(total) - (float(world) - theta_val)) < 1e-6

        # The frame term makes the total pitch-sensitivity differ from the
        # geometry-only sensitivity by ~ -1 rad/rad (the deleted feedback path).
        dth = 1e-4
        th0, th1 = jnp.float64(0.03), jnp.float64(0.03 + dth)
        d_total = float(
            wand_angle_from_state(pos_d, th1, fwd_pivot, DEFAULT_WAND_LENGTH)
            - wand_angle_from_state(pos_d, th0, fwd_pivot, DEFAULT_WAND_LENGTH)
        ) / dth
        d_world = float(
            wand_world_angle_from_height(
                -compute_foil_ned_depth(
                    pos_d, float(fwd_pivot[0]), float(fwd_pivot[2]), th1, 0.0
                ), DEFAULT_WAND_LENGTH)
            - wand_world_angle_from_height(
                -compute_foil_ned_depth(
                    pos_d, float(fwd_pivot[0]), float(fwd_pivot[2]), th0, 0.0
                ), DEFAULT_WAND_LENGTH)
        ) / dth
        assert abs((d_total - d_world) - (-1.0)) < 1e-3

    def test_wave_path_also_hull_relative(self, origin_pivot):
        """The -theta correction applies to the wave-aware path too.

        With the pivot at the origin the world-frame fixed-point solution is
        pitch-independent (pivot depth and pivot NED-north are both independent
        of theta), so the whole pitch dependence of the wave result is the
        hull-frame term: waves(theta) == waves(0) - theta.
        """
        wave_field = WaveField.from_params(WaveParams.regular(0.15, 4.0))
        pos_d = jnp.float64(-0.5)
        fwd_speed = jnp.float64(5.0)
        t = 1.3
        base = wand_angle_from_state_waves(
            pos_d, jnp.float64(0.0), fwd_speed, t, wave_field,
            origin_pivot, DEFAULT_WAND_LENGTH,
        )
        for theta_val in [-0.08, 0.05, 0.11]:
            theta = jnp.float64(theta_val)
            waves = wand_angle_from_state_waves(
                pos_d, theta, fwd_speed, t, wave_field,
                origin_pivot, DEFAULT_WAND_LENGTH,
            )
            assert abs(float(waves - base) - (-theta_val)) < 1e-6, (
                f"wave-path frame term wrong at theta={theta_val}"
            )
