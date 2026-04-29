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
                    # Angle should be in valid range
                    assert 0.0 <= float(result) <= np.pi / 2 + 1e-10

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
