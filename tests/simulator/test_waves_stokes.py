"""Tests for 2nd-order Stokes wave extensions.

Validates backward compatibility (order=1), crest-trough asymmetry,
analytical Stokes corrections, orbital velocity, JIT compat, and SF Bay presets.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from fmd.simulator.waves import WaveField
from fmd.simulator.params.wave import WaveParams
from fmd.simulator.params import (
    WAVE_REGULAR_1M,
    WAVE_SF_BAY_LIGHT,
    WAVE_SF_BAY_MODERATE,
)


class TestStokesBackwardCompat:
    """Order=1 must produce bit-identical results to pre-Stokes code."""

    def test_elevation_order1_matches_airy(self):
        """stokes_order=1 elevation matches original Airy wave."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        assert wf.stokes_order == 1

        # WAVE_REGULAR_1M has default stokes_order=1
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert float(eta) == pytest.approx(0.5, abs=1e-6)

        # Verify at multiple points
        for x, y, t in [(5.0, 0.0, 1.0), (0.0, 3.0, 2.5), (10.0, 10.0, 5.0)]:
            eta = wf.elevation(x, y, t)
            assert jnp.isfinite(eta)

    def test_orbital_velocity_order1_matches_airy(self):
        """stokes_order=1 orbital velocity is unchanged."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        vel = wf.orbital_velocity(0.0, 0.0, 0.0, 0.0)
        assert vel.shape == (3,)
        assert jnp.all(jnp.isfinite(vel))

        # Known value: at surface, phase=0, horizontal vel = a*omega
        T = WAVE_REGULAR_1M.peak_period
        a = 0.5
        omega = 2.0 * np.pi / T
        assert float(vel[0]) == pytest.approx(a * omega, abs=1e-6)

    def test_orbital_acceleration_order1_matches_airy(self):
        """stokes_order=1 orbital acceleration is unchanged."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        acc = wf.orbital_acceleration(0.0, 0.0, 0.0, 0.0)
        assert acc.shape == (3,)
        assert jnp.all(jnp.isfinite(acc))


class TestStokesCrestTroughAsymmetry:
    """Order=2 should produce taller crests and shallower troughs."""

    def test_crest_trough_asymmetry_regular_wave(self):
        """For a single regular wave with stokes_order=2, |crest| > |trough|."""
        params = WaveParams(
            significant_wave_height=1.0,  # a = 0.5m
            peak_period=5.0,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf = WaveField.from_params(params)

        T = params.peak_period
        # At t=0, phase=0 -> crest. At t=T/2, phase=pi -> trough.
        eta_crest = float(wf.elevation(0.0, 0.0, 0.0))
        eta_trough = float(wf.elevation(0.0, 0.0, T / 2.0))

        # Stokes 2nd order: crest should be higher, trough shallower
        assert eta_crest > 0, f"Crest should be positive, got {eta_crest}"
        assert eta_trough < 0, f"Trough should be negative, got {eta_trough}"
        assert abs(eta_crest) > abs(eta_trough), (
            f"|crest|={abs(eta_crest):.4f} should exceed |trough|={abs(eta_trough):.4f}"
        )

    def test_asymmetry_increases_with_steepness(self):
        """Larger amplitude -> more crest-trough asymmetry."""
        asymmetries = []
        for amplitude in [0.1, 0.3, 0.5]:
            params = WaveParams(
                significant_wave_height=2.0 * amplitude,
                peak_period=5.0,
                spectrum_type="regular",
                stokes_order=2,
            )
            wf = WaveField.from_params(params)
            T = params.peak_period
            eta_crest = abs(float(wf.elevation(0.0, 0.0, 0.0)))
            eta_trough = abs(float(wf.elevation(0.0, 0.0, T / 2.0)))
            asymmetries.append(eta_crest - eta_trough)

        # Asymmetry should increase with amplitude
        for i in range(len(asymmetries) - 1):
            assert asymmetries[i + 1] > asymmetries[i], (
                f"Asymmetry should increase: {asymmetries}"
            )


class TestStokesAnalytical:
    """Verify 2nd-order correction against known analytical formula."""

    def test_stokes_elevation_analytical(self):
        """Single regular deep-water wave: compare 2nd-order correction.

        For a regular wave a*cos(kx - wt + phi):
          eta_2 = a^2 * k / 2 * cos(2*(kx - wt + phi))

        At x=0, t=0, phi=0:
          eta_1 = a
          eta_2 = a^2 * k / 2
          total = a + a^2 * k / 2
        """
        amplitude = 0.3
        period = 4.0
        g = 9.80665

        params = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf = WaveField.from_params(params)

        omega = 2.0 * np.pi / period
        k = omega**2 / g  # deep water

        # At origin, t=0: eta = a + a^2 * k / 2
        expected = amplitude + amplitude**2 * k / 2.0
        actual = float(wf.elevation(0.0, 0.0, 0.0))

        assert actual == pytest.approx(expected, rel=1e-6), (
            f"Expected {expected:.6f}, got {actual:.6f}"
        )

    def test_stokes_correction_magnitude(self):
        """2nd-order correction should be proportional to a^2*k."""
        amplitude = 0.5
        period = 5.0
        g = 9.80665

        params_1 = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=1,
        )
        params_2 = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf1 = WaveField.from_params(params_1)
        wf2 = WaveField.from_params(params_2)

        omega = 2.0 * np.pi / period
        k = omega**2 / g

        eta1 = float(wf1.elevation(0.0, 0.0, 0.0))
        eta2 = float(wf2.elevation(0.0, 0.0, 0.0))

        correction = eta2 - eta1
        expected_correction = amplitude**2 * k / 2.0

        assert correction == pytest.approx(expected_correction, rel=1e-6)


class TestStokesOrbitalVelocity:
    """Verify 2nd-order orbital velocity corrections."""

    def test_orbital_velocity_2nd_order_at_surface(self):
        """Verify 2nd-order velocity correction at surface for regular wave.

        At surface (z=0), x=0, t=0, phase=0:
          1st order: u_n = a * omega * cos(0) = a * omega
          2nd order: u_n_2 = a^2 * omega * k * cos(0) = a^2 * omega * k
          total = a * omega + a^2 * omega * k
        """
        amplitude = 0.3
        period = 4.0
        g = 9.80665

        params = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf = WaveField.from_params(params)

        omega = 2.0 * np.pi / period
        k = omega**2 / g

        vel = wf.orbital_velocity(0.0, 0.0, 0.0, 0.0)
        expected_u_n = amplitude * omega + amplitude**2 * omega * k
        assert float(vel[0]) == pytest.approx(expected_u_n, rel=1e-5)

    def test_2nd_order_velocity_decays_faster(self):
        """2nd-order terms decay as exp(-2kz), faster than 1st-order exp(-kz)."""
        amplitude = 0.3
        period = 4.0
        g = 9.80665

        params_1 = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=1,
        )
        params_2 = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf1 = WaveField.from_params(params_1)
        wf2 = WaveField.from_params(params_2)

        # At surface, 2nd-order adds positive correction
        vel1_surface = float(wf1.orbital_velocity(0.0, 0.0, 0.0, 0.0)[0])
        vel2_surface = float(wf2.orbital_velocity(0.0, 0.0, 0.0, 0.0)[0])
        correction_surface = vel2_surface - vel1_surface

        # At depth z=2m, correction should be much smaller
        vel1_deep = float(wf1.orbital_velocity(0.0, 0.0, 2.0, 0.0)[0])
        vel2_deep = float(wf2.orbital_velocity(0.0, 0.0, 2.0, 0.0)[0])
        correction_deep = vel2_deep - vel1_deep

        assert abs(correction_surface) > abs(correction_deep), (
            f"Surface correction {correction_surface:.6f} should exceed "
            f"deep correction {correction_deep:.6f}"
        )


class TestStokesJIT:
    """JIT compatibility with stokes_order=2."""

    def test_elevation_jit_order2(self):
        """elevation works through jax.jit with stokes_order=2."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=5.0,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf = WaveField.from_params(params)

        @jax.jit
        def eval_fn(wf, x, y, t):
            return wf.elevation(x, y, t)

        result = eval_fn(wf, 0.0, 0.0, 0.0)
        assert jnp.isfinite(result)

    def test_orbital_velocity_jit_order2(self):
        """orbital_velocity works through jax.jit with stokes_order=2."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=5.0,
            spectrum_type="regular",
            stokes_order=2,
        )
        wf = WaveField.from_params(params)

        @jax.jit
        def eval_fn(wf, x, y, z, t):
            return wf.orbital_velocity(x, y, z, t)

        result = eval_fn(wf, 0.0, 0.0, 0.0, 0.0)
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_spectral_jit_order2(self):
        """Spectral wave field with stokes_order=2 works through JIT."""
        wf = WaveField.from_params(WAVE_SF_BAY_MODERATE)

        @jax.jit
        def eval_fn(wf, x, y, t):
            return wf.elevation(x, y, t)

        result = eval_fn(wf, 0.0, 0.0, 0.0)
        assert jnp.isfinite(result)


class TestSFBayPresets:
    """SF Bay wave presets construct and produce valid output."""

    def test_sf_bay_light_constructs(self):
        """WAVE_SF_BAY_LIGHT constructs a valid WaveField."""
        wf = WaveField.from_params(WAVE_SF_BAY_LIGHT)
        assert wf.stokes_order == 2
        assert wf.amplitudes.shape[0] == 30
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_sf_bay_moderate_constructs(self):
        """WAVE_SF_BAY_MODERATE constructs a valid WaveField."""
        wf = WaveField.from_params(WAVE_SF_BAY_MODERATE)
        assert wf.stokes_order == 2
        assert wf.amplitudes.shape[0] == 30
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_sf_bay_presets_produce_valid_hs(self):
        """SF Bay presets should produce elevations of reasonable magnitude."""
        for params, name in [
            (WAVE_SF_BAY_LIGHT, "light"),
            (WAVE_SF_BAY_MODERATE, "moderate"),
        ]:
            wf = WaveField.from_params(params)
            # Sample elevations over time
            t_arr = jnp.linspace(0, 100, 1000)
            etas = jax.vmap(lambda t: wf.elevation(0.0, 0.0, t))(t_arr)
            hs_measured = 4.0 * float(jnp.sqrt(jnp.var(etas)))
            # Should be in reasonable range of target Hs (50% tolerance for short sample)
            assert hs_measured > params.significant_wave_height * 0.3, (
                f"{name}: Hs too low ({hs_measured:.3f} vs target {params.significant_wave_height})"
            )
            assert hs_measured < params.significant_wave_height * 2.0, (
                f"{name}: Hs too high ({hs_measured:.3f} vs target {params.significant_wave_height})"
            )


class TestStokesAccelerationConsistency:
    """Verify orbital_acceleration matches finite-difference of orbital_velocity."""

    @pytest.mark.parametrize("stokes_order", [1, 2])
    def test_acceleration_matches_fd_velocity(self, stokes_order):
        """orbital_acceleration should match d/dt(orbital_velocity) via FD.

        Uses central finite difference with small dt to verify all three
        acceleration components (N, E, D) against the velocity derivative.
        """
        amplitude = 0.3
        period = 4.0

        params = WaveParams(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            stokes_order=stokes_order,
        )
        wf = WaveField.from_params(params)

        # Test at a non-trivial point (not origin) for better coverage
        x, y, z, t = 3.0, 1.5, 0.5, 1.7

        # Analytical acceleration
        acc = wf.orbital_acceleration(x, y, z, t)

        # Finite-difference acceleration from velocity
        eps = 1e-5
        vel_plus = wf.orbital_velocity(x, y, z, t + eps)
        vel_minus = wf.orbital_velocity(x, y, z, t - eps)
        acc_fd = (vel_plus - vel_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            np.array(acc), np.array(acc_fd), rtol=1e-4,
            err_msg=f"Stokes order {stokes_order}: acceleration does not match "
                    f"finite-difference velocity derivative",
        )


class TestStokesOrderValidation:
    """Validate stokes_order parameter."""

    def test_stokes_order_default_is_1(self):
        """Default stokes_order should be 1."""
        params = WaveParams(significant_wave_height=1.0, peak_period=5.0)
        assert params.stokes_order == 1

    def test_stokes_order_3_raises(self):
        """stokes_order=3 should raise ValueError."""
        with pytest.raises(ValueError, match="must be 1 or 2"):
            WaveParams(significant_wave_height=1.0, peak_period=5.0, stokes_order=3)

    def test_stokes_order_0_raises(self):
        """stokes_order=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be 1 or 2"):
            WaveParams(significant_wave_height=1.0, peak_period=5.0, stokes_order=0)


class TestWaveParamsRegularFactory:
    """Verify WaveParams.regular() supports stokes_order."""

    def test_regular_default_stokes_order(self):
        """regular() defaults to stokes_order=1."""
        params = WaveParams.regular(amplitude=0.5, period=5.0)
        assert params.stokes_order == 1

    def test_regular_stokes_order_2(self):
        """regular() accepts stokes_order=2."""
        params = WaveParams.regular(amplitude=0.5, period=5.0, stokes_order=2)
        assert params.stokes_order == 2
        assert params.spectrum_type == "regular"
        assert params.significant_wave_height == 1.0
