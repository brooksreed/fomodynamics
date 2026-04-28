"""Tests for ocean wave field module.

Comprehensive tests for dispersion relation, regular/spectral waves,
JIT compatibility, directional spreading, and parameter validation.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from fmd.simulator.waves import (
    WaveField,
    cos2s_spreading,
    dispersion_relation,
    jonswap_spectrum,
    pierson_moskowitz_spectrum,
)
from fmd.simulator.params import (
    WAVE_CALM,
    WAVE_MODERATE,
    WAVE_REGULAR_1M,
    WaveParams,
)


# ============================================================================
# Dispersion Relation Tests
# ============================================================================


class TestDispersionRelation:
    """Tests for the linear dispersion relation solver."""

    def test_deep_water(self):
        """Deep water: k = omega^2 / g for infinite depth."""
        g = 9.80665
        omega = jnp.array([0.5, 1.0, 2.0, 3.0])
        k = dispersion_relation(omega, depth=float("inf"), g=g)

        expected = omega**2 / g
        assert jnp.allclose(k, expected, atol=1e-10)

    def test_finite_depth_converges(self):
        """Finite depth: result satisfies omega^2 = g * k * tanh(k * h)."""
        g = 9.80665
        h = 10.0
        omega = jnp.array([0.5, 1.0, 1.5, 2.0])
        k = dispersion_relation(omega, depth=h, g=g)

        # Verify the dispersion relation is satisfied
        residual = omega**2 - g * k * jnp.tanh(k * h)
        assert jnp.allclose(residual, 0.0, atol=1e-10)

    def test_shallow_water_limit(self):
        """Shallow water limit: k approaches omega / sqrt(g * h)."""
        g = 9.80665
        h = 0.5  # Shallow depth
        # Use low frequencies where shallow water approximation is valid (kh << 1)
        omega = jnp.array([0.05, 0.1, 0.15])
        k = dispersion_relation(omega, depth=h, g=g)

        # Shallow water: omega^2 = g*k*tanh(kh) ≈ g*k*(kh) = g*k^2*h
        # So k ≈ omega / sqrt(g*h)
        k_shallow = omega / jnp.sqrt(g * h)

        # For very low frequencies in shallow water, should approach this limit
        # Use loose tolerance since we're testing the asymptotic behavior
        assert jnp.allclose(k, k_shallow, rtol=0.1)


# ============================================================================
# Regular Wave Tests
# ============================================================================


class TestRegularWave:
    """Tests for single-component regular (Airy) wave evaluation."""

    def test_amplitude_at_origin(self):
        """Elevation at (0,0,0) with phase=0 should equal amplitude."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        eta = wf.elevation(0.0, 0.0, 0.0)
        # WAVE_REGULAR_1M: amplitude=0.5m, Hs=1.0m
        assert float(eta) == pytest.approx(0.5, abs=1e-6)

    def test_period(self):
        """Elevation at t=T should equal elevation at t=0."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        T = WAVE_REGULAR_1M.peak_period  # 5.0 s

        eta_0 = wf.elevation(0.0, 0.0, 0.0)
        eta_T = wf.elevation(0.0, 0.0, T)
        assert float(eta_T) == pytest.approx(float(eta_0), abs=1e-6)

    def test_half_period(self):
        """Elevation at t=T/2 should be -amplitude."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        T = WAVE_REGULAR_1M.peak_period

        eta_half = wf.elevation(0.0, 0.0, T / 2.0)
        assert float(eta_half) == pytest.approx(-0.5, abs=1e-6)

    def test_wavelength(self):
        """Elevation at (lambda,0,0) should equal elevation at (0,0,0)."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        T = WAVE_REGULAR_1M.peak_period
        g = WAVE_REGULAR_1M.g

        # Deep water wavelength: lambda = g * T^2 / (2 * pi)
        wavelength = g * T**2 / (2.0 * np.pi)

        eta_0 = wf.elevation(0.0, 0.0, 0.0)
        eta_lambda = wf.elevation(wavelength, 0.0, 0.0)
        assert float(eta_lambda) == pytest.approx(float(eta_0), abs=1e-6)

    def test_orbital_velocity_magnitude_at_surface(self):
        """Orbital velocity at surface z=0: horizontal magnitude = a * omega."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        T = WAVE_REGULAR_1M.peak_period
        amplitude = 0.5
        omega = 2.0 * np.pi / T

        # At (0,0,z=0,t=0), phase=0, cos(0)=1 for horizontal, sin(0)=0 for vertical
        vel = wf.orbital_velocity(0.0, 0.0, 0.0, 0.0)

        # Wave propagates in North direction (direction=0), so u_n = a*omega*cos(0) = a*omega
        expected_horizontal = amplitude * omega
        assert float(vel[0]) == pytest.approx(expected_horizontal, abs=1e-6)

    def test_velocity_decays_with_depth(self):
        """Velocity at z=2m should be less than at z=0."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        t = 0.0

        vel_surface = wf.orbital_velocity(0.0, 0.0, 0.0, t)
        vel_deep = wf.orbital_velocity(0.0, 0.0, 2.0, t)

        mag_surface = jnp.linalg.norm(vel_surface)
        mag_deep = jnp.linalg.norm(vel_deep)

        assert float(mag_deep) < float(mag_surface)

    def test_zero_amplitude_gives_zero_elevation(self):
        """Near-zero amplitude should give near-zero elevation."""
        params = WaveParams.regular(amplitude=1e-10, period=5.0)
        wf = WaveField.from_params(params)

        eta = wf.elevation(10.0, 5.0, 3.0)
        assert float(jnp.abs(eta)) < 1e-8

    def test_direction_east(self):
        """Wave propagating East (pi/2) should have zero North velocity at crest."""
        # Wave propagating East: direction = pi/2
        params = WaveParams.regular(amplitude=0.5, period=5.0, direction=np.pi / 2.0)
        wf = WaveField.from_params(params)

        vel = wf.orbital_velocity(0.0, 0.0, 0.0, 0.0)

        # North component should be zero (wave moves East)
        assert float(jnp.abs(vel[0])) < 1e-6
        # East component should be nonzero
        assert float(jnp.abs(vel[1])) > 0.1


# ============================================================================
# Spectral Statistics Tests
# ============================================================================


class TestSpectralStatistics:
    """Tests for spectral sea state statistics."""

    @pytest.mark.slow
    def test_significant_wave_height(self):
        """For WAVE_MODERATE (Hs=1.0m), Hs from timeseries should be approximately correct."""
        wf = WaveField.from_params(WAVE_MODERATE)

        # Generate a long timeseries to get good statistics
        t = jnp.linspace(0, 600, 6000)  # 10 minutes, 0.1s sampling

        @jax.jit
        def eval_elevation(wf, ti):
            return wf.elevation(0.0, 0.0, ti)

        eta = jax.vmap(lambda ti: eval_elevation(wf, ti))(t)

        # Hs = 4 * sqrt(m0), where m0 = variance of elevation
        m0 = jnp.var(eta)
        hs_measured = 4.0 * jnp.sqrt(m0)

        # Stochastic: 30% tolerance
        assert float(hs_measured) == pytest.approx(1.0, rel=0.3)

    @pytest.mark.slow
    def test_mean_elevation_near_zero(self):
        """Mean elevation should be near zero for spectral sea state."""
        wf = WaveField.from_params(WAVE_MODERATE)

        t = jnp.linspace(0, 600, 6000)

        @jax.jit
        def eval_elevation(wf, ti):
            return wf.elevation(0.0, 0.0, ti)

        eta = jax.vmap(lambda ti: eval_elevation(wf, ti))(t)

        mean_eta = float(jnp.mean(eta))
        hs = WAVE_MODERATE.significant_wave_height

        # Mean should be within 10% of Hs from zero
        assert abs(mean_eta) < 0.1 * hs


# ============================================================================
# Directional Spreading Tests
# ============================================================================


class TestDirectionalSpreading:
    """Tests for directional spreading function and multi-directional waves."""

    def test_long_crested_single_direction(self):
        """Long-crested wave (num_dirs=1) has single direction."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
            spectrum_type="jonswap",
            num_directions=1,
            spreading_exponent=0.0,
        )
        wf = WaveField.from_params(params)
        assert wf.directions.shape == (1,)

    def test_directional_wave_field_constructs(self):
        """Creating a directional wave field with spreading works without error."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
            spectrum_type="jonswap",
            num_directions=8,
            spreading_exponent=10.0,
        )
        wf = WaveField.from_params(params)
        assert wf.directions.shape == (8,)
        assert wf.amplitudes.shape[1] == 8

        # Should be able to evaluate without error
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_cos2s_normalization(self):
        """cos2s spreading weights should sum to 1."""
        theta = jnp.linspace(-jnp.pi, jnp.pi, 36, endpoint=False)
        weights = cos2s_spreading(theta, theta_mean=0.0, s=10.0)

        assert float(jnp.sum(weights)) == pytest.approx(1.0, abs=1e-6)

    def test_cos2s_peak_at_mean_direction(self):
        """cos2s spreading should peak at the mean direction."""
        theta = jnp.linspace(-jnp.pi, jnp.pi, 36, endpoint=False)
        weights = cos2s_spreading(theta, theta_mean=0.0, s=10.0)

        # Peak should be at (or very near) theta=0
        peak_idx = int(jnp.argmax(weights))
        assert abs(float(theta[peak_idx])) < 0.2  # Within ~11 degrees of 0


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Tests for reproducibility of wave field generation."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical wave fields."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
            spectrum_type="jonswap",
            seed=42,
        )
        wf1 = WaveField.from_params(params)
        wf2 = WaveField.from_params(params)

        # Same elevations at multiple points
        for x, y, t in [(0.0, 0.0, 0.0), (10.0, 5.0, 3.0), (-5.0, 20.0, 10.0)]:
            eta1 = wf1.elevation(x, y, t)
            eta2 = wf2.elevation(x, y, t)
            assert float(eta1) == pytest.approx(float(eta2), abs=1e-10)

    def test_different_seed_different_result(self):
        """Different seeds should produce different wave fields."""
        params1 = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
            spectrum_type="jonswap",
            seed=42,
        )
        params2 = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
            spectrum_type="jonswap",
            seed=99,
        )
        wf1 = WaveField.from_params(params1)
        wf2 = WaveField.from_params(params2)

        # Should differ at some point (check a few to be safe)
        any_different = False
        for x, y, t in [(0.0, 0.0, 0.0), (10.0, 5.0, 3.0), (-5.0, 20.0, 10.0)]:
            eta1 = float(wf1.elevation(x, y, t))
            eta2 = float(wf2.elevation(x, y, t))
            if abs(eta1 - eta2) > 1e-6:
                any_different = True
                break

        assert any_different, "Different seeds should produce different elevations"


# ============================================================================
# JIT Compatibility Tests
# ============================================================================


class TestJITCompatibility:
    """Tests for JAX JIT compatibility of WaveField methods."""

    def test_elevation_through_jit(self):
        """elevation works through jax.jit."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)

        @jax.jit
        def eval_fn(wf, x, y, t):
            return wf.elevation(x, y, t)

        result = eval_fn(wf, 0.0, 0.0, 0.0)
        assert jnp.isfinite(result)
        assert float(result) == pytest.approx(0.5, abs=1e-6)

    def test_orbital_velocity_through_jit(self):
        """orbital_velocity works through jax.jit."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)

        @jax.jit
        def eval_fn(wf, x, y, z, t):
            return wf.orbital_velocity(x, y, z, t)

        result = eval_fn(wf, 0.0, 0.0, 0.0, 0.0)
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_vmap_over_positions(self):
        """vmap over positions works for elevation."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)

        @jax.jit
        def eval_fn(wf, x, y, t):
            return wf.elevation(x, y, t)

        xs = jnp.linspace(0.0, 100.0, 20)
        ys = jnp.zeros(20)
        ts = jnp.ones(20) * 2.0

        etas = jax.vmap(lambda x, y, t: eval_fn(wf, x, y, t))(xs, ys, ts)

        assert etas.shape == (20,)
        assert jnp.all(jnp.isfinite(etas))


# ============================================================================
# WaveField Construction Tests
# ============================================================================


class TestWaveFieldConstruction:
    """Tests for WaveField construction methods."""

    def test_from_params_jonswap(self):
        """WaveField.from_params works for JONSWAP spectrum."""
        wf = WaveField.from_params(WAVE_CALM)
        assert wf.amplitudes.shape[0] > 0
        assert jnp.all(jnp.isfinite(wf.amplitudes))
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_from_params_pierson_moskowitz(self):
        """WaveField.from_params works for Pierson-Moskowitz spectrum."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
            spectrum_type="pierson_moskowitz",
        )
        wf = WaveField.from_params(params)
        assert wf.amplitudes.shape[0] > 0
        assert jnp.all(jnp.isfinite(wf.amplitudes))
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_from_params_regular(self):
        """WaveField.from_params works for regular wave."""
        wf = WaveField.from_params(WAVE_REGULAR_1M)
        assert wf.amplitudes.shape == (1, 1)
        assert wf.frequencies.shape == (1,)
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_regular_convenience_method(self):
        """WaveField.regular convenience method works."""
        wf = WaveField.regular(amplitude=0.5, period=5.0)
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert float(eta) == pytest.approx(0.5, abs=1e-6)

    def test_regular_convenience_with_direction(self):
        """WaveField.regular with custom direction works."""
        wf = WaveField.regular(amplitude=0.3, period=4.0, direction=np.pi / 4.0)
        assert wf.directions.shape == (1,)
        assert float(wf.directions[0]) == pytest.approx(np.pi / 4.0, abs=1e-6)
        eta = wf.elevation(0.0, 0.0, 0.0)
        assert jnp.isfinite(eta)

    def test_regular_factory_params(self):
        """WaveParams.regular factory produces valid params."""
        params = WaveParams.regular(amplitude=0.5, period=5.0)
        assert params.num_components == 1
        assert params.spectrum_type == "regular"
        assert params.significant_wave_height == pytest.approx(1.0)  # 2 * amplitude
        assert params.peak_period == 5.0


# ============================================================================
# WaveParams Validation Tests
# ============================================================================


class TestWaveParams:
    """Tests for WaveParams parameter validation."""

    def test_valid_construction(self):
        """Valid params should construct without error."""
        params = WaveParams(
            significant_wave_height=1.0,
            peak_period=6.0,
        )
        assert params.significant_wave_height == 1.0
        assert params.peak_period == 6.0

    def test_negative_hs_raises(self):
        """Negative significant wave height should raise ValueError."""
        with pytest.raises(ValueError):
            WaveParams(significant_wave_height=-1.0, peak_period=6.0)

    def test_zero_hs_raises(self):
        """Zero significant wave height should raise ValueError."""
        with pytest.raises(ValueError):
            WaveParams(significant_wave_height=0.0, peak_period=6.0)

    def test_negative_tp_raises(self):
        """Negative peak period should raise ValueError."""
        with pytest.raises(ValueError):
            WaveParams(significant_wave_height=1.0, peak_period=-6.0)

    def test_zero_tp_raises(self):
        """Zero peak period should raise ValueError."""
        with pytest.raises(ValueError):
            WaveParams(significant_wave_height=1.0, peak_period=0.0)

    def test_gamma_must_be_greater_than_one(self):
        """Gamma <= 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 1.0"):
            WaveParams(significant_wave_height=1.0, peak_period=6.0, gamma=1.0)

        with pytest.raises(ValueError, match="must be > 1.0"):
            WaveParams(significant_wave_height=1.0, peak_period=6.0, gamma=0.5)

    def test_num_components_lower_bound(self):
        """num_components < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\[1, 200\\]"):
            WaveParams(significant_wave_height=1.0, peak_period=6.0, num_components=0)

    def test_num_components_upper_bound(self):
        """num_components > 200 should raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\[1, 200\\]"):
            WaveParams(significant_wave_height=1.0, peak_period=6.0, num_components=201)

    def test_num_components_valid_bounds(self):
        """num_components at boundaries should work."""
        p1 = WaveParams(significant_wave_height=1.0, peak_period=6.0, num_components=1)
        assert p1.num_components == 1

        p200 = WaveParams(significant_wave_height=1.0, peak_period=6.0, num_components=200)
        assert p200.num_components == 200

    def test_regular_factory(self):
        """Regular factory should set num_components=1 and spectrum_type='regular'."""
        params = WaveParams.regular(amplitude=0.5, period=5.0)
        assert params.num_components == 1
        assert params.spectrum_type == "regular"
        assert params.significant_wave_height == pytest.approx(1.0)  # 2 * 0.5

    def test_invalid_spectrum_type_raises(self):
        """Invalid spectrum type should raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            WaveParams(
                significant_wave_height=1.0,
                peak_period=6.0,
                spectrum_type="invalid",
            )
