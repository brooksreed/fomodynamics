"""Tests for trajectory generators in fmd.simulator.trajectories."""

import numpy as np
import pytest
import jax.numpy as jnp

from fmd.simulator.trajectories import (
    cartpole_sinusoidal_tracking,
    cartpole_trapezoidal_tracking,
)


class TestCartpoleSinusoidalTracking:
    """Tests for cartpole_sinusoidal_tracking trajectory generator."""

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        num_points = 301
        times, x_refs, u_refs = cartpole_sinusoidal_tracking(
            amplitude=0.5,
            period=3.0,
            duration=6.0,
            num_points=num_points,
        )

        assert times.shape == (num_points,)
        assert x_refs.shape == (num_points, 4)
        assert u_refs.shape == (num_points, 1)

    def test_time_bounds(self):
        """Test that time array spans [0, duration]."""
        duration = 6.0
        times, _, _ = cartpole_sinusoidal_tracking(duration=duration)

        assert float(times[0]) == pytest.approx(0.0)
        assert float(times[-1]) == pytest.approx(duration)

    def test_sinusoidal_position(self):
        """Test that cart position follows sinusoidal pattern."""
        amplitude = 0.5
        period = 3.0
        duration = 6.0
        times, x_refs, _ = cartpole_sinusoidal_tracking(
            amplitude=amplitude,
            period=period,
            duration=duration,
            num_points=301,
        )

        # Check amplitude bounds
        x_positions = np.array(x_refs[:, 0])
        assert np.max(np.abs(x_positions)) <= amplitude + 1e-10

        # Check that position is zero at t=0 (sin(0) = 0)
        assert float(x_refs[0, 0]) == pytest.approx(0.0, abs=1e-10)

        # Check peaks occur at expected times (t = period/4, 3*period/4, etc.)
        times_np = np.array(times)
        quarter_period_idx = np.argmin(np.abs(times_np - period / 4))
        three_quarter_idx = np.argmin(np.abs(times_np - 3 * period / 4))

        # At t=period/4, position should be at +amplitude
        assert float(x_refs[quarter_period_idx, 0]) == pytest.approx(
            amplitude, rel=0.01
        )

        # At t=3*period/4, position should be at -amplitude
        assert float(x_refs[three_quarter_idx, 0]) == pytest.approx(
            -amplitude, rel=0.01
        )

    def test_velocity_consistency(self):
        """Test that velocity is consistent with position derivative."""
        amplitude = 0.5
        period = 3.0
        times, x_refs, _ = cartpole_sinusoidal_tracking(
            amplitude=amplitude,
            period=period,
            duration=6.0,
            num_points=301,
        )

        omega = 2 * np.pi / period
        times_np = np.array(times)

        # Expected velocity: amplitude * omega * cos(omega * t)
        expected_velocity = amplitude * omega * np.cos(omega * times_np)
        actual_velocity = np.array(x_refs[:, 1])

        np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=1e-10)

    def test_pole_stays_upright(self):
        """Test that pole angle and angular velocity are zero (upright)."""
        times, x_refs, _ = cartpole_sinusoidal_tracking()

        # theta (angle) should be zero
        np.testing.assert_allclose(x_refs[:, 2], 0.0, atol=1e-10)

        # theta_dot (angular velocity) should be zero
        np.testing.assert_allclose(x_refs[:, 3], 0.0, atol=1e-10)

    def test_feedforward_control_is_zero(self):
        """Test that feedforward control is zero."""
        times, x_refs, u_refs = cartpole_sinusoidal_tracking()

        np.testing.assert_allclose(u_refs, 0.0, atol=1e-10)

    def test_default_parameters(self):
        """Test that default parameters match specification."""
        times, x_refs, u_refs = cartpole_sinusoidal_tracking()

        # Default: amplitude=0.5, period=3.0, duration=6.0, num_points=301
        assert times.shape == (301,)
        assert float(times[-1]) == pytest.approx(6.0)

        # Max position should be 0.5
        assert np.max(np.abs(x_refs[:, 0])) == pytest.approx(0.5, rel=0.01)

    def test_returns_jax_arrays(self):
        """Test that outputs are JAX arrays."""
        times, x_refs, u_refs = cartpole_sinusoidal_tracking()

        assert isinstance(times, jnp.ndarray)
        assert isinstance(x_refs, jnp.ndarray)
        assert isinstance(u_refs, jnp.ndarray)


class TestCartpoleTrapezoidalTracking:
    """Tests for cartpole_trapezoidal_tracking trajectory generator."""

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        num_points = 301
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=1.0,
            max_velocity=0.5,
            num_points=num_points,
        )

        assert times.shape == (num_points,)
        assert x_refs.shape == (num_points, 4)
        assert u_refs.shape == (num_points, 1)

    def test_round_trip_returns_to_start(self):
        """Test that trajectory returns to origin."""
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=1.0,
            max_velocity=0.5,
            accel_time=0.5,
            num_points=501,  # More points for accuracy
        )

        # Should start near x=0
        assert float(x_refs[0, 0]) == pytest.approx(0.0, abs=0.01)

        # Should end near x=0
        assert float(x_refs[-1, 0]) == pytest.approx(0.0, abs=0.05)

        # Velocity should be near zero at start and end
        assert float(x_refs[0, 1]) == pytest.approx(0.0, abs=0.05)
        assert float(x_refs[-1, 1]) == pytest.approx(0.0, abs=0.05)

    def test_reaches_target_distance(self):
        """Test that trajectory reaches the target distance."""
        distance = 1.0
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=distance,
            max_velocity=0.5,
            num_points=501,
        )

        # Max position should be approximately the target distance
        max_pos = np.max(np.array(x_refs[:, 0]))
        assert max_pos == pytest.approx(distance, rel=0.1)

    def test_velocity_within_bounds(self):
        """Test that velocity stays within max_velocity bounds."""
        max_velocity = 0.5
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=1.0,
            max_velocity=max_velocity,
            num_points=301,
        )

        velocities = np.array(x_refs[:, 1])
        # Allow small overshoot due to smoothing
        assert np.max(np.abs(velocities)) <= max_velocity * 1.1

    def test_theta_follows_quasi_static_physics(self):
        """Test that theta follows the quasi-static approximation during acceleration."""
        g = 9.8
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=1.0,
            max_velocity=0.5,
            accel_time=0.5,
            g=g,
            num_points=301,
        )

        times_np = np.array(times)
        x_refs_np = np.array(x_refs)

        # Compute acceleration from velocity
        x_dot = x_refs_np[:, 1]
        x_ddot = np.gradient(x_dot, times_np)

        # Expected theta from quasi-static: theta = -arctan(x_ddot / g)
        expected_theta = -np.arctan(x_ddot / g)
        actual_theta = x_refs_np[:, 2]

        # Should match reasonably well (not exact due to numerical differentiation)
        np.testing.assert_allclose(actual_theta, expected_theta, atol=0.02)

    def test_theta_near_zero_during_cruise(self):
        """Test that theta is near zero when velocity is constant (cruise phase)."""
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=2.0,  # Larger distance for longer cruise
            max_velocity=0.5,
            accel_time=0.3,
            num_points=501,
        )

        times_np = np.array(times)
        x_refs_np = np.array(x_refs)

        # Find true cruise phase: where velocity is very close to max AND acceleration is near zero
        x_dot = x_refs_np[:, 1]
        x_ddot = np.gradient(x_dot, times_np)

        # Cruise means: velocity near max AND acceleration near zero
        cruise_mask = (np.abs(x_dot - 0.5) < 0.02) & (np.abs(x_ddot) < 0.1)

        if np.any(cruise_mask):
            cruise_theta = x_refs_np[cruise_mask, 2]
            # Theta should be near zero during cruise (constant velocity)
            assert np.max(np.abs(cruise_theta)) < 0.02  # Less than ~1 degree

    def test_feedforward_matches_expected(self):
        """Test that feedforward control matches (m_c + m_p) * x_ddot."""
        mass_cart = 1.0
        mass_pole = 0.1
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=1.0,
            max_velocity=0.5,
            mass_cart=mass_cart,
            mass_pole=mass_pole,
            include_feedforward=True,
            num_points=301,
        )

        times_np = np.array(times)
        x_refs_np = np.array(x_refs)
        u_refs_np = np.array(u_refs)

        # Compute expected feedforward
        x_dot = x_refs_np[:, 1]
        x_ddot = np.gradient(x_dot, times_np)
        expected_ff = (mass_cart + mass_pole) * x_ddot

        # Should match
        np.testing.assert_allclose(u_refs_np.flatten(), expected_ff, atol=0.1)

    def test_returns_jax_arrays(self):
        """Test that outputs are JAX arrays."""
        times, x_refs, u_refs = cartpole_trapezoidal_tracking()

        assert isinstance(times, jnp.ndarray)
        assert isinstance(x_refs, jnp.ndarray)
        assert isinstance(u_refs, jnp.ndarray)

    def test_invalid_distance_raises_error(self):
        """Test that non-positive distance raises ValueError."""
        with pytest.raises(ValueError, match="distance must be positive"):
            cartpole_trapezoidal_tracking(distance=0.0)

        with pytest.raises(ValueError, match="distance must be positive"):
            cartpole_trapezoidal_tracking(distance=-1.0)

    def test_default_parameters(self):
        """Test that default parameters produce valid trajectory."""
        times, x_refs, u_refs = cartpole_trapezoidal_tracking()

        # Default: distance=1.0, max_velocity=0.5, accel_time=0.5, pause_time=0.5
        assert times.shape == (301,)

        # Check position range
        x_positions = np.array(x_refs[:, 0])
        assert np.min(x_positions) >= -0.1  # Should stay near origin or positive
        assert np.max(x_positions) <= 1.1   # Should reach near distance=1.0

    def test_theta_nonzero_during_acceleration(self):
        """Test that theta is non-zero during acceleration phases (key differentiator from sinusoidal)."""
        times, x_refs, u_refs = cartpole_trapezoidal_tracking(
            distance=1.0,
            max_velocity=0.5,
            accel_time=0.5,
            num_points=301,
        )

        times_np = np.array(times)
        x_refs_np = np.array(x_refs)

        # During initial acceleration (first ~0.5s), theta should be negative
        # (pole leans backward when accelerating forward)
        early_indices = times_np < 0.4
        early_theta = x_refs_np[early_indices, 2]

        # Should have some non-zero angles
        max_theta_early = np.max(np.abs(early_theta))
        assert max_theta_early > 0.01  # At least ~0.6 degrees
