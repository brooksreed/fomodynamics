"""Tests for MeasurementSensor and WandSensor."""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import create_moth_measurement
from fmd.simulator.sensors import MeasurementSensor, WandSensor
from fmd.simulator.components.moth_wand import wand_angle_from_state
from fmd.simulator.environment import Environment
from fmd.simulator.waves import WaveField
from fmd.simulator.params.wave import WaveParams


class TestMeasurementSensor:
    """Tests for MeasurementSensor wrapping, noise, and key splitting."""

    def test_init_state_is_none(self):
        """Basic sensor state is None."""
        meas = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas)
        assert sensor.init_state() is None

    def test_sense_returns_correct_shapes(self):
        """Sense returns (y_noisy, y_clean, None, key_new) with correct shapes."""
        meas = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas)

        x = jnp.array([0.0, 0.0, 0.0, 0.0, 10.0])
        key = jax.random.PRNGKey(42)

        y_noisy, y_clean, state_new, key_new = sensor.sense(x, 0.0, None, None, key)

        assert y_noisy.shape == (5,)
        assert y_clean.shape == (5,)
        assert state_new is None
        assert key_new.shape == key.shape

    def test_clean_measurement_matches_model(self):
        """Clean measurement from sensor matches model.measure() directly."""
        meas = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas)

        x = jnp.array([-1.0, 0.05, 0.1, 0.02, 10.0])
        key = jax.random.PRNGKey(42)

        _, y_clean, _, _ = sensor.sense(x, 0.0, None, None, key)
        y_direct = meas.measure(x, jnp.zeros(2), 0.0)

        np.testing.assert_allclose(y_clean, y_direct, atol=1e-12)

    def test_noisy_differs_from_clean(self):
        """Noisy measurement differs from clean (noise is applied)."""
        meas = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas)

        x = jnp.array([-1.0, 0.05, 0.1, 0.02, 10.0])
        key = jax.random.PRNGKey(42)

        y_noisy, y_clean, _, _ = sensor.sense(x, 0.0, None, None, key)

        # With R = 0.01*I, noise should be present
        assert not np.allclose(y_noisy, y_clean, atol=1e-10)

    def test_different_keys_produce_different_noise(self):
        """Different PRNG keys produce different noisy measurements."""
        meas = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas)

        x = jnp.array([-1.0, 0.05, 0.1, 0.02, 10.0])

        y1, _, _, _ = sensor.sense(x, 0.0, None, None, jax.random.PRNGKey(1))
        y2, _, _, _ = sensor.sense(x, 0.0, None, None, jax.random.PRNGKey(2))

        assert not np.allclose(y1, y2, atol=1e-10)

    def test_key_is_split(self):
        """Returned key differs from input key (key splitting works)."""
        meas = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas)

        x = jnp.array([0.0, 0.0, 0.0, 0.0, 10.0])
        key_in = jax.random.PRNGKey(42)

        _, _, _, key_out = sensor.sense(x, 0.0, None, None, key_in)

        assert not np.array_equal(key_in, key_out)


class TestWandSensor:
    """Tests for WandSensor: calm + wave, noise, key splitting, variants."""

    @pytest.fixture()
    def wand_pivot(self):
        return jnp.array([1.5, 0.0, -0.3])

    @pytest.fixture()
    def state(self):
        """Typical foiling state: boat above water, small pitch."""
        return jnp.array([-0.5, 0.05, 0.0, 0.0, 10.0])

    def _make_sensor(self, wand_pivot, include_speed_pitch=False):
        """Create a WandSensor with appropriate R matrix."""
        if include_speed_pitch:
            R = jnp.diag(jnp.array([0.09, 8e-5, 3e-4]))
        else:
            R = jnp.array([[3e-4]])
        return WandSensor(
            wand_pivot_position=wand_pivot,
            wand_length=1.175,
            R=R,
            include_speed_pitch=include_speed_pitch,
        )

    def test_init_state_is_none(self, wand_pivot):
        """WandSensor has no dynamic state."""
        sensor = self._make_sensor(wand_pivot)
        assert sensor.init_state() is None

    def test_wand_only_calm_water_shape(self, wand_pivot, state):
        """Wand-only sensor returns shape (1,) measurements."""
        sensor = self._make_sensor(wand_pivot, include_speed_pitch=False)
        y_noisy, y_clean, s, key_new = sensor.sense(
            state, 0.0, None, None, jax.random.PRNGKey(42)
        )
        assert y_clean.shape == (1,)
        assert y_noisy.shape == (1,)
        assert s is None

    def test_speed_pitch_wand_shape(self, wand_pivot, state):
        """Speed+pitch+wand sensor returns shape (3,) measurements."""
        sensor = self._make_sensor(wand_pivot, include_speed_pitch=True)
        y_noisy, y_clean, _, _ = sensor.sense(
            state, 0.0, None, None, jax.random.PRNGKey(42)
        )
        assert y_clean.shape == (3,)
        assert y_noisy.shape == (3,)

    def test_calm_water_matches_geometry(self, wand_pivot, state):
        """Calm-water clean measurement matches direct geometry computation."""
        sensor = self._make_sensor(wand_pivot, include_speed_pitch=False)
        _, y_clean, _, _ = sensor.sense(
            state, 0.0, None, None, jax.random.PRNGKey(42)
        )
        expected = wand_angle_from_state(
            state[0], state[1], wand_pivot, 1.175
        )
        np.testing.assert_allclose(y_clean[0], expected, atol=1e-10)

    def test_speed_pitch_values(self, wand_pivot, state):
        """Speed and pitch channels match true state directly."""
        sensor = self._make_sensor(wand_pivot, include_speed_pitch=True)
        _, y_clean, _, _ = sensor.sense(
            state, 0.0, None, None, jax.random.PRNGKey(42)
        )
        np.testing.assert_allclose(y_clean[0], state[4], atol=1e-10)  # speed = u
        np.testing.assert_allclose(y_clean[1], state[1], atol=1e-10)  # pitch = theta

    def test_noisy_differs_from_clean(self, wand_pivot, state):
        """Noise is applied to measurements."""
        sensor = self._make_sensor(wand_pivot)
        y_noisy, y_clean, _, _ = sensor.sense(
            state, 0.0, None, None, jax.random.PRNGKey(42)
        )
        assert not np.allclose(y_noisy, y_clean, atol=1e-10)

    def test_different_keys_different_noise(self, wand_pivot, state):
        """Different PRNG keys produce different noisy measurements."""
        sensor = self._make_sensor(wand_pivot)
        y1, _, _, _ = sensor.sense(state, 0.0, None, None, jax.random.PRNGKey(1))
        y2, _, _, _ = sensor.sense(state, 0.0, None, None, jax.random.PRNGKey(2))
        assert not np.allclose(y1, y2, atol=1e-10)

    def test_key_is_split(self, wand_pivot, state):
        """Returned key differs from input key."""
        sensor = self._make_sensor(wand_pivot)
        key_in = jax.random.PRNGKey(42)
        _, _, _, key_out = sensor.sense(state, 0.0, None, None, key_in)
        assert not np.array_equal(key_in, key_out)

    def test_wave_aware_differs_from_calm(self, wand_pivot, state):
        """With waves, wand angle differs from calm-water computation."""
        sensor = self._make_sensor(wand_pivot)

        wave_params = WaveParams.regular(amplitude=0.3, period=8.0)
        wave_field = WaveField.from_params(wave_params)
        env = Environment(wave_field=wave_field)

        # Use t > 0 to get nonzero wave elevation at some position
        _, y_calm, _, _ = sensor.sense(
            state, 1.0, None, None, jax.random.PRNGKey(42)
        )
        _, y_wave, _, _ = sensor.sense(
            state, 1.0, env, None, jax.random.PRNGKey(42)
        )
        # Wave should change the wand angle
        assert not np.allclose(y_calm[0], y_wave[0], atol=1e-6)

    def test_calm_water_env_without_waves(self, wand_pivot, state):
        """Environment with no wave_field gives same result as None env."""
        sensor = self._make_sensor(wand_pivot)
        env_no_waves = Environment(wave_field=None)

        _, y_none, _, _ = sensor.sense(
            state, 0.0, None, None, jax.random.PRNGKey(42)
        )
        _, y_env, _, _ = sensor.sense(
            state, 0.0, env_no_waves, None, jax.random.PRNGKey(42)
        )
        np.testing.assert_allclose(y_none, y_env, atol=1e-12)
