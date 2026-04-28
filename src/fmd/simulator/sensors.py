"""Sensor implementations for the closed-loop pipeline.

Provides MeasurementSensor, which wraps a MeasurementModel from
blur.estimation.measurement to produce noisy and clean measurements
with proper PRNG key management.

Also provides WandSensor, a wave-aware sensor that computes wand angle
measurements from true state + wave field.

Example:
    from fmd.estimation import create_moth_measurement
    from fmd.simulator.sensors import MeasurementSensor, WandSensor

    meas_model = create_moth_measurement("speed_pitch_height", ...)
    sensor = MeasurementSensor(measurement_model=meas_model, num_controls=2)
"""

from __future__ import annotations


import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from fmd.estimation.measurement import MeasurementModel
from fmd.simulator.components.moth_wand import wand_angle_from_state_waves
from fmd.simulator.moth_3d import ConstantSchedule

# Moth3D state indices (mirror moth_3d.py)
_POS_D = 0
_THETA = 1
_U = 4


class MeasurementSensor(eqx.Module):
    """Sensor that wraps a MeasurementModel for the closed-loop pipeline.

    Generates noisy measurements via MeasurementModel.noisy_measure()
    and clean measurements via MeasurementModel.measure(). Manages
    JAX PRNG key splitting internally.

    Attributes:
        measurement_model: The underlying MeasurementModel.
        num_controls: Number of control inputs (for dummy u vector).
    """

    measurement_model: MeasurementModel
    num_controls: int = eqx.field(static=True, default=2)

    def sense(
        self, x_true: Array, t: float, env, sensor_state, key: Array
    ) -> tuple[Array, Array, None, Array]:
        """Generate noisy and clean measurements from true state.

        Args:
            x_true: True state vector.
            t: Current simulation time.
            env: Environment (unused by basic measurement sensor).
            sensor_state: Sensor state (unused, always None).
            key: JAX PRNG key.

        Returns:
            Tuple of (y_noisy, y_clean, None, key_new).
        """
        key, subkey = jax.random.split(key)
        # Use a dummy control for measurement (MeasurementModel API requires u)
        u_dummy = jnp.zeros(self.num_controls)
        y_noisy = self.measurement_model.noisy_measure(x_true, u_dummy, subkey, t)
        y_clean = self.measurement_model.measure(x_true, u_dummy, t)
        return y_noisy, y_clean, None, key

    def init_state(self):
        """Return initial sensor state (None for basic sensor)."""
        return None


class WandSensor(eqx.Module):
    """Wave-aware wand angle sensor for the closed-loop pipeline.

    Produces [fwd_speed, pitch, wand_angle] or [wand_angle] depending on
    ``include_speed_pitch``. Uses ``wand_angle_from_state_waves()`` for
    wave-aware computation when an environment with a wave field is provided.

    Attributes:
        wand_pivot_position: Wand pivot [x, y, z] in body frame (m).
        wand_length: Physical wand length from pivot to float (m).
        heel_angle: Static heel angle (rad).
        n_iterations: Fixed-point iterations for wave-aware computation.
        include_speed_pitch: If True, output is [speed, pitch, wand_angle];
            if False, output is [wand_angle].
        R: Measurement noise covariance matrix.
        fwd_speed_func: Schedule eqx.Module with __call__(t) -> forward
            speed (m/s), used for N_pivot computation in wave-aware mode.
            Default: ConstantSchedule(10.0).
    """

    wand_pivot_position: Array
    wand_length: float = eqx.field(static=True)
    R: Array
    heel_angle: float = eqx.field(static=True, default=0.0)
    n_iterations: int = eqx.field(static=True, default=5)
    include_speed_pitch: bool = eqx.field(static=True, default=False)
    fwd_speed_func: ConstantSchedule = ConstantSchedule(10.0)

    def sense(
        self, x_true: Array, t: float, env, sensor_state, key: Array
    ) -> tuple[Array, Array, None, Array]:
        """Generate noisy and clean wand measurements from true state.

        Args:
            x_true: True state vector [pos_d, theta, w, q, u].
            t: Current simulation time.
            env: Environment (may contain wave_field).
            sensor_state: Sensor state (unused, always None).
            key: JAX PRNG key.

        Returns:
            Tuple of (y_noisy, y_clean, None, key_new).
        """
        key, subkey = jax.random.split(key)

        wave_field = env.wave_field if env is not None else None
        wand_angle = wand_angle_from_state_waves(
            x_true[_POS_D], x_true[_THETA], self.fwd_speed_func(t), t,
            wave_field, self.wand_pivot_position, self.wand_length,
            self.heel_angle, self.n_iterations,
        )

        if self.include_speed_pitch:
            y_clean = jnp.array([x_true[_U], x_true[_THETA], wand_angle])
        else:
            y_clean = jnp.array([wand_angle])

        noise = jax.random.multivariate_normal(
            subkey, jnp.zeros(y_clean.shape[0]), self.R
        )
        y_noisy = y_clean + noise
        return y_noisy, y_clean, None, key

    def init_state(self):
        """Return initial sensor state (None — no sensor dynamics)."""
        return None
