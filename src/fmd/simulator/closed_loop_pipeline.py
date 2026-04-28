"""Closed-loop simulation pipeline with Protocol-based interfaces.

Provides a composable simulation pipeline using Sensor, Estimator, and
Controller Protocol interfaces. The pipeline runs a jax.lax.scan loop
with sense -> estimate -> control -> dynamics ordering.

Example:
    from fmd.simulator.closed_loop_pipeline import simulate_closed_loop
    from fmd.simulator.sensors import MeasurementSensor
    from fmd.simulator.estimators import EKFEstimator
    from fmd.simulator.controllers import LQRController

    result = simulate_closed_loop(
        system=moth,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        dt=0.005,
        duration=5.0,
        rng_key=jax.random.PRNGKey(42),
    )
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fmd.simulator.moth_forces_extract import MothForceLog
    from fmd.simulator.params import MothParams

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.integrator import rk4_step, SimulationResult


class ClosedLoopScanResult(NamedTuple):
    """Pure-JAX result from closed-loop scan (no numpy, no force extraction).

    All fields are JAX arrays, suitable for vmap/jit composition.
    """

    times: Array  # (T,)
    true_states: Array  # (T+1, n)
    est_states: Array  # (T+1, n)
    controls: Array  # (T, m)
    covariance_traces: Array  # (T+1,)
    covariance_diagonals: Array  # (T+1, n)
    estimation_errors: Array  # (T+1, n)
    measurements_clean: Array  # (T, p)
    measurements_noisy: Array  # (T, p)
    innovations: Array  # (T, p)


@runtime_checkable
class Sensor(Protocol):
    """Protocol for sensor models in the closed-loop pipeline."""

    def sense(
        self, x_true: Array, t: float, env: Any, sensor_state: Any, key: Array
    ) -> tuple[Array, Array, Any, Array]:
        """Generate noisy and clean measurements from true state.

        Args:
            x_true: True state vector.
            t: Current simulation time.
            env: Optional environment (e.g., wave field).
            sensor_state: Sensor-specific state (e.g., None for basic sensors).
            key: JAX PRNG key.

        Returns:
            Tuple of (y_noisy, y_clean, sensor_state_new, key_new).
        """
        ...

    def init_state(self) -> Any:
        """Return initial sensor state."""
        ...


@runtime_checkable
class Estimator(Protocol):
    """Protocol for state estimators in the closed-loop pipeline."""

    def estimate(
        self,
        est_state: Any,
        y: Array,
        u_prev: Array,
        system: JaxDynamicSystem,
        t: float,
    ) -> tuple[Array, Any, Array]:
        """Update state estimate from measurement and previous control.

        Args:
            est_state: Estimator-specific state (e.g., (x_est, P) for EKF).
            y: Noisy measurement vector.
            u_prev: Control applied at previous timestep.
            system: Dynamic system for prediction.
            t: Current simulation time.

        Returns:
            Tuple of (x_est, est_state_new, innovation).
        """
        ...

    def init_state(self, x0_est: Array, P0: Array) -> Any:
        """Return initial estimator state.

        Args:
            x0_est: Initial state estimate.
            P0: Initial covariance (or similar uncertainty).

        Returns:
            Initial estimator state.
        """
        ...


@runtime_checkable
class Controller(Protocol):
    """Protocol for controllers in the closed-loop pipeline."""

    def control(self, x_est: Array, t: float) -> Array:
        """Compute control input from estimated state.

        Args:
            x_est: State estimate.
            t: Current simulation time.

        Returns:
            Control vector.
        """
        ...


@dataclass
class ClosedLoopResult:
    """Results from a closed-loop simulation.

    Attributes:
        times: Simulation time at each step, shape (n_steps,)
        true_states: True system states, shape (n_steps+1, n)
        est_states: EKF state estimates, shape (n_steps+1, n)
        controls: Applied controls at each step, shape (n_steps, m)
        covariance_traces: Trace of covariance at each step, shape (n_steps+1,)
        covariance_diagonals: Diagonal of covariance, shape (n_steps+1, n)
        estimation_errors: true - estimated states, shape (n_steps+1, n)
        params: Model params used for the simulation (None if not provided)
        force_log: Per-component force breakdown (None if params not provided)
        heel_angle: Heel angle used for geometry calcs (rad)
        measurements_clean: h(x_true), shape (n_steps, p)
        measurements_noisy: y, shape (n_steps, p)
        innovations: y - h(x_pred), shape (n_steps, p)
        trim_state: Trim operating point (None if not provided)
        trim_control: Trim control input (None if not provided)
        lqr_K: LQR gain matrix (None if not provided)
        measurement_output_names: Names of measurement outputs
        measurement_state_index_map: Mapping from output names to state indices
        metadata: Additional controller/estimator-specific data
    """

    times: np.ndarray
    true_states: np.ndarray
    est_states: np.ndarray
    controls: np.ndarray
    covariance_traces: np.ndarray
    covariance_diagonals: np.ndarray
    estimation_errors: np.ndarray
    params: Optional[MothParams] = None
    force_log: Optional[MothForceLog] = None
    heel_angle: float = 0.0
    measurements_clean: Optional[np.ndarray] = None
    measurements_noisy: Optional[np.ndarray] = None
    innovations: Optional[np.ndarray] = None
    trim_state: Optional[np.ndarray] = None
    trim_control: Optional[np.ndarray] = None
    lqr_K: Optional[np.ndarray] = None
    measurement_output_names: Optional[tuple[str, ...]] = None
    measurement_state_index_map: Optional[dict[str, Optional[int]]] = None
    metadata: Optional[dict] = None


def _closed_loop_scan(
    system: JaxDynamicSystem,
    sensor: Sensor,
    estimator: Estimator,
    controller: Controller,
    x0_true: Array,
    x0_est: Array,
    P0: Array,
    dt: float,
    n_steps: int,
    rng_key: Array,
    W_true: Optional[Array] = None,
    env=None,
    u_prev_init: Optional[Array] = None,
    measurement_noise_override: Optional[Array] = None,
) -> ClosedLoopScanResult:
    """JAX-pure closed-loop scan. Returns JAX arrays only.

    This is the inner loop of ``simulate_closed_loop``, extracted so it
    can be vmapped for batched sweeps. No numpy conversion, no force
    extraction, no metadata — pure JAX from input to output.

    Args:
        system: Dynamic system instance.
        sensor: Sensor implementing the Sensor protocol.
        estimator: Estimator implementing the Estimator protocol.
        controller: Controller implementing the Controller protocol.
        x0_true: Initial true state.
        x0_est: Initial estimated state.
        P0: Initial covariance.
        dt: Simulation timestep (seconds).
        n_steps: Number of simulation steps.
        rng_key: JAX PRNG key for noise generation.
        W_true: Optional true process noise covariance.
        env: Optional environment (e.g., wave field).
        u_prev_init: Initial control. If None, inferred from controller.
        measurement_noise_override: Optional pre-generated noise array (n_steps, p).

    Returns:
        ClosedLoopScanResult with pure JAX arrays.
    """
    n = x0_true.shape[0]

    # Initialize sensor and estimator states
    sensor_state = sensor.init_state()
    est_state = estimator.init_state(x0_est, P0)

    # Initialize u_prev
    if u_prev_init is None:
        u_prev_init = controller.control(x0_est, 0.0)

    # Convert noise override to JAX array for use inside scan
    _noise_override = (
        jnp.asarray(measurement_noise_override)
        if measurement_noise_override is not None
        else None
    )

    def step_fn(carry, i):
        x_true, sensor_state_c, est_state_c, u_prev, key = carry
        t = i * dt

        # 1. Sense: generate measurement from true state
        y_noisy, y_clean, sensor_state_new, key = sensor.sense(
            x_true, t, env, sensor_state_c, key
        )

        # 1b. Override measurement noise if provided
        if _noise_override is not None:
            y_noisy = y_clean + _noise_override[i]

        # 2. Estimate: update state estimate
        x_est, est_state_new, innovation = estimator.estimate(
            est_state_c, y_noisy, u_prev, system, t
        )

        # 3. Control: compute control from estimated state
        u = controller.control(x_est, t)

        # 4. Dynamics: propagate true state
        x_true_new = rk4_step(system, x_true, u, dt, t, env=env)

        # 4b. Add process disturbance to true state
        if W_true is not None:
            key, subkey_w = jax.random.split(key)
            w = jax.random.multivariate_normal(subkey_w, jnp.zeros(n), W_true)
            x_true_new = x_true_new + w

        # Extract covariance info from estimator state.
        _, P_new = est_state_new

        new_carry = (x_true_new, sensor_state_new, est_state_new, u, key)
        outputs = (
            x_true_new,
            x_est,
            u,
            jnp.trace(P_new),
            jnp.diag(P_new),
            x_true - x_est,
            y_clean,
            y_noisy,
            innovation,
        )
        return new_carry, outputs

    init_carry = (x0_true, sensor_state, est_state, u_prev_init, rng_key)
    _, (
        true_rest, est_rest, controls,
        traces_rest, diags_rest, errors_rest,
        meas_clean, meas_noisy, innovations,
    ) = jax.lax.scan(step_fn, init_carry, jnp.arange(n_steps))

    # Prepend initial values to get (n_steps+1, ...) arrays
    true_states = jnp.concatenate([x0_true[None], true_rest], axis=0)
    est_states = jnp.concatenate([x0_est[None], est_rest], axis=0)
    cov_traces = jnp.concatenate([jnp.trace(P0)[None], traces_rest], axis=0)
    cov_diags = jnp.concatenate([jnp.diag(P0)[None], diags_rest], axis=0)
    est_errors = jnp.concatenate([(x0_true - x0_est)[None], errors_rest], axis=0)

    times = jnp.arange(n_steps) * dt

    return ClosedLoopScanResult(
        times=times,
        true_states=true_states,
        est_states=est_states,
        controls=controls,
        covariance_traces=cov_traces,
        covariance_diagonals=cov_diags,
        estimation_errors=est_errors,
        measurements_clean=meas_clean,
        measurements_noisy=meas_noisy,
        innovations=innovations,
    )


def simulate_closed_loop(
    system: JaxDynamicSystem,
    sensor: Sensor,
    estimator: Estimator,
    controller: Controller,
    x0_true: Array,
    x0_est: Array,
    P0: Array,
    dt: float,
    duration: float,
    rng_key: jax.random.PRNGKey,
    params=None,
    W_true: Optional[Array] = None,
    env=None,
    measurement_model=None,
    trim_state: Optional[Array] = None,
    trim_control: Optional[Array] = None,
    u_trim: Optional[Array] = None,
    measurement_noise_override: Optional[Array] = None,
) -> ClosedLoopResult:
    """Run closed-loop simulation: sense -> estimate -> control -> dynamics.

    At each timestep:
      1. Sensor: generate noisy measurement from true state
      2. Estimator: update state estimate from measurement + previous control
      3. Controller: compute control from estimated state
      4. Dynamics: propagate true state with RK4

    Uses ``jax.lax.scan`` to compile the entire simulation loop.

    Args:
        system: Dynamic system instance.
        sensor: Sensor implementing the Sensor protocol.
        estimator: Estimator implementing the Estimator protocol.
        controller: Controller implementing the Controller protocol.
        x0_true: Initial true state.
        x0_est: Initial estimated state.
        P0: Initial covariance.
        dt: Simulation timestep (seconds).
        duration: Total simulation duration (seconds).
        rng_key: JAX PRNG key for noise generation.
        params: Optional model params. If provided, force extraction is included.
        W_true: Optional true process noise covariance. If provided,
            Gaussian noise is added to the true state after each RK4 step.
        env: Optional environment (e.g., wave field).
        measurement_model: Optional MeasurementModel for metadata extraction
            (output_names, state_index_map, num_outputs).
        trim_state: Optional trim state for result metadata.
        trim_control: Optional trim control for result metadata.
        u_trim: Initial control (u_prev for first step). If None, uses
            trim_control or zeros.
        measurement_noise_override: Optional pre-generated measurement noise
            array of shape (n_steps, p) where p is the measurement dimension.
            If provided, the noisy measurement is computed as
            ``y_clean + noise_override[i]`` instead of using the sensor's
            random noise. This enables fair cross-config comparisons with
            shared noise realizations.

    Returns:
        ClosedLoopResult with full trajectory data.
    """
    n = len(x0_true)
    if W_true is not None:
        if W_true.shape != (n, n):
            raise ValueError(f"W_true must be ({n}, {n}), got {W_true.shape}")

    n_steps = int(round(duration / dt))

    # Initialize u_prev
    if u_trim is not None:
        u_prev_init = u_trim
    elif trim_control is not None:
        u_prev_init = jnp.asarray(trim_control)
    else:
        u_prev_init = None  # _closed_loop_scan will infer from controller

    scan_result = _closed_loop_scan(
        system=system,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        x0_est=x0_est,
        P0=P0,
        dt=dt,
        n_steps=n_steps,
        rng_key=rng_key,
        W_true=W_true,
        env=env,
        u_prev_init=u_prev_init,
        measurement_noise_override=measurement_noise_override,
    )

    # Extract forces if params provided
    force_log = None
    heel_angle = float(getattr(getattr(system, 'main_foil', None), 'heel_angle', 0.0))
    if params is not None:
        from fmd.simulator.moth_forces_extract import extract_forces

        sim_result = SimulationResult(
            times=np.asarray(scan_result.times),
            states=np.asarray(scan_result.true_states[1:]),
            controls=np.asarray(scan_result.controls),
        )
        force_log = extract_forces(system, sim_result)

    # Extract measurement metadata
    output_names = None
    state_index_map = None
    num_outputs = None
    if measurement_model is not None:
        output_names = measurement_model.output_names
        state_index_map = measurement_model.state_index_map or None
        num_outputs = measurement_model.num_outputs

    return ClosedLoopResult(
        times=np.asarray(scan_result.times),
        true_states=np.asarray(scan_result.true_states),
        est_states=np.asarray(scan_result.est_states),
        controls=np.asarray(scan_result.controls),
        covariance_traces=np.asarray(scan_result.covariance_traces),
        covariance_diagonals=np.asarray(scan_result.covariance_diagonals),
        estimation_errors=np.asarray(scan_result.estimation_errors),
        params=params,
        force_log=force_log,
        heel_angle=heel_angle,
        measurements_clean=np.asarray(scan_result.measurements_clean) if n_steps > 0 else np.empty((0, num_outputs or 0)),
        measurements_noisy=np.asarray(scan_result.measurements_noisy) if n_steps > 0 else np.empty((0, num_outputs or 0)),
        innovations=np.asarray(scan_result.innovations) if n_steps > 0 else np.empty((0, num_outputs or 0)),
        trim_state=np.asarray(trim_state) if trim_state is not None else None,
        trim_control=np.asarray(trim_control) if trim_control is not None else None,
        measurement_output_names=output_names,
        measurement_state_index_map=state_index_map,
    )
