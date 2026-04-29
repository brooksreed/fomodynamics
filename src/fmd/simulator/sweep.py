"""vmap sweep framework for batched simulation.

Provides composable vmap-based batching over parameter variations,
initial conditions, and wave seeds. Three sweep idioms compose freely
via cross-product expansion through ``build_sweep_inputs``.

For JAX-pure controllers (LQR), use ``sweep_closed_loop`` for vmap
speedup. For MPC/CasADi controllers, use ``sweep_closed_loop_sequential``
which falls back to a Python for-loop.

Example:
    from fmd.simulator import Moth3D, sweep_open_loop, stack_systems
    from fmd.simulator.params import MOTH_BIEKER_V3
    import attrs

    # Sweep over 3 different masses
    systems = [
        Moth3D(attrs.evolve(MOTH_BIEKER_V3, hull_mass=m))
        for m in [60.0, 70.0, 80.0]
    ]
    batched_sys = stack_systems(systems)
    ics = jnp.tile(systems[0].default_state(), (3, 1))
    result = sweep_open_loop(batched_sys, ics, dt=0.01, duration=1.0)
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from typing import NamedTuple, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.closed_loop_pipeline import (
    ClosedLoopResult,
    ClosedLoopScanResult,
    Controller,
    Estimator,
    Sensor,
    _closed_loop_scan,
    simulate_closed_loop,
)
from fmd.simulator.environment import Environment
from fmd.simulator.integrator import simulate


# ============================================================================
# Result types
# ============================================================================


class SweepResult(NamedTuple):
    """Batched open-loop simulation results.

    Attributes:
        times: Time points shared across batch, shape (T,).
        states: State trajectories, shape (N, T, n_states).
        controls: Control histories, shape (N, T, n_controls).
    """

    times: Array  # (T,)
    states: Array  # (N, T, n_states)
    controls: Array  # (N, T, n_controls)


class ClosedLoopSweepResult(NamedTuple):
    """Batched closed-loop simulation results.

    Attributes:
        times: Time points, shape (T,) shared across batch.
        true_states: True state trajectories, shape (N, T+1, n).
        est_states: Estimated state trajectories, shape (N, T+1, n).
        controls: Control histories, shape (N, T, m).
        covariance_traces: Covariance traces, shape (N, T+1).
    """

    times: Array  # (T,)
    true_states: Array  # (N, T+1, n)
    est_states: Array  # (N, T+1, n)
    controls: Array  # (N, T, m)
    covariance_traces: Array  # (N, T+1)


class ClosedLoopSweepResultFull(NamedTuple):
    """Batched closed-loop simulation results with all scan fields.

    Extended version of ``ClosedLoopSweepResult`` that includes estimation
    errors, covariance diagonals, measurements, and innovations.

    Attributes:
        times: Time points, shape (T,) shared across batch.
        true_states: True state trajectories, shape (N, T+1, n).
        est_states: Estimated state trajectories, shape (N, T+1, n).
        controls: Control histories, shape (N, T, m).
        covariance_traces: Covariance traces, shape (N, T+1).
        covariance_diagonals: Covariance diagonals, shape (N, T+1, n).
        estimation_errors: Estimation errors, shape (N, T+1, n).
        measurements_clean: Clean measurements, shape (N, T, p).
        measurements_noisy: Noisy measurements, shape (N, T, p).
        innovations: Measurement innovations, shape (N, T, p).
    """

    times: Array  # (T,)
    true_states: Array  # (N, T+1, n)
    est_states: Array  # (N, T+1, n)
    controls: Array  # (N, T, m)
    covariance_traces: Array  # (N, T+1)
    covariance_diagonals: Array  # (N, T+1, n)
    estimation_errors: Array  # (N, T+1, n)
    measurements_clean: Array  # (N, T, p)
    measurements_noisy: Array  # (N, T, p)
    innovations: Array  # (N, T, p)


class SweepLabels(NamedTuple):
    """Index arrays for reshaping flat batch back to sweep grid.

    Attributes:
        param_index: Parameter variation index for each sim, shape (N,).
        ic_index: Initial condition index for each sim, shape (N,).
        seed_index: Wave seed index for each sim, shape (N,).
        shape: Grid shape as (n_params, n_ics, n_seeds).
    """

    param_index: np.ndarray  # (N,)
    ic_index: np.ndarray  # (N,)
    seed_index: np.ndarray  # (N,)
    shape: tuple[int, int, int]  # (n_params, n_ics, n_seeds)


# ============================================================================
# Batching helpers
# ============================================================================


def _get_static_fields(module: eqx.Module) -> dict:
    """Extract static field names and values from an Equinox module.

    Static fields are those marked with ``eqx.field(static=True)``.
    They are part of the module's treedef, not its leaves.
    """
    import dataclasses

    static = {}
    for f in dataclasses.fields(module):
        if f.metadata.get("static", False):
            static[f.name] = getattr(module, f.name)
    return static


def _validate_static_fields(modules: list[eqx.Module], label: str = "module") -> None:
    """Validate that all modules share identical static fields.

    Args:
        modules: List of Equinox modules to validate.
        label: Label for error messages.

    Raises:
        ValueError: If any static field differs across instances.
    """
    if len(modules) < 2:
        return

    ref_statics = _get_static_fields(modules[0])
    for i, m in enumerate(modules[1:], 1):
        m_statics = _get_static_fields(m)
        for key, ref_val in ref_statics.items():
            m_val = m_statics.get(key)
            if ref_val != m_val:
                raise ValueError(
                    f"Static field '{key}' differs across {label} instances: "
                    f"instance 0 has {ref_val!r}, instance {i} has {m_val!r}. "
                    f"All instances must share identical static fields for vmap."
                )


def stack_systems(systems: list[JaxDynamicSystem]) -> JaxDynamicSystem:
    """Stack multiple dynamic systems into a batched pytree for vmap.

    Uses ``jax.tree.map(jnp.stack, ...)`` to stack non-static leaves.
    Validates that all instances share identical static fields.

    Args:
        systems: List of JaxDynamicSystem instances with identical structure.

    Returns:
        A single JaxDynamicSystem with batched array leaves.

    Raises:
        ValueError: If systems have mismatched static fields.
    """
    if not systems:
        raise ValueError("Cannot stack empty list of systems")
    if len(systems) == 1:
        # Add batch dimension of 1
        return jax.tree.map(lambda x: jnp.expand_dims(x, 0), systems[0])

    _validate_static_fields(systems, label="system")
    return jax.tree.map(lambda *xs: jnp.stack(xs), *systems)


def stack_envs(envs: list[Environment]) -> Environment:
    """Stack multiple environments into a batched pytree for vmap.

    All environments must either have ``wave_field=None`` or all have
    a WaveField with matching shapes (same num_components, num_directions).

    Args:
        envs: List of Environment instances.

    Returns:
        A single Environment with batched wave field arrays.

    Raises:
        ValueError: If environments have mixed None/non-None wave fields
            or mismatched static fields.
    """
    if not envs:
        raise ValueError("Cannot stack empty list of environments")
    if len(envs) == 1:
        return jax.tree.map(lambda x: jnp.expand_dims(x, 0), envs[0])

    # Check consistency: all None or all have WaveField
    has_waves = [e.wave_field is not None for e in envs]
    if any(has_waves) and not all(has_waves):
        raise ValueError(
            "Mixed wave_field: some environments have WaveField, others have None. "
            "All must be consistent for vmap batching."
        )

    if all(has_waves):
        # Validate WaveField static fields match
        wave_fields = [e.wave_field for e in envs]
        _validate_static_fields(wave_fields, label="WaveField")

    return jax.tree.map(lambda *xs: jnp.stack(xs), *envs)


def make_ics(base_state: Array, perturbations: Array) -> Array:
    """Create initial conditions from a base state plus perturbations.

    Args:
        base_state: Reference state vector, shape (n_states,).
        perturbations: Perturbation matrix, shape (K, n_states).

    Returns:
        Initial conditions array, shape (K, n_states).
    """
    return base_state[None, :] + perturbations


def build_sweep_inputs(
    systems: Union[JaxDynamicSystem, list[JaxDynamicSystem]],
    ics: Union[Array, list[Array]],
    envs: Union[Environment, list[Environment], None] = None,
) -> tuple[JaxDynamicSystem, Array, Optional[Environment], SweepLabels]:
    """Build cross-product sweep inputs from parameter/IC/env variations.

    Handles cross-product expansion: M params x K ICs x J seeds = N total sims.
    Single inputs (not lists) are broadcast — no variation on that axis.

    Args:
        systems: Single system (broadcast) or list of M systems.
        ics: Single IC array shape (n,) or stacked (K, n).
        envs: None, single Environment, or list of J environments.
            Passing ``None`` or a single ``Environment`` means no
            variation on the seed axis (J=1). Lists of ``None`` are
            collapsed to a single ``None``.

    Returns:
        Tuple of (batched_system, batched_ics, batched_env, labels).
    """
    # Normalize to lists
    if not isinstance(systems, list):
        systems_list = [systems]
    else:
        systems_list = systems

    if isinstance(ics, (list, tuple)):
        ics_array = jnp.stack(ics)
    else:
        ics_array = ics
    # Ensure 2D: (K, n_states)
    if ics_array.ndim == 1:
        ics_array = ics_array[None, :]

    if envs is None:
        envs_list: list[Optional[Environment]] = [None]
    elif not isinstance(envs, list):
        envs_list = [envs]
    else:
        envs_list = envs

    M = len(systems_list)
    K = ics_array.shape[0]
    J = len(envs_list)
    N = M * K * J

    # Build flat lists via cross-product: param (outer) x ic (middle) x seed (inner)
    flat_systems = []
    flat_ics = []
    flat_envs = []
    param_idx = np.empty(N, dtype=np.intp)
    ic_idx = np.empty(N, dtype=np.intp)
    seed_idx = np.empty(N, dtype=np.intp)

    idx = 0
    for m in range(M):
        for k in range(K):
            for j in range(J):
                flat_systems.append(systems_list[m])
                flat_ics.append(ics_array[k])
                if envs_list[j] is not None:
                    flat_envs.append(envs_list[j])
                param_idx[idx] = m
                ic_idx[idx] = k
                seed_idx[idx] = j
                idx += 1

    batched_system = stack_systems(flat_systems)
    batched_ics = jnp.stack(flat_ics)

    batched_env: Optional[Environment] = None
    if flat_envs:
        batched_env = stack_envs(flat_envs)

    labels = SweepLabels(
        param_index=param_idx,
        ic_index=ic_idx,
        seed_index=seed_idx,
        shape=(M, K, J),
    )

    return batched_system, batched_ics, batched_env, labels


# ============================================================================
# Core sweep functions
# ============================================================================


def sweep_open_loop(
    system: JaxDynamicSystem,
    initial_states: Array,
    dt: float,
    duration: float,
    control=None,
    env: Optional[Environment] = None,
) -> SweepResult:
    """Run batched open-loop simulations via vmap.

    Args:
        system: Batched system (from ``stack_systems``) with N-dim leaves.
        initial_states: Batched initial states, shape (N, n_states).
        dt: Timestep (seconds), shared across batch.
        duration: Simulation duration (seconds), shared across batch.
        control: Shared control schedule (not batched). None for default.
        env: Batched environment (N) or None.

    Returns:
        SweepResult with batched trajectories.
    """

    def _sim_one(sys, x0, e):
        return simulate(sys, x0, dt, duration, control=control, env=e)

    if env is not None:
        result = jax.vmap(_sim_one)(system, initial_states, env)
    else:
        # When env is None, we can't vmap over it — use a lambda that ignores it
        def _sim_no_env(sys, x0):
            return simulate(sys, x0, dt, duration, control=control, env=None)

        result = jax.vmap(_sim_no_env)(system, initial_states)

    # result is a SimulationResult with batched arrays
    # times: (N, T) — but all identical, take first
    return SweepResult(
        times=result.times[0],
        states=result.states,
        controls=result.controls,
    )


def sweep_closed_loop(
    system: JaxDynamicSystem,
    sensor: Sensor,
    estimator: Estimator,
    controller: Controller,
    x0_true: Array,
    x0_est: Array,
    P0: Array,
    dt: float,
    duration: float,
    rng_key: Array,
    env: Optional[Environment] = None,
    W_true: Optional[Array] = None,
    u_prev_init: Optional[Array] = None,
    measurement_noise_override: Optional[Array] = None,
) -> ClosedLoopSweepResult:
    """Run batched closed-loop simulations via vmap.

    Uses ``_closed_loop_scan`` internally (JAX-pure). The sensor,
    estimator, and controller are shared (not batched). System, initial
    states, and env are batched over dim 0.

    For MPC/CasADi controllers, use ``sweep_closed_loop_sequential`` instead.

    Args:
        system: Batched system with N-dim leaves.
        sensor: Shared sensor (not batched).
        estimator: Shared estimator (not batched).
        controller: Shared controller (not batched).
        x0_true: Batched true initial states, shape (N, n).
        x0_est: Batched estimated initial states, shape (N, n).
        P0: Shared initial covariance, shape (n, n).
        dt: Timestep (seconds).
        duration: Duration (seconds).
        rng_key: Single PRNG key — split into N keys internally.
        env: Batched environment (N) or None.
        W_true: Shared process noise covariance or None.
        u_prev_init: Shared initial control or None.
        measurement_noise_override: Shared noise override or None.

    Returns:
        ClosedLoopSweepResult with batched trajectories.
    """
    N = x0_true.shape[0]
    n_steps = int(round(duration / dt))
    rng_keys = jax.random.split(rng_key, N)

    def _scan_one(sys, x0_t, x0_e, key, e):
        return _closed_loop_scan(
            system=sys,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_t,
            x0_est=x0_e,
            P0=P0,
            dt=dt,
            n_steps=n_steps,
            rng_key=key,
            W_true=W_true,
            env=e,
            u_prev_init=u_prev_init,
            measurement_noise_override=measurement_noise_override,
        )

    if env is not None:
        scan_results = jax.vmap(
            _scan_one,
            in_axes=(0, 0, 0, 0, 0),
        )(system, x0_true, x0_est, rng_keys, env)
    else:

        def _scan_no_env(sys, x0_t, x0_e, key):
            return _closed_loop_scan(
                system=sys,
                sensor=sensor,
                estimator=estimator,
                controller=controller,
                x0_true=x0_t,
                x0_est=x0_e,
                P0=P0,
                dt=dt,
                n_steps=n_steps,
                rng_key=key,
                W_true=W_true,
                env=None,
                u_prev_init=u_prev_init,
                measurement_noise_override=measurement_noise_override,
            )

        scan_results = jax.vmap(_scan_no_env)(
            system, x0_true, x0_est, rng_keys
        )

    # scan_results is a ClosedLoopScanResult with batched arrays
    return ClosedLoopSweepResult(
        times=scan_results.times[0],  # (T,) shared
        true_states=scan_results.true_states,  # (N, T+1, n)
        est_states=scan_results.est_states,  # (N, T+1, n)
        controls=scan_results.controls,  # (N, T, m)
        covariance_traces=scan_results.covariance_traces,  # (N, T+1)
    )


def sweep_closed_loop_full(
    system: JaxDynamicSystem,
    sensor: Sensor,
    estimator: Estimator,
    controller: Controller,
    x0_true: Array,
    x0_est: Array,
    P0: Array,
    dt: float,
    duration: float,
    rng_key: Array,
    env: Optional[Environment] = None,
    W_true: Optional[Array] = None,
    u_prev_init: Optional[Array] = None,
    measurement_noise_override: Optional[Array] = None,
) -> ClosedLoopSweepResultFull:
    """Run batched closed-loop simulations via vmap, returning all fields.

    Like ``sweep_closed_loop`` but returns the full ``ClosedLoopSweepResultFull``
    including estimation errors, covariance diagonals, measurements, and
    innovations.

    Args:
        system: Batched system with N-dim leaves.
        sensor: Shared sensor (not batched).
        estimator: Shared estimator (not batched).
        controller: Shared controller (not batched).
        x0_true: Batched true initial states, shape (N, n).
        x0_est: Batched estimated initial states, shape (N, n).
        P0: Shared initial covariance, shape (n, n).
        dt: Timestep (seconds).
        duration: Duration (seconds).
        rng_key: Single PRNG key — split into N keys internally.
        env: Batched environment (N) or None.
        W_true: Shared process noise covariance or None.
        u_prev_init: Shared initial control or None.
        measurement_noise_override: Shared noise override or None.

    Returns:
        ClosedLoopSweepResultFull with all scan fields batched.
    """
    N = x0_true.shape[0]
    n_steps = int(round(duration / dt))
    rng_keys = jax.random.split(rng_key, N)

    def _scan_one(sys, x0_t, x0_e, key, e):
        return _closed_loop_scan(
            system=sys,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_t,
            x0_est=x0_e,
            P0=P0,
            dt=dt,
            n_steps=n_steps,
            rng_key=key,
            W_true=W_true,
            env=e,
            u_prev_init=u_prev_init,
            measurement_noise_override=measurement_noise_override,
        )

    if env is not None:
        scan_results = jax.vmap(
            _scan_one,
            in_axes=(0, 0, 0, 0, 0),
        )(system, x0_true, x0_est, rng_keys, env)
    else:

        def _scan_no_env(sys, x0_t, x0_e, key):
            return _closed_loop_scan(
                system=sys,
                sensor=sensor,
                estimator=estimator,
                controller=controller,
                x0_true=x0_t,
                x0_est=x0_e,
                P0=P0,
                dt=dt,
                n_steps=n_steps,
                rng_key=key,
                W_true=W_true,
                env=None,
                u_prev_init=u_prev_init,
                measurement_noise_override=measurement_noise_override,
            )

        scan_results = jax.vmap(_scan_no_env)(
            system, x0_true, x0_est, rng_keys
        )

    return ClosedLoopSweepResultFull(
        times=scan_results.times[0],
        true_states=scan_results.true_states,
        est_states=scan_results.est_states,
        controls=scan_results.controls,
        covariance_traces=scan_results.covariance_traces,
        covariance_diagonals=scan_results.covariance_diagonals,
        estimation_errors=scan_results.estimation_errors,
        measurements_clean=scan_results.measurements_clean,
        measurements_noisy=scan_results.measurements_noisy,
        innovations=scan_results.innovations,
    )


def sweep_closed_loop_sequential(
    system: JaxDynamicSystem,
    sensor: Sensor,
    estimator: Estimator,
    controller: Controller,
    x0_trues: list[Array],
    x0_ests: list[Array],
    P0: Array,
    dt: float,
    duration: float,
    rng_keys: list[Array],
    envs: Optional[list] = None,
    **kwargs,
) -> list[ClosedLoopResult]:
    """Run closed-loop simulations sequentially (fallback for MPC/CasADi).

    Simple for-loop calling ``simulate_closed_loop`` per simulation.
    For JAX-pure controllers, prefer ``sweep_closed_loop`` for vmap speedup.

    Args:
        system: Single (unbatched) system instance.
        sensor: Sensor instance.
        estimator: Estimator instance.
        controller: Controller instance.
        x0_trues: List of true initial states.
        x0_ests: List of estimated initial states.
        P0: Shared initial covariance.
        dt: Timestep (seconds).
        duration: Duration (seconds).
        rng_keys: List of PRNG keys (one per simulation).
        envs: Optional list of environments (one per sim) or None.
        **kwargs: Additional keyword arguments passed to simulate_closed_loop.

    Returns:
        List of ClosedLoopResult instances.
    """
    N = len(x0_trues)
    results = []
    for i in range(N):
        env_i = envs[i] if envs is not None else None
        result = simulate_closed_loop(
            system=system,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_trues[i],
            x0_est=x0_ests[i],
            P0=P0,
            dt=dt,
            duration=duration,
            rng_key=rng_keys[i],
            env=env_i,
            **kwargs,
        )
        results.append(result)
    return results


# ============================================================================
# Metrics
# ============================================================================


def compute_sweep_metrics(
    states: Array,
    trim_state: Array,
    controls: Array,
    pos_d_idx: int = 0,
    theta_idx: int = 1,
) -> dict[str, Array]:
    """Compute performance metrics over a batch of simulations.

    Pure JAX, vmapped over batch dimension. Mirrors the metrics in
    ``moth_metrics.py`` but operates on JAX arrays.

    Args:
        states: State trajectories, shape (N, T, n_states).
            For closed-loop results, pass ``true_states[:, 1:]``.
        trim_state: Trim state vector, shape (n_states,).
        controls: Control histories, shape (N, T, n_controls).
        pos_d_idx: Index of the position state in the state vector.
            Default 0 (Moth3D convention: pos_d).
        theta_idx: Index of the pitch angle state in the state vector.
            Default 1 (Moth3D convention: theta).

    Returns:
        Dict of metric name -> Array of shape (N,).
    """

    def _metrics_one(s, c):
        # s: (T, n_states), c: (T, n_controls)
        pos_d_err = s[:, pos_d_idx] - trim_state[pos_d_idx]
        theta_err = s[:, theta_idx] - trim_state[theta_idx]

        rms_pos_d = jnp.sqrt(jnp.mean(pos_d_err**2))
        rms_theta = jnp.sqrt(jnp.mean(theta_err**2))
        max_pos_d_error = jnp.max(jnp.abs(pos_d_err))

        # Control effort: RMS of control rates
        du = jnp.diff(c, axis=0)
        control_effort = jnp.sqrt(jnp.mean(du**2))

        has_nan = jnp.any(jnp.isnan(s)).astype(jnp.float64)

        return rms_pos_d, rms_theta, max_pos_d_error, control_effort, has_nan

    results = jax.vmap(_metrics_one)(states, controls)

    return {
        "rms_pos_d": results[0],
        "rms_theta": results[1],
        "max_pos_d_error": results[2],
        "control_effort": results[3],
        "has_nan": results[4],
    }


def metrics_to_dataframe(
    metrics: dict[str, Array],
    labels: Optional[SweepLabels] = None,
) -> pd.DataFrame:
    """Convert sweep metrics to a pandas DataFrame.

    Args:
        metrics: Dict of metric name -> Array of shape (N,).
        labels: Optional SweepLabels for adding index columns.

    Returns:
        DataFrame with one row per simulation.
    """
    data = {k: np.asarray(v) for k, v in metrics.items()}
    df = pd.DataFrame(data)

    if labels is not None:
        df["param_index"] = labels.param_index
        df["ic_index"] = labels.ic_index
        df["seed_index"] = labels.seed_index

    return df
