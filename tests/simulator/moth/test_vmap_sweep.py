"""Tests for vmap sweep framework.

Tests batched simulation via vmap, including stack helpers,
open-loop and closed-loop sweeps, and metrics computation.

All tests use short sims: dt=0.01, duration=0.5, small batches (3-5).
"""

from fmd.simulator import _config  # noqa: F401

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import create_moth_measurement, ExtendedKalmanFilter
from fmd.simulator import (
    ClosedLoopSweepResult,
    ConstantSchedule,
    Environment,
    Moth3D,
    SweepLabels,
    SweepResult,
    simulate,
    simulate_closed_loop,
)
from fmd.simulator.controllers import LQRController
from fmd.simulator.estimators import EKFEstimator
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.moth_metrics import compute_metrics
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.params.wave import WaveParams
from fmd.simulator.sensors import MeasurementSensor
from fmd.simulator.sweep import (
    build_sweep_inputs,
    compute_sweep_metrics,
    make_ics,
    metrics_to_dataframe,
    stack_envs,
    stack_systems,
    sweep_closed_loop,
    sweep_closed_loop_sequential,
    sweep_open_loop,
)

# Short sim parameters for fast tests
DT = 0.01
DURATION = 0.5
N_STEPS = int(round(DURATION / DT))


@pytest.fixture(scope="module")
def base_moth():
    """A standard Moth3D for testing."""
    return Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))


@pytest.fixture(scope="module")
def lqr_setup():
    """Shared LQR design + closed-loop pipeline components."""
    lqr_result = design_moth_lqr(u_forward=10.0)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))

    meas_model = create_moth_measurement("full_state")
    sensor = MeasurementSensor(measurement_model=meas_model, num_controls=2)

    Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
    ekf = ExtendedKalmanFilter(dt=DT)
    estimator = EKFEstimator(
        ekf=ekf, measurement_model=meas_model, Q_ekf=Q_ekf, num_controls=2
    )

    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    controller = LQRController(
        K=jnp.array(lqr_result.K),
        x_trim=jnp.array(lqr_result.trim.state),
        u_trim=jnp.array(lqr_result.trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    P0 = jnp.eye(5) * 0.01

    return moth, sensor, estimator, controller, lqr_result, P0


# ============================================================================
# Helpers (unit tests)
# ============================================================================


class TestStackSystems:
    """Tests for stack_systems helper."""

    def test_stack_systems_produces_batched_pytree(self):
        """3 Moth3D with different masses produce batched leaf shapes."""
        systems = []
        for m in [60.0, 65.0, 70.0]:
            p = attrs.evolve(MOTH_BIEKER_V3, hull_mass=m)
            systems.append(Moth3D(p, u_forward=ConstantSchedule(10.0)))

        batched = stack_systems(systems)

        # total_mass should be a (3,) array
        assert batched.total_mass.shape == (3,)
        np.testing.assert_allclose(
            batched.total_mass,
            [60.0 + MOTH_BIEKER_V3.sailor_mass,
             65.0 + MOTH_BIEKER_V3.sailor_mass,
             70.0 + MOTH_BIEKER_V3.sailor_mass],
        )
        # Static fields remain single values
        assert batched.state_names == ("pos_d", "theta", "w", "q", "u")
        assert isinstance(batched, Moth3D)

    def test_stack_systems_rejects_mismatched_statics(self):
        """Different surge_enabled raises ValueError."""
        s1 = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0), surge_enabled=True)
        s2 = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0), surge_enabled=False)

        with pytest.raises(ValueError, match="surge_enabled"):
            stack_systems([s1, s2])


class TestStackEnvs:
    """Tests for stack_envs helper."""

    def test_stack_envs_produces_batched_wavefield(self):
        """3 environments with different seeds produce batched arrays."""
        envs = []
        for seed in [42, 43, 44]:
            wp = WaveParams(
                significant_wave_height=0.3,
                peak_period=3.0,
                seed=seed,
                num_components=8,
                num_directions=4,
            )
            envs.append(Environment.with_waves(wp))

        batched = stack_envs(envs)

        assert batched.wave_field is not None
        # phases should be (3, N, M) — 3 envs batched
        assert batched.wave_field.phases.shape[0] == 3
        assert batched.wave_field.phases.shape[1] == 8  # num_components
        assert batched.wave_field.phases.shape[2] == 4  # num_directions


class TestMakeIcs:
    """Tests for make_ics helper."""

    def test_make_ics(self, base_moth):
        """Verify shape from base + perturbations."""
        base = base_moth.default_state()
        perturbations = jnp.array([
            [0.01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0, 0.0],
            [-0.01, 0.0, 0.0, 0.0, 0.0],
        ])

        ics = make_ics(base, perturbations)

        assert ics.shape == (3, 5)
        # First IC should be base + first perturbation
        np.testing.assert_allclose(ics[0], base + perturbations[0])


class TestBuildSweepInputs:
    """Tests for build_sweep_inputs helper."""

    def test_build_sweep_inputs_cross_product(self):
        """2x3x2=12 total, verify shapes and label indices."""
        # 2 parameter variations
        systems = [
            Moth3D(attrs.evolve(MOTH_BIEKER_V3, hull_mass=m), u_forward=ConstantSchedule(10.0))
            for m in [60.0, 70.0]
        ]

        # 3 ICs
        base = systems[0].default_state()
        ics = jnp.stack([
            base + jnp.array([0.01, 0.0, 0.0, 0.0, 0.0]),
            base + jnp.array([0.0, 0.01, 0.0, 0.0, 0.0]),
            base + jnp.array([-0.01, 0.0, 0.0, 0.0, 0.0]),
        ])

        # 2 wave seeds
        envs = []
        for seed in [42, 43]:
            wp = WaveParams(
                significant_wave_height=0.3,
                peak_period=3.0,
                seed=seed,
                num_components=8,
                num_directions=4,
            )
            envs.append(Environment.with_waves(wp))

        batched_sys, batched_ics, batched_env, labels = build_sweep_inputs(
            systems, ics, envs
        )

        N = 2 * 3 * 2  # = 12
        assert batched_ics.shape == (N, 5)
        assert batched_sys.total_mass.shape == (N,)
        assert batched_env is not None
        assert batched_env.wave_field.phases.shape[0] == N

        assert labels.shape == (2, 3, 2)
        assert len(labels.param_index) == N
        assert len(labels.ic_index) == N
        assert len(labels.seed_index) == N

        # Verify cross-product ordering: param outer, ic middle, seed inner
        assert labels.param_index[0] == 0
        assert labels.ic_index[0] == 0
        assert labels.seed_index[0] == 0
        assert labels.seed_index[1] == 1  # second sim varies seed
        assert labels.ic_index[2] == 1  # third sim is next IC


# ============================================================================
# Open-loop integration tests
# ============================================================================


class TestSweepOpenLoop:
    """Tests for sweep_open_loop."""

    def test_sweep_open_loop_ic_only(self, base_moth):
        """3 ICs, verify result shape (3, T, 5) and states differ."""
        x0 = base_moth.default_state()
        ics = jnp.stack([
            x0 + jnp.array([0.02, 0.0, 0.0, 0.0, 0.0]),
            x0 + jnp.array([0.0, 0.02, 0.0, 0.0, 0.0]),
            x0 + jnp.array([-0.02, 0.0, 0.0, 0.0, 0.0]),
        ])

        # Stack same system 3 times for IC sweep
        batched = stack_systems([base_moth] * 3)
        result = sweep_open_loop(batched, ics, dt=DT, duration=DURATION)

        assert isinstance(result, SweepResult)
        T = N_STEPS + 1  # simulate returns T+1 states (initial + N steps)
        assert result.states.shape == (3, T, 5)
        assert result.controls.shape[0] == 3
        assert result.times.shape[0] == T

        # States should differ due to different ICs
        assert not jnp.allclose(result.states[0], result.states[1])
        assert not jnp.allclose(result.states[0], result.states[2])

    def test_sweep_open_loop_matches_sequential(self, base_moth):
        """3 sims via sweep vs 3 sequential simulate() calls match."""
        x0 = base_moth.default_state()
        ics = [
            x0 + jnp.array([0.02, 0.0, 0.0, 0.0, 0.0]),
            x0 + jnp.array([0.0, 0.02, 0.0, 0.0, 0.0]),
            x0 + jnp.array([-0.02, 0.0, 0.0, 0.0, 0.0]),
        ]

        # Sequential
        seq_results = [
            simulate(base_moth, ic, DT, DURATION)
            for ic in ics
        ]

        # Batched
        batched = stack_systems([base_moth] * 3)
        batched_ics = jnp.stack(ics)
        sweep_result = sweep_open_loop(batched, batched_ics, dt=DT, duration=DURATION)

        for i in range(3):
            np.testing.assert_allclose(
                sweep_result.states[i],
                seq_results[i].states,
                atol=1e-10,
                err_msg=f"Sweep vs sequential mismatch for sim {i}",
            )

    def test_sweep_open_loop_param_variation(self):
        """3 masses, verify results differ."""
        systems = []
        for m in [60.0, 65.0, 70.0]:
            p = attrs.evolve(MOTH_BIEKER_V3, hull_mass=m)
            systems.append(Moth3D(p, u_forward=ConstantSchedule(10.0)))

        batched = stack_systems(systems)
        x0 = systems[0].default_state()
        ics = jnp.tile(x0, (3, 1))

        result = sweep_open_loop(batched, ics, dt=DT, duration=DURATION)

        assert result.states.shape[0] == 3
        # Different masses should produce different trajectories
        assert not jnp.allclose(result.states[0], result.states[1])
        assert not jnp.allclose(result.states[0], result.states[2])

    def test_sweep_open_loop_wave_seeds(self, base_moth):
        """3 wave seeds, verify results differ."""
        envs = []
        for seed in [42, 43, 44]:
            wp = WaveParams(
                significant_wave_height=0.3,
                peak_period=3.0,
                seed=seed,
                num_components=8,
                num_directions=4,
            )
            envs.append(Environment.with_waves(wp))

        batched_sys = stack_systems([base_moth] * 3)
        batched_env = stack_envs(envs)
        x0 = base_moth.default_state()
        ics = jnp.tile(x0, (3, 1))

        result = sweep_open_loop(
            batched_sys, ics, dt=DT, duration=DURATION, env=batched_env
        )

        assert result.states.shape[0] == 3
        # Different wave seeds should produce different trajectories
        assert not jnp.allclose(result.states[0], result.states[1])
        assert not jnp.allclose(result.states[0], result.states[2])

    def test_sweep_open_loop_composed(self):
        """2x2x2=8 sims, verify shapes."""
        systems = [
            Moth3D(attrs.evolve(MOTH_BIEKER_V3, hull_mass=m), u_forward=ConstantSchedule(10.0))
            for m in [60.0, 70.0]
        ]

        base = systems[0].default_state()
        ics = jnp.stack([
            base + jnp.array([0.02, 0.0, 0.0, 0.0, 0.0]),
            base + jnp.array([-0.02, 0.0, 0.0, 0.0, 0.0]),
        ])

        envs = []
        for seed in [42, 43]:
            wp = WaveParams(
                significant_wave_height=0.3,
                peak_period=3.0,
                seed=seed,
                num_components=8,
                num_directions=4,
            )
            envs.append(Environment.with_waves(wp))

        batched_sys, batched_ics, batched_env, labels = build_sweep_inputs(
            systems, ics, envs
        )

        result = sweep_open_loop(
            batched_sys, batched_ics, dt=DT, duration=DURATION, env=batched_env
        )

        N = 2 * 2 * 2  # = 8
        T = N_STEPS + 1
        assert result.states.shape == (N, T, 5)
        assert labels.shape == (2, 2, 2)


# ============================================================================
# Closed-loop integration tests
# ============================================================================


class TestSweepClosedLoop:
    """Tests for sweep_closed_loop."""

    def test_sweep_closed_loop_wave_seeds(self, lqr_setup):
        """3 wave seeds with LQR, verify shapes and results differ."""
        moth, sensor, estimator, controller, lqr_result, P0 = lqr_setup
        trim = lqr_result.trim

        envs = []
        for seed in [42, 43, 44]:
            wp = WaveParams(
                significant_wave_height=0.3,
                peak_period=3.0,
                seed=seed,
                num_components=8,
                num_directions=4,
            )
            envs.append(Environment.with_waves(wp))

        N = 3
        batched_sys = stack_systems([moth] * N)
        batched_env = stack_envs(envs)

        x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.02)
        x0_est = jnp.array(trim.state)

        result = sweep_closed_loop(
            system=batched_sys,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=jnp.tile(x0_true, (N, 1)),
            x0_est=jnp.tile(x0_est, (N, 1)),
            P0=P0,
            dt=DT,
            duration=DURATION,
            rng_key=jax.random.PRNGKey(0),
            env=batched_env,
            u_prev_init=jnp.array(trim.control),
        )

        assert isinstance(result, ClosedLoopSweepResult)
        assert result.true_states.shape == (N, N_STEPS + 1, 5)
        assert result.est_states.shape == (N, N_STEPS + 1, 5)
        assert result.controls.shape == (N, N_STEPS, 2)
        assert result.covariance_traces.shape == (N, N_STEPS + 1)
        assert result.times.shape == (N_STEPS,)

        # Different wave seeds should produce different trajectories
        assert not jnp.allclose(result.true_states[0], result.true_states[1])

    def test_sweep_closed_loop_matches_sequential(self, lqr_setup):
        """Compare vmapped vs sequential simulate_closed_loop on true_states."""
        moth, sensor, estimator, controller, lqr_result, P0 = lqr_setup
        trim = lqr_result.trim

        N = 3
        x0_true_base = jnp.array(trim.state).at[0].set(trim.state[0] + 0.02)
        x0_est = jnp.array(trim.state)
        u_prev = jnp.array(trim.control)

        # Different perturbations for each sim
        perturbations = jnp.array([
            [0.01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0, 0.0],
            [-0.01, 0.0, 0.0, 0.0, 0.0],
        ])
        x0_trues = x0_true_base[None, :] + perturbations

        rng_key = jax.random.PRNGKey(99)
        rng_keys = jax.random.split(rng_key, N)

        # Sequential
        seq_results = []
        for i in range(N):
            r = simulate_closed_loop(
                system=moth,
                sensor=sensor,
                estimator=estimator,
                controller=controller,
                x0_true=x0_trues[i],
                x0_est=x0_est,
                P0=P0,
                dt=DT,
                duration=DURATION,
                rng_key=rng_keys[i],
                u_trim=u_prev,
            )
            seq_results.append(r)

        # Batched
        batched_sys = stack_systems([moth] * N)
        sweep_result = sweep_closed_loop(
            system=batched_sys,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_trues,
            x0_est=jnp.tile(x0_est, (N, 1)),
            P0=P0,
            dt=DT,
            duration=DURATION,
            rng_key=rng_key,
            u_prev_init=u_prev,
        )

        for i in range(N):
            np.testing.assert_allclose(
                sweep_result.true_states[i],
                seq_results[i].true_states,
                atol=1e-8,
                err_msg=f"Sweep vs sequential mismatch for sim {i}",
            )


# ============================================================================
# Metrics tests
# ============================================================================


class TestMetrics:
    """Tests for compute_sweep_metrics and metrics_to_dataframe."""

    def test_compute_sweep_metrics_shapes(self, base_moth):
        """Verify each metric has shape (N,)."""
        N = 3
        systems = [base_moth] * N
        batched = stack_systems(systems)

        x0 = base_moth.default_state()
        ics = jnp.stack([
            x0 + jnp.array([0.02, 0.0, 0.0, 0.0, 0.0]),
            x0 + jnp.array([0.0, 0.02, 0.0, 0.0, 0.0]),
            x0 + jnp.array([-0.02, 0.0, 0.0, 0.0, 0.0]),
        ])

        result = sweep_open_loop(batched, ics, dt=DT, duration=DURATION)

        # Use states[1:] to skip initial (matching closed-loop convention)
        metrics = compute_sweep_metrics(
            states=result.states[:, 1:, :],
            trim_state=x0,
            controls=result.controls[:, 1:, :],  # skip initial control too
        )

        assert set(metrics.keys()) == {
            "rms_pos_d", "rms_theta", "max_pos_d_error",
            "control_effort", "has_nan",
        }
        for name, val in metrics.items():
            assert val.shape == (N,), f"Metric {name} has wrong shape: {val.shape}"

    def test_metrics_to_dataframe(self):
        """Verify DataFrame columns and row count."""
        N = 3
        # Create dummy metrics
        metrics = {
            "rms_pos_d": jnp.array([0.01, 0.02, 0.03]),
            "rms_theta": jnp.array([0.001, 0.002, 0.003]),
            "max_pos_d_error": jnp.array([0.05, 0.06, 0.07]),
            "control_effort": jnp.array([0.1, 0.2, 0.3]),
            "has_nan": jnp.array([0.0, 0.0, 0.0]),
        }

        labels = SweepLabels(
            param_index=np.array([0, 0, 0]),
            ic_index=np.array([0, 1, 2]),
            seed_index=np.array([0, 0, 0]),
            shape=(1, 3, 1),
        )

        df = metrics_to_dataframe(metrics, labels)

        assert len(df) == N
        assert "rms_pos_d" in df.columns
        assert "rms_theta" in df.columns
        assert "param_index" in df.columns
        assert "ic_index" in df.columns
        assert "seed_index" in df.columns

        # Without labels
        df_no_labels = metrics_to_dataframe(metrics)
        assert len(df_no_labels) == N
        assert "param_index" not in df_no_labels.columns

    def test_metrics_match_sequential(self, lqr_setup):
        """JAX metrics vs numpy compute_metrics() on individual sims."""
        moth, sensor, estimator, controller, lqr_result, P0 = lqr_setup
        trim = lqr_result.trim

        N = 3
        x0_true_base = jnp.array(trim.state).at[0].set(trim.state[0] + 0.02)
        x0_est = jnp.array(trim.state)
        u_prev = jnp.array(trim.control)

        perturbations = jnp.array([
            [0.01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0, 0.0],
            [-0.01, 0.0, 0.0, 0.0, 0.0],
        ])
        x0_trues = x0_true_base[None, :] + perturbations

        rng_key = jax.random.PRNGKey(42)
        rng_keys = jax.random.split(rng_key, N)

        # Run sequential sims and compute numpy metrics
        seq_results = []
        numpy_metrics = []
        for i in range(N):
            r = simulate_closed_loop(
                system=moth,
                sensor=sensor,
                estimator=estimator,
                controller=controller,
                x0_true=x0_trues[i],
                x0_est=x0_est,
                P0=P0,
                dt=DT,
                duration=DURATION,
                rng_key=rng_keys[i],
                u_trim=u_prev,
            )
            seq_results.append(r)
            numpy_metrics.append(compute_metrics(r, np.array(trim.state)))

        # Run batched sweep
        batched_sys = stack_systems([moth] * N)
        sweep_result = sweep_closed_loop(
            system=batched_sys,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_trues,
            x0_est=jnp.tile(x0_est, (N, 1)),
            P0=P0,
            dt=DT,
            duration=DURATION,
            rng_key=rng_key,
            u_prev_init=u_prev,
        )

        # Compute JAX metrics
        jax_metrics = compute_sweep_metrics(
            states=sweep_result.true_states[:, 1:, :],
            trim_state=jnp.array(trim.state),
            controls=sweep_result.controls,
        )

        # Compare
        for i in range(N):
            np.testing.assert_allclose(
                jax_metrics["rms_pos_d"][i],
                numpy_metrics[i]["rms_pos_d"],
                rtol=1e-4,
                err_msg=f"rms_pos_d mismatch for sim {i}",
            )
            np.testing.assert_allclose(
                jax_metrics["rms_theta"][i],
                numpy_metrics[i]["rms_theta"],
                rtol=1e-4,
                err_msg=f"rms_theta mismatch for sim {i}",
            )
            np.testing.assert_allclose(
                jax_metrics["control_effort"][i],
                numpy_metrics[i]["control_effort"],
                rtol=1e-4,
                err_msg=f"control_effort mismatch for sim {i}",
            )


# ============================================================================
# Sequential fallback test
# ============================================================================


class TestSweepSequential:
    """Tests for sweep_closed_loop_sequential."""

    def test_sweep_sequential_runs(self, lqr_setup):
        """Verify sweep_closed_loop_sequential returns correct number of results."""
        moth, sensor, estimator, controller, lqr_result, P0 = lqr_setup
        trim = lqr_result.trim

        N = 3
        x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.02)
        x0_est = jnp.array(trim.state)

        x0_trues = [x0_true + jnp.array([i * 0.01, 0, 0, 0, 0]) for i in range(N)]
        x0_ests = [x0_est] * N
        rng_keys = list(jax.random.split(jax.random.PRNGKey(42), N))

        results = sweep_closed_loop_sequential(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_trues=x0_trues,
            x0_ests=x0_ests,
            P0=P0,
            dt=DT,
            duration=DURATION,
            rng_keys=rng_keys,
            u_trim=jnp.array(trim.control),
        )

        assert len(results) == N
        for r in results:
            assert r.true_states.shape == (N_STEPS + 1, 5)
            assert r.controls.shape == (N_STEPS, 2)
