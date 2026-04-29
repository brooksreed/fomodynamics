"""Moth wave-body interaction tests.

Includes tests for:
- Basic calm-vs-waves behavior
- Per-foil differential wave effects (Phase 2)
- Horizontal orbital velocity effects
- Encounter frequency verification
- Backward compatibility
- LQG with env parameter
- Wave aux output verification
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fmd.simulator import simulate, Moth3D, ConstantSchedule, ConstantControl, Environment
from fmd.simulator.params import (
    MOTH_BIEKER_V3,
    WAVE_REGULAR_1M,
    WaveParams,
)
from fmd.simulator.closed_loop_pipeline import simulate_closed_loop
from fmd.simulator.sensors import MeasurementSensor
from fmd.simulator.estimators import EKFEstimator
from fmd.simulator.controllers import LQRController
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.waves import WaveField


class TestMothInWaves:
    """Tests for moth behavior in wave fields."""

    @pytest.fixture(scope="class")
    def moth(self):
        return Moth3D(MOTH_BIEKER_V3)

    @pytest.fixture(scope="class")
    def trim_result(self):
        result = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
        )
        assert result.success, f"Trim failed: residual={result.residual:.2e}"
        return result

    @pytest.fixture
    def wave_env(self):
        return Environment.with_waves(WAVE_REGULAR_1M)

    def test_calm_vs_waves_differ(self, moth, trim_result, wave_env):
        """Calm and wave simulations produce different trajectories."""
        state0 = jnp.array(trim_result.state)
        control = ConstantControl(jnp.array(trim_result.control))

        result_calm = simulate(moth, state0, dt=0.005, duration=2.0, control=control)
        result_wave = simulate(moth, state0, dt=0.005, duration=2.0, env=wave_env, control=control)

        diff = jnp.max(jnp.abs(result_calm.states - result_wave.states))
        assert float(diff) > 0.001, f"Calm and wave trajectories should differ, max diff={diff}"

    def test_wave_elevation_changes_foil_depth(self, moth, trim_result):
        """Wave crest (eta>0) should make foil shallower."""
        state0 = jnp.array(trim_result.state)
        control = ConstantControl(jnp.array(trim_result.control))

        # Large regular wave to see clear effect
        env = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=5.0))
        result = simulate(moth, state0, dt=0.005, duration=2.0, env=env, control=control)

        # pos_d should show wave influence
        pos_d = result.states[:, 0]
        pos_d_range = float(jnp.max(pos_d) - jnp.min(pos_d))
        assert pos_d_range > 0.01, f"Wave should affect heave, range={pos_d_range}"

    def test_env_none_matches_no_env(self, moth, trim_result):
        """env=None should match omitting env entirely."""
        state0 = jnp.array(trim_result.state)
        control = ConstantControl(jnp.array(trim_result.control))

        result_none = simulate(moth, state0, dt=0.005, duration=1.0, env=None, control=control)
        result_omit = simulate(moth, state0, dt=0.005, duration=1.0, control=control)

        assert jnp.allclose(result_none.states, result_omit.states, atol=1e-12)


class TestDifferentialWaveEffects:
    """Tests for per-foil wave queries (Phase 2)."""

    @pytest.fixture(scope="class")
    def trim_result(self):
        result = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
        )
        assert result.success
        return result

    def test_differential_elevation(self, trim_result):
        """Main foil and rudder experience different wave elevations.

        Use wavelength ~ 2 * foil_separation so foils are in opposite wave phases.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # Foil separation: main_foil ~0.55m forward, rudder ~-1.75m aft -> ~2.3m apart
        # Want wavelength ~ 2 * 2.3 = 4.6m -> deep water: T = sqrt(2*pi*L/g) ~ 1.72s
        # Use T=1.7s for wavelength ~4.5m
        params = WaveParams.regular(amplitude=0.15, period=1.7, direction=0.0)
        env = Environment.with_waves(params)

        aux = moth.compute_aux(state0, control, t=0.0, env=env)
        aux_names = moth.aux_names
        eta_main_idx = aux_names.index("wave_eta_main")
        eta_rudder_idx = aux_names.index("wave_eta_rudder")

        eta_main = float(aux[eta_main_idx])
        eta_rudder = float(aux[eta_rudder_idx])

        # They should differ because foils are at different NED positions
        assert abs(eta_main - eta_rudder) > 0.001, (
            f"Differential wave elevation too small: main={eta_main:.4f}, rudder={eta_rudder:.4f}"
        )

    def test_differential_forces(self, trim_result):
        """Wave forces differ between main foil and rudder."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # Regular wave with wavelength ~ foil separation
        params = WaveParams.regular(amplitude=0.15, period=1.7, direction=0.0)
        env = Environment.with_waves(params)

        # Check at t=0.5s where there's a phase offset
        aux_wave = moth.compute_aux(state0, control, t=0.5, env=env)
        aux_calm = moth.compute_aux(state0, control, t=0.5)

        # Main foil lift should differ from calm
        main_lift_idx = moth.aux_names.index("main_lift_aero")
        rudder_lift_idx = moth.aux_names.index("rudder_lift_aero")

        main_lift_wave = float(aux_wave[main_lift_idx])
        main_lift_calm = float(aux_calm[main_lift_idx])
        rudder_lift_wave = float(aux_wave[rudder_lift_idx])
        rudder_lift_calm = float(aux_calm[rudder_lift_idx])

        # At least one foil should show a measurable force difference
        main_diff = abs(main_lift_wave - main_lift_calm)
        rudder_diff = abs(rudder_lift_wave - rudder_lift_calm)
        assert max(main_diff, rudder_diff) > 0.1, (
            f"Wave should affect forces: main_diff={main_diff:.2f}, rudder_diff={rudder_diff:.2f}"
        )


class TestHorizontalOrbitalVelocity:
    """Tests for horizontal orbital velocity effects."""

    @pytest.fixture(scope="class")
    def trim_result(self):
        result = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
        )
        assert result.success
        return result

    def test_horizontal_velocity_nonzero(self, trim_result):
        """u_orbital_body should be nonzero in waves."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        params = WaveParams.regular(amplitude=0.3, period=5.0, direction=0.0)
        env = Environment.with_waves(params)

        aux = moth.compute_aux(state0, control, t=0.5, env=env)
        u_orb_main_idx = moth.aux_names.index("wave_u_orbital_main")
        u_orb = float(aux[u_orb_main_idx])

        assert abs(u_orb) > 0.001, f"Horizontal orbital velocity should be nonzero: {u_orb}"

    def test_horizontal_velocity_affects_forces(self, trim_result):
        """Forces with horizontal velocity should differ from vertical-only.

        We verify this by checking that u_eff != u_forward in the force
        calculation produces different lift.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        # Strong regular wave
        params = WaveParams.regular(amplitude=0.3, period=5.0, direction=0.0)
        env = Environment.with_waves(params)

        # Get aux at a time where orbital velocity is significant
        aux_wave = moth.compute_aux(state0, control, t=0.0, env=env)
        aux_calm = moth.compute_aux(state0, control, t=0.0)

        main_lift_idx = moth.aux_names.index("main_lift_aero")
        lift_wave = float(aux_wave[main_lift_idx])
        lift_calm = float(aux_calm[main_lift_idx])

        # The difference includes both horizontal and vertical effects
        assert abs(lift_wave - lift_calm) > 0.1, (
            f"Wave orbital velocity should affect lift: wave={lift_wave:.2f}, calm={lift_calm:.2f}"
        )


class TestEncounterFrequency:
    """Verify encounter frequency in wave aux signals."""

    def test_encounter_frequency_head_seas(self):
        """10 m/s head seas, Tp=3s -> f_e ~ 1/3 + 10/14 ~ 1.05 Hz.

        Run 5s sim, FFT wave_eta_main aux signal, check peak near 1.05 Hz.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
        )
        assert trim.success

        # Head seas: waves propagate toward the boat (from north, direction=pi)
        # At 10 m/s with Tp=3s, deep water lambda = g*T^2/(2*pi) ~ 14.0m
        # f_encounter = f_wave + V/lambda = 1/3 + 10/14 ~ 1.05 Hz
        params = WaveParams.regular(amplitude=0.15, period=3.0, direction=np.pi)
        env = Environment.with_waves(params)

        state0 = jnp.array(trim.state)
        control = ConstantControl(jnp.array(trim.control))

        dt = 0.005
        duration = 5.0
        result = simulate(moth, state0, dt=dt, duration=duration, env=env, control=control)

        # Extract wave_eta_main from aux
        from fmd.simulator.integrator import compute_aux_trajectory
        aux = compute_aux_trajectory(moth, result, env=env)
        eta_signal = np.array(aux["wave_eta_main"])

        # FFT
        n = len(eta_signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        spectrum = np.abs(np.fft.rfft(eta_signal))

        # Find peak frequency (skip DC)
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_freq = freqs[peak_idx]

        # Expected encounter frequency ~ 1.05 Hz
        expected_fe = 1.0 / 3.0 + 10.0 / 14.0
        assert abs(peak_freq - expected_fe) < 0.15, (
            f"Peak freq {peak_freq:.2f} Hz should be near {expected_fe:.2f} Hz"
        )


class TestWaveAuxOutputs:
    """Verify all 6 wave aux outputs are populated correctly."""

    @pytest.fixture(scope="class")
    def trim_result(self):
        result = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
        )
        assert result.success
        return result

    def test_wave_aux_zero_in_calm(self, trim_result):
        """All wave aux should be zero without env."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        aux = moth.compute_aux(state0, control, t=1.0)
        wave_names = [n for n in moth.aux_names if n.startswith("wave_")]
        assert len(wave_names) == 6

        for name in wave_names:
            idx = moth.aux_names.index(name)
            assert float(aux[idx]) == 0.0, f"{name} should be 0 in calm, got {float(aux[idx])}"

    def test_wave_aux_nonzero_in_waves(self, trim_result):
        """All wave aux should be nonzero with active wave field."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = jnp.array(trim_result.state)
        control = jnp.array(trim_result.control)

        params = WaveParams.regular(amplitude=0.3, period=5.0, direction=0.0)
        env = Environment.with_waves(params)

        # Check at t=0.5 (not t=0 where some components may happen to be zero)
        aux = moth.compute_aux(state0, control, t=0.5, env=env)
        wave_names = [n for n in moth.aux_names if n.startswith("wave_")]
        nonzero_count = 0
        for name in wave_names:
            idx = moth.aux_names.index(name)
            if abs(float(aux[idx])) > 1e-6:
                nonzero_count += 1

        # At least eta and some orbital velocities should be nonzero
        assert nonzero_count >= 4, (
            f"Expected at least 4 nonzero wave aux outputs, got {nonzero_count}"
        )


class TestEncounterFrequencyHeadVsFollowing:
    """Head seas encounter frequency should be higher than following seas."""

    def test_head_seas_higher_encounter_freq(self):
        """Same Tp/amplitude/speed: head seas peak > following seas peak."""
        moth = Moth3D(MOTH_BIEKER_V3)
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
        )
        assert trim.success

        state0 = jnp.array(trim.state)
        control = ConstantControl(jnp.array(trim.control))

        dt = 0.005
        duration = 5.0

        # Head seas: direction=pi
        head_params = WaveParams.regular(amplitude=0.15, period=3.0, direction=np.pi)
        env_head = Environment.with_waves(head_params)
        result_head = simulate(moth, state0, dt=dt, duration=duration, env=env_head, control=control)

        # Following seas: direction=0
        follow_params = WaveParams.regular(amplitude=0.15, period=3.0, direction=0.0)
        env_follow = Environment.with_waves(follow_params)
        result_follow = simulate(moth, state0, dt=dt, duration=duration, env=env_follow, control=control)

        from fmd.simulator.integrator import compute_aux_trajectory

        aux_head = compute_aux_trajectory(moth, result_head, env=env_head)
        aux_follow = compute_aux_trajectory(moth, result_follow, env=env_follow)

        eta_head = np.array(aux_head["wave_eta_main"])
        eta_follow = np.array(aux_follow["wave_eta_main"])

        def peak_freq(signal, dt):
            n = len(signal)
            freqs = np.fft.rfftfreq(n, d=dt)
            spectrum = np.abs(np.fft.rfft(signal))
            # Skip DC
            peak_idx = np.argmax(spectrum[1:]) + 1
            return freqs[peak_idx]

        f_head = peak_freq(eta_head, dt)
        f_follow = peak_freq(eta_follow, dt)

        assert f_head > f_follow, (
            f"Head seas encounter freq ({f_head:.2f} Hz) should exceed "
            f"following seas ({f_follow:.2f} Hz)"
        )


class TestClosedLoopWithEnv:
    """Test that simulate_closed_loop accepts and propagates env parameter."""

    def _build_pipeline(self, moth, meas, lqr, Q_ekf):
        """Build pipeline components for testing."""
        from fmd.estimation import ExtendedKalmanFilter
        sensor = MeasurementSensor(measurement_model=meas, num_controls=2)
        ekf = ExtendedKalmanFilter(dt=0.005)
        estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)
        u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
        controller = LQRController(
            K=jnp.array(lqr.K),
            x_trim=jnp.array(lqr.trim.state),
            u_trim=jnp.array(lqr.trim.control),
            u_min=u_min, u_max=u_max,
        )
        return sensor, estimator, controller

    def test_closed_loop_calm_env_matches_no_env(self):
        """Calm environment should match no-env result exactly."""
        from fmd.simulator.moth_lqr import design_moth_lqr
        from fmd.estimation import create_moth_measurement

        lqr = design_moth_lqr(u_forward=10.0, dt=0.005)
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
        meas = create_moth_measurement("full_state")
        trim_state = jnp.array(lqr.trim.state)
        x0_true = trim_state.at[0].set(trim_state[0] + 0.03)

        P0 = jnp.eye(5) * 0.1
        Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
        key = jax.random.PRNGKey(42)

        sensor, estimator, controller = self._build_pipeline(moth, meas, lqr, Q_ekf)

        # No env
        result_no_env = simulate_closed_loop(
            moth, sensor, estimator, controller,
            x0_true, trim_state, P0,
            dt=0.005, duration=1.0, rng_key=key,
            u_trim=jnp.array(lqr.trim.control),
        )

        # Calm env
        calm_env = Environment.calm()
        result_calm = simulate_closed_loop(
            moth, sensor, estimator, controller,
            x0_true, trim_state, P0,
            dt=0.005, duration=1.0, rng_key=key,
            env=calm_env,
            u_trim=jnp.array(lqr.trim.control),
        )

        assert np.allclose(result_no_env.true_states, result_calm.true_states, atol=1e-10), (
            "Calm env should match no-env"
        )

    def test_closed_loop_accepts_wave_env(self):
        """simulate_closed_loop should run with a wave environment without error."""
        from fmd.simulator.moth_lqr import design_moth_lqr
        from fmd.estimation import create_moth_measurement

        lqr = design_moth_lqr(u_forward=10.0, dt=0.005)
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
        meas = create_moth_measurement("full_state")
        trim_state = jnp.array(lqr.trim.state)
        x0_true = trim_state.at[0].set(trim_state[0] + 0.03)

        P0 = jnp.eye(5) * 0.1
        Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))

        env = Environment.with_waves(WaveParams.regular(amplitude=0.1, period=3.0))

        sensor, estimator, controller = self._build_pipeline(moth, meas, lqr, Q_ekf)

        result = simulate_closed_loop(
            moth, sensor, estimator, controller,
            x0_true, trim_state, P0,
            dt=0.005, duration=2.0,
            rng_key=jax.random.PRNGKey(42),
            env=env,
            u_trim=jnp.array(lqr.trim.control),
        )

        # Should complete without NaN
        assert np.all(np.isfinite(result.true_states)), "Closed-loop with waves should not produce NaN"
        assert result.true_states.shape[0] > 1
