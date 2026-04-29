"""Tests for wave LQG scenarios (Phase 4).

Verifies scenario stability, wave aux outputs, encounter frequency,
and script smoke tests.
"""

import numpy as np
import pytest

import attrs

from fmd.simulator.moth_scenarios import (
    ScenarioConfig,
    SCENARIOS,
    run_scenario,
    HEAD_SEAS_SF_BAY,
    FOLLOWING_SEAS_SF_BAY,
    HEAD_SEAS_SF_BAY_LIGHT,
)


class TestWaveScenarioStability:
    """All wave scenarios should run 10s without NaN or divergence."""

    @pytest.mark.parametrize("scenario_name", [
        "head_seas_sf_bay",
        "following_seas_sf_bay",
        "head_seas_sf_bay_light",
    ])
    def test_scenario_runs_without_nan(self, scenario_name):
        """Scenario runs to completion without NaN."""
        config = SCENARIOS[scenario_name]
        result = run_scenario(config)

        assert result.true_states.shape[0] > 1, "Should have multiple steps"
        assert np.all(np.isfinite(result.true_states)), (
            f"{scenario_name}: true states contain NaN/inf"
        )
        assert np.all(np.isfinite(result.controls)), (
            f"{scenario_name}: controls contain NaN/inf"
        )


class TestWaveAuxNonZero:
    """Wave aux outputs should be non-zero when env has waves."""

    def test_head_seas_wave_aux(self):
        """Head seas scenario should have non-zero wave aux."""
        result = run_scenario(HEAD_SEAS_SF_BAY)

        # Extract aux for a mid-simulation point
        from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
        from fmd.simulator.params import MOTH_BIEKER_V3
        from fmd.simulator.environment import Environment

        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
        env = Environment.with_waves(HEAD_SEAS_SF_BAY.wave_params)

        import jax.numpy as jnp
        mid_idx = len(result.times) // 2
        state = jnp.array(result.true_states[mid_idx])
        control = jnp.array(result.controls[mid_idx])
        t = float(result.times[mid_idx])

        aux = moth.compute_aux(state, control, t=t, env=env)
        eta_main_idx = moth.aux_names.index("wave_eta_main")
        eta_rudder_idx = moth.aux_names.index("wave_eta_rudder")

        # At least one elevation should be nonzero
        assert abs(float(aux[eta_main_idx])) > 1e-6 or abs(float(aux[eta_rudder_idx])) > 1e-6, (
            "Wave elevation should be non-zero in wave scenario"
        )


class TestHeadVsFollowingSeas:
    """Wave scenarios produce larger RMS disturbances than calm."""

    def test_waves_increase_disturbance_vs_calm(self):
        """Head seas RMS pos_d variation > calm baseline.

        Use the second half of the simulation (after initial perturbation
        transient settles) and compute RMS of pos_d variation around its mean.
        """
        result_head = run_scenario(HEAD_SEAS_SF_BAY)
        calm_config = attrs.evolve(HEAD_SEAS_SF_BAY, name="calm_ref", wave_params=None)
        result_calm = run_scenario(calm_config)

        # Use second half to avoid initial perturbation transient
        n = len(result_head.true_states)
        half = n // 2

        pos_d_head = result_head.true_states[half:, 0]
        pos_d_calm = result_calm.true_states[half:, 0]

        rms_head = np.sqrt(np.mean((pos_d_head - np.mean(pos_d_head))**2))
        rms_calm = np.sqrt(np.mean((pos_d_calm - np.mean(pos_d_calm))**2))

        assert rms_head > rms_calm, (
            f"Head seas RMS ({rms_head:.4f}) should exceed calm RMS ({rms_calm:.4f})"
        )


class TestEncounterFrequencyVerification:
    """Verify encounter frequency from aux data."""

    def test_encounter_frequency_spectrum(self):
        """FFT of wave elevation should show encounter frequency peak."""
        result = run_scenario(HEAD_SEAS_SF_BAY)

        # Extract wave_eta_main trajectory
        from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
        from fmd.simulator.params import MOTH_BIEKER_V3
        from fmd.simulator.environment import Environment
        from fmd.simulator.integrator import compute_aux_trajectory, SimulationResult

        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
        env = Environment.with_waves(HEAD_SEAS_SF_BAY.wave_params)

        sim_result = SimulationResult(
            times=result.times,
            states=result.true_states[1:],
            controls=result.controls,
        )
        aux = compute_aux_trajectory(moth, sim_result, env=env)
        eta_signal = np.array(aux["wave_eta_main"])

        # FFT
        dt = HEAD_SEAS_SF_BAY.dt
        n = len(eta_signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        spectrum = np.abs(np.fft.rfft(eta_signal))

        # Find peak (skip DC and very low frequencies)
        min_freq_idx = max(1, int(0.3 / (freqs[1] - freqs[0])))  # Above 0.3 Hz
        peak_idx = np.argmax(spectrum[min_freq_idx:]) + min_freq_idx
        peak_freq = freqs[peak_idx]

        # For head seas at 10 m/s, Tp=3s, lambda~14m:
        # f_e = 1/Tp + V/lambda = 1/3 + 10/14 ~ 1.05 Hz
        # For spectral waves, the peak may be broader
        assert 0.5 < peak_freq < 1.5, (
            f"Peak frequency {peak_freq:.2f} Hz outside expected range"
        )


class TestLiftLagWithWavesLQG:
    """Combined lift_lag + waves + LQG should work together."""

    def test_lift_lag_waves_lqg_no_nan(self):
        """LQG scenario with lift_lag=True and waves completes without NaN."""
        config = attrs.evolve(
            HEAD_SEAS_SF_BAY_LIGHT,
            name="lift_lag_waves",
            enable_lift_lag=True,
            duration=5.0,
        )
        result = run_scenario(config)

        assert result.true_states.shape[1] == 7, (
            f"State dimension should be 7 with lift lag, got {result.true_states.shape[1]}"
        )
        assert np.all(np.isfinite(result.true_states)), (
            "Lift lag + waves LQG should not produce NaN"
        )
        assert np.all(np.isfinite(result.controls)), (
            "Lift lag + waves LQG controls should not produce NaN"
        )


