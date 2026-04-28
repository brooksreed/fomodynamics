"""Tests for the shared closed-loop analysis module.

Tests compute_extended_metrics, format_metrics_table,
find_interesting_window, compute_leeward_tip_depth,
compute_wave_vs_calm_metrics, and plotting functions.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from fmd.analysis.closed_loop import (
    compute_extended_metrics,
    compute_wave_vs_calm_metrics,
    compute_leeward_tip_depth,
    format_metrics_table,
    find_interesting_window,
    plot_config_overlay,
    plot_single_dashboard,
)


# ---------------------------------------------------------------------------
# Synthetic ClosedLoopResult fixture
# ---------------------------------------------------------------------------

class _FakeParams:
    """Minimal params mock with foil geometry."""
    hull_mass = 35.0
    sailor_mass = 75.0
    sailor_position = (0.0, 0.0, 0.0)
    main_foil_position = (0.0, 0.0, 0.8)
    rudder_position = (-2.3, 0.0, 0.6)
    main_foil_span = 1.1
    rudder_span = 0.4
    bowsprit_position = (2.0, 0.0, -0.3)


class _FakeResult:
    """Minimal ClosedLoopResult-like object for testing."""

    def __init__(self, n_steps=200, dt=0.005, trim_state=None, noise_scale=0.01):
        self.dt = dt
        self.times = np.arange(n_steps) * dt
        n = 5
        m = 2

        if trim_state is None:
            trim_state = np.array([-1.39, np.radians(2.0), 0.0, 0.0, 10.0])
        self.trim_state_val = trim_state

        # True states: trim + small sinusoidal perturbation
        base = np.tile(trim_state, (n_steps + 1, 1))
        t_ext = np.concatenate([[0.0], self.times])
        for i in range(n):
            base[:, i] += noise_scale * np.sin(2 * np.pi * t_ext / 0.5 + i)
        self.true_states = base
        self.est_states = base + np.random.default_rng(0).normal(0, 0.001, base.shape)

        # Controls: trim + small perturbation
        ctrl_trim = np.array([np.radians(5.0), np.radians(1.0)])
        self.controls = np.tile(ctrl_trim, (n_steps, 1))
        self.controls += noise_scale * 0.1 * np.sin(
            2 * np.pi * self.times[:, None] / 0.3
        )

        self.estimation_errors = self.true_states - self.est_states
        self.covariance_traces = np.ones(n_steps + 1) * 0.01
        self.covariance_diagonals = np.ones((n_steps + 1, n)) * 0.002
        self.params = _FakeParams()
        self.heel_angle = np.radians(30.0)
        self.force_log = None
        self.measurements_clean = None
        self.measurements_noisy = None
        self.innovations = None
        self.trim_state = trim_state
        self.trim_control = ctrl_trim
        self.lqr_K = None
        self.measurement_output_names = None
        self.measurement_state_index_map = None
        self.metadata = None


@pytest.fixture
def trim_state():
    return np.array([-1.39, np.radians(2.0), 0.0, 0.0, 10.0])


@pytest.fixture
def fake_result(trim_state):
    return _FakeResult(n_steps=2000, dt=0.005, trim_state=trim_state)


@pytest.fixture
def fake_aux():
    """Minimal aux dict with wave keys."""
    n = 2000
    return {
        "wave_eta_main": np.sin(np.linspace(0, 10, n)) * 0.1,
        "wave_eta_rudder": np.sin(np.linspace(0, 10, n) + 0.5) * 0.1,
        "main_drag_aero": np.ones(n) * 5.0,
        "rudder_drag_aero": np.ones(n) * 1.5,
        "main_strut_drag": np.ones(n) * 0.8,
        "rudder_strut_drag": np.ones(n) * 0.3,
        "main_strut_immersion": np.ones(n) * 0.5,
        "rudder_strut_immersion": np.ones(n) * 0.3,
        "hull_drag": np.zeros(n),
        "hull_buoyancy": np.zeros(n),
    }


# ---------------------------------------------------------------------------
# Tests: compute_extended_metrics
# ---------------------------------------------------------------------------

class TestComputeExtendedMetrics:
    def test_returns_expected_keys(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        assert "ride_height" in m
        assert "pitch" in m
        assert "speed" in m
        assert "foil_tip_depth" in m
        assert "control_effort" in m
        assert "control_rate" in m
        assert "settling_time" in m
        assert "has_nan" in m
        assert "rms_pos_d" in m
        assert "rms_theta" in m

    def test_ride_height_stats(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        rh = m["ride_height"]
        assert "mean" in rh
        assert "std" in rh
        assert "rms_error" in rh
        assert rh["std"] >= 0
        assert rh["rms_error"] >= 0

    def test_pitch_stats(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        p = m["pitch"]
        assert "mean_rad" in p
        assert "std_deg" in p
        assert p["std_rad"] >= 0

    def test_control_effort_per_channel(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        ce = m["control_effort"]
        assert "main_flap" in ce
        assert "rudder_elevator" in ce
        assert ce["main_flap"]["std_rad"] >= 0

    def test_control_rate(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        cr = m["control_rate"]
        assert "main_flap" in cr
        assert cr["main_flap"]["rms_rad_per_s"] >= 0

    def test_foil_tip_depth(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        ft = m["foil_tip_depth"]
        assert "min" in ft
        assert "breach_fraction" in ft
        assert "breach_count" in ft
        assert 0 <= ft["breach_fraction"] <= 1

    def test_no_nan_for_clean_data(self, fake_result, trim_state, fake_aux):
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        assert m["has_nan"] is False

    def test_nan_detection(self, trim_state, fake_aux):
        result = _FakeResult(n_steps=100, trim_state=trim_state)
        result.true_states[50, 0] = np.nan
        m = compute_extended_metrics(result, trim_state, fake_aux)
        assert m["has_nan"] is True

    def test_settling_time_for_already_settled(self, trim_state, fake_aux):
        """If signal stays within threshold from t=0, settling time is 0."""
        result = _FakeResult(n_steps=200, trim_state=trim_state, noise_scale=0.001)
        m = compute_extended_metrics(result, trim_state, fake_aux, dt=0.005)
        # With noise_scale=0.001, pos_d error < 0.01 threshold always
        assert m["settling_time"] < 1.0


# ---------------------------------------------------------------------------
# Tests: compute_wave_vs_calm_metrics
# ---------------------------------------------------------------------------

class TestComputeWaveVsCalmMetrics:
    def test_returns_expected_keys(self, fake_result, trim_state, fake_aux):
        calm_result = _FakeResult(n_steps=2000, trim_state=trim_state, noise_scale=0.001)
        m = compute_wave_vs_calm_metrics(
            fake_result, calm_result, fake_aux, fake_aux, trim_state
        )
        assert "state_statistics" in m
        assert "control_statistics" in m
        assert "drag_decomposition" in m
        assert "speed_equilibrium" in m

    def test_state_statistics_structure(self, fake_result, trim_state, fake_aux):
        calm_result = _FakeResult(n_steps=2000, trim_state=trim_state)
        m = compute_wave_vs_calm_metrics(
            fake_result, calm_result, fake_aux, fake_aux, trim_state
        )
        ss = m["state_statistics"]
        assert "pos_d" in ss
        assert "theta" in ss
        assert "u" in ss
        pd = ss["pos_d"]
        assert "wave_mean" in pd
        assert "calm_mean" in pd
        assert "delta_mean" in pd

    def test_drag_decomposition(self, fake_result, trim_state, fake_aux):
        calm_result = _FakeResult(n_steps=2000, trim_state=trim_state)
        m = compute_wave_vs_calm_metrics(
            fake_result, calm_result, fake_aux, fake_aux, trim_state
        )
        dd = m["drag_decomposition"]
        assert "main_drag_aero" in dd
        assert "difference" in dd["main_drag_aero"]


# ---------------------------------------------------------------------------
# Tests: format_metrics_table
# ---------------------------------------------------------------------------

class TestFormatMetricsTable:
    def test_basic_formatting(self):
        metrics = {
            ("config_a", "calm"): {
                "rms_pos_d": 0.01,
                "rms_theta": 0.005,
                "overall_control_effort": 0.1,
                "settling_time": 2.0,
                "has_nan": False,
            },
            ("config_b", "waves"): {
                "rms_pos_d": 0.02,
                "rms_theta": 0.01,
                "overall_control_effort": 0.2,
                "settling_time": 3.0,
                "has_nan": True,
            },
        }
        table = format_metrics_table(metrics)
        assert "config_a" in table
        assert "config_b" in table
        assert "calm" in table
        assert "waves" in table
        assert "YES" in table  # has_nan for config_b

    def test_header_present(self):
        metrics = {
            ("x", "y"): {
                "rms_pos_d": 0.01,
                "rms_theta": 0.005,
                "overall_control_effort": 0.1,
                "settling_time": 1.0,
                "has_nan": False,
            },
        }
        table = format_metrics_table(metrics)
        assert "Config" in table
        assert "RMS pos_d" in table
        assert "RMS theta" in table

    def test_extended_metrics_format(self, fake_result, trim_state, fake_aux):
        """Table formatting works with compute_extended_metrics output."""
        m = compute_extended_metrics(fake_result, trim_state, fake_aux)
        metrics = {("test_config", "calm"): m}
        table = format_metrics_table(metrics)
        assert "test_config" in table
        lines = table.strip().split("\n")
        assert len(lines) >= 3  # header + separator + 1 data row


# ---------------------------------------------------------------------------
# Tests: find_interesting_window
# ---------------------------------------------------------------------------

class TestFindInterestingWindow:
    def test_returns_valid_window(self, trim_state):
        """Window is within simulation time bounds."""
        result_a = _FakeResult(n_steps=2000, trim_state=trim_state, noise_scale=0.01)
        results = {"a": result_a}
        auxs = {"a": {}}
        t_start, t_end = find_interesting_window(results, auxs, trim_state)
        assert t_start >= 0.0
        assert t_end > t_start
        assert t_end <= float(result_a.times[-1]) + 0.001

    def test_finds_spike(self, trim_state):
        """Window should locate a region with a large disturbance."""
        result = _FakeResult(n_steps=4000, dt=0.005, trim_state=trim_state, noise_scale=0.001)
        # Inject a large disturbance at t=15s (index 3000)
        result.true_states[3000:3200, 0] += 0.5  # big pos_d excursion
        results = {"spike": result}
        auxs = {"spike": {}}
        t_start, t_end = find_interesting_window(results, auxs, trim_state, window_size=2.0)
        # The spike is at t=15s; window should contain it
        assert t_start <= 15.0
        assert t_end >= 15.0

    def test_short_simulation(self, trim_state):
        """If simulation is shorter than window, returns full range."""
        result = _FakeResult(n_steps=100, dt=0.005, trim_state=trim_state)
        results = {"short": result}
        auxs = {"short": {}}
        t_start, t_end = find_interesting_window(results, auxs, trim_state, window_size=5.0)
        assert t_start == pytest.approx(0.0)
        assert t_end == pytest.approx(float(result.times[-1]))


# ---------------------------------------------------------------------------
# Tests: compute_leeward_tip_depth
# ---------------------------------------------------------------------------

class TestComputeLeewayTipDepth:
    def test_matches_moth_forces_formula(self):
        """Verify against the canonical computation in moth_forces."""
        pos_d = np.array([-1.39, -1.40, -1.38])
        theta = np.array([0.035, 0.04, 0.03])
        foil_x = 0.0
        foil_z = 0.8
        foil_span = 1.1
        heel_angle = np.radians(30.0)

        tip = compute_leeward_tip_depth(pos_d, theta, foil_x, foil_z, foil_span, heel_angle)

        # Manual: center_depth = pos_d + foil_z*cos(heel)*cos(theta) - foil_x*sin(theta)
        # tip = center_depth - (foil_span/2)*sin(heel)
        center = pos_d + foil_z * np.cos(heel_angle) * np.cos(theta)
        expected = center - (foil_span / 2.0) * np.sin(heel_angle)
        np.testing.assert_allclose(tip, expected, atol=1e-10)

    def test_deeper_pos_d_means_deeper_tip(self):
        """More positive pos_d (deeper CG) means deeper foil tip."""
        theta = np.array([0.035, 0.035])
        pos_d_shallow = np.array([-1.5])
        pos_d_deep = np.array([-1.2])
        tip_shallow = compute_leeward_tip_depth(pos_d_shallow, theta[:1], 0.0, 0.8, 1.1, np.radians(30))
        tip_deep = compute_leeward_tip_depth(pos_d_deep, theta[:1], 0.0, 0.8, 1.1, np.radians(30))
        assert tip_deep[0] > tip_shallow[0]

    def test_no_heel_no_span_offset(self):
        """With zero heel angle, leeward tip depth equals center depth."""
        pos_d = np.array([-1.39])
        theta = np.array([0.035])
        tip = compute_leeward_tip_depth(pos_d, theta, 0.0, 0.8, 1.1, heel_angle=0.0)
        center = pos_d + 0.8 * np.cos(0.0) * np.cos(theta)
        np.testing.assert_allclose(tip, center, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Plotting (smoke tests - verify they return Figure)
# ---------------------------------------------------------------------------

class TestPlotConfigOverlay:
    def test_returns_figure(self, fake_result, trim_state):
        results = {"cfg_a": fake_result}
        fig = plot_config_overlay(results, trim_state)
        assert fig is not None
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_time_window(self, fake_result, trim_state):
        results = {"cfg_a": fake_result}
        fig = plot_config_overlay(results, trim_state, t_start=1.0, t_end=5.0)
        assert fig is not None
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_multiple_configs(self, trim_state):
        results = {
            "a": _FakeResult(n_steps=200, trim_state=trim_state),
            "b": _FakeResult(n_steps=200, trim_state=trim_state),
        }
        fig = plot_config_overlay(results, trim_state)
        assert fig is not None
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSingleDashboard:
    def test_returns_figure(self, fake_result, trim_state):
        fig = plot_single_dashboard(fake_result, trim_state)
        assert fig is not None
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_calm_overlay(self, fake_result, trim_state):
        calm = _FakeResult(n_steps=2000, trim_state=trim_state, noise_scale=0.001)
        fig = plot_single_dashboard(
            fake_result, trim_state,
            calm_result=calm,
        )
        assert fig is not None
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_aux(self, fake_result, trim_state, fake_aux):
        fig = plot_single_dashboard(fake_result, trim_state, aux=fake_aux)
        assert fig is not None
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)
