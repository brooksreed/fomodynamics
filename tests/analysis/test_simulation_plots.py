"""Tests for the unified plotting system (L0, L1, L2 layers).

Uses Agg backend to avoid GUI windows. Focuses on function signatures,
return types, and structural correctness rather than visual output.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_times():
    return np.linspace(0, 5, 100)


@pytest.fixture
def sample_states(sample_times):
    """5-state trajectory (mimics Moth 3DOF)."""
    n = len(sample_times)
    return np.column_stack([
        np.sin(sample_times) * 0.1,            # pos_d
        np.sin(sample_times * 2) * 0.05,        # theta
        np.cos(sample_times) * 0.3,             # w
        np.cos(sample_times * 2) * 0.2,         # q
        10.0 + np.sin(sample_times * 0.5) * 0.1, # u
    ])


@pytest.fixture
def sample_controls(sample_times):
    """2-control trajectory (mimics Moth 3DOF)."""
    n = len(sample_times)
    return np.column_stack([
        np.sin(sample_times) * 0.05,  # main flap
        np.cos(sample_times) * 0.02,  # rudder elevator
    ])


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_style(self):
        from fmd.analysis.plots import FMD_STYLE, apply_fmd_style, style_axis, get_colors, savefig_and_close
        assert isinstance(FMD_STYLE, dict)
        assert callable(apply_fmd_style)
        assert callable(style_axis)
        assert callable(get_colors)
        assert callable(savefig_and_close)

    def test_import_windowing(self):
        from fmd.analysis.plots import compute_adaptive_window, apply_window
        assert callable(compute_adaptive_window)
        assert callable(apply_window)

    def test_import_primitives(self):
        from fmd.analysis.plots import (
            time_series_panels, multi_trace_panel, error_with_envelope,
            bar_comparison, overlay_with_difference, step_indicator,
        )
        for fn in [time_series_panels, multi_trace_panel, error_with_envelope,
                    bar_comparison, overlay_with_difference, step_indicator]:
            assert callable(fn)

    def test_import_simulation(self):
        from fmd.analysis.plots import (
            state_trajectory, estimation_error, estimation_error_with_covariance,
            nees_plot, covariance_diagonal, control_effort, solver_diagnostics,
            trajectory_comparison, bar_comparison_by_variant, multi_trace_grid,
        )
        for fn in [state_trajectory, estimation_error, nees_plot, covariance_diagonal,
                    control_effort, solver_diagnostics, trajectory_comparison,
                    bar_comparison_by_variant, multi_trace_grid]:
            assert callable(fn)

    def test_import_convenience(self):
        from fmd.analysis.plots import (
            plot_simulation_result, plot_lqg_result, plot_sweep_category,
            MOTH_3DOF_STATE_LABELS, MOTH_3DOF_STATE_TRANSFORMS,
        )
        assert callable(plot_simulation_result)
        assert callable(plot_lqg_result)
        assert callable(plot_sweep_category)
        assert len(MOTH_3DOF_STATE_LABELS) == 5
        assert len(MOTH_3DOF_STATE_TRANSFORMS) == 5

    def test_backward_compat_imports(self):
        """Existing import paths must still work."""
        from fmd.analysis.plots import plot_time_series, plot_polar, radians_to_degrees
        from fmd.analysis.plots import _get_display_info, _autosize_figure_to_screen
        from fmd.analysis import plot_time_series as pts
        assert callable(pts)


# ---------------------------------------------------------------------------
# Style tests
# ---------------------------------------------------------------------------

class TestStyle:
    def test_get_colors_small(self):
        from fmd.analysis.plots import get_colors
        colors = get_colors(5)
        assert len(colors) == 5

    def test_get_colors_large(self):
        from fmd.analysis.plots import get_colors
        colors = get_colors(15)
        assert len(colors) == 15

    def test_apply_fmd_style(self):
        from fmd.analysis.plots import apply_fmd_style
        with apply_fmd_style():
            assert plt.rcParams["grid.alpha"] == 0.3

    def test_style_axis(self):
        from fmd.analysis.plots import style_axis
        fig, ax = plt.subplots()
        style_axis(ax, xlabel="X", ylabel="Y", title="Test")
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_title() == "Test"
        plt.close(fig)

    def test_savefig_and_close(self, tmp_path):
        from fmd.analysis.plots import savefig_and_close
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test_output.png"
        savefig_and_close(fig, str(out))
        assert out.exists()
        assert out.stat().st_size > 0
        # Figure should be closed (no longer in active figures)
        assert fig not in [plt.figure(num) for num in plt.get_fignums()]


# ---------------------------------------------------------------------------
# Windowing tests
# ---------------------------------------------------------------------------

class TestWindowing:
    def test_empty_traces(self):
        from fmd.analysis.plots import compute_adaptive_window
        result = compute_adaptive_window([])
        assert result["xlim"] is None
        assert not result["has_divergent"]

    def test_single_stable_trace(self):
        from fmd.analysis.plots import compute_adaptive_window
        trace = {
            "times": np.linspace(0, 5, 100),
            "values": np.sin(np.linspace(0, 5, 100)),
            "divergence_time": None,
            "stable": True,
        }
        result = compute_adaptive_window([trace], reference_value=0.0)
        assert result["xlim"] is not None
        assert result["ylim"] is not None
        assert not result["has_divergent"]

    def test_divergent_trace(self):
        from fmd.analysis.plots import compute_adaptive_window
        trace = {
            "times": np.linspace(0, 5, 100),
            "values": np.exp(np.linspace(0, 5, 100)),
            "divergence_time": 2.0,
            "stable": False,
        }
        result = compute_adaptive_window([trace])
        assert result["has_divergent"]
        assert len(result["divergence_times"]) == 1
        # Time window should be truncated near divergence
        assert result["xlim"][1] < 5.0

    def test_all_empty_times(self):
        from fmd.analysis.plots import compute_adaptive_window
        trace = {
            "times": np.array([]),
            "values": np.array([]),
            "divergence_time": None,
            "stable": True,
        }
        result = compute_adaptive_window([trace])
        assert result["xlim"] is None
        assert not result["has_divergent"]


# ---------------------------------------------------------------------------
# L0 Primitive tests
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_time_series_panels(self, sample_times):
        from fmd.analysis.plots import time_series_panels
        values = [np.sin(sample_times), np.cos(sample_times)]
        fig, axes = time_series_panels(
            sample_times, values,
            ylabels=["sin", "cos"],
            title="Test",
        )
        assert len(axes) == 2
        plt.close(fig)

    def test_time_series_panels_single(self, sample_times):
        from fmd.analysis.plots import time_series_panels
        fig, axes = time_series_panels(sample_times, [np.sin(sample_times)])
        assert len(axes) == 1
        plt.close(fig)

    def test_time_series_panels_with_reference(self, sample_times):
        from fmd.analysis.plots import time_series_panels
        fig, axes = time_series_panels(
            sample_times,
            [np.sin(sample_times)],
            reference_lines=[(0.5, "ref")],
        )
        plt.close(fig)

    def test_error_with_envelope(self, sample_times):
        from fmd.analysis.plots import error_with_envelope
        fig, ax = plt.subplots()
        error = np.sin(sample_times) * 0.1
        error_with_envelope(ax, sample_times, error, ylabel="test error")
        plt.close(fig)

    def test_bar_comparison(self):
        from fmd.analysis.plots import bar_comparison
        fig, ax = plt.subplots()
        bar_comparison(ax, ["A", "B", "C"], [1.0, 2.0, 3.0], ylabel="values")
        plt.close(fig)

    def test_overlay_with_difference(self, sample_times):
        from fmd.analysis.plots import overlay_with_difference
        a = [np.sin(sample_times), np.cos(sample_times)]
        b = [np.sin(sample_times) + 0.01, np.cos(sample_times) - 0.01]
        fig, axes = overlay_with_difference(sample_times, a, b, ylabels=["s1", "s2"])
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_step_indicator(self, sample_times):
        from fmd.analysis.plots import step_indicator
        fig, ax = plt.subplots()
        flags = np.random.choice([True, False], size=len(sample_times))
        step_indicator(ax, sample_times, flags, ylabel="Status")
        plt.close(fig)


# ---------------------------------------------------------------------------
# L1 Simulation-aware tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_state_trajectory(self, sample_times, sample_states):
        from fmd.analysis.plots import state_trajectory
        fig = state_trajectory(
            sample_times, sample_states,
            state_labels=["a", "b", "c", "d", "e"],
            title="Test Trajectory",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_state_trajectory_with_transforms(self, sample_times, sample_states):
        from fmd.analysis.plots import state_trajectory
        transforms = [None, np.degrees, None, np.degrees, None]
        fig = state_trajectory(
            sample_times, sample_states,
            state_labels=["a", "b (deg)", "c", "d (deg/s)", "e"],
            state_transforms=transforms,
        )
        plt.close(fig)

    def test_state_trajectory_with_reference(self, sample_times, sample_states):
        from fmd.analysis.plots import state_trajectory
        ref = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        fig = state_trajectory(
            sample_times, sample_states,
            state_labels=["a", "b", "c", "d", "e"],
            reference_state=ref,
        )
        plt.close(fig)

    def test_state_trajectory_prepend_t0(self, sample_times, sample_states):
        """states with N+1 rows should still work."""
        from fmd.analysis.plots import state_trajectory
        extra_state = np.vstack([sample_states[0:1], sample_states])
        fig = state_trajectory(sample_times, extra_state)
        plt.close(fig)

    def test_estimation_error(self, sample_times, sample_states):
        from fmd.analysis.plots import estimation_error
        est = sample_states + np.random.randn(*sample_states.shape) * 0.01
        fig = estimation_error(
            sample_times, sample_states, est,
            state_labels=["a", "b", "c", "d", "e"],
        )
        plt.close(fig)

    def test_nees_plot(self, sample_times):
        from fmd.analysis.plots import nees_plot
        nees = np.abs(np.random.randn(len(sample_times))) * 5 + 3
        fig = nees_plot(sample_times, nees, n_states=5)
        plt.close(fig)

    def test_covariance_diagonal(self, sample_times):
        from fmd.analysis.plots import covariance_diagonal
        P_diags = np.abs(np.random.randn(len(sample_times), 5)) + 0.01
        fig = covariance_diagonal(
            sample_times, P_diags,
            state_labels=["a", "b", "c", "d", "e"],
        )
        plt.close(fig)

    def test_control_effort(self, sample_times, sample_controls):
        from fmd.analysis.plots import control_effort
        fig = control_effort(
            sample_times, sample_controls,
            control_labels=["flap (deg)", "elev (deg)"],
            control_transforms=[np.degrees, np.degrees],
        )
        plt.close(fig)

    def test_control_effort_with_bounds(self, sample_times, sample_controls):
        from fmd.analysis.plots import control_effort
        fig = control_effort(
            sample_times, sample_controls,
            control_labels=["flap", "elev"],
            bounds=[(-10, 15), (-3, 6)],
        )
        plt.close(fig)

    def test_estimation_error_with_covariance(self, sample_times, sample_states):
        from fmd.analysis.plots import estimation_error_with_covariance
        n = len(sample_times)
        n_states = sample_states.shape[1]
        # Known errors: constant per state
        errors = np.ones((n, n_states)) * 0.1
        errors[:, 1] = 0.05  # smaller error on state 1 (angle)
        # Known covariance diags: sigma^2 values
        cov_diags = np.ones((n, n_states)) * 0.04  # sigma = 0.2
        cov_diags[:, 1] = 0.01  # sigma = 0.1 for angle state

        fig = estimation_error_with_covariance(
            sample_times, errors, cov_diags,
            state_labels=["a", "b (deg)", "c", "d (deg/s)", "e"],
        )
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == n_states
        plt.close(fig)

    def test_estimation_error_with_covariance_transforms(self, sample_times, sample_states):
        from fmd.analysis.plots import estimation_error_with_covariance
        n = len(sample_times)
        n_states = sample_states.shape[1]
        errors = np.random.randn(n, n_states) * 0.01
        cov_diags = np.abs(np.random.randn(n, n_states)) * 0.01 + 0.001
        transforms = [None, np.degrees, None, np.degrees, None]

        fig = estimation_error_with_covariance(
            sample_times, errors, cov_diags,
            state_labels=["a", "b (deg)", "c", "d (deg/s)", "e"],
            state_transforms=transforms,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == n_states
        plt.close(fig)

    def test_solver_diagnostics(self):
        from fmd.analysis.plots import solver_diagnostics
        n = 50
        solve_times = np.random.rand(n) * 0.01
        converged = np.random.choice([True, False], size=n, p=[0.9, 0.1])
        fig = solver_diagnostics(solve_times, converged)
        plt.close(fig)

    def test_solver_diagnostics_empty(self):
        from fmd.analysis.plots import solver_diagnostics
        fig = solver_diagnostics(np.array([]), np.array([], dtype=bool))
        plt.close(fig)

    def test_trajectory_comparison(self, sample_times, sample_states):
        from fmd.analysis.plots import trajectory_comparison
        states_b = sample_states + np.random.randn(*sample_states.shape) * 1e-10
        fig = trajectory_comparison(
            sample_times, sample_states, states_b,
            label_a="JAX", label_b="CasADi",
            state_labels=["a", "b", "c", "d", "e"],
        )
        plt.close(fig)

    def test_bar_comparison_by_variant(self):
        from fmd.analysis.plots import bar_comparison_by_variant
        variant_errors = {
            "full_state": np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            "speed_pitch_height": np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
        }
        fig = bar_comparison_by_variant(
            variant_errors,
            state_labels=["a", "b", "c", "d", "e"],
        )
        plt.close(fig)

    def test_multi_trace_grid(self, sample_times):
        from fmd.analysis.plots import multi_trace_grid
        traces1 = [
            {"times": sample_times, "values": np.sin(sample_times),
             "divergence_time": None, "stable": True, "label": "A"},
            {"times": sample_times, "values": np.cos(sample_times),
             "divergence_time": None, "stable": True, "label": "B"},
        ]
        traces2 = [
            {"times": sample_times, "values": sample_times * 0.1,
             "divergence_time": None, "stable": True, "label": "C"},
        ]
        fig, axes = multi_trace_grid(
            [traces1, traces2],
            reference_values=[0.0, 0.0],
            panel_labels=["Panel 1", "Panel 2"],
            title="Grid Test",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# L2 Convenience tests
# ---------------------------------------------------------------------------

class TestConvenience:
    def test_plot_simulation_result(self, sample_times, sample_states):
        from fmd.analysis.plots import plot_simulation_result
        from collections import namedtuple
        Result = namedtuple("Result", ["times", "states", "controls"])
        result = Result(sample_times, sample_states, np.zeros((len(sample_times), 2)))
        fig = plot_simulation_result(result, title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_simulation_result_with_transforms(self, sample_times, sample_states):
        from fmd.analysis.plots import (
            plot_simulation_result,
            MOTH_3DOF_STATE_LABELS,
            MOTH_3DOF_STATE_TRANSFORMS,
        )
        from collections import namedtuple
        Result = namedtuple("Result", ["times", "states", "controls"])
        result = Result(sample_times, sample_states, np.zeros((len(sample_times), 2)))
        fig = plot_simulation_result(
            result,
            state_labels=MOTH_3DOF_STATE_LABELS,
            state_transforms=MOTH_3DOF_STATE_TRANSFORMS,
        )
        plt.close(fig)

    def test_plot_lqg_result(self, sample_times, sample_states, sample_controls):
        from fmd.analysis.plots import plot_lqg_result
        from dataclasses import dataclass, field
        from fmd.simulator.params import MOTH_BIEKER_V3
        from fmd.simulator.moth_forces_extract import MothForceLog

        n = len(sample_times)

        @dataclass
        class FakeLQG:
            times: np.ndarray
            true_states: np.ndarray
            est_states: np.ndarray
            controls: np.ndarray
            covariance_traces: np.ndarray
            covariance_diagonals: np.ndarray
            estimation_errors: np.ndarray
            params: object = None
            force_log: object = None
            heel_angle: float = 0.0
            measurements_clean: np.ndarray = None
            measurements_noisy: np.ndarray = None
            innovations: np.ndarray = None

        force_log = MothForceLog(
            times=sample_times,
            main_foil_force=np.random.randn(n, 3) * 100,
            main_foil_moment=np.random.randn(n, 3),
            rudder_force=np.random.randn(n, 3) * 50,
            rudder_moment=np.random.randn(n, 3),
            sail_force=np.random.randn(n, 3) * 20,
            sail_moment=np.random.randn(n, 3),
            hull_drag_force=np.random.randn(n, 3) * 10,
            hull_drag_moment=np.random.randn(n, 3),
            gravity_force=np.tile([0.0, 0.0, 900.0], (n, 1)),
            strut_main_force=np.random.randn(n, 3) * 5,
            strut_main_moment=np.random.randn(n, 3),
            strut_rudder_force=np.random.randn(n, 3) * 3,
            strut_rudder_moment=np.random.randn(n, 3),
        )

        result = FakeLQG(
            times=sample_times,
            true_states=sample_states,
            est_states=sample_states + np.random.randn(*sample_states.shape) * 0.01,
            controls=sample_controls,
            covariance_traces=np.abs(np.random.randn(n)) + 0.01,
            covariance_diagonals=np.abs(np.random.randn(n, 5)) + 0.001,
            estimation_errors=np.random.randn(n, 5) * 0.01,
            params=MOTH_BIEKER_V3,
            force_log=force_log,
            heel_angle=np.deg2rad(30.0),
            measurements_clean=np.random.randn(n, 5) * 0.01,
            measurements_noisy=np.random.randn(n, 5) * 0.1,
            innovations=np.random.randn(n, 5) * 0.05,
        )
        figs = plot_lqg_result(result)
        assert "foiling_dashboard" in figs
        assert "estimation_dashboard" in figs
        for fig in figs.values():
            plt.close(fig)

    def test_plot_sweep_category(self, sample_times, sample_states):
        from fmd.analysis.plots import plot_sweep_category
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            description: str = "run_1"

        @dataclass
        class FakeResult:
            times: np.ndarray = None
            states: np.ndarray = None

        @dataclass
        class FakeSweepEntry:
            result: object = None
            config: object = None
            stable: bool = True
            divergence_time: float = None

        results = [
            FakeSweepEntry(
                result=FakeResult(times=sample_times, states=sample_states),
                config=FakeConfig(description="run_A"),
            ),
            FakeSweepEntry(
                result=FakeResult(times=sample_times, states=sample_states * 1.1),
                config=FakeConfig(description="run_B"),
            ),
        ]
        fig, axes = plot_sweep_category(
            results,
            state_indices=[0, 1, 4],
            state_transforms=[None, np.degrees, None],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_moth_3dof_constants(self):
        from fmd.analysis.plots import (
            MOTH_3DOF_STATE_LABELS,
            MOTH_3DOF_STATE_TRANSFORMS,
            MOTH_3DOF_CONTROL_LABELS,
            MOTH_3DOF_CONTROL_TRANSFORMS,
        )
        assert len(MOTH_3DOF_STATE_LABELS) == 5
        assert len(MOTH_3DOF_STATE_TRANSFORMS) == 5
        assert len(MOTH_3DOF_CONTROL_LABELS) == 2
        assert len(MOTH_3DOF_CONTROL_TRANSFORMS) == 2
        # Transforms for angle states should be np.degrees
        assert MOTH_3DOF_STATE_TRANSFORMS[1] is np.degrees
        assert MOTH_3DOF_STATE_TRANSFORMS[3] is np.degrees
