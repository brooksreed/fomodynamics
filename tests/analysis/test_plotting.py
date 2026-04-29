"""Smoke tests for plotting modules.

These tests verify that plotting modules can be imported and that
basic function signatures work. They do NOT test actual plot rendering
to avoid matplotlib backend issues in CI/headless environments.
"""

import numpy as np
import pandas as pd
import pytest


class TestPlotImports:
    """Test that plotting modules can be imported."""

    def test_import_plots_module(self):
        """Test direct import of plots module."""
        from fmd.analysis import plots

        assert hasattr(plots, "plot_time_series")
        assert hasattr(plots, "plot_polar")
        assert hasattr(plots, "radians_to_degrees")

    def test_import_from_analysis_package(self):
        """Test that plotting functions are exported from fmd.analysis."""
        from fmd.analysis import plot_time_series, plot_polar

        assert callable(plot_time_series)
        assert callable(plot_polar)

    def test_import_matplotlib_dependencies(self):
        """Test that matplotlib dependencies are available."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        assert hasattr(plt, "subplots")
        assert hasattr(mdates, "AutoDateLocator")


class TestHelperFunctions:
    """Test helper functions that don't require matplotlib rendering."""

    def test_radians_to_degrees(self):
        """Test radians_to_degrees conversion."""
        from fmd.analysis.plots import radians_to_degrees

        # Test basic conversion
        # Note: radians_to_degrees wraps to [-180, 180), so pi -> -180
        angles_rad = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
        angles_deg = radians_to_degrees(angles_rad)

        np.testing.assert_allclose(angles_deg, [0, 90, -180, -90], atol=1e-10)

    def test_radians_to_degrees_wrapping(self):
        """Test that radians_to_degrees wraps correctly."""
        from fmd.analysis.plots import radians_to_degrees

        # Values that should wrap to [-180, 180)
        angles_rad = np.array([2 * np.pi, -2 * np.pi, 3 * np.pi])
        angles_deg = radians_to_degrees(angles_rad)

        # All should be wrapped to [-180, 180)
        assert np.all(angles_deg >= -180)
        assert np.all(angles_deg < 180)

    def test_get_display_info(self):
        """Test _get_display_info helper function."""
        from fmd.analysis.plots import _get_display_info

        # Test known column
        display_name, unit, multiplier = _get_display_info("speed", "generic")
        assert isinstance(display_name, str)
        assert isinstance(unit, str)
        assert isinstance(multiplier, (int, float))

        # Test unknown column (should return defaults)
        display_name, unit, multiplier = _get_display_info("unknown_column", "generic")
        assert display_name == "unknown_column"
        assert multiplier == 1.0


class TestPlotFunctionSignatures:
    """Test that plot functions accept expected arguments.

    These tests verify function signatures without actually rendering plots.
    We use mocking to prevent matplotlib from creating windows.
    """

    def test_plot_time_series_requires_time_column(self):
        """Test that plot_time_series raises error without time column."""
        from fmd.analysis.plots import plot_time_series

        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="requires a 'time' column"):
            plot_time_series(df)

    def test_plot_time_series_requires_numeric_columns(self):
        """Test that plot_time_series raises error without numeric columns."""
        from fmd.analysis.plots import plot_time_series

        df = pd.DataFrame({"time": [1, 2, 3], "category": ["a", "b", "c"]})

        with pytest.raises(ValueError, match="No numeric columns to plot"):
            plot_time_series(df)

    def test_plot_time_series_x_length_mismatch(self):
        """Test that plot_time_series raises error for x length mismatch."""
        from fmd.analysis.plots import plot_time_series

        df = pd.DataFrame({"time": [1, 2, 3], "value": [1, 2, 3]})
        x = pd.Series([1, 2])  # Wrong length

        with pytest.raises(ValueError, match="x must be the same length"):
            plot_time_series(df, x=x)

    def test_plot_time_series_accepts_datastream(self, monkeypatch):
        """Test that plot_time_series accepts DataStream input."""
        from fmd.analysis.plots import plot_time_series
        from fmd.analysis.core import DataStream

        # Mock plt.show() to prevent actual rendering
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        df = pd.DataFrame({"time": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
        stream = DataStream(df, name="test_stream")

        # Should not raise
        plot_time_series(stream)
        plt.close("all")

    def test_plot_polar_accepts_dataframe(self, monkeypatch):
        """Test that plot_polar accepts DataFrame input."""
        from fmd.analysis.plots import plot_polar

        # Mock plt.show() to prevent actual rendering
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        df = pd.DataFrame({
            "angle": [0, np.pi / 2, np.pi],
            "radius": [1.0, 2.0, 3.0],
        })

        # Should not raise
        plot_polar(df, angle_col="angle", radius_col="radius")
        plt.close("all")

    def test_plot_polar_accepts_datastream(self, monkeypatch):
        """Test that plot_polar accepts DataStream input."""
        from fmd.analysis.plots import plot_polar
        from fmd.analysis.core import DataStream

        # Mock plt.show() to prevent actual rendering
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        df = pd.DataFrame({
            "angle": [0, np.pi / 2, np.pi],
            "radius": [1.0, 2.0, 3.0],
        })
        stream = DataStream(df, name="test_stream")

        # Should not raise
        plot_polar(stream, angle_col="angle", radius_col="radius")
        plt.close("all")


class TestAutosizeHelper:
    """Test the _autosize_figure_to_screen helper."""

    def test_autosize_function_exists(self):
        """Test that _autosize_figure_to_screen exists and is callable."""
        from fmd.analysis.plots import _autosize_figure_to_screen

        assert callable(_autosize_figure_to_screen)

    def test_autosize_handles_headless_gracefully(self, monkeypatch):
        """Test that _autosize_figure_to_screen doesn't crash in headless mode."""
        from fmd.analysis.plots import _autosize_figure_to_screen
        import matplotlib.pyplot as plt

        # Create a figure (in headless mode, this uses Agg backend)
        fig, ax = plt.subplots()

        # Should not raise even in headless/Agg backend
        _autosize_figure_to_screen(fig)
        plt.close(fig)
