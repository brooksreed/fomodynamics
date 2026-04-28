"""Tests for fmd.analysis.operations (analysis-only utilities).

This file exists to keep fmd.core numpy-only. Anything that requires SciPy or
pandas-index interpolation lives under fmd.analysis.
"""

import numpy as np
import pandas as pd
import pytest

from fmd.analysis.operations import (
    circular_std,
    circular_interpolate,
    linear_interpolate,
    interpolate_with_max_gap,
)


class TestCircularStd:
    """Tests for circular_std function."""

    def test_std_same_direction(self):
        """Std of same direction is near zero."""
        angles = np.radians([45, 45, 45])
        result = circular_std(angles)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_std_spread(self):
        """Std of spread angles is larger."""
        angles = np.radians([0, 90, 180, 270])
        result = circular_std(angles)
        assert result > 1.0


class TestCircularInterpolate:
    """Tests for circular_interpolate function."""

    def test_interpolate_across_zero(self):
        """Interpolation goes through 0 deg not 180 deg."""
        series = pd.Series([np.radians(350), np.radians(10)], index=[0.0, 2.0])
        new_index = pd.Index([0.0, 1.0, 2.0])
        result = circular_interpolate(series, new_index)
        assert abs(result.iloc[1]) < np.radians(5)

    def test_interpolate_preserves_endpoints(self):
        """Endpoints are preserved."""
        series = pd.Series([np.radians(45), np.radians(90)], index=[0.0, 1.0])
        new_index = pd.Index([0.0, 0.5, 1.0])
        result = circular_interpolate(series, new_index)
        assert result.iloc[0] == pytest.approx(np.radians(45), abs=1e-6)
        assert result.iloc[2] == pytest.approx(np.radians(90), abs=1e-6)

    def test_interpolate_with_nan_inside_gap(self):
        """Interpolation handles NaNs and still follows the short arc."""
        series = pd.Series([np.radians(350), np.nan, np.radians(10)], index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.0, 1.0, 2.0])
        result = circular_interpolate(series, new_index)
        assert not np.isnan(result.iloc[1])
        assert abs(result.iloc[1]) < np.radians(5)

    def test_interpolate_opposite_angles_is_ambiguous(self):
        """Halfway between opposite headings is ambiguous; return NaN."""
        series = pd.Series([0.0, np.pi], index=[0.0, 1.0])
        new_index = pd.Index([0.0, 0.5, 1.0])
        result = circular_interpolate(series, new_index)
        assert np.isnan(result.iloc[1])


class TestLinearInterpolate:
    """Tests for linear_interpolate function."""

    def test_linear_interpolate(self):
        series = pd.Series([0.0, 10.0], index=[0.0, 1.0])
        new_index = pd.Index([0.0, 0.5, 1.0])
        result = linear_interpolate(series, new_index)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(5.0)
        assert result.iloc[2] == pytest.approx(10.0)


class TestInterpolateWithMaxGap:
    """Tests for interpolate_with_max_gap function."""

    def test_respects_max_gap(self):
        series = pd.Series([1.0, 2.0], index=[0.0, 5.0])
        new_index = pd.Index([0.0, 2.5, 5.0])
        result = interpolate_with_max_gap(series, new_index, max_gap=3.0)
        assert not np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])

    def test_allows_small_gaps(self):
        series = pd.Series([1.0, 2.0], index=[0.0, 1.0])
        new_index = pd.Index([0.0, 0.5, 1.0])
        result = interpolate_with_max_gap(series, new_index, max_gap=3.0)
        assert not np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])


