"""Tests for circular math operations."""

import pytest
import numpy as np

from fmd.core import (
    wrap_angle,
    unwrap_angle,
    circular_subtract,
    circular_mean,
    angle_difference_to_vector,
)


class TestWrapAngle:
    """Tests for wrap_angle function."""

    def test_wrap_positive_overflow(self):
        """Test wrapping angle > pi."""
        result = wrap_angle(2 * np.pi)  # 360 deg
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_wrap_negative_overflow(self):
        """Test wrapping angle < -pi."""
        result = wrap_angle(-2 * np.pi)  # -360 deg
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_wrap_just_over_pi(self):
        """Test wrapping angle just over pi."""
        result = wrap_angle(np.pi + 0.1)
        expected = -np.pi + 0.1
        assert result == pytest.approx(expected, abs=1e-10)

    def test_wrap_array(self):
        """Test wrapping array of angles."""
        angles = np.array([0, np.pi, 2*np.pi, -np.pi, 3*np.pi])
        result = wrap_angle(angles)
        expected = np.array([0, -np.pi, 0, -np.pi, -np.pi])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_wrap_custom_range(self):
        """Test wrapping with custom range [0, 2pi)."""
        result = wrap_angle(-np.pi/2, low=0, high=2*np.pi)
        expected = 3*np.pi/2  # -90 deg -> 270 deg
        assert result == pytest.approx(expected, abs=1e-10)

    def test_wrap_already_in_range(self):
        """Test that in-range values are unchanged."""
        angles = np.array([-np.pi/2, 0, np.pi/2])
        result = wrap_angle(angles)
        np.testing.assert_array_almost_equal(result, angles, decimal=10)


class TestUnwrapAngle:
    """Tests for unwrap_angle function."""

    def test_unwrap_discontinuity(self):
        """Test unwrapping across 360 deg discontinuity."""
        # Angles that go 170 deg, 180 deg, -170 deg (which is really 190 deg)
        angles = np.array([170, 180, -170]) * np.pi / 180
        result = unwrap_angle(angles)

        # Should be monotonically increasing
        assert result[1] > result[0]
        assert result[2] > result[1]


class TestCircularSubtract:
    """Tests for circular_subtract function."""

    def test_subtract_across_zero(self):
        """Test subtraction across 0 deg boundary."""
        # 1 deg - 359 deg should be 2 deg, not -358 deg
        a = np.radians(1)
        b = np.radians(359)
        result = circular_subtract(a, b)
        expected = np.radians(2)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_subtract_across_zero_negative(self):
        """Test subtraction giving negative result."""
        # 359 deg - 1 deg should be -2 deg, not 358 deg
        a = np.radians(359)
        b = np.radians(1)
        result = circular_subtract(a, b)
        expected = np.radians(-2)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_subtract_same_value(self):
        """Test subtracting same value gives zero."""
        a = np.radians(180)
        result = circular_subtract(a, a)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_subtract_arrays(self):
        """Test subtracting arrays."""
        a = np.radians([1, 359, 180])
        b = np.radians([359, 1, 180])
        result = circular_subtract(a, b)
        expected = np.radians([2, -2, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_leeway_calculation(self):
        """Test typical leeway calculation (heading - COG)."""
        # Heading 45 deg, COG 40 deg -> leeway 5 deg
        heading = np.radians(45)
        cog = np.radians(40)
        leeway = circular_subtract(heading, cog)
        assert leeway == pytest.approx(np.radians(5), abs=1e-6)

        # Heading 5 deg, COG 355 deg -> leeway 10 deg
        heading = np.radians(5)
        cog = np.radians(355)
        leeway = circular_subtract(heading, cog)
        assert leeway == pytest.approx(np.radians(10), abs=1e-6)


class TestCircularMean:
    """Tests for circular_mean function."""

    def test_mean_across_zero(self):
        """Test mean of angles across 0 deg."""
        # Mean of 350 deg and 10 deg should be 0 deg, not 180 deg
        angles = np.radians([350, 10])
        result = circular_mean(angles)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_mean_opposite_directions(self):
        """Mean of opposite directions is undefined -> deterministic 0.0 (hard contract)."""
        angles = np.radians([0, 180])
        result = circular_mean(angles)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_mean_same_direction(self):
        """Test mean of same direction."""
        angles = np.radians([45, 45, 45])
        result = circular_mean(angles)
        assert result == pytest.approx(np.radians(45), abs=1e-6)

    def test_weighted_mean(self):
        """Test weighted circular mean."""
        angles = np.radians([0, 90])
        weights = np.array([3, 1])  # Weight towards 0 deg
        result = circular_mean(angles, weights)
        # Should be between 0 deg and 45 deg, closer to 0 deg
        assert 0 <= result < np.radians(45)


class TestAngleDifferenceToVector:
    """Tests for angle_difference_to_vector function."""

    def test_target_ahead(self):
        """Test when target is directly ahead."""
        heading = np.radians(45)
        target = np.radians(45)
        lateral, longitudinal = angle_difference_to_vector(heading, target)

        assert lateral == pytest.approx(0.0, abs=1e-10)
        assert longitudinal == pytest.approx(1.0, abs=1e-10)

    def test_target_right(self):
        """Test when target is 90 deg to the right."""
        heading = np.radians(0)
        target = np.radians(90)
        lateral, longitudinal = angle_difference_to_vector(heading, target)

        assert lateral == pytest.approx(1.0, abs=1e-10)  # Right
        assert longitudinal == pytest.approx(0.0, abs=1e-10)

    def test_target_left(self):
        """Test when target is 90 deg to the left."""
        heading = np.radians(0)
        target = np.radians(-90)
        lateral, longitudinal = angle_difference_to_vector(heading, target)

        assert lateral == pytest.approx(-1.0, abs=1e-10)  # Left
        assert longitudinal == pytest.approx(0.0, abs=1e-10)
