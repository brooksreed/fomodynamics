"""Tests for the unit system."""

import math
import pytest
import numpy as np

from fmd.core import (
    QuantityType,
    QUANTITY_TYPES,
    VARIABLE_SCHEMA,
    VARIABLE_ALIASES,
    get_quantity_type,
    resolve_alias,
    convert_to_si,
    convert_from_si,
    get_display_name,
    is_circular,
)


class TestQuantityType:
    """Tests for QuantityType dataclass."""

    def test_basic_quantity_type(self):
        """Test creating a basic quantity type."""
        qty = QuantityType("speed", "m/s")
        assert qty.name == "speed"
        assert qty.si_unit == "m/s"
        assert qty.is_circular is False
        assert qty.wrap_range is None

    def test_circular_quantity_type(self):
        """Test creating a circular quantity type."""
        qty = QuantityType("angle", "rad", is_circular=True)
        assert qty.is_circular is True
        # Should get default wrap range
        assert qty.wrap_range == (-math.pi, math.pi)

    def test_circular_with_custom_range(self):
        """Test circular quantity with custom wrap range."""
        qty = QuantityType("angle", "rad", is_circular=True, wrap_range=(0, 2*math.pi))
        assert qty.wrap_range == (0, 2*math.pi)

    def test_standard_quantity_types(self):
        """Test that standard quantity types are defined."""
        assert "speed" in QUANTITY_TYPES
        assert "angle" in QUANTITY_TYPES
        assert "angular_velocity" in QUANTITY_TYPES

        assert QUANTITY_TYPES["angle"].is_circular is True
        assert QUANTITY_TYPES["speed"].is_circular is False
        assert QUANTITY_TYPES["angular_velocity"].is_circular is False


class TestVariableSchema:
    """Tests for variable schema and aliases."""

    def test_ned_frame_variables_defined(self):
        """Test that NED frame variables are in schema."""
        ned_vars = ["roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate"]
        for var in ned_vars:
            assert var in VARIABLE_SCHEMA

    def test_navigation_variables_defined(self):
        """Test that navigation variables are in schema."""
        nav_vars = ["sog", "cog", "latitude", "longitude"]
        for var in nav_vars:
            assert var in VARIABLE_SCHEMA

    def test_attitude_variables_are_angles(self):
        """Test that attitude variables are marked as angles."""
        assert VARIABLE_SCHEMA["roll"] == "angle"
        assert VARIABLE_SCHEMA["pitch"] == "angle"
        assert VARIABLE_SCHEMA["yaw"] == "angle"
        assert VARIABLE_SCHEMA["cog"] == "angle"

    def test_sailing_aliases(self):
        """Test that sailing aliases map to standard names."""
        assert "heel" in VARIABLE_ALIASES
        assert "trim" in VARIABLE_ALIASES
        assert "hdg_true" in VARIABLE_ALIASES

        assert VARIABLE_ALIASES["heel"] == ("roll", 1.0)
        assert VARIABLE_ALIASES["trim"] == ("pitch", 1.0)
        assert VARIABLE_ALIASES["hdg_true"] == ("yaw", 1.0)


class TestGetQuantityType:
    """Tests for get_quantity_type function."""

    def test_get_known_quantity(self):
        """Test getting quantity type for known variable."""
        qty = get_quantity_type("roll")
        assert qty.name == "angle"
        assert qty.is_circular is True

    def test_get_unknown_quantity_raises(self):
        """Test that unknown variable raises KeyError."""
        with pytest.raises(KeyError):
            get_quantity_type("unknown_variable")


class TestResolveAlias:
    """Tests for resolve_alias function."""

    def test_resolve_alias(self):
        """Test resolving a known alias."""
        std_name, sign = resolve_alias("heel")
        assert std_name == "roll"
        assert sign == 1.0

    def test_resolve_non_alias(self):
        """Test resolving a name that's not an alias."""
        std_name, sign = resolve_alias("roll")
        assert std_name == "roll"
        assert sign == 1.0

    def test_resolve_unknown_name(self):
        """Test resolving an unknown name (returns as-is)."""
        std_name, sign = resolve_alias("some_custom_var")
        assert std_name == "some_custom_var"
        assert sign == 1.0


class TestConversions:
    """Tests for unit conversion functions."""

    def test_convert_knots_to_si(self):
        """Test converting knots to m/s."""
        result = convert_to_si(1.0, "speed", "kts")
        assert result == pytest.approx(0.514444, rel=1e-4)

    def test_convert_degrees_to_si(self):
        """Test converting degrees to radians."""
        result = convert_to_si(180.0, "angle", "deg")
        assert result == pytest.approx(math.pi, rel=1e-6)

    def test_convert_unknown_unit_raises(self):
        """Test that unknown unit raises KeyError."""
        with pytest.raises(KeyError):
            convert_to_si(1.0, "speed", "unknown_unit")

    def test_convert_from_si_speed(self):
        """Test converting m/s to display units (knots)."""
        value, unit = convert_from_si(1.0, "speed")
        assert unit == "kts"
        assert value == pytest.approx(1.94384, rel=1e-4)

    def test_convert_from_si_angle(self):
        """Test converting radians to display units (degrees)."""
        value, unit = convert_from_si(math.pi, "angle")
        assert unit == "deg"
        assert value == pytest.approx(180.0, rel=1e-6)

    def test_round_trip_conversion(self):
        """Test that to_si -> from_si gives original value."""
        original_knots = 10.0
        si_value = convert_to_si(original_knots, "speed", "kts")
        back, _ = convert_from_si(si_value, "speed")
        assert back == pytest.approx(original_knots, rel=1e-4)


class TestDisplayName:
    """Tests for get_display_name function."""

    def test_sailing_display_names(self):
        """Test display names in sailing domain."""
        assert get_display_name("roll", "sailing") == "heel"
        assert get_display_name("pitch", "sailing") == "trim"
        assert get_display_name("yaw", "sailing") == "heading"

    def test_generic_display_names(self):
        """Test display names in generic domain."""
        assert get_display_name("roll", "generic") == "roll"
        assert get_display_name("pitch", "generic") == "pitch"

    def test_unknown_variable_returns_as_is(self):
        """Test that unknown variable returns its name."""
        assert get_display_name("custom_var", "sailing") == "custom_var"


class TestIsCircular:
    """Tests for is_circular function."""

    def test_angles_are_circular(self):
        """Test that angle variables are circular."""
        assert is_circular("roll") is True
        assert is_circular("pitch") is True
        assert is_circular("yaw") is True
        assert is_circular("cog") is True

    def test_rates_are_not_circular(self):
        """Test that angular rates are not circular."""
        assert is_circular("roll_rate") is False
        assert is_circular("pitch_rate") is False

    def test_other_variables_not_circular(self):
        """Test that non-angle variables are not circular."""
        assert is_circular("sog") is False
        assert is_circular("latitude") is False

    def test_unknown_variable_not_circular(self):
        """Test that unknown variables default to not circular."""
        assert is_circular("unknown_var") is False
