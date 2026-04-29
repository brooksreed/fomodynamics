"""Tests for fmd.analysis.units module."""

import math
import pytest

from fmd.core.units import (
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


class TestQuantityTypes:
    """Test quantity type definitions."""

    def test_speed_type(self):
        """Speed quantity type is defined correctly."""
        speed = QUANTITY_TYPES["speed"]
        assert speed.si_unit == "m/s"
        assert not speed.is_circular

    def test_angle_type(self):
        """Angle quantity type is circular."""
        angle = QUANTITY_TYPES["angle"]
        assert angle.si_unit == "rad"
        assert angle.is_circular
        assert angle.wrap_range == (-math.pi, math.pi)


class TestVariableSchema:
    """Test variable schema mapping."""

    def test_attitude_variables(self):
        """Attitude variables are defined."""
        assert "roll" in VARIABLE_SCHEMA
        assert "pitch" in VARIABLE_SCHEMA
        assert "yaw" in VARIABLE_SCHEMA
        assert VARIABLE_SCHEMA["roll"] == "angle"

    def test_navigation_variables(self):
        """Navigation variables are defined."""
        assert "sog" in VARIABLE_SCHEMA
        assert "cog" in VARIABLE_SCHEMA
        assert VARIABLE_SCHEMA["sog"] == "speed"


class TestVariableAliases:
    """Test domain-specific aliases."""

    def test_sailing_aliases(self):
        """Sailing terminology maps to NED frame."""
        assert "heel" in VARIABLE_ALIASES
        assert "trim" in VARIABLE_ALIASES
        assert VARIABLE_ALIASES["heel"] == ("roll", 1.0)


class TestConversions:
    """Test unit conversions."""

    def test_knots_to_ms(self):
        """Convert knots to m/s."""
        result = convert_to_si(1.0, "speed", "kts")
        assert pytest.approx(result) == 0.514444

    def test_degrees_to_radians(self):
        """Convert degrees to radians."""
        result = convert_to_si(180.0, "angle", "deg")
        assert pytest.approx(result) == math.pi

    def test_ms_to_knots(self):
        """Convert m/s to knots for display."""
        value, unit = convert_from_si(1.0, "speed")
        assert pytest.approx(value) == 1.0 / 0.514444
        assert unit == "kts"


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_quantity_type(self):
        """Get quantity type for variable."""
        qty = get_quantity_type("roll")
        assert qty.name == "angle"
        assert qty.is_circular

    def test_resolve_alias(self):
        """Resolve sailing alias."""
        name, mult = resolve_alias("heel")
        assert name == "roll"
        assert mult == 1.0

    def test_resolve_non_alias(self):
        """Non-alias returns itself."""
        name, mult = resolve_alias("roll")
        assert name == "roll"
        assert mult == 1.0

    def test_is_circular(self):
        """Check circular detection."""
        assert is_circular("roll")
        assert is_circular("yaw")
        assert not is_circular("sog")
        assert not is_circular("unknown")

    def test_get_display_name(self):
        """Get domain-specific display names."""
        assert get_display_name("roll", "sailing") == "heel"
        assert get_display_name("roll", "generic") == "roll"
