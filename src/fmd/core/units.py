"""Unit system with quantity types, variable schemas, and conversions.

Design principles:
- SI units internally (m/s, rad, m, s)
- Convert to display units (knots, degrees) only for output
- Circular quantities (angles) marked for special math handling
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantityType:
    """Definition of a physical quantity type."""
    
    name: str
    si_unit: str
    is_circular: bool = False
    wrap_range: tuple[float, float] | None = None  # For circular quantities
    
    def __post_init__(self):
        if self.is_circular and self.wrap_range is None:
            # Default wrap range for circular quantities
            object.__setattr__(self, 'wrap_range', (-math.pi, math.pi))


# Standard quantity types
QUANTITY_TYPES: dict[str, QuantityType] = {
    "speed": QuantityType("speed", "m/s"),
    "distance": QuantityType("distance", "m"),
    "angle": QuantityType("angle", "rad", is_circular=True),
    "angular_velocity": QuantityType("angular_velocity", "rad/s"),
    "position": QuantityType("position", "deg"),  # lat/lon stay in degrees
    "acceleration": QuantityType("acceleration", "m/s^2"),
    "time": QuantityType("time", "s"),
    "dimensionless": QuantityType("dimensionless", ""),
}


# Variable name -> quantity type mapping (standard NED frame names)
VARIABLE_SCHEMA: dict[str, str] = {
    # Time
    "time": "time",
    
    # Position/Navigation
    "latitude": "position",
    "longitude": "position",
    "sog": "speed",           # Speed over ground
    "cog": "angle",           # Course over ground
    
    # Attitude (NED frame: North-East-Down)
    # Roll: rotation around X (longitudinal), + = right side down
    # Pitch: rotation around Y (lateral), + = nose up
    # Yaw: rotation around Z (vertical), + = clockwise from North
    "roll": "angle",
    "pitch": "angle",
    "yaw": "angle",
    
    # Angular rates
    "roll_rate": "angular_velocity",
    "pitch_rate": "angular_velocity",
    "yaw_rate": "angular_velocity",
    
    # Accelerations (body frame)
    "accel_x": "acceleration",
    "accel_y": "acceleration",
    "accel_z": "acceleration",
}


# Domain-specific aliases -> (standard_name, sign_multiplier)
# Allows mapping domain terminology to standard NED frame
VARIABLE_ALIASES: dict[str, tuple[str, float]] = {
    # Sailing terminology
    "heel": ("roll", 1.0),        # heel = roll (starboard down positive)
    "trim": ("pitch", 1.0),       # trim = pitch (bow up positive)
    "hdg_true": ("yaw", 1.0),     # true heading = yaw
    
    # Common alternatives
    "heading": ("yaw", 1.0),
    "course": ("cog", 1.0),
    "speed": ("sog", 1.0),
}


# Display names per domain (standard_name -> display_name)
DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "sailing": {
        "roll": "heel",
        "pitch": "trim", 
        "yaw": "heading",
        "sog": "boat speed",
        "cog": "course",
    },
    "aviation": {
        "yaw": "heading",
        "sog": "ground speed",
        "cog": "track",
    },
    "generic": {},  # Use standard names
}


# Conversion factors: (quantity_type, source_unit) -> multiplier to SI
CONVERSIONS_TO_SI: dict[tuple[str, str], float] = {
    # Speed
    ("speed", "kts"): 0.514444,       # knots -> m/s
    ("speed", "knots"): 0.514444,
    ("speed", "mph"): 0.44704,        # mph -> m/s
    ("speed", "km/h"): 1.0 / 3.6,     # km/h -> m/s
    ("speed", "kmh"): 1.0 / 3.6,
    ("speed", "m/s"): 1.0,
    
    # Angles
    ("angle", "deg"): math.pi / 180.0,
    ("angle", "degrees"): math.pi / 180.0,
    ("angle", "rad"): 1.0,
    
    # Angular velocity
    ("angular_velocity", "deg/s"): math.pi / 180.0,
    ("angular_velocity", "rad/s"): 1.0,
    
    # Distance
    ("distance", "nm"): 1852.0,       # nautical miles -> m
    ("distance", "km"): 1000.0,
    ("distance", "mi"): 1609.344,     # miles -> m
    ("distance", "ft"): 0.3048,
    ("distance", "m"): 1.0,
    
    # Acceleration
    ("acceleration", "g"): 9.80665,   # g -> m/s^2
    ("acceleration", "m/s^2"): 1.0,
    
    # Position (stays in degrees)
    ("position", "deg"): 1.0,
    ("position", "degrees"): 1.0,
    
    # Time
    ("time", "s"): 1.0,
    ("time", "ms"): 0.001,
    ("time", "min"): 60.0,
    ("time", "h"): 3600.0,
}


# Conversion factors: quantity_type -> (display_unit, multiplier from SI)
CONVERSIONS_FROM_SI: dict[str, tuple[str, float]] = {
    "speed": ("kts", 1.0 / 0.514444),  # m/s -> knots (exact reciprocal of to-SI factor)
    "angle": ("deg", 180.0 / math.pi),  # rad -> degrees
    "angular_velocity": ("deg/s", 180.0 / math.pi),
    "distance": ("m", 1.0),
    "acceleration": ("m/s^2", 1.0),
    "position": ("deg", 1.0),
    "time": ("s", 1.0),
    "dimensionless": ("", 1.0),
}


def get_quantity_type(variable_name: str) -> QuantityType:
    """Get the QuantityType for a variable name.
    
    Args:
        variable_name: Standard variable name (e.g., 'roll', 'sog')
        
    Returns:
        QuantityType for the variable
        
    Raises:
        KeyError: If variable name is not in schema
    """
    qty_name = VARIABLE_SCHEMA[variable_name]
    return QUANTITY_TYPES[qty_name]


def resolve_alias(name: str) -> tuple[str, float]:
    """Resolve a variable alias to its standard name and sign multiplier.
    
    Args:
        name: Variable name (may be alias or standard)
        
    Returns:
        (standard_name, sign_multiplier) - if not an alias, returns (name, 1.0)
    """
    if name in VARIABLE_ALIASES:
        return VARIABLE_ALIASES[name]
    return (name, 1.0)


def convert_to_si(value: float, quantity_type: str, source_unit: str) -> float:
    """Convert a value from source units to SI.
    
    Args:
        value: The value to convert
        quantity_type: Type of quantity (e.g., 'speed', 'angle')
        source_unit: Source unit (e.g., 'kts', 'deg')
        
    Returns:
        Value in SI units
        
    Raises:
        KeyError: If conversion not found
    """
    key = (quantity_type, source_unit.lower())
    if key not in CONVERSIONS_TO_SI:
        raise KeyError(f"No conversion for ({quantity_type}, {source_unit})")
    return value * CONVERSIONS_TO_SI[key]


def convert_from_si(value: float, quantity_type: str) -> tuple[float, str]:
    """Convert a value from SI to display units.
    
    Args:
        value: The value in SI units
        quantity_type: Type of quantity (e.g., 'speed', 'angle')
        
    Returns:
        (converted_value, unit_string)
    """
    if quantity_type not in CONVERSIONS_FROM_SI:
        return (value, "")
    unit, multiplier = CONVERSIONS_FROM_SI[quantity_type]
    return (value * multiplier, unit)


def get_display_name(variable_name: str, domain: str = "generic") -> str:
    """Get the display name for a variable in a specific domain.
    
    Args:
        variable_name: Standard variable name
        domain: Domain context ('sailing', 'aviation', 'generic')
        
    Returns:
        Display name for the variable
    """
    domain_names = DISPLAY_NAMES.get(domain, {})
    return domain_names.get(variable_name, variable_name)


def is_circular(variable_name: str) -> bool:
    """Check if a variable is circular (wrapping).
    
    Args:
        variable_name: Standard variable name
        
    Returns:
        True if the variable is circular (e.g., angles)
    """
    try:
        qty_type = get_quantity_type(variable_name)
        return qty_type.is_circular
    except KeyError:
        return False

