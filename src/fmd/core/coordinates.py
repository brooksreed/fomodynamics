"""NED coordinate frame definitions and conventions.

This module defines the coordinate frame conventions used throughout fomodynamics.

NED Frame (North-East-Down):
    - X/North: Points north (positive northward)
    - Y/East: Points east (positive eastward)
    - Z/Down: Points down (positive downward, negative = altitude)

Body Frame:
    - X/Forward: Points forward along vehicle longitudinal axis
    - Y/Right: Points to starboard/right
    - Z/Down: Points downward

Rotation Convention:
    - Roll: Rotation about X (longitudinal), positive = right side down
    - Pitch: Rotation about Y (lateral), positive = nose up
    - Yaw: Rotation about Z (vertical), positive = clockwise from north

Quaternion Convention:
    - Scalar-first: [qw, qx, qy, qz]
    - Hamilton multiplication convention
"""

# Axis indices for NED frame
NED_NORTH = 0
NED_EAST = 1
NED_DOWN = 2

# Axis indices for body frame
BODY_X = 0  # Forward
BODY_Y = 1  # Right/Starboard
BODY_Z = 2  # Down

# Frame axis names for documentation and display
NED_AXES = ("North", "East", "Down")
BODY_AXES = ("X/Forward", "Y/Right", "Z/Down")

# Euler angle indices
ROLL = 0
PITCH = 1
YAW = 2

__all__ = [
    "NED_NORTH", "NED_EAST", "NED_DOWN",
    "BODY_X", "BODY_Y", "BODY_Z",
    "NED_AXES", "BODY_AXES",
    "ROLL", "PITCH", "YAW",
]
