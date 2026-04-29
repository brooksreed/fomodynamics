"""fmd.core - Shared math, units, and abstractions for fomodynamics.

This package provides the foundational utilities used across the fomodynamics
package:
- Quaternion math (scalar-first Hamilton convention)
- Rotation utilities (DCM, Euler angles)
- NED coordinate frame definitions
- Physical unit system with SI internals
- Circular math operations for angles
- Base abstractions for dynamic systems
- JAX configuration (GPU auto-detection, memory management)

Submodule Dependencies (for CasADi compatibility):
    - quaternion: numpy only
    - rotation: numpy only (imports from quaternion)
    - coordinates: stdlib only
    - units: stdlib only
    - operations: numpy only
    - abc: numpy only
    - jax_config: jax (optional GPU support)

Import fmd.simulator or fmd.analysis packages separately - they are not
included here to avoid circular dependencies and unnecessary imports.
"""

# Quaternion operations (numpy only, CasADi-compatible)
from fmd.core.quaternion import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_derivative,
    rotate_vector,
    rotate_vector_inverse,
    identity_quat,
    quaternion_distance,
    # Also export rotation functions directly
    quat_to_dcm,
    dcm_to_quat,
    quat_to_euler,
    euler_to_quat,
)

# Coordinate frame definitions
from fmd.core.coordinates import (
    NED_NORTH, NED_EAST, NED_DOWN,
    BODY_X, BODY_Y, BODY_Z,
    NED_AXES, BODY_AXES,
    ROLL, PITCH, YAW,
)

# Unit system (stdlib only)
from fmd.core.units import (
    QuantityType,
    QUANTITY_TYPES,
    VARIABLE_SCHEMA,
    VARIABLE_ALIASES,
    DISPLAY_NAMES,
    CONVERSIONS_TO_SI,
    CONVERSIONS_FROM_SI,
    get_quantity_type,
    resolve_alias,
    convert_to_si,
    convert_from_si,
    get_display_name,
    is_circular,
)

# Circular math operations (numpy only)
from fmd.core.operations import (
    wrap_angle,
    unwrap_angle,
    circular_subtract,
    circular_mean,
    angle_difference_to_vector,
)

# Base abstractions
from fmd.core.abc import DynamicSystem, DynamicsProtocol

# JAX configuration (GPU auto-detection, memory management, dtype)
from fmd.core.jax_config import (
    configure_jax,
    get_device_info,
    is_gpu_available,
    DeviceInfo,
    DEFAULT_GPU_MEMORY_FRACTION,
    FMD_DTYPE,
    FMD_NP_DTYPE,
    get_fmd_dtype,
    get_fmd_np_dtype,
)

__all__ = [
    # JAX configuration
    "configure_jax",
    "get_device_info",
    "is_gpu_available",
    "DeviceInfo",
    "DEFAULT_GPU_MEMORY_FRACTION",
    "FMD_DTYPE",
    "FMD_NP_DTYPE",
    "get_fmd_dtype",
    "get_fmd_np_dtype",
    # Quaternion operations
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_derivative",
    "rotate_vector",
    "rotate_vector_inverse",
    "identity_quat",
    "quaternion_distance",
    # Rotation utilities
    "quat_to_dcm",
    "dcm_to_quat",
    "quat_to_euler",
    "euler_to_quat",
    # Coordinate frames
    "NED_NORTH", "NED_EAST", "NED_DOWN",
    "BODY_X", "BODY_Y", "BODY_Z",
    "NED_AXES", "BODY_AXES",
    "ROLL", "PITCH", "YAW",
    # Unit system
    "QuantityType",
    "QUANTITY_TYPES",
    "VARIABLE_SCHEMA",
    "VARIABLE_ALIASES",
    "DISPLAY_NAMES",
    "CONVERSIONS_TO_SI",
    "CONVERSIONS_FROM_SI",
    "get_quantity_type",
    "resolve_alias",
    "convert_to_si",
    "convert_from_si",
    "get_display_name",
    "is_circular",
    # Circular math
    "wrap_angle",
    "unwrap_angle",
    "circular_subtract",
    "circular_mean",
    "angle_difference_to_vector",
    # Base abstractions
    "DynamicSystem",
    "DynamicsProtocol",
]
