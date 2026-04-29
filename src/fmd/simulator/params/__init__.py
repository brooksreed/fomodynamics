"""Parameter management for fomodynamics simulator models.

This package provides immutable, validated parameter classes for all
public simulator models. Parameters are frozen attrs classes with JAX
PyTree compatibility for domain randomization and gradient-based
optimization.

Surface: generic primitives + moth + simple/teaching models
(boat_2d, box_1d, cartpole, pendulum, planar_quadrotor) + wave params.

Example:
    >>> from fmd.simulator.params import Boat2DParams, SIMPLE_MOTORBOAT
    >>>
    >>> boat = Boat2D(SIMPLE_MOTORBOAT)
    >>>
    >>> params = Boat2DParams(
    ...     mass=150.0, izz=80.0,
    ...     drag_surge=50.0, drag_sway=400.0, drag_yaw=100.0,
    ... )

For JAX PyTree registration (domain randomization, vmap):
    >>> import fmd.simulator.params._pytree  # Side-effect import
"""

# Base infrastructure
from fmd.simulator.params.base import (
    STANDARD_GRAVITY,
    WATER_DENSITY_FRESH,
    WATER_DENSITY_SALT,
    AIR_DENSITY_SL,
    is_finite,
    is_finite_array,
    is_valid_inertia,
    positive,
    non_negative,
    positive_array,
    is_3vector,
    to_float_array,
)

# Public param classes
from fmd.simulator.params.boat_2d import Boat2DParams
from fmd.simulator.params.rigid_body import RigidBody6DOFParams
from fmd.simulator.params.pendulum import SimplePendulumParams
from fmd.simulator.params.cartpole import CartpoleParams
from fmd.simulator.params.planar_quadrotor import PlanarQuadrotorParams
from fmd.simulator.params.box_1d import (
    Box1DParams,
    Box1DFrictionParams,
    BOX1D_DEFAULT,
    BOX1D_FRICTION_DEFAULT,
)
from fmd.simulator.params.wave import WaveParams
from fmd.simulator.params.moth import MothParams

# Public presets
from fmd.simulator.params.presets import (
    BOAT2D_TEST_DEFAULT,
    SIMPLE_MOTORBOAT,
    DISPLACEMENT_SAILBOAT,
    RIGIDBODY_TEST_SYMMETRIC,
    RIGIDBODY_TEST_ASYMMETRIC,
    PENDULUM_1M,
    PENDULUM_2M,
    SECONDS_PENDULUM,
    CARTPOLE_CLASSIC,
    CARTPOLE_HEAVY_POLE,
    CARTPOLE_LONG_POLE,
    PLANAR_QUAD_CRAZYFLIE,
    PLANAR_QUAD_TEST_DEFAULT,
    PLANAR_QUAD_HEAVY,
    WAVE_CALM,
    WAVE_MODERATE,
    WAVE_REGULAR_1M,
    WAVE_SF_BAY_LIGHT,
    WAVE_SF_BAY_MODERATE,
    MOTH_BIEKER_V3,
)

__all__ = [
    # Constants
    "STANDARD_GRAVITY",
    "WATER_DENSITY_FRESH",
    "WATER_DENSITY_SALT",
    "AIR_DENSITY_SL",
    # Validators
    "is_finite",
    "is_finite_array",
    "is_valid_inertia",
    "positive",
    "non_negative",
    "positive_array",
    "is_3vector",
    "to_float_array",
    # Param classes
    "Boat2DParams",
    "RigidBody6DOFParams",
    "SimplePendulumParams",
    "CartpoleParams",
    "PlanarQuadrotorParams",
    "Box1DParams",
    "Box1DFrictionParams",
    "WaveParams",
    "MothParams",
    # Box1D presets
    "BOX1D_DEFAULT",
    "BOX1D_FRICTION_DEFAULT",
    # Boat2D presets
    "BOAT2D_TEST_DEFAULT",
    "SIMPLE_MOTORBOAT",
    "DISPLACEMENT_SAILBOAT",
    # RigidBody presets
    "RIGIDBODY_TEST_SYMMETRIC",
    "RIGIDBODY_TEST_ASYMMETRIC",
    # Pendulum presets
    "PENDULUM_1M",
    "PENDULUM_2M",
    "SECONDS_PENDULUM",
    # Cartpole presets
    "CARTPOLE_CLASSIC",
    "CARTPOLE_HEAVY_POLE",
    "CARTPOLE_LONG_POLE",
    # PlanarQuadrotor presets
    "PLANAR_QUAD_CRAZYFLIE",
    "PLANAR_QUAD_TEST_DEFAULT",
    "PLANAR_QUAD_HEAVY",
    # Wave presets
    "WAVE_CALM",
    "WAVE_MODERATE",
    "WAVE_REGULAR_1M",
    "WAVE_SF_BAY_LIGHT",
    "WAVE_SF_BAY_MODERATE",
    # Moth preset
    "MOTH_BIEKER_V3",
]
