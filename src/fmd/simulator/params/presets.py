"""Preset parameter configurations for fomodynamics simulator models.

This module provides canonical parameter instances that can be imported
and used directly across tests, notebooks, and documentation.

Import presets directly:
    from fmd.simulator.params import SIMPLE_MOTORBOAT, MOTH_BIEKER_V3

All presets are frozen (immutable) instances. To modify, use attrs.evolve()
or the with_* methods on the parameter classes.

All presets use SI units and match the model parameter conventions.
"""

import numpy as np

from fmd.simulator.params.boat_2d import Boat2DParams
from fmd.simulator.params.rigid_body import RigidBody6DOFParams
from fmd.simulator.params.pendulum import SimplePendulumParams
from fmd.simulator.params.cartpole import CartpoleParams
from fmd.simulator.params.planar_quadrotor import PlanarQuadrotorParams
from fmd.simulator.params.moth import MothParams
from fmd.simulator.params.wave import WaveParams


# ============================================================================
# Boat2D Presets
# ============================================================================

BOAT2D_TEST_DEFAULT = Boat2DParams(
    mass=100.0,
    izz=50.0,
    drag_surge=10.0,
    drag_sway=20.0,
    drag_yaw=5.0,
)
"""Reference Boat2D parameters for unit tests.

A generic test boat with moderate drag values. Matches the parameters
historically used in the test suite.

Properties:
    - Surge time constant: 10.0 s
    - Yaw time constant: 10.0 s
    - Steady state surge at 100N thrust: 10.0 m/s
"""

SIMPLE_MOTORBOAT = Boat2DParams(
    mass=125.0,
    izz=100.0,
    drag_surge=60.0,
    drag_sway=500.0,
    drag_yaw=135.0,
)
"""Small inflatable/aluminum motorboat (~10ft).

Representative parameters for a small recreational motorboat.
Higher sway drag than surge gives realistic sideslip behavior.

Properties:
    - Surge time constant: ~2.1 s
    - Yaw time constant: ~0.74 s
"""

DISPLACEMENT_SAILBOAT = Boat2DParams(
    mass=3000.0,
    izz=8000.0,
    drag_surge=200.0,
    drag_sway=2000.0,
    drag_yaw=500.0,
)
"""Displacement sailboat (~30ft).

Representative parameters for a medium displacement sailboat.

Properties:
    - Surge time constant: 15.0 s
    - Yaw time constant: 16.0 s
"""


# ============================================================================
# RigidBody6DOF Presets
# ============================================================================

RIGIDBODY_TEST_SYMMETRIC = RigidBody6DOFParams(
    mass=1.0,
    inertia=np.array([1.0, 1.0, 1.0]),
)
"""Symmetric rigid body for basic tests.

Equal inertia in all axes - simplest case for verification.
"""

RIGIDBODY_TEST_ASYMMETRIC = RigidBody6DOFParams(
    mass=1.0,
    inertia=np.array([1.0, 2.0, 3.0]),
)
"""Asymmetric rigid body for intermediate axis instability tests.

The [1, 2, 3] inertia is specifically chosen to demonstrate the
Dzhanibekov effect (tennis racket theorem) - spinning about the
intermediate axis (Iyy = 2) is unstable.
"""


# ============================================================================
# SimplePendulum Presets
# ============================================================================

PENDULUM_1M = SimplePendulumParams(length=1.0)
"""1-meter reference pendulum.

Standard test pendulum. Period for small oscillations: ~2.006 s
"""

PENDULUM_2M = SimplePendulumParams(length=2.0)
"""2-meter pendulum for scaling tests."""

SECONDS_PENDULUM = SimplePendulumParams(length=0.994)
"""Seconds pendulum (half-period = 1 second).

A seconds pendulum has a period of exactly 2 seconds, meaning the
pendulum takes 1 second to swing from one extreme to the other.
At standard gravity, this requires L = g/pi^2 ≈ 0.994 m.
"""


# ============================================================================
# Cartpole Presets
# ============================================================================

CARTPOLE_CLASSIC = CartpoleParams(
    mass_cart=1.0,
    mass_pole=0.1,
    pole_length=0.5,
)
"""Classic OpenAI Gym Cartpole parameters.

Standard parameters from Barto, Sutton, Anderson (1983) and OpenAI Gym.
The pole is relatively light (10% of cart mass) making it challenging
but tractable for basic control and RL.

Properties:
    - Total mass: 1.1 kg
    - Mass ratio: 0.091
    - Linearized period: ~1.42 s
"""

CARTPOLE_HEAVY_POLE = CartpoleParams(
    mass_cart=1.0,
    mass_pole=0.5,
    pole_length=0.5,
)
"""Cartpole with heavier pole (more challenging to balance).

Same cart mass and pole length as classic, but 5x heavier pole.
This makes balancing more challenging as the pole has more momentum.

Properties:
    - Total mass: 1.5 kg
    - Mass ratio: 0.333
    - Linearized period: ~1.42 s
"""

CARTPOLE_LONG_POLE = CartpoleParams(
    mass_cart=1.0,
    mass_pole=0.1,
    pole_length=1.0,
)
"""Cartpole with longer pole.

Double the pole length of the classic. Longer poles are easier to
balance (like balancing a broomstick vs a pencil) due to slower
rotational dynamics.

Properties:
    - Total mass: 1.1 kg
    - Linearized period: ~2.01 s (slower, easier to balance)
"""

# ============================================================================
# PlanarQuadrotor Presets
# ============================================================================

PLANAR_QUAD_CRAZYFLIE = PlanarQuadrotorParams(
    mass=0.030,
    arm_length=0.0397,
    inertia_pitch=1.4e-5,
)
"""Crazyflie 2.1 parameters (2D planar projection).

Based on the Bitcraze Crazyflie 2.1 nano quadcopter, a popular
platform for control research. Very light and agile.

Properties:
    - Hover thrust per rotor: ~0.147 N
    - Very low inertia enables fast pitch response
"""

PLANAR_QUAD_TEST_DEFAULT = PlanarQuadrotorParams(
    mass=1.0,
    arm_length=0.25,
    inertia_pitch=0.01,
)
"""Simple test quadrotor with round numbers.

1 kg mass with 25 cm arms. Easy to verify calculations mentally.

Properties:
    - Hover thrust per rotor: ~4.9 N
    - Moment for 1 rad/s^2 pitch accel: 0.01 N*m
"""

PLANAR_QUAD_HEAVY = PlanarQuadrotorParams(
    mass=2.0,
    arm_length=0.3,
    inertia_pitch=0.04,
)
"""Heavier quadrotor for stress testing.

Representative of a larger photography drone in 2D.

Properties:
    - Hover thrust per rotor: ~9.8 N
    - Slower pitch response due to higher inertia
"""

# ============================================================================
# Moth Sailboat Presets
# ============================================================================

MOTH_BIEKER_V3 = MothParams(
    # Hull — non-sailor boat mass (hull shell + rig + foils + rigging)
    hull_mass=50.0,
    hull_inertia=np.array([91.1, 118.6, 31.3]),  # Estimated via estimate_moth_inertia.py
    hull_length=3.355,  # 11ft LOA = 3.355m
    hull_beam=0.42,  # 42cm max external width (measured; not waterline beam)

    # Sailor
    sailor_mass=75.0,
    sailor_position=np.array([-0.30, 0.0, -0.2]),  # Aft, above hull CG (measured: sailor CG 2.29m from bow)

    # Main foil (measured Bieker V3 dimensions)
    main_foil_span=0.95,  # 37.5" measured
    main_foil_chord=0.089,  # 3.5" measured
    main_foil_area=0.08455,  # 0.95 * 0.089
    main_foil_cl_alpha=6.0,
    main_foil_cl0=0.15,  # Camber lift: moth foils have cambered sections
    main_foil_cd0=0.006,
    main_foil_cd0_section=0.004,
    main_foil_cd0_parasitic=0.002,
    main_foil_oswald=0.85,
    main_foil_flap_effectiveness=0.5,
    main_foil_cd_flap=0.15,
    main_strut_chord=0.09,
    main_strut_thickness=0.013,
    main_strut_cd_pressure=0.01,
    main_strut_cf_skin=0.003,

    # Rudder (measured dimensions)
    rudder_span=0.68,  # 27" measured
    rudder_chord=0.075,  # 3" measured
    rudder_area=0.051,  # 0.68 * 0.075
    rudder_elevator_min=np.radians(-3.0),
    rudder_elevator_max=np.radians(6.0),
    rudder_cl_alpha=5.5,
    rudder_cd0=0.008,
    rudder_oswald=0.85,
    rudder_strut_chord=0.07,
    rudder_strut_thickness=0.010,
    rudder_strut_cd_pressure=0.01,
    rudder_strut_cf_skin=0.003,

    # Wand (measured from Bieker V3)
    wand_length=1.175,  # midpoint of 0.95-1.40m adjustable range
    wand_gearing_ratio=130.0 / 170.0,  # 170mm rod, output tap at 130mm

    # Sail (class rules)
    sail_area=8.0,

    # Hull geometry (hull-datum reference frame)
    hull_depth=0.45,
    hull_cg_above_bottom=0.82,  # From component CG estimate (estimate_moth_inertia)
    hull_cg_from_bow=1.99,  # Direct measurement (mast stump 110cm + 89cm to CG)
    main_foil_strut_depth=1.03,  # Main strut below hull bottom (measured)
    rudder_strut_depth=0.95,  # Rudder strut below hull bottom
    wing_rack_span=2.25,  # Tip-to-tip
    wing_dihedral=0.5236,  # 30 degrees

    # Hull-datum structural positions
    main_foil_from_bow=1.57,  # Main vertical case (measured)
    wing_rack_from_bow=2.0,
    rudder_from_bow=3.855,  # hull 3.355 + gantry 0.5
    sail_ce_hull_datum=np.array([2.5, 0.0, 2.0]),  # CE 1.55m above deck; aft past CG per aerodynamic estimates
    bowsprit_hull_datum=np.array([0.0, 0.0, 0.45]),  # Bow, deck height
    wand_pivot_hull_datum=np.array([0.0, 0.0, 0.35]),  # Bow, below deck

    # Sail thrust: lookup table calibrated via CasADi/IPOPT trim solver
    sail_thrust_coeff=69.7,
    sail_thrust_speeds=(6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0),
    sail_thrust_values=(46.8, 49.0, 54.0, 60.7, 69.7, 80.5, 98.4, 113.6, 126.6, 143.2, 161.5, 182.6, 202.1, 227.1, 240.9),
)
"""Mackay Bieker Moth V3 parameters.

Geometry based on measured Bieker Moth V3 dimensions and engineering
estimation. Positions are stored in hull-datum coordinates (x aft from
bow, z up from hull bottom) and exposed as body-frame @properties.

Sources:
    - Measured foil dimensions (span, chord)
    - Hull geometry from boat inspection
    - Inertia estimated via scripts/estimate_moth_inertia.py

Properties:
    - Total mass: 125 kg (50 kg boat + 75 kg sailor)
    - Main foil AR: ~10.7
    - Rudder AR: ~9.1
    - hull_contact_depth: 0.94 m (system CG to hull bottom)
    - Derived main_foil_position: [0.55, 0, 1.82] (body FRD)
    - Derived rudder_position: [-1.755, 0, 1.77] (body FRD)
    - Derived sail_ce_position: [-0.40, 0, -1.18] (body FRD)
    - main_foil_cl0: 0.15 (cambered foil section)
"""


# ============================================================================
# Wave Presets
# ============================================================================

WAVE_CALM = WaveParams(
    significant_wave_height=0.3,
    peak_period=4.0,
    spectrum_type="jonswap",
    num_components=30,
)
"""Calm sea state (Hs=0.3m, Tp=4s). Light chop typical of sheltered waters."""

WAVE_MODERATE = WaveParams(
    significant_wave_height=1.0,
    peak_period=6.0,
    spectrum_type="jonswap",
    num_components=30,
)
"""Moderate sea state (Hs=1.0m, Tp=6s). Typical coastal wind-sea conditions."""

WAVE_REGULAR_1M = WaveParams.regular(
    amplitude=0.5,
    period=5.0,
    direction=0.0,
)
"""Regular 1m wave (A=0.5m, T=5s). Single-component Airy wave for testing."""

WAVE_SF_BAY_LIGHT = WaveParams(
    significant_wave_height=0.3,
    peak_period=2.5,
    spectrum_type="jonswap",
    gamma=4.0,
    num_components=30,
    stokes_order=2,
)
"""SF Bay light chop (Hs=0.3m, Tp=2.5s, JONSWAP gamma=4.0, Stokes 2nd-order).

Short-period steep chop typical of light wind conditions on SF Bay.
Wavelength ~10m, comparable to moth foil separation (2.3m).
"""

WAVE_SF_BAY_MODERATE = WaveParams(
    significant_wave_height=0.5,
    peak_period=3.0,
    spectrum_type="jonswap",
    gamma=4.0,
    num_components=30,
    stokes_order=2,
)
"""SF Bay moderate chop (Hs=0.5m, Tp=3.0s, JONSWAP gamma=4.0, Stokes 2nd-order).

Moderate wind-driven chop on SF Bay. Wavelength ~14m with steep crests.
Significant differential wave effects on moth foils.
"""
