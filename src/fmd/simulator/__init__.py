"""fmd.simulator - 6-DOF rigid body dynamics simulation (JAX).

This package provides JIT-compiled, differentiable physics simulation:
- RK4 numerical integration with jax.lax.scan
- 6-DOF rigid body with quaternion attitude
- Force accumulator pattern for modular force/moment components
- Automatic differentiation for gradient-based optimization

Public surface: generic infrastructure, the moth model, and the
simple/teaching models (pendulum, cartpole, planar quadrotor, box_1d,
boat_2d).

Note: fmd.simulator enforces JAX float64 mode. Import fmd.simulator
before any other JAX code to ensure this is set correctly.

Example:
    from fmd.simulator import RigidBody6DOF, simulate, ConstantControl
    from fmd.simulator.components import Gravity
    import jax.numpy as jnp

    body = RigidBody6DOF(mass=10.0, inertia=jnp.array([1, 1, 1]),
                         components=[Gravity(10.0)])
    result = simulate(body, body.default_state(), dt=0.01, duration=10.0)
"""

# Ensure float64 before any JAX imports
from fmd.simulator import _config  # noqa: F401

# Base classes
from fmd.simulator.base import JaxDynamicSystem as DynamicSystem

# Quaternion utilities
from fmd.simulator.quaternion import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_derivative,
    quat_to_dcm,
    quat_to_euler,
    euler_to_quat,
    rotate_vector,
    rotate_vector_inverse,
    identity_quat,
)

# Control interfaces
from fmd.simulator.control import (
    ControlSchedule,
    ConstantControl,
    ZeroControl,
    PiecewiseConstantControl,
    LinearInterpolatedControl,
)

# Integration
from fmd.simulator.integrator import (
    SimulationResult,
    RichSimulationResult,
    result_with_meta,
    compute_aux_trajectory,
    euler_step,
    rk4_step,
    semi_implicit_euler_step,
    simulate,
    simulate_euler,
    simulate_euler_substepped,
    simulate_noisy,
    simulate_symplectic,
    simulate_symplectic_substepped,
    simulate_trajectory,
)

# ── Generic / teaching models ──────────────────────────────────────────
from fmd.simulator.pendulum import SimplePendulumJax as SimplePendulum
from fmd.simulator.boat_2d import Boat2DJax as Boat2D
from fmd.simulator.rigid_body import (
    RigidBody6DOFJax as RigidBody6DOF,
    create_state_jax as create_state,
    POS_N, POS_E, POS_D,
    VEL_U, VEL_V, VEL_W,
    QUAT_W, QUAT_X, QUAT_Y, QUAT_Z,
    OMEGA_P, OMEGA_Q, OMEGA_R,
    NUM_STATES,
)
from fmd.simulator.cartpole import (
    CartpoleJax as Cartpole,
    X as CART_X, X_DOT as CART_X_DOT,
    THETA as CART_THETA, THETA_DOT as CART_THETA_DOT,
    FORCE as CART_FORCE,
)
from fmd.simulator.planar_quadrotor import (
    PlanarQuadrotorJax as PlanarQuadrotor,
    X as PQUAD_X, Z as PQUAD_Z, THETA as PQUAD_THETA,
    X_DOT as PQUAD_X_DOT, Z_DOT as PQUAD_Z_DOT, THETA_DOT as PQUAD_THETA_DOT,
    T1 as PQUAD_T1, T2 as PQUAD_T2,
)
from fmd.simulator.box_1d import (
    Box1DJax as Box1D,
    Box1DFrictionJax as Box1DFriction,
    X as BOX1D_X, X_DOT as BOX1D_X_DOT,
    FORCE as BOX1D_FORCE,
)
# Boat2D state indices
from fmd.simulator.boat_2d import (
    X as BOAT_X, Y as BOAT_Y, PSI as BOAT_PSI,
    U as BOAT_U, V as BOAT_V, R as BOAT_R,
)

# ── Moth model ────────────────────────────────────────────────────────
from fmd.simulator.moth_3d import (
    Moth3D,
    MothGeometry,
    ConstantSchedule,
    ConstantArraySchedule,
    POS_D as MOTH_POS_D, THETA as MOTH_THETA,
    W as MOTH_W, Q as MOTH_Q, U as MOTH_U,
    MAIN_FLAP as MOTH_MAIN_FLAP, RUDDER_ELEVATOR as MOTH_RUDDER_ELEVATOR,
)
from fmd.simulator.moth_forces_extract import extract_forces, MothForceLog
from fmd.simulator.moth_lqr import (
    MOTH_DEFAULT_DT,
    MothTrimLQR,
    design_moth_lqr,
    design_moth_gain_schedule,
    MothGainScheduledController,
    validate_simulation_dt,
)
from fmd.simulator.closed_loop_pipeline import (
    ClosedLoopResult,
    ClosedLoopScanResult,
    simulate_closed_loop,
)
from fmd.simulator.sweep import (
    SweepResult,
    ClosedLoopSweepResult,
    ClosedLoopSweepResultFull,
    SweepLabels,
    stack_systems,
    stack_envs,
    make_ics,
    build_sweep_inputs,
    sweep_open_loop,
    sweep_closed_loop,
    sweep_closed_loop_full,
    sweep_closed_loop_sequential,
    compute_sweep_metrics,
    metrics_to_dataframe,
)
from fmd.simulator.sensors import MeasurementSensor, WandSensor
from fmd.simulator.estimators import EKFEstimator, PassthroughEstimator
from fmd.simulator.controllers import (
    LQRController as PipelineLQRController,
    MechanicalWandController,
)
from fmd.simulator.moth_scenarios import (
    ScenarioConfig,
    run_scenario,
    run_wand_scenario,
    create_baseline_config,
    create_speed_pitch_wand_config,
    create_wand_only_config,
    create_mechanical_wand_config,
    SCENARIOS,
    BASELINE,
    SURFACE_BREACH,
    HEAD_SEAS_SF_BAY,
    FOLLOWING_SEAS_SF_BAY,
    HEAD_SEAS_SF_BAY_LIGHT,
)
from fmd.simulator.moth_metrics import compute_metrics

# Linearization utilities
from fmd.simulator.linearize import (
    linearize,
    discretize_zoh,
    discretize_euler,
    controllability_matrix,
    is_controllable,
    observability_matrix,
    is_observable,
)

# LQR controllers
from fmd.simulator.lqr import (
    LQRController,
    TrajectoryLQRController,
    TVLQRController,
    lqr,
    compute_lqr_gain,
)

# Output utilities (logging renamed to avoid stdlib conflict)
from fmd.simulator.output import (
    LogWriter,
    result_to_dataframe,
    result_to_datastream,
)

# Wave field and environment
from fmd.simulator.waves import WaveField
from fmd.simulator.environment import Environment

# Components subpackage
from fmd.simulator import components

# Constraints subpackage
from fmd.simulator import constraints

# Trajectories subpackage
from fmd.simulator import trajectories
from fmd.simulator.trajectories import (
    circle_trajectory_2d,
    figure_eight_trajectory_2d,
    step_trajectory_2d,
    cartpole_swing_up_trajectory,
    cartpole_stabilization_trajectory,
    cartpole_sinusoidal_tracking,
    cartpole_trapezoidal_tracking,
    lane_change_trajectory,
    circular_track_trajectory,
    slalom_trajectory,
    hold_final_value,
    resample_trajectory,
)

# Trim — generic SciPy trimmer
from fmd.simulator.trim import (
    TrimResult,
    trim_residual,
    find_trim,
)

# Trim — CasADi Moth trim solver
from fmd.simulator.trim_casadi import (
    CasadiTrimResult,
    CalibrationTrimResult,
    PhaseInfo,
    CharacteristicScales,
    DEFAULT_SCALES,
    find_casadi_trim,
    find_casadi_trim_sweep,
    find_moth_trim,
    calibrate_moth_thrust,
    validate_trim_result,
    validate_thrust_sweep,
)

# Validation toolkit (generic)
from fmd.simulator.validation import (
    SimCase,
    CaseDiagnostics,
    compute_diagnostics,
)

# Moth-specific validation case factories live in fmd.simulator.moth_validation
from fmd.simulator.moth_validation import (
    run_case,
    case_trim_equilibrium,
    case_flap_impulse,
    case_elevator_impulse,
)

# Clean component exports (no "Jax" prefix for common usage)
from fmd.simulator.components import (
    JaxForceElement as ForceElement,
    JaxGravity as Gravity,
)

__all__ = [
    "DynamicSystem",
    "quat_multiply", "quat_conjugate", "quat_normalize", "quat_derivative",
    "quat_to_dcm", "quat_to_euler", "euler_to_quat",
    "rotate_vector", "rotate_vector_inverse", "identity_quat",
    "ControlSchedule", "ConstantControl", "ZeroControl",
    "PiecewiseConstantControl", "LinearInterpolatedControl",
    "SimulationResult", "RichSimulationResult", "result_with_meta", "compute_aux_trajectory",
    "euler_step", "rk4_step", "semi_implicit_euler_step",
    "simulate", "simulate_euler", "simulate_euler_substepped", "simulate_noisy",
    "simulate_symplectic", "simulate_symplectic_substepped", "simulate_trajectory",
    "SimplePendulum", "Boat2D", "RigidBody6DOF", "create_state",
    "Cartpole", "PlanarQuadrotor",
    "Box1D", "Box1DFriction",
    "WaveField", "Environment",
    "ForceElement", "Gravity",
    "POS_N", "POS_E", "POS_D", "VEL_U", "VEL_V", "VEL_W",
    "QUAT_W", "QUAT_X", "QUAT_Y", "QUAT_Z", "OMEGA_P", "OMEGA_Q", "OMEGA_R",
    "NUM_STATES", "BOAT_X", "BOAT_Y", "BOAT_PSI", "BOAT_U", "BOAT_V", "BOAT_R",
    "CART_X", "CART_X_DOT", "CART_THETA", "CART_THETA_DOT", "CART_FORCE",
    "PQUAD_X", "PQUAD_Z", "PQUAD_THETA", "PQUAD_X_DOT", "PQUAD_Z_DOT", "PQUAD_THETA_DOT",
    "PQUAD_T1", "PQUAD_T2",
    "BOX1D_X", "BOX1D_X_DOT", "BOX1D_FORCE",
    "LogWriter", "result_to_dataframe", "result_to_datastream",
    "linearize", "discretize_zoh", "discretize_euler",
    "controllability_matrix", "is_controllable",
    "observability_matrix", "is_observable",
    "LQRController", "TrajectoryLQRController", "TVLQRController",
    "lqr", "compute_lqr_gain",
    "TrimResult", "trim_residual", "find_trim",
    "CasadiTrimResult", "CalibrationTrimResult", "PhaseInfo",
    "CharacteristicScales", "DEFAULT_SCALES",
    "find_casadi_trim", "find_casadi_trim_sweep",
    "calibrate_moth_thrust",
    "validate_trim_result", "validate_thrust_sweep",
    "SimCase", "CaseDiagnostics", "run_case", "compute_diagnostics",
    "components", "constraints",
    # ── Moth ──
    "Moth3D", "MothGeometry", "ConstantSchedule", "ConstantArraySchedule",
    "MOTH_POS_D", "MOTH_THETA", "MOTH_W", "MOTH_Q", "MOTH_U",
    "MOTH_MAIN_FLAP", "MOTH_RUDDER_ELEVATOR",
    "extract_forces", "MothForceLog",
    "find_moth_trim",
    "MOTH_DEFAULT_DT",
    "MothTrimLQR", "design_moth_lqr", "design_moth_gain_schedule",
    "MothGainScheduledController", "validate_simulation_dt",
    "ClosedLoopResult", "ClosedLoopScanResult", "simulate_closed_loop",
    "SweepResult", "ClosedLoopSweepResult", "ClosedLoopSweepResultFull", "SweepLabels",
    "stack_systems", "stack_envs", "make_ics", "build_sweep_inputs",
    "sweep_open_loop", "sweep_closed_loop", "sweep_closed_loop_full",
    "sweep_closed_loop_sequential",
    "compute_sweep_metrics", "metrics_to_dataframe",
    "MeasurementSensor", "WandSensor",
    "EKFEstimator", "PassthroughEstimator",
    "PipelineLQRController", "MechanicalWandController",
    "ScenarioConfig", "run_scenario", "run_wand_scenario",
    "create_baseline_config", "create_speed_pitch_wand_config",
    "create_wand_only_config", "create_mechanical_wand_config",
    "SCENARIOS", "BASELINE", "SURFACE_BREACH",
    "HEAD_SEAS_SF_BAY", "FOLLOWING_SEAS_SF_BAY", "HEAD_SEAS_SF_BAY_LIGHT",
    "compute_metrics",
    "case_trim_equilibrium", "case_flap_impulse", "case_elevator_impulse",
    "trajectories",
    "circle_trajectory_2d", "figure_eight_trajectory_2d", "step_trajectory_2d",
    "cartpole_swing_up_trajectory", "cartpole_stabilization_trajectory",
    "cartpole_sinusoidal_tracking", "cartpole_trapezoidal_tracking",
    "lane_change_trajectory", "circular_track_trajectory", "slalom_trajectory",
    "hold_final_value", "resample_trajectory",
]
