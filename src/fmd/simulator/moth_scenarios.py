"""Moth closed-loop scenario configuration and runner.

Provides a frozen attrs class for scenario configuration and a runner
function that wires up trim, LQR, measurement model, and closed-loop
simulation using the composable pipeline.

Also provides factory functions for wand-based sensor/estimator/controller
triples:

- ``create_speed_pitch_wand_config`` — WandSensor + EKF + LQR
- ``create_wand_only_config`` — WandSensor (wand only) + EKF + LQR
- ``create_mechanical_wand_config`` — WandSensor + Passthrough + MechanicalWand

Example:
    from fmd.simulator.moth_scenarios import SCENARIOS, run_scenario

    result = run_scenario(SCENARIOS["baseline"])
    print(f"Final pos_d error: {result.true_states[-1, 0] - result.trim_state[0]:.4f}")
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from typing import Optional

import attrs

import jax
import jax.numpy as jnp
import numpy as np

from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
from fmd.simulator.closed_loop_pipeline import (
    ClosedLoopResult,
    Controller,
    Estimator,
    Sensor,
    simulate_closed_loop,
)
from fmd.simulator.controllers import (
    LQRController,
    MechanicalWandController,
    PIDController,
)
from fmd.simulator.estimators import EKFEstimator, PassthroughEstimator
from fmd.simulator.moth_lqr import design_moth_lqr, MothTrimLQR
from fmd.simulator.params import MothParams, MOTH_BIEKER_V3
from fmd.simulator.params.wave import WaveParams
from fmd.simulator.components.moth_wand import DEFAULT_WAND_LENGTH
from fmd.simulator.sensors import MeasurementSensor, WandSensor


@attrs.frozen
class ScenarioConfig:
    """Configuration for a closed-loop scenario.

    Attributes:
        name: Scenario identifier.
        u_forward: Forward speed (m/s).
        heel_angle: Static heel angle (rad). Default 30 deg (nominal foiling).
        dt: Simulation timestep (s).
        duration: Simulation duration (s).
        measurement_variant: Sensor suite name for create_moth_measurement.
        target_theta: If set, fix theta during trim search.
        target_pos_d: If set, fix pos_d during trim search.
        perturbation: 5-element tuple of state perturbations from trim,
            or None for no perturbation.
        P0_scale: Scale factor for initial EKF covariance (P0 = I * P0_scale).
        Q_ekf_diag: Diagonal of EKF process noise covariance.
        lqr_Q_diag: Diagonal of LQR state cost (4-state subsystem).
            None uses default [100, 100, 10, 10].
        lqr_R_diag: Diagonal of LQR control cost.
            None uses default [2, 2].
        W_true_diag: Diagonal of true process noise covariance for the
            plant dynamics. If set, Gaussian noise w ~ N(0, diag(W_true_diag))
            is added to the true state at each step. Separate from Q_ekf_diag
            (the EKF's belief about process noise). Default None (no disturbance).
        params: MothParams for the simulation.
        seed: RNG seed for measurement noise.
    """

    name: str
    u_forward: float = 10.0
    heel_angle: float = np.deg2rad(30.0)
    dt: float = 0.005
    duration: float = 10.0
    measurement_variant: str = "speed_pitch_height"
    target_theta: Optional[float] = None
    target_pos_d: Optional[float] = None
    perturbation: Optional[tuple[float, ...]] = None
    P0_scale: float = 0.1
    Q_ekf_diag: tuple[float, ...] = (1e-4, 1e-4, 1e-3, 1e-3, 1e-4)
    lqr_Q_diag: Optional[tuple[float, ...]] = None
    lqr_R_diag: Optional[tuple[float, ...]] = None
    W_true_diag: Optional[tuple[float, ...]] = None
    wave_params: Optional[WaveParams] = None
    enable_lift_lag: bool = False
    params: MothParams = MOTH_BIEKER_V3
    seed: int = 42


def run_scenario(config: ScenarioConfig) -> ClosedLoopResult:
    """Run a complete closed-loop scenario from config.

    Steps:
        1. Design LQR at trim (with optional target_theta)
        2. Create Moth3D and measurement model
        3. Build Sensor, Estimator, Controller
        4. Build initial states with perturbation
        5. Run simulate_closed_loop
        6. Return ClosedLoopResult

    Args:
        config: Scenario configuration.

    Returns:
        ClosedLoopResult with full trajectory data.
    """
    from fmd.estimation import ExtendedKalmanFilter, create_moth_measurement

    # LQR weights
    Q_lqr = np.diag(config.lqr_Q_diag) if config.lqr_Q_diag is not None else None
    R_lqr = np.diag(config.lqr_R_diag) if config.lqr_R_diag is not None else None

    # 1. Design LQR at trim
    lqr = design_moth_lqr(
        params=config.params,
        u_forward=config.u_forward,
        Q=Q_lqr,
        R=R_lqr,
        dt=config.dt,
        target_theta=config.target_theta,
        target_pos_d=config.target_pos_d,
        heel_angle=config.heel_angle,
    )

    # 2. Create Moth3D and measurement model
    moth = Moth3D(
        config.params,
        u_forward=ConstantSchedule(config.u_forward),
        heel_angle=config.heel_angle,
        enable_lift_lag=config.enable_lift_lag,
    )

    if config.measurement_variant == "full_state":
        meas = create_moth_measurement("full_state", num_states=moth.num_states)
    else:
        meas = create_moth_measurement(
            config.measurement_variant,
            bowsprit_position=config.params.bowsprit_position,
            heel_angle=config.heel_angle,
            num_states=moth.num_states,
        )

    # 3. Build initial states
    trim_state_5 = jnp.array(lqr.trim.state)  # always 5 states from trim solver

    # Pad trim state for lift lag (filter states initialize to trim alpha_eff)
    if moth.enable_lift_lag:
        from fmd.simulator.moth_3d import _MOTH_AUX_NAMES
        moth_5 = Moth3D(config.params, u_forward=ConstantSchedule(config.u_forward),
                       heel_angle=config.heel_angle, enable_lift_lag=False)
        trim_aux_5 = moth_5.compute_aux(trim_state_5, jnp.array(lqr.trim.control), t=0.0)
        main_alpha_trim = float(trim_aux_5[_MOTH_AUX_NAMES.index("main_alpha_eff")])
        rudder_alpha_trim = float(trim_aux_5[_MOTH_AUX_NAMES.index("rudder_alpha_eff")])
        trim_state = jnp.concatenate([trim_state_5, jnp.array([main_alpha_trim, rudder_alpha_trim])])
    else:
        trim_state = trim_state_5

    if config.perturbation is not None:
        pert = jnp.array(config.perturbation)
        if len(config.perturbation) < len(trim_state):
            pert = jnp.concatenate([pert, jnp.zeros(len(trim_state) - len(config.perturbation))])
        x0_true = trim_state + pert
    else:
        x0_true = trim_state

    x0_est = trim_state
    n = len(trim_state)
    P0 = jnp.eye(n) * config.P0_scale
    # Pad Q_ekf_diag to match state dimension (extra states get small noise)
    q_diag = jnp.array(config.Q_ekf_diag)
    if len(config.Q_ekf_diag) < n:
        q_diag = jnp.concatenate([q_diag, jnp.ones(n - len(config.Q_ekf_diag)) * 1e-3])
    Q_ekf = jnp.diag(q_diag)

    # 4. Build W_true if configured
    W_true = None
    if config.W_true_diag is not None:
        w_diag = jnp.array(config.W_true_diag)
        if len(config.W_true_diag) < n:
            w_diag = jnp.concatenate([w_diag, jnp.ones(n - len(config.W_true_diag)) * 1e-8])
        W_true = jnp.diag(w_diag)

    # 5. Build environment if wave_params configured
    env = None
    if config.wave_params is not None:
        from fmd.simulator.environment import Environment
        env = Environment.with_waves(config.wave_params)

    # 6. Build pipeline components
    sensor = MeasurementSensor(measurement_model=meas, num_controls=moth.num_controls)
    ekf = ExtendedKalmanFilter(dt=config.dt)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=moth.num_controls)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    K = jnp.array(lqr.K)
    controller = LQRController(
        K=K, x_trim=trim_state, u_trim=jnp.array(lqr.trim.control),
        u_min=u_min, u_max=u_max,
    )

    # 7. Run closed-loop simulation
    result = simulate_closed_loop(
        system=moth,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        x0_est=x0_est,
        P0=P0,
        dt=config.dt,
        duration=config.duration,
        rng_key=jax.random.PRNGKey(config.seed),
        params=config.params,
        W_true=W_true,
        env=env,
        measurement_model=meas,
        trim_state=trim_state,
        trim_control=jnp.array(lqr.trim.control),
        u_trim=jnp.array(lqr.trim.control),
    )

    # Store LQR gain matrix on result for downstream analysis
    result.lqr_K = np.asarray(K)

    return result


# ---------------------------------------------------------------------------
# Geometry helpers (re-exported from moth_forces for backward compatibility)
# ---------------------------------------------------------------------------

from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d  # noqa: E402


# ---------------------------------------------------------------------------
# Wand scenario factory functions
# ---------------------------------------------------------------------------

# Noise defaults (from plan)
_R_WAND = 3e-4        # wand angle variance: sigma ~1 deg (0.017 rad)
_R_SPEED = 0.09       # forward speed variance: sigma ~0.3 m/s
_R_PITCH = 8e-5       # pitch variance: sigma ~0.5 deg (0.009 rad)


def create_speed_pitch_wand_config(
    lqr: MothTrimLQR,
    *,
    params: MothParams = MOTH_BIEKER_V3,
    heel_angle: float = np.deg2rad(30.0),
    dt: float = 0.005,
    Q_ekf_diag: tuple[float, ...] = (1e-4, 1e-4, 1e-3, 1e-3, 1e-4),
) -> tuple[Sensor, Estimator, Controller]:
    """Create speed+pitch+wand sensor/estimator/controller triple.

    Uses WandSensor(include_speed_pitch=True) + EKF + LQR.

    Args:
        lqr: Pre-computed LQR design (shared trim across configs).
        params: MothParams for geometry.
        heel_angle: Static heel angle (rad).
        dt: Timestep for EKF.
        Q_ekf_diag: EKF process noise diagonal.

    Returns:
        Tuple of (sensor, estimator, controller).
    """
    from fmd.estimation import ExtendedKalmanFilter, create_moth_measurement

    u_forward = lqr.u_forward
    wand_pivot = params.wand_pivot_position

    # Sensor: wave-aware wand + speed + pitch
    R_sensor = jnp.diag(jnp.array([_R_SPEED, _R_PITCH, _R_WAND]))
    sensor = WandSensor(
        wand_pivot_position=jnp.asarray(wand_pivot),
        wand_length=DEFAULT_WAND_LENGTH,
        heel_angle=heel_angle,
        include_speed_pitch=True,
        R=R_sensor,
        fwd_speed_func=ConstantSchedule(u_forward),
    )

    # Measurement model (calm-water, for EKF internal use)
    R_meas = jnp.diag(jnp.array([_R_SPEED, _R_PITCH, _R_WAND]))
    meas = create_moth_measurement(
        "speed_pitch_wand",
        wand_pivot_position=wand_pivot,
        heel_angle=heel_angle,
        R=R_meas,
    )

    # Estimator
    Q_ekf = jnp.diag(jnp.array(Q_ekf_diag))
    ekf = ExtendedKalmanFilter(dt=dt)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)

    # Controller
    moth = Moth3D(params, u_forward=ConstantSchedule(u_forward), heel_angle=heel_angle)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    trim_state = jnp.array(lqr.trim.state)
    controller = LQRController(
        K=jnp.array(lqr.K),
        x_trim=trim_state,
        u_trim=jnp.array(lqr.trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    return sensor, estimator, controller


def create_wand_only_config(
    lqr: MothTrimLQR,
    *,
    params: MothParams = MOTH_BIEKER_V3,
    heel_angle: float = np.deg2rad(30.0),
    dt: float = 0.005,
    Q_ekf_diag: tuple[float, ...] = (1e-4, 1e-4, 1e-2, 1e-2, 1e-3),
) -> tuple[Sensor, Estimator, Controller]:
    """Create wand-only sensor/estimator/controller triple.

    Uses WandSensor(include_speed_pitch=False) + EKF + LQR.
    Marginal observability: 1 measurement for 5 states. Larger Q_ekf
    for w, q, u lets the filter trust dynamics for unobserved states.

    Args:
        lqr: Pre-computed LQR design (shared trim across configs).
        params: MothParams for geometry.
        heel_angle: Static heel angle (rad).
        dt: Timestep for EKF.
        Q_ekf_diag: EKF process noise diagonal. Default has larger
            values for w, q, u (unobserved by wand).

    Returns:
        Tuple of (sensor, estimator, controller).
    """
    from fmd.estimation import ExtendedKalmanFilter, create_moth_measurement

    u_forward = lqr.u_forward
    wand_pivot = params.wand_pivot_position

    # Sensor: wave-aware wand only
    R_sensor = jnp.diag(jnp.array([_R_WAND]))
    sensor = WandSensor(
        wand_pivot_position=jnp.asarray(wand_pivot),
        wand_length=DEFAULT_WAND_LENGTH,
        heel_angle=heel_angle,
        include_speed_pitch=False,
        R=R_sensor,
        fwd_speed_func=ConstantSchedule(u_forward),
    )

    # Measurement model (calm-water, for EKF internal use)
    R_meas = jnp.diag(jnp.array([_R_WAND]))
    meas = create_moth_measurement(
        "wand_only",
        wand_pivot_position=wand_pivot,
        heel_angle=heel_angle,
        R=R_meas,
    )

    # Estimator
    Q_ekf = jnp.diag(jnp.array(Q_ekf_diag))
    ekf = ExtendedKalmanFilter(dt=dt)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)

    # Controller
    moth = Moth3D(params, u_forward=ConstantSchedule(u_forward), heel_angle=heel_angle)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    trim_state = jnp.array(lqr.trim.state)
    controller = LQRController(
        K=jnp.array(lqr.K),
        x_trim=trim_state,
        u_trim=jnp.array(lqr.trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    return sensor, estimator, controller


def _build_wand_sensor_and_passthrough(
    lqr: MothTrimLQR,
    *,
    params: MothParams,
    heel_angle: float,
) -> tuple[WandSensor, "PassthroughEstimator", Array, Array, float]:
    """Build the wand-only sensor/estimator/bounds triple shared by the
    mechanical-wand and PID-wand factories.

    This is a pure refactor: both ``create_mechanical_wand_config`` and
    ``create_pid_wand_config`` previously constructed identical
    ``WandSensor`` / ``PassthroughEstimator`` instances and read the
    same control bounds + elevator trim. Extracting the shared work
    here keeps the two factories' behaviour byte-identical while
    avoiding duplication.

    Args:
        lqr: Pre-computed LQR design (for trim control and forward speed).
        params: MothParams for geometry.
        heel_angle: Static heel angle (rad).

    Returns:
        Tuple of (sensor, estimator, u_min, u_max, elevator_trim).
    """
    u_forward = lqr.u_forward
    wand_pivot = params.wand_pivot_position

    # Sensor: wave-aware wand only
    R_sensor = jnp.diag(jnp.array([_R_WAND]))
    sensor = WandSensor(
        wand_pivot_position=jnp.asarray(wand_pivot),
        wand_length=DEFAULT_WAND_LENGTH,
        heel_angle=heel_angle,
        include_speed_pitch=False,
        R=R_sensor,
        fwd_speed_func=ConstantSchedule(u_forward),
    )

    # Estimator: passthrough (wand angle -> slot 0)
    estimator = PassthroughEstimator(n_states=5)

    # Controller bounds (and elevator trim, shared across both factories)
    moth = Moth3D(params, u_forward=ConstantSchedule(u_forward), heel_angle=heel_angle)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    elevator_trim = float(lqr.trim.control[1])

    return sensor, estimator, u_min, u_max, elevator_trim


def create_mechanical_wand_config(
    lqr: MothTrimLQR,
    *,
    params: MothParams = MOTH_BIEKER_V3,
    heel_angle: float = np.deg2rad(30.0),
    linkage_overrides: Optional[dict] = None,
) -> tuple[Sensor, Estimator, Controller]:
    """Create mechanical wand sensor/estimator/controller triple.

    Uses WandSensor(include_speed_pitch=False) + PassthroughEstimator
    + MechanicalWandController. The wand angle is mechanically linked
    to the flap; elevator is held at trim.

    The default pullrod_offset of 0.005m is tuned so that the linkage
    produces approximately the correct flap angle at the trim wand angle,
    eliminating steady-state ride height offset.

    Args:
        lqr: Pre-computed LQR design (for trim control and bounds).
        params: MothParams for geometry.
        heel_angle: Static heel angle (rad).
        linkage_overrides: Optional dict of WandLinkage field overrides
            (e.g., pullrod_offset, gearing_ratio). If pullrod_offset is
            not in overrides, the tuned default of 0.005 is used.

    Returns:
        Tuple of (sensor, estimator, controller).
    """
    from fmd.simulator.components.moth_wand import create_wand_linkage

    sensor, estimator, u_min, u_max, elevator_trim = _build_wand_sensor_and_passthrough(
        lqr, params=params, heel_angle=heel_angle
    )

    # Controller: mechanical linkage
    # Default pullrod_offset=0.005 tuned to eliminate steady-state offset
    overrides = {"pullrod_offset": 0.005}
    if linkage_overrides:
        overrides.update(linkage_overrides)
    linkage = create_wand_linkage(**overrides)

    controller = MechanicalWandController(
        linkage=linkage,
        elevator_trim=elevator_trim,
        u_min=u_min,
        u_max=u_max,
    )

    return sensor, estimator, controller


# Default PID gains for the wand-only PID controller.
#
# Conservative starting point chosen to give a stable baseline for the
# wand_vs_pid_waves report. The wand-only LQG flap gain on pos_d at
# u=10 m/s is ~1.28 rad/m; we run at roughly half that magnitude on
# the proportional term to leave margin under wave forcing.
#
# Kd is zero by design: the only state the controller sees is the
# wand angle, which in waves is already a high-bandwidth signal (wave
# orbital motion couples directly into wand-tip elevation). A finite
# Kd amplifies that noise and destabilises the boat (verified during
# tuning — Kd=0.15 with Kp=1.5 drove pitch RMS to ~0.5 rad). Users
# who want derivative action should low-pass the wand signal first
# or estimate vertical velocity through an EKF.
_DEFAULT_PID_KP = 0.6     # rad flap per m height error
_DEFAULT_PID_KI = 0.1     # rad flap per m*s
_DEFAULT_PID_KD = 0.0     # rad flap per (m/s)


def create_pid_wand_config(
    lqr: MothTrimLQR,
    *,
    params: MothParams = MOTH_BIEKER_V3,
    heel_angle: float = np.deg2rad(30.0),
    dt: float = 0.005,
    Kp: float = _DEFAULT_PID_KP,
    Ki: float = _DEFAULT_PID_KI,
    Kd: float = _DEFAULT_PID_KD,
    Kb: Optional[float] = None,
    target_pos_d: Optional[float] = None,
) -> tuple[Sensor, Estimator, Controller]:
    """Create wand-only PID sensor/estimator/controller triple.

    Uses ``WandSensor(include_speed_pitch=False)`` + ``PassthroughEstimator``
    + ``PIDController``. The PID acts on a closed-form nonlinear estimate
    of ride height from wand angle (trim-attitude assumption: theta=0,
    heel=0). At construction time, a ``wand_angle_offset`` calibration is
    computed so the inversion exactly reproduces ``pos_d_target`` when
    evaluated at the trim wand angle.

    Args:
        lqr: Pre-computed LQR design (for trim state/control).
        params: MothParams for geometry.
        heel_angle: Static heel angle used for the sensor (rad).
            The PID inversion still assumes heel=0 — this argument
            controls only the wave-aware sensor's geometry.
        dt: Simulation timestep, used for the PID integral and derivative.
        Kp, Ki, Kd: PID gains. Defaults are conservative starting values.
        Kb: Anti-windup back-calculation gain. ``None`` (default) selects
            the Aström recipe ``1/Ki`` when ``Ki > 0`` else ``0``. Set
            to ``0.0`` to disable anti-windup.
        target_pos_d: Optional override for the PID's setpoint (m, NED).
            ``None`` (default) uses ``lqr.trim.state[0]`` — the natural
            trim ride height. When provided, the closed-form inversion
            offset is recomputed so the round-trip identity holds at
            the new target: ``estimate_pos_d(trim_wand_angle) ==
            target_pos_d``. Note that "exact at trim" no longer means
            "exact at the natural trim wand angle" — it means "exact
            at the requested deeper/shallower target once the
            integrator settles". Useful for setting a safety margin
            below the natural trim (e.g.
            ``target_pos_d = compute_tip_at_surface_pos_d() - 0.30``).

    Returns:
        Tuple of (sensor, estimator, controller).
    """
    from fmd.simulator.components.moth_wand import wand_angle_from_state

    sensor, estimator, u_min, u_max, elevator_trim = _build_wand_sensor_and_passthrough(
        lqr, params=params, heel_angle=heel_angle
    )
    wand_pivot = params.wand_pivot_position

    # Trim values
    trim_state = jnp.array(lqr.trim.state)
    trim_control = jnp.array(lqr.trim.control)
    pos_d_target = (
        float(target_pos_d) if target_pos_d is not None else float(trim_state[0])
    )
    flap_trim = float(trim_control[0])

    # Inversion constants (trim attitude assumption: theta=0, heel=0)
    wand_length = DEFAULT_WAND_LENGTH
    wand_pivot_z_body = float(wand_pivot[2])

    # Calibrate angle offset so the inversion is exact at the trim
    # wand angle. The trim wand angle is computed from the *actual* trim
    # attitude (with heel_angle, trim theta) so the wand sensor at trim
    # is consistent with what the EKF-free PID will see.
    trim_wand_angle = float(
        wand_angle_from_state(
            pos_d=trim_state[0],
            theta=trim_state[1],
            wand_pivot_position=jnp.asarray(wand_pivot),
            wand_length=wand_length,
            heel_angle=heel_angle,
        )
    )
    pos_d_est_without_offset = (
        -wand_pivot_z_body - wand_length * np.cos(trim_wand_angle)
    )
    wand_angle_offset = pos_d_target - pos_d_est_without_offset

    # Sanity check: round-trip identity at trim
    pos_d_check = (
        -wand_pivot_z_body
        - wand_length * np.cos(trim_wand_angle)
        + wand_angle_offset
    )
    # ``assert`` is stripped under ``python -O`` and raises the wrong
    # exception type for a public-API precondition — escalate to
    # ``ValueError`` so the round-trip identity is always enforced.
    if abs(pos_d_check - pos_d_target) > 1e-9:
        raise ValueError(
            f"PID wand-angle calibration failed: "
            f"pos_d_check={pos_d_check}, pos_d_target={pos_d_target}"
        )

    # Anti-windup gain. Aström back-calculation recipe Kb = 1 / Ki when
    # the integral gain is on; otherwise Kb has no effect and we set it
    # to 0 so the (Kb * 0)-saturation term zeroes out cleanly.
    if Kb is None:
        kb_value = (1.0 / Ki) if Ki > 0.0 else 0.0
    else:
        kb_value = float(Kb)

    controller = PIDController(
        Kp=jnp.array(Kp),
        Ki=jnp.array(Ki),
        Kd=jnp.array(Kd),
        Kb=jnp.array(kb_value),
        dt=jnp.array(dt),
        pos_d_target=jnp.array(pos_d_target),
        flap_trim=jnp.array(flap_trim),
        elevator_trim=jnp.array(elevator_trim),
        wand_length=jnp.array(wand_length),
        wand_pivot_z_body=jnp.array(wand_pivot_z_body),
        wand_angle_offset=jnp.array(wand_angle_offset),
        u_min=u_min,
        u_max=u_max,
    )

    return sensor, estimator, controller


def create_baseline_config(
    lqr: MothTrimLQR,
    *,
    params: MothParams = MOTH_BIEKER_V3,
    heel_angle: float = np.deg2rad(30.0),
    dt: float = 0.005,
    Q_ekf_diag: tuple[float, ...] = (1e-4, 1e-4, 1e-3, 1e-3, 1e-4),
) -> tuple[Sensor, Estimator, Controller]:
    """Create baseline speed+pitch+height sensor/estimator/controller triple.

    Uses MeasurementSensor(speed_pitch_height) + EKF + LQR. This is
    the reference configuration for comparison against wand variants.

    Args:
        lqr: Pre-computed LQR design (shared trim across configs).
        params: MothParams for geometry.
        heel_angle: Static heel angle (rad).
        dt: Timestep for EKF.
        Q_ekf_diag: EKF process noise diagonal.

    Returns:
        Tuple of (sensor, estimator, controller).
    """
    from fmd.estimation import ExtendedKalmanFilter, create_moth_measurement

    u_forward = lqr.u_forward

    meas = create_moth_measurement(
        "speed_pitch_height",
        bowsprit_position=params.bowsprit_position,
        heel_angle=heel_angle,
    )

    sensor = MeasurementSensor(measurement_model=meas, num_controls=2)

    Q_ekf = jnp.diag(jnp.array(Q_ekf_diag))
    ekf = ExtendedKalmanFilter(dt=dt)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)

    moth = Moth3D(params, u_forward=ConstantSchedule(u_forward), heel_angle=heel_angle)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    trim_state = jnp.array(lqr.trim.state)
    controller = LQRController(
        K=jnp.array(lqr.K),
        x_trim=trim_state,
        u_trim=jnp.array(lqr.trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    return sensor, estimator, controller


def run_wand_scenario(
    config_name: str,
    sensor: Sensor,
    estimator: Estimator,
    controller: Controller,
    lqr: MothTrimLQR,
    *,
    params: MothParams = MOTH_BIEKER_V3,
    heel_angle: float = np.deg2rad(30.0),
    dt: float = 0.005,
    duration: float = 10.0,
    perturbation: Optional[tuple[float, ...]] = None,
    wave_params: Optional[WaveParams] = None,
    seed: int = 42,
    measurement_model=None,
    measurement_noise_override=None,
) -> ClosedLoopResult:
    """Run a closed-loop scenario with pre-built SEC components.

    This is a lower-level runner than ``run_scenario`` that accepts
    pre-built Sensor/Estimator/Controller triples from the factory
    functions.

    Args:
        config_name: Scenario name for logging.
        sensor: Pre-built Sensor.
        estimator: Pre-built Estimator.
        controller: Pre-built Controller.
        lqr: LQR design result (for trim state/control).
        params: MothParams for dynamics.
        heel_angle: Static heel angle (rad).
        dt: Simulation timestep (s).
        duration: Simulation duration (s).
        perturbation: State perturbation from trim, or None.
        wave_params: Wave parameters, or None for calm water.
        seed: RNG seed.
        measurement_model: Optional measurement model for result metadata.
        measurement_noise_override: Optional pre-generated measurement noise
            array of shape (n_steps, p). If provided, overrides sensor noise
            for fair cross-config comparisons.

    Returns:
        ClosedLoopResult with full trajectory data.
    """
    u_forward = lqr.u_forward
    moth = Moth3D(params, u_forward=ConstantSchedule(u_forward), heel_angle=heel_angle)

    trim_state = jnp.array(lqr.trim.state)
    trim_control = jnp.array(lqr.trim.control)

    if perturbation is not None:
        pert = jnp.array(perturbation)
        x0_true = trim_state + pert
    else:
        x0_true = trim_state

    x0_est = trim_state
    n = len(trim_state)
    P0 = jnp.eye(n) * 0.1

    env = None
    if wave_params is not None:
        from fmd.simulator.environment import Environment
        env = Environment.with_waves(wave_params)

    result = simulate_closed_loop(
        system=moth,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        x0_est=x0_est,
        P0=P0,
        dt=dt,
        duration=duration,
        rng_key=jax.random.PRNGKey(seed),
        params=params,
        env=env,
        measurement_model=measurement_model,
        trim_state=trim_state,
        trim_control=trim_control,
        u_trim=trim_control,
        measurement_noise_override=measurement_noise_override,
    )

    result.lqr_K = np.asarray(lqr.K) if hasattr(lqr, 'K') else None

    return result


# ---------------------------------------------------------------------------
# Named scenarios
# ---------------------------------------------------------------------------

BASELINE = ScenarioConfig(
    name="baseline",
    perturbation=(0.05, np.radians(-2.0), 0.0, 0.0, 0.0),
)

SURFACE_BREACH = ScenarioConfig(
    name="surface_breach",
    target_theta=0.0,
    target_pos_d=compute_tip_at_surface_pos_d(),
    duration=30.0,
    W_true_diag=(1e-7, 1e-8, 1e-6, 1e-6, 1e-8),
)

# ---------------------------------------------------------------------------
# Wave disturbance scenarios
# ---------------------------------------------------------------------------

from fmd.simulator.params.presets import WAVE_SF_BAY_MODERATE, WAVE_SF_BAY_LIGHT  # noqa: E402

HEAD_SEAS_SF_BAY = ScenarioConfig(
    name="head_seas_sf_bay",
    perturbation=(0.05, np.radians(-2.0), 0.0, 0.0, 0.0),
    wave_params=attrs.evolve(WAVE_SF_BAY_MODERATE, mean_direction=np.pi),
)

FOLLOWING_SEAS_SF_BAY = ScenarioConfig(
    name="following_seas_sf_bay",
    perturbation=(0.05, np.radians(-2.0), 0.0, 0.0, 0.0),
    wave_params=attrs.evolve(WAVE_SF_BAY_MODERATE, mean_direction=0.0),
)

HEAD_SEAS_SF_BAY_LIGHT = ScenarioConfig(
    name="head_seas_sf_bay_light",
    perturbation=(0.05, np.radians(-2.0), 0.0, 0.0, 0.0),
    wave_params=attrs.evolve(WAVE_SF_BAY_LIGHT, mean_direction=np.pi),
)


SCENARIOS: dict[str, ScenarioConfig] = {
    "baseline": BASELINE,
    "surface_breach": SURFACE_BREACH,
    "head_seas_sf_bay": HEAD_SEAS_SF_BAY,
    "following_seas_sf_bay": FOLLOWING_SEAS_SF_BAY,
    "head_seas_sf_bay_light": HEAD_SEAS_SF_BAY_LIGHT,
}
