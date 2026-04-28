"""fmd.estimation - State estimation for dynamic systems.

This package provides Kalman Filter infrastructure for state estimation:
- Measurement models for defining observation equations
- Linear Kalman Filter for linear systems
- Extended Kalman Filter for nonlinear systems

All estimators are JAX-compatible and can be JIT-compiled.

Key Concepts:
    MeasurementModel: Defines y = h(x, u, t) and includes R (noise covariance)
    LinearMeasurementModel: y = H @ x + D @ u (analytical Jacobian)
    FullStateMeasurement: Observe all states directly (for testing)
    KalmanFilter: Discrete-time linear Kalman filter
    ExtendedKalmanFilter: EKF for nonlinear systems (RK4 propagation)

Example:
    from fmd.estimation import KalmanFilter, LinearMeasurementModel
    import jax.numpy as jnp

    # Define measurement model (observe position and velocity)
    H = jnp.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]])
    R = jnp.eye(2) * 0.01

    model = LinearMeasurementModel(
        output_names=("position", "velocity"),
        H=H,
        R=R,
    )

    # Use Kalman Filter
    kf = KalmanFilter()
    x_upd, P_upd = kf.step(x, P, A, B, Q, u, y, model.H, model.R)
"""

from fmd.estimation.measurement import (
    MeasurementModel,
    LinearMeasurementModel,
    FullStateMeasurement,
)
from fmd.estimation.kalman import KalmanFilter
from fmd.estimation.ekf import ExtendedKalmanFilter
from fmd.estimation.moth_measurements import (
    SpeedPitchHeightMeasurement,
    SpeedPitchRateHeightMeasurement,
    SpeedPitchRateHeightAccelMeasurement,
    WandAngleMeasurement,
    SpeedPitchWandMeasurement,
    create_moth_measurement,
    bow_ride_height,
)

__all__ = [
    "MeasurementModel",
    "LinearMeasurementModel",
    "FullStateMeasurement",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "SpeedPitchHeightMeasurement",
    "SpeedPitchRateHeightMeasurement",
    "SpeedPitchRateHeightAccelMeasurement",
    "WandAngleMeasurement",
    "SpeedPitchWandMeasurement",
    "create_moth_measurement",
    "bow_ride_height",
]
