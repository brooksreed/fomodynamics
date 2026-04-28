"""Extended Kalman Filter (EKF) for nonlinear state estimation.

This module provides a JAX-compatible EKF that uses nonlinear dynamics
(via RK4 integration) for state propagation and nonlinear measurement
models for updates.

The EKF linearizes the dynamics and measurement equations at each step:

Predict step:
    x_pred = rk4_step(system, x, u, dt, t)        # Nonlinear propagation
    F = I + A * dt                                  # Euler-discretized Jacobian
    P_pred = F @ P @ F.T + Q

Update step (using Joseph form for numerical stability):
    y_pred = measurement_model.measure(x_pred, u, t)
    H = measurement_model.get_measurement_jacobian(x_pred, u, t)
    K = P_pred @ H.T @ (H @ P_pred @ H.T + R)^{-1}
    x_upd = x_pred + K @ (y - y_pred)
    P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

Example:
    from fmd.estimation import ExtendedKalmanFilter, FullStateMeasurement
    from fmd.simulator import SimplePendulum, rk4_step
    from fmd.simulator.params import PENDULUM_1M
    import jax.numpy as jnp

    system = SimplePendulum(PENDULUM_1M)
    meas = FullStateMeasurement.for_system(system, R=jnp.eye(2) * 0.01)

    ekf = ExtendedKalmanFilter(dt=0.01)
    x = jnp.array([0.5, 0.0])
    P = jnp.eye(2)
    Q = jnp.eye(2) * 0.001
    u = jnp.zeros(0)

    # Single step with measurement
    x_upd, P_upd = ekf.step(x, P, system, u, Q, y, meas, t=0.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import Array

from fmd.simulator.integrator import rk4_step

if TYPE_CHECKING:
    from fmd.simulator.base import JaxDynamicSystem
    from fmd.estimation.measurement import MeasurementModel


class ExtendedKalmanFilter(eqx.Module):
    """Extended Kalman Filter for nonlinear systems.

    A functional-style EKF that uses RK4 integration for nonlinear state
    propagation and Euler-discretized Jacobians for covariance propagation.
    All methods are pure functions -- no internal mutation.

    The filter uses:
    - ``rk4_step`` from ``blur.simulator.integrator`` for state propagation
    - ``system.get_state_jacobian`` for continuous-time A matrix (via autodiff)
    - Euler discretization: F = I + A * dt
    - ``measurement_model.measure`` for nonlinear measurement prediction
    - ``measurement_model.get_measurement_jacobian`` for H matrix (via autodiff)
    - ``jax.scipy.linalg.solve`` for numerically stable Kalman gain
    - Joseph form for covariance update
    - Symmetry enforcement on P after each update
    - Optional covariance floor to prevent P collapse

    **Known limitation — Euler Jacobian discretization**: The covariance
    propagation uses first-order Euler (F = I + A*dt) while state
    propagation uses fourth-order RK4. This mismatch systematically
    inflates the predicted covariance, biasing the NEES below n. For
    the Moth model at 10 m/s, the fastest mode is ~280 rad/s giving
    |lambda*dt|=1.4 at dt=0.005. This is well within the Euler stability
    limit and provides acceptable accuracy. A matrix exponential
    discretization would improve consistency further.

    Note: No built-in divergence detection (NaN/Inf). Users should monitor
    state estimates and covariance trace externally if divergence is a concern.

    Attributes:
        dt: Discretization timestep (seconds).
        P_min_diag: Minimum diagonal value for covariance floor (default 0.0,
            meaning no floor). Set to a small positive value (e.g. 1e-8) to
            prevent covariance collapse.
        angular_measurement_indices: Tuple of measurement output indices that
            are angular (wrapped to [-pi, pi]). Innovation for these indices
            is wrapped via arctan2. Default empty (no wrapping).
        S_reg: Innovation covariance regularization (default 1e-6). Added to
            the diagonal of S = H @ P_pred @ H.T + R before computing the
            Kalman gain. Acts as a minimum innovation uncertainty floor,
            making the filter slightly more conservative when S is
            near-singular. Set to 0.0 to disable. The default 1e-6 is
            ~80x smaller than the smallest typical R entry (~8e-5 for
            pitch variance) and has negligible impact on well-conditioned
            updates.

    Example:
        ekf = ExtendedKalmanFilter(dt=0.01)

        # Predict
        x_pred, P_pred = ekf.predict(x, P, system, u, Q, t=0.0)

        # Update with measurement
        x_upd, P_upd = ekf.update(x_pred, P_pred, y, measurement_model, u, t=0.0)

        # Or combined
        x_upd, P_upd = ekf.step(x, P, system, u, Q, y, measurement_model, t=0.0)

        # With angular wrapping on measurement index 1 (e.g. pitch):
        ekf = ExtendedKalmanFilter(dt=0.01, angular_measurement_indices=(1,))
    """

    dt: float
    P_min_diag: float = eqx.field(static=True, default=0.0)
    angular_measurement_indices: tuple[int, ...] = eqx.field(
        static=True, default=()
    )
    S_reg: float = eqx.field(static=True, default=1e-6)

    def __check_init__(self):
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.P_min_diag < 0:
            raise ValueError(f"P_min_diag must be non-negative, got {self.P_min_diag}")
        if self.S_reg < 0:
            raise ValueError(f"S_reg must be non-negative, got {self.S_reg}")

    def predict(
        self,
        x: Array,
        P: Array,
        system: "JaxDynamicSystem",
        u: Array,
        Q: Array,
        t: float = 0.0,
        apply_post_step: bool = False,
    ) -> tuple[Array, Array]:
        """Perform EKF prediction step with nonlinear dynamics.

        Propagates the state through the full nonlinear dynamics using RK4,
        and propagates covariance using the Euler-discretized Jacobian.

        Args:
            x: Current state estimate of shape (n,)
            P: Current state covariance of shape (n, n)
            system: JaxDynamicSystem instance providing dynamics and Jacobians
            u: Control input of shape (m,)
            Q: Process noise covariance of shape (n, n)
            t: Current simulation time (default 0.0)
            apply_post_step: If True, apply ``system.post_step`` to the
                predicted state (e.g. angle wrapping). Useful during
                predict-only steps when no measurement update follows.
                Default False for backward compatibility.

        Returns:
            Tuple of (x_pred, P_pred):
                x_pred: Predicted state estimate of shape (n,)
                P_pred: Predicted state covariance of shape (n, n)
        """
        n = x.shape[0]

        # Nonlinear state propagation via RK4
        x_pred = rk4_step(system, x, u, self.dt, t)

        # Continuous-time Jacobian at current state
        A = system.get_state_jacobian(x, u, t)

        # Euler discretization: F = I + A * dt
        F = jnp.eye(n) + A * self.dt

        # Covariance propagation
        P_pred = F @ P @ F.T + Q

        # Symmetry enforcement (guards against floating-point drift)
        P_pred = 0.5 * (P_pred + P_pred.T)

        # Covariance floor (prevents collapse when Q is very small)
        # Only floor diagonal elements to preserve off-diagonal correlations
        if self.P_min_diag > 0:
            diag_vals = jnp.diag(P_pred)
            floor_correction = jnp.maximum(self.P_min_diag - diag_vals, 0.0)
            P_pred = P_pred + jnp.diag(floor_correction)

        # Apply post_step (e.g. angle wrapping) if system is provided.
        # This prevents theta from drifting outside [-pi, pi] during
        # predict-only steps (measurement dropout).
        if apply_post_step:
            x_pred = system.post_step(x_pred)

        return x_pred, P_pred

    def update(
        self,
        x_pred: Array,
        P_pred: Array,
        y: Array,
        measurement_model: "MeasurementModel",
        u: Array,
        t: float = 0.0,
        system: "JaxDynamicSystem | None" = None,
    ) -> tuple[Array, Array]:
        """Perform EKF measurement update step.

        Uses the nonlinear measurement model for innovation computation
        and its Jacobian for the Kalman gain calculation.

        Args:
            x_pred: Predicted state estimate of shape (n,)
            P_pred: Predicted state covariance of shape (n, n)
            y: Measurement vector of shape (p,)
            measurement_model: MeasurementModel providing measure() and
                get_measurement_jacobian()
            u: Control input of shape (m,)
            t: Current simulation time (default 0.0)
            system: Optional JaxDynamicSystem. If provided, ``post_step``
                is applied to the updated state (e.g. angle wrapping).

        Returns:
            Tuple of (x_upd, P_upd):
                x_upd: Updated state estimate of shape (n,)
                P_upd: Updated state covariance of shape (n, n)
        """
        n = x_pred.shape[0]

        # Nonlinear measurement prediction
        y_pred = measurement_model.measure(x_pred, u, t)

        # Measurement Jacobian at predicted state
        H = measurement_model.get_measurement_jacobian(x_pred, u, t)

        # Innovation (with angular wrapping for specified indices)
        innovation = y - y_pred
        for idx in self.angular_measurement_indices:
            innovation = innovation.at[idx].set(
                jnp.arctan2(
                    jnp.sin(innovation[idx]), jnp.cos(innovation[idx])
                )
            )

        # Innovation covariance
        R = measurement_model.R
        S = H @ P_pred @ H.T + R

        # Regularize S to prevent near-singular solves
        if self.S_reg > 0:
            S = S + self.S_reg * jnp.eye(S.shape[0])

        # Kalman gain: K = P_pred @ H.T @ S^{-1}
        # Solve S.T @ K.T = H @ P_pred.T for K.T (same pattern as KalmanFilter)
        K_T = jla.solve(S.T, H @ P_pred.T, assume_a='pos')
        K = K_T.T

        # State update
        x_upd = x_pred + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = jnp.eye(n) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        # Symmetry enforcement (guards against floating-point drift)
        P_upd = 0.5 * (P_upd + P_upd.T)

        # Covariance floor (prevents collapse when Q is very small)
        # Only floor diagonal elements to preserve off-diagonal correlations
        if self.P_min_diag > 0:
            diag_vals = jnp.diag(P_upd)
            floor_correction = jnp.maximum(self.P_min_diag - diag_vals, 0.0)
            P_upd = P_upd + jnp.diag(floor_correction)

        # Apply post_step (e.g. angle wrapping) if system is provided
        if system is not None:
            x_upd = system.post_step(x_upd)

        return x_upd, P_upd

    def step(
        self,
        x: Array,
        P: Array,
        system: "JaxDynamicSystem",
        u: Array,
        Q: Array,
        y: Array,
        measurement_model: "MeasurementModel",
        t: float = 0.0,
    ) -> tuple[Array, Array]:
        """Perform combined predict and update step.

        Convenience method that calls predict() followed by update().

        Args:
            x: Current state estimate of shape (n,)
            P: Current state covariance of shape (n, n)
            system: JaxDynamicSystem instance providing dynamics and Jacobians
            u: Control input of shape (m,)
            Q: Process noise covariance of shape (n, n)
            y: Measurement vector of shape (p,)
            measurement_model: MeasurementModel providing measure() and
                get_measurement_jacobian()
            t: Current simulation time (default 0.0)

        Returns:
            Tuple of (x_upd, P_upd):
                x_upd: Updated state estimate of shape (n,)
                P_upd: Updated state covariance of shape (n, n)
        """
        x_pred, P_pred = self.predict(x, P, system, u, Q, t)
        x_upd, P_upd = self.update(
            x_pred, P_pred, y, measurement_model, u, t, system=system
        )
        return x_upd, P_upd
