"""Discrete-time Linear Kalman Filter implementation.

This module provides a JAX-compatible Kalman Filter that can be JIT-compiled
and used for state estimation in linear systems.

The filter implements the standard predict-update cycle:

Predict step:
    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Q

Update step (using Joseph form for numerical stability):
    K = P_pred @ H.T @ (H @ P_pred @ H.T + R)^{-1}
    x_upd = x_pred + K @ (y - H @ x_pred)
    P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

Example:
    from fmd.estimation import KalmanFilter
    import jax.numpy as jnp

    kf = KalmanFilter()

    # System matrices
    A = jnp.array([[1.0, dt], [0.0, 1.0]])  # State transition
    B = jnp.array([[0.0], [dt/m]])           # Control input
    Q = jnp.eye(2) * 0.01                    # Process noise
    H = jnp.array([[1.0, 0.0]])              # Measurement (position only)
    R = jnp.array([[0.1]])                   # Measurement noise

    # Initial state and covariance
    x = jnp.zeros(2)
    P = jnp.eye(2)

    # Single step
    u = jnp.array([1.0])
    y = jnp.array([0.5])
    x_new, P_new = kf.step(x, P, A, B, Q, u, y, H, R)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import Array


class KalmanFilter(eqx.Module):
    """Discrete-time Linear Kalman Filter.

    A functional-style Kalman Filter implementation that is fully JIT-compatible.
    All methods are pure functions that return updated state and covariance,
    with no internal mutation.

    The filter uses:
    - `jax.scipy.linalg.solve` instead of matrix inverse for numerical stability
    - Joseph form for covariance update to maintain positive semi-definiteness

    Attributes:
        None - this is a stateless filter. All state is passed as arguments.

    Example:
        kf = KalmanFilter()

        # Predict
        x_pred, P_pred = kf.predict(x, P, A, B, Q, u)

        # Update with measurement
        x_upd, P_upd = kf.update(x_pred, P_pred, y, H, R)

        # Or combined
        x_upd, P_upd = kf.step(x, P, A, B, Q, u, y, H, R)
    """

    def predict(
        self,
        x: Array,
        P: Array,
        A: Array,
        B: Array,
        Q: Array,
        u: Array,
    ) -> tuple[Array, Array]:
        """Perform Kalman filter prediction step.

        Propagates state and covariance through the linear system dynamics:
            x_pred = A @ x + B @ u
            P_pred = A @ P @ A.T + Q

        Args:
            x: Current state estimate of shape (n,)
            P: Current state covariance of shape (n, n)
            A: State transition matrix of shape (n, n)
            B: Control input matrix of shape (n, m)
            Q: Process noise covariance of shape (n, n)
            u: Control input of shape (m,)

        Returns:
            Tuple of (x_pred, P_pred):
                x_pred: Predicted state estimate of shape (n,)
                P_pred: Predicted state covariance of shape (n, n)
        """
        x_pred = A @ x + B @ u
        P_pred = A @ P @ A.T + Q
        return x_pred, P_pred

    def update(
        self,
        x_pred: Array,
        P_pred: Array,
        y: Array,
        H: Array,
        R: Array,
    ) -> tuple[Array, Array]:
        """Perform Kalman filter measurement update step.

        Updates state and covariance using the measurement:
            innovation = y - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ S^{-1}  (computed via solve)
            x_upd = x_pred + K @ innovation
            P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T  (Joseph form)

        The Joseph form for covariance update is used for numerical stability.
        It guarantees the result is symmetric and positive semi-definite even
        with finite precision arithmetic.

        Args:
            x_pred: Predicted state estimate of shape (n,)
            P_pred: Predicted state covariance of shape (n, n)
            y: Measurement vector of shape (p,)
            H: Measurement matrix of shape (p, n)
            R: Measurement noise covariance of shape (p, p)

        Returns:
            Tuple of (x_upd, P_upd):
                x_upd: Updated state estimate of shape (n,)
                P_upd: Updated state covariance of shape (n, n)
        """
        n = x_pred.shape[0]

        # Innovation (measurement residual)
        innovation = y - H @ x_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain: K = P_pred @ H.T @ S^{-1}
        # Use solve for numerical stability: K @ S = P_pred @ H.T
        # Solve S.T @ K.T = (P_pred @ H.T).T = H @ P_pred.T = H @ P_pred (P symmetric)
        K_T = jla.solve(S.T, H @ P_pred.T, assume_a='pos')
        K = K_T.T

        # State update
        x_upd = x_pred + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        # P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
        I_KH = jnp.eye(n) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        # Enforce symmetry (numerical drift from finite precision)
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd

    def step(
        self,
        x: Array,
        P: Array,
        A: Array,
        B: Array,
        Q: Array,
        u: Array,
        y: Array,
        H: Array,
        R: Array,
    ) -> tuple[Array, Array]:
        """Perform combined predict and update step.

        Convenience method that combines prediction and update in a single call.
        Equivalent to calling predict() followed by update().

        Args:
            x: Current state estimate of shape (n,)
            P: Current state covariance of shape (n, n)
            A: State transition matrix of shape (n, n)
            B: Control input matrix of shape (n, m)
            Q: Process noise covariance of shape (n, n)
            u: Control input of shape (m,)
            y: Measurement vector of shape (p,)
            H: Measurement matrix of shape (p, n)
            R: Measurement noise covariance of shape (p, p)

        Returns:
            Tuple of (x_upd, P_upd):
                x_upd: Updated state estimate of shape (n,)
                P_upd: Updated state covariance of shape (n, n)
        """
        x_pred, P_pred = self.predict(x, P, A, B, Q, u)
        x_upd, P_upd = self.update(x_pred, P_pred, y, H, R)
        return x_upd, P_upd
