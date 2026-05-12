"""Controller implementations for the closed-loop pipeline.

Provides LQRController, MechanicalWandController, and PIDController.

Example:
    from fmd.simulator.controllers import LQRController, MechanicalWandController

    controller = LQRController(
        K=K, x_trim=x_trim, u_trim=u_trim,
        u_min=u_min, u_max=u_max,
    )
    u = controller.control(x_est, t)
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from fmd.simulator.components.moth_wand import WandLinkage


class LQRController(eqx.Module):
    """LQR state-feedback controller with saturation.

    Computes: u = u_trim - K @ (x_est[:n_ctrl] - x_trim[:n_ctrl])
    then clips to [u_min, u_max].

    Attributes:
        K: LQR gain matrix, shape (m, n_ctrl).
        x_trim: Trim state, shape (n,).
        u_trim: Trim control, shape (m,).
        u_min: Lower control bounds, shape (m,).
        u_max: Upper control bounds, shape (m,).
    """

    K: Array
    x_trim: Array
    u_trim: Array
    u_min: Array
    u_max: Array

    def control(self, x_est: Array, t: float) -> Array:
        """Compute saturated LQR control.

        Args:
            x_est: Estimated state vector.
            t: Current simulation time (unused for LQR).

        Returns:
            Control vector, clipped to bounds.
        """
        n_ctrl = self.K.shape[1]
        u = self.u_trim - self.K @ (x_est[:n_ctrl] - self.x_trim[:n_ctrl])
        return jnp.clip(u, self.u_min, self.u_max)


class MechanicalWandController(eqx.Module):
    """Mechanical wand-to-flap controller with saturation.

    Uses WandLinkage to convert wand angle to flap deflection.
    Elevator is held at a fixed trim value.

    The wand angle is read from ``x_est[0]`` (slot 0), as placed there
    by PassthroughEstimator.

    Attributes:
        linkage: WandLinkage instance for angle-to-flap conversion.
        elevator_trim: Fixed elevator angle from trim solver (rad).
        u_min: Lower control bounds [flap_min, elevator_min].
        u_max: Upper control bounds [flap_max, elevator_max].
    """

    linkage: WandLinkage
    elevator_trim: float
    u_min: Array
    u_max: Array

    def control(self, x_est: Array, t: float) -> Array:
        """Compute saturated mechanical wand control.

        Args:
            x_est: Estimated state vector (wand angle at index 0).
            t: Current simulation time (unused).

        Returns:
            Control vector [flap, elevator], clipped to bounds.
        """
        wand_angle = x_est[0]
        flap = self.linkage.compute(wand_angle)
        u = jnp.array([flap, self.elevator_trim])
        return jnp.clip(u, self.u_min, self.u_max)


class PIDControllerState(NamedTuple):
    """Persistent state for the wand-only PIDController.

    Attributes:
        integrator: Accumulated height-error integral (m * s).
        prev_err: Previous height error (m).
    """

    integrator: Array
    prev_err: Array


class PIDController(eqx.Module):
    """PID controller on wand-derived ride height (wand-only feedback).

    Inverts the wand-angle measurement to a closed-form ride-height
    estimate (assuming trim attitude — no pitch/roll feedback), then
    applies PID on height error. Single output: main-flap command on
    top of trim. Elevator is held at its trim value.

    Closed-form height inversion (calm water, theta=0, heel=0):

    .. code-block:: text

        pos_d_est = -wand_pivot_z_body - wand_length * cos(wand_angle)
                    + wand_angle_offset

    ``wand_angle_offset`` is a per-construction calibration that
    guarantees the inversion reproduces ``pos_d_target`` at the trim
    wand angle (round-trip identity), so the controller has zero
    steady-state bias at trim.

    The wand angle is read from ``x_est[0]`` (slot 0), as placed there
    by PassthroughEstimator.

    Attributes:
        Kp: Proportional gain (rad-flap per m-height-error).
        Ki: Integral gain (rad-flap per m-height-error-second).
        Kd: Derivative gain (rad-flap per (m-height-error / second)).
        dt: Control timestep (s) — must match simulation dt.
        pos_d_target: Target ride height (m, NED).
        flap_trim: Trim flap command (rad).
        elevator_trim: Trim elevator command (rad).
        wand_length: Physical wand length (m).
        wand_pivot_z_body: z-component of wand pivot in body frame (m).
        wand_angle_offset: Calibration offset (m) so the inversion
            reproduces ``pos_d_target`` at the trim wand angle.
        u_min: Lower control bounds [flap_min, elevator_min].
        u_max: Upper control bounds [flap_max, elevator_max].
    """

    Kp: Array
    Ki: Array
    Kd: Array
    dt: Array
    pos_d_target: Array
    flap_trim: Array
    elevator_trim: Array
    wand_length: Array
    wand_pivot_z_body: Array
    wand_angle_offset: Array
    u_min: Array
    u_max: Array

    def estimate_pos_d(self, wand_angle: Array) -> Array:
        """Closed-form inversion: wand angle -> estimated ride height.

        Assumes trim attitude (theta=0, heel=0). The offset is calibrated
        at construction time so this evaluates to ``pos_d_target`` when
        ``wand_angle = trim_wand_angle``.

        Args:
            wand_angle: Wand angle (rad). 0 = vertical, pi/2 = horizontal.

        Returns:
            Estimated ride height ``pos_d`` (m, NED — negative = above water).
        """
        return (
            -self.wand_pivot_z_body
            - self.wand_length * jnp.cos(wand_angle)
            + self.wand_angle_offset
        )

    def control(
        self,
        x_est: Array,
        t: float,
        ctrl_state: PIDControllerState | None = None,
    ) -> tuple[Array, PIDControllerState]:
        """Compute saturated PID flap command from wand-derived height.

        Args:
            x_est: Estimated state vector (wand angle at index 0).
            t: Current simulation time (unused — PID timing comes from dt).
            ctrl_state: Persistent PID state. If None, starts from zero
                integrator and zero previous error (used at t=0).

        Returns:
            Tuple of (control vector [flap, elevator], new ctrl_state).
        """
        if ctrl_state is None:
            ctrl_state = PIDControllerState(
                integrator=jnp.zeros_like(self.Kp),
                prev_err=jnp.zeros_like(self.Kp),
            )

        wand_angle = x_est[0]
        pos_d_est = self.estimate_pos_d(wand_angle)
        # Both pos_d values are negative in NED at trim; positive err = boat
        # has sunk below the target, so flap must increase (more lift).
        height_err = pos_d_est - self.pos_d_target

        integrator_new = ctrl_state.integrator + height_err * self.dt
        derivative = (height_err - ctrl_state.prev_err) / self.dt

        u_flap = (
            self.flap_trim
            + self.Kp * height_err
            + self.Ki * integrator_new
            + self.Kd * derivative
        )
        u = jnp.array([u_flap, self.elevator_trim])
        u = jnp.clip(u, self.u_min, self.u_max)

        ctrl_state_new = PIDControllerState(
            integrator=integrator_new, prev_err=height_err
        )
        return u, ctrl_state_new

    def init_controller_state(self) -> PIDControllerState:
        """Return initial PID controller state (zero integrator, zero prev_err)."""
        return PIDControllerState(
            integrator=jnp.zeros_like(self.Kp),
            prev_err=jnp.zeros_like(self.Kp),
        )
