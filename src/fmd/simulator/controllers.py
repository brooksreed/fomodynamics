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


def _pid_pos_d_estimate(
    wand_angle: float,
    *,
    wand_pivot_z: float,
    wand_length: float,
    heel_angle: float,
    wand_angle_offset: float,
) -> float:
    """Closed-form inversion: wand angle -> estimated ride height.

    Module-level helper so tests can call exactly the same math that
    ``PIDController.estimate_pos_d`` uses, without re-deriving it.

    Args:
        wand_angle: Wand angle (rad).
        wand_pivot_z: z-component of wand pivot in body frame (m).
        wand_length: Physical wand length (m).
        heel_angle: Static heel angle (rad).
        wand_angle_offset: Calibration offset (m).

    Returns:
        Estimated pos_d (m, NED).
    """
    import math
    return (
        -wand_pivot_z * math.cos(heel_angle)
        - wand_length * math.cos(wand_angle)
        + wand_angle_offset
    )


class PIDController(eqx.Module):
    """PID controller on wand-derived ride height (wand-only feedback).

    Inverts the wand-angle measurement to a closed-form ride-height
    estimate (assuming trim attitude — theta=0, constant heel), then
    applies PID on height error. Single output: main-flap command on
    top of trim. Elevator is held at its trim value.

    Closed-form height inversion (calm water, theta=0, constant heel):

    .. code-block:: text

        pos_d_est = -wand_pivot_z_body * cos(heel_angle)
                    - wand_length * cos(wand_angle)
                    + wand_angle_offset

    The ``cos(heel_angle)`` factor projects the body-z pivot offset
    onto NED-down, making the inversion pos_d-agnostic: under
    ``theta = trim_theta`` and constant heel, the round-trip identity
    ``estimate_pos_d(wand_angle_from_state(pos_d, trim_theta, heel)) == pos_d``
    holds for ANY pos_d (not just at the natural trim). This means
    ``target_pos_d`` overrides work without per-target re-tuning.

    ``wand_angle_offset`` is a per-construction calibration that
    absorbs the trim_theta residual (the ``z_p * cos(heel) * (cos(theta_trim) - 1)``
    and ``-x_p * sin(theta_trim)`` terms). At construction, offset is
    chosen so ``estimate_pos_d(trim_wand_angle) == natural_pos_d``.
    The only remaining bias is dynamic pitch perturbation away from
    trim_theta, which the integrator averages out.

    The wand angle is read from ``x_est[0]`` (slot 0), as placed there
    by PassthroughEstimator.

    Attributes:
        Kp: Proportional gain (rad-flap per m-height-error).
        Ki: Integral gain (rad-flap per m-height-error-second).
        Kd: Derivative gain (rad-flap per (m-height-error / second)).
        Kb: Back-calculation anti-windup gain (1/s — same units as
            ``1 / Ki * dt``). When ``Ki > 0`` the controller uses the
            Aström back-calculation recipe:
            ``integrator += (err - Kb * (u_unsat - u_sat)) * dt``.
            With the default ``Kb = 1 / Ki`` the integrator stops
            growing as soon as the flap saturates. Set ``Kb = 0`` for
            classical (no anti-windup) behaviour.
        dt: Control timestep (s) — must match simulation dt.
        pos_d_target: Target ride height (m, NED).
        flap_trim: Trim flap command (rad).
        elevator_trim: Trim elevator command (rad).
        wand_length: Physical wand length (m).
        wand_pivot_z_body: z-component of wand pivot in body frame (m).
        heel_angle: Static heel angle (rad). Used to project the body-z
            pivot component onto NED-down in the inversion formula.
        wand_angle_offset: Calibration offset (m) so the inversion
            reproduces ``natural_pos_d`` at the trim wand angle, absorbing
            the trim_theta residual.
        u_min: Lower control bounds [flap_min, elevator_min].
        u_max: Upper control bounds [flap_max, elevator_max].
    """

    Kp: Array
    Ki: Array
    Kd: Array
    Kb: Array
    dt: Array
    pos_d_target: Array
    flap_trim: Array
    elevator_trim: Array
    wand_length: Array
    wand_pivot_z_body: Array
    heel_angle: eqx.field(static=True)
    wand_angle_offset: Array
    u_min: Array
    u_max: Array

    def estimate_pos_d(self, wand_angle: Array) -> Array:
        """Closed-form inversion: wand angle -> estimated ride height.

        Assumes trim attitude (theta=0) and constant heel angle. The
        ``cos(heel_angle)`` factor on the z_p term makes this pos_d-agnostic:
        for any pos_d reached under theta=trim_theta and constant heel, the
        round-trip ``estimate_pos_d(wand_angle_from_state(pos_d, trim_theta,
        heel)) == pos_d`` holds to numerical precision. The offset is
        calibrated at construction time to absorb the trim_theta residual.

        Args:
            wand_angle: Wand angle (rad). 0 = vertical, pi/2 = horizontal.

        Returns:
            Estimated ride height ``pos_d`` (m, NED — negative = above water).
        """
        import math
        # math.cos(heel_angle) is constant at trace time (heel_angle is a
        # Python float stored as a static field, not a JAX array).
        return (
            -self.wand_pivot_z_body * math.cos(self.heel_angle)
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

        Cold-start D-term note: on the very first call ``ctrl_state.prev_err``
        is zero, so the derivative term is ``height_err / dt`` rather than a
        true error rate. For the default ``Kd = 0`` this is moot, but tuners
        experimenting with ``Kd > 0`` should expect a one-step "D kick"
        proportional to the initial error / dt — pre-seeding ``prev_err``
        to ``height_err`` (or low-pass-filtering the derivative) avoids it.

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

        derivative = (height_err - ctrl_state.prev_err) / self.dt

        # First propose the unsaturated control using the *previous*
        # integrator so we can measure how much the limiter clips it.
        u_unsat_flap = (
            self.flap_trim
            + self.Kp * height_err
            + self.Ki * ctrl_state.integrator
            + self.Kd * derivative
        )
        u_sat_flap = jnp.clip(u_unsat_flap, self.u_min[0], self.u_max[0])
        u_excess = u_unsat_flap - u_sat_flap

        # Aström back-calculation: when the flap is saturated, subtract
        # ``Kb * excess`` from the integrator update so it stops growing
        # once the actuator can no longer follow the command. Outside
        # saturation, ``u_excess = 0`` and this reduces to the classical
        # update ``integrator += err * dt``.
        integrator_new = ctrl_state.integrator + (
            height_err - self.Kb * u_excess
        ) * self.dt

        # Re-evaluate with the updated integrator so the next-step view
        # of the actuator command reflects the wind-up correction.
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
