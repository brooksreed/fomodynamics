"""Moth LQR controller design and gain scheduling.

Provides functions to design LQR controllers at Moth 3DOF trim points
and a gain-scheduled controller that interpolates gains across speeds.

When surge_enabled=True (default), surge runs open-loop in the plant
(not controllable by flap/elevator). LQR is designed on the controllable
4-state subsystem [pos_d, theta, w, q] and the gain is embedded into
the full 5-state format with zero gain on the U column.

Example:
    from fmd.simulator.moth_lqr import design_moth_lqr, design_moth_gain_schedule
    from fmd.simulator.moth_lqr import MothGainScheduledController

    # Single speed design
    result = design_moth_lqr(u_forward=10.0)
    print(f"Gain K shape: {result.K.shape}")

    # Multi-speed gain schedule
    schedule = design_moth_gain_schedule()
    controller = MothGainScheduledController.from_gain_schedule(schedule)
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from scipy import linalg

from fmd.simulator.control import ControlSchedule
from fmd.simulator.linearize import linearize, discretize_zoh
from fmd.simulator.lqr import compute_lqr_gain
from fmd.simulator.trim_casadi import find_moth_trim, CasadiTrimResult
from fmd.simulator.params import MothParams, MOTH_BIEKER_V3
from fmd.simulator.moth_3d import Moth3D, ConstantSchedule, POS_D, THETA, W, Q, U as MOTH_U

# Default Moth 3DOF simulation/control timestep (seconds).
# At 10 m/s the fast pitch eigenvalue is ~280 rad/s, giving a max stable
# RK4 dt of 9.6 ms.  5 ms provides adequate margin across 6-20 m/s.
MOTH_DEFAULT_DT = 0.005

# Speed points: 8 speeds from 6-20 m/s, dense around typical foiling range
DEFAULT_SPEEDS_MS = [6.0, 8.0, 10.0, 11.0, 12.0, 14.0, 17.0, 20.0]

# Controllable state indices (without surge when surge_enabled=False)
_CTRL_STATES = [POS_D, THETA, W, Q]  # [0, 1, 2, 3]


@dataclass
class MothTrimLQR:
    """Holds trim result + LQR design data for a single speed point.

    K is always (2, 5) -- the full 5-state gain matrix. When surge is
    disabled, K[:, U] is zero (no feedback on the uncontrollable state).

    Attributes:
        trim: CasadiTrimResult from find_moth_trim
        u_forward: Forward speed (m/s)
        A: Continuous-time state matrix (5, 5)
        B: Continuous-time input matrix (5, 2)
        Ad: Discrete-time state matrix (5, 5)
        Bd: Discrete-time input matrix (5, 2)
        K: LQR feedback gain (2, 5) -- full state format
        Q: State cost matrix used for design
        R: Control cost matrix (2, 2)
        dt: Discretization timestep (s)
        closed_loop_eigenvalues: Eigenvalues of controllable closed-loop
    """

    trim: CasadiTrimResult
    u_forward: float
    A: np.ndarray
    B: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    K: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    dt: float
    closed_loop_eigenvalues: np.ndarray


def design_moth_lqr(
    params: MothParams = MOTH_BIEKER_V3,
    u_forward: float = 10.0,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    dt: float = MOTH_DEFAULT_DT,
    target_theta: Optional[float] = None,
    target_pos_d: Optional[float] = None,
    heel_angle: float = np.deg2rad(30.0),
) -> MothTrimLQR:
    """Design LQR at a single speed trim point.

    When surge_enabled=False, designs on the 4-state controllable
    subsystem [pos_d, theta, w, q] and embeds into 5-state gain.

    Default Q: diag([100, 100, 10, 10]) for 4-state design
    Default R: diag([50, 500])

    Args:
        params: Moth parameter set.
        u_forward: Forward speed (m/s).
        Q: State cost matrix. Shape (4,4) when surge disabled, (5,5) when enabled.
            Default prioritizes height and pitch regulation.
        R: Control cost matrix (2, 2). Default diag([50, 500]) penalizes
            control action to produce smoother control with minimal impact
            on breach fraction.
        dt: Discretization timestep (s).
        target_theta: If set, fix theta to this value during trim search
            instead of letting the optimizer find it.
        target_pos_d: If set, fix pos_d to this value during trim search.
        heel_angle: Static heel angle (rad). Default 30 deg (nominal foiling).
            Must match the heel_angle used for simulation so that the LQR
            design and simulation operate at the same trim/linearization point.

    Returns:
        MothTrimLQR with trim, linearization, and LQR data.
    """
    moth = Moth3D(params, u_forward=ConstantSchedule(u_forward), heel_angle=heel_angle)

    # CasADi solver is robust — no continuation seeding needed
    trim = find_moth_trim(
        params, u_forward=u_forward,
        target_theta=target_theta, target_pos_d=target_pos_d,
        heel_angle=heel_angle,
    )

    A_full, B_full = linearize(
        moth, jnp.array(trim.state), jnp.array(trim.control)
    )
    Ad_full, Bd_full = discretize_zoh(A_full, B_full, dt)

    A_full_np = np.asarray(A_full)
    B_full_np = np.asarray(B_full)
    Ad_full_np = np.asarray(Ad_full)
    Bd_full_np = np.asarray(Bd_full)

    # Always design on 4-state controllable subsystem [pos_d, theta, w, q].
    # Surge is not directly controllable by flap/elevator — it runs
    # open-loop in the plant. The gain matrix is embedded into 5-state
    # format with zero gain on the U column.
    keep = _CTRL_STATES
    Ad_r = Ad_full_np[np.ix_(keep, keep)]
    Bd_r = Bd_full_np[keep, :]

    if Q is None:
        Q = np.diag([100.0, 100.0, 10.0, 10.0])
    if R is None:
        R = np.diag([50.0, 500.0])

    K_r = np.asarray(compute_lqr_gain(Ad_r, Bd_r, Q, R, discrete=True))

    # Embed into 5-state gain: zero gain on U column
    K_5 = np.zeros((2, 5))
    K_5[:, _CTRL_STATES] = K_r

    # Closed-loop eigenvalues of the controllable subsystem
    A_cl_r = Ad_r - Bd_r @ K_r
    eigs = linalg.eigvals(A_cl_r)

    return MothTrimLQR(
        trim=trim,
        u_forward=u_forward,
        A=A_full_np,
        B=B_full_np,
        Ad=Ad_full_np,
        Bd=Bd_full_np,
        K=K_5,
        Q=Q,
        R=R,
        dt=dt,
        closed_loop_eigenvalues=eigs,
    )


def design_moth_gain_schedule(
    params: MothParams = MOTH_BIEKER_V3,
    speeds_ms: Optional[list] = None,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    dt: float = MOTH_DEFAULT_DT,
    target_theta: Optional[float] = None,
    target_pos_d: Optional[float] = None,
    heel_angle: float = np.deg2rad(30.0),
) -> list[MothTrimLQR]:
    """Design LQR at multiple speeds for gain scheduling.

    Uses trim continuation: each speed's trim result seeds the next speed's
    initial guess, which is essential for convergence at high speeds (>12 m/s).

    Args:
        params: Moth parameter set.
        speeds_ms: List of forward speeds (m/s). Default: 6-20 m/s range.
        Q: State cost matrix (4,4 or 5,5 depending on surge mode).
        R: Control cost matrix (2, 2).
        dt: Discretization timestep (s).
        target_theta: If set, fix theta to this value during trim search
            for every speed point.
        target_pos_d: If set, fix pos_d to this value during trim search
            for every speed point.
        heel_angle: Static heel angle (rad) used for all design points.

    Returns:
        List of MothTrimLQR, one per speed, sorted by speed.
    """
    if speeds_ms is None:
        speeds_ms = DEFAULT_SPEEDS_MS
    results = []
    for spd in speeds_ms:
        result = design_moth_lqr(
            params=params,
            u_forward=spd,
            Q=Q,
            R=R,
            dt=dt,
            target_theta=target_theta,
            target_pos_d=target_pos_d,
            heel_angle=heel_angle,
        )
        results.append(result)
    return results


class MothGainScheduledController(ControlSchedule):
    """Gain-scheduled LQR controller that interpolates K, x_trim, u_trim by speed.

    Linearly interpolates between pre-designed LQR gains based on the
    current forward speed (state[U]). Clamps to the nearest design point
    if speed is outside the scheduled range.

    Attributes:
        speeds: (N,) design speeds in m/s, sorted ascending
        Ks: (N, m, n) gain matrices at each speed, n=5 (full state)
        x_trims: (N, n) trim states at each speed
        u_trims: (N, m) trim controls at each speed
        design_dt: Timestep (s) used during LQR design (discretization)
    """

    speeds: Array  # (N,) speeds in m/s
    Ks: Array  # (N, m, n) gain matrices
    x_trims: Array  # (N, n) trim states
    u_trims: Array  # (N, m) trim controls
    design_dt: float  # Timestep used during LQR design

    def __call__(self, t: float, state: Array) -> Array:
        """Compute control: interpolate K, x_trim, u_trim by state[U].

        Args:
            t: Current time (unused).
            state: Current state vector [pos_d, theta, w, q, u].

        Returns:
            Control vector: u_trim_interp - K_interp @ (state - x_trim_interp)
        """
        u_speed = state[MOTH_U]
        u_clamped = jnp.clip(u_speed, self.speeds[0], self.speeds[-1])

        # Find interpolation indices
        idx = jnp.searchsorted(self.speeds, u_clamped, side="right") - 1
        idx = jnp.clip(idx, 0, len(self.speeds) - 2)

        # Linear interpolation
        t0 = self.speeds[idx]
        t1 = self.speeds[idx + 1]
        alpha = (u_clamped - t0) / (t1 - t0 + 1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        K_interp = self.Ks[idx] + alpha * (self.Ks[idx + 1] - self.Ks[idx])
        x_trim_interp = self.x_trims[idx] + alpha * (self.x_trims[idx + 1] - self.x_trims[idx])
        u_trim_interp = self.u_trims[idx] + alpha * (self.u_trims[idx + 1] - self.u_trims[idx])

        return u_trim_interp - K_interp @ (state - x_trim_interp)

    @classmethod
    def from_gain_schedule(
        cls, schedule: list[MothTrimLQR]
    ) -> "MothGainScheduledController":
        """Build controller from a list of MothTrimLQR design points.

        Args:
            schedule: List of MothTrimLQR from design_moth_gain_schedule.
                Must be sorted by u_forward ascending.

        Returns:
            MothGainScheduledController ready for simulation.

        Raises:
            ValueError: If schedule entries have inconsistent dt values.
        """
        schedule = sorted(schedule, key=lambda s: s.u_forward)

        # Validate all entries have the same dt
        dts = {s.dt for s in schedule}
        if len(dts) > 1:
            raise ValueError(
                f"Schedule entries have inconsistent dt values: {sorted(dts)}. "
                "All entries must use the same discretization timestep."
            )
        design_dt = schedule[0].dt

        speeds = jnp.array([s.u_forward for s in schedule])
        Ks = jnp.array([s.K for s in schedule])
        x_trims = jnp.array([s.trim.state for s in schedule])
        u_trims = jnp.array([s.trim.control for s in schedule])
        return cls(
            speeds=speeds, Ks=Ks, x_trims=x_trims,
            u_trims=u_trims, design_dt=design_dt,
        )


def validate_simulation_dt(
    controller: MothGainScheduledController,
    sim_dt: float,
    tolerance: float = 0.2,
) -> bool:
    """Warn if sim_dt differs from controller design_dt by more than tolerance.

    Call this before simulation to catch timestep mismatches between the
    LQR design and the simulation integrator.

    Args:
        controller: Gain-scheduled controller with a design_dt attribute.
        sim_dt: Simulation timestep (s) that will be used with ``simulate()``.
        tolerance: Maximum relative difference before warning (default 0.2 = 20%).

    Returns:
        True if sim_dt is within tolerance, False otherwise.
    """
    rel_diff = abs(sim_dt - controller.design_dt) / controller.design_dt
    if rel_diff > tolerance:
        warnings.warn(
            f"Simulation dt={sim_dt} differs from controller design_dt="
            f"{controller.design_dt} by {rel_diff:.0%} (tolerance {tolerance:.0%}). "
            f"Controller gains may not be appropriate for this timestep.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True
