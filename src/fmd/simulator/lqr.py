"""LQR (Linear Quadratic Regulator) controllers.

This module provides LQR feedback controllers for stabilization and
trajectory tracking. Controllers inherit from ControlSchedule and
can be used directly in simulation.

Control law: u = u_ref - K @ (x - x_ref)

The negative sign convention means K is the stabilizing feedback gain.
For a stable closed-loop system, eigenvalues of (A - B @ K) should have
negative real parts (continuous) or magnitude < 1 (discrete).

Tuning and Numerical Stability
------------------------------
The Q/R weight ratio controls closed-loop response speed. Aggressive tuning
(high Q/R ratio) creates fast closed-loop eigenvalues that may require very
small simulation timesteps with explicit integrators like RK4.

**Key constraint**: RK4 requires dt < 2.785 / |λ_max| where λ_max is the
fastest closed-loop eigenvalue.

For small vehicles with high control authority (e.g., Crazyflie quadrotor
with mass=27g, inertia=1.4e-5 kg⋅m²), aggressive tuning can create
eigenvalues at ~9000 rad/s, requiring dt < 0.3ms for stability.

**Recommendations**:
- For dt=10ms: Use R diagonal values 100x larger than "textbook" values
- For aggressive tracking: Reduce timestep to dt < 1ms
- Check closed-loop eigenvalues: ``scipy.linalg.eigvals(A - B @ K)``

See ``scripts/analyze_lqr_stability.py`` for detailed stability analysis.

For detailed guidance on controller tuning, timestep selection, and
integrator choice, see ``docs/public/control_guide.md``.

Example:
    from fmd.simulator import Cartpole, simulate
    from fmd.simulator.lqr import LQRController
    from fmd.simulator.params import CARTPOLE_CLASSIC
    import jax.numpy as jnp

    cartpole = Cartpole(CARTPOLE_CLASSIC)
    x_eq = cartpole.upright_state()
    u_eq = jnp.array([0.0])

    # Design LQR controller
    Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0]))  # State weights
    R = jnp.array([[0.1]])  # Control weight

    controller = LQRController.from_linearization(
        cartpole, x_eq, u_eq, Q, R
    )

    # Simulate from perturbed initial condition
    x0 = jnp.array([0.0, 0.0, 0.1, 0.0])  # Small angle
    result = simulate(cartpole, x0, dt=0.01, duration=5.0, control=controller)
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy import linalg
from typing import Optional, TYPE_CHECKING

from fmd.simulator.control import ControlSchedule
from fmd.simulator.linearize import linearize, discretize_zoh

if TYPE_CHECKING:
    from fmd.simulator.base import JaxDynamicSystem


def solve_continuous_are(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
) -> Array:
    """Solve the continuous-time algebraic Riccati equation.

    Finds P such that:
        A'P + PA - PBR^{-1}B'P + Q = 0

    Args:
        A: State matrix (n, n)
        B: Input matrix (n, m)
        Q: State cost matrix (n, n), must be positive semi-definite
        R: Control cost matrix (m, m), must be positive definite

    Returns:
        P: Solution matrix (n, n)

    Note:
        Uses scipy.linalg.solve_continuous_are internally.
        Different scipy versions may yield slightly different results
        for ill-conditioned problems.
    """
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    Q_np = np.asarray(Q)
    R_np = np.asarray(R)

    P_np = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)

    return jnp.array(P_np)


def solve_discrete_are(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
) -> Array:
    """Solve the discrete-time algebraic Riccati equation.

    Finds P such that:
        A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0

    Args:
        A: State matrix (n, n)
        B: Input matrix (n, m)
        Q: State cost matrix (n, n), must be positive semi-definite
        R: Control cost matrix (m, m), must be positive definite

    Returns:
        P: Solution matrix (n, n)

    Note:
        Uses scipy.linalg.solve_discrete_are internally.
    """
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    Q_np = np.asarray(Q)
    R_np = np.asarray(R)

    P_np = linalg.solve_discrete_are(A_np, B_np, Q_np, R_np)

    return jnp.array(P_np)


def compute_lqr_gain(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    discrete: bool = False,
) -> Array:
    """Compute LQR feedback gain K.

    For continuous-time:
        K = R^{-1} B' P

    For discrete-time:
        K = (R + B'PB)^{-1} B' P A

    where P is the solution to the algebraic Riccati equation.

    Args:
        A: State matrix (n, n)
        B: Input matrix (n, m)
        Q: State cost matrix (n, n)
        R: Control cost matrix (m, m)
        discrete: If True, solve discrete-time ARE

    Returns:
        K: Feedback gain matrix (m, n)

    Note:
        The control law is u = -K @ x (or u = u_ref - K @ (x - x_ref)).
        The closed-loop system is x_dot = (A - B @ K) @ x.
    """
    if discrete:
        P = solve_discrete_are(A, B, Q, R)
        # K = (R + B'PB)^{-1} B' P A
        R_np = np.asarray(R)
        B_np = np.asarray(B)
        P_np = np.asarray(P)
        A_np = np.asarray(A)

        BtPB = B_np.T @ P_np @ B_np
        K_np = np.linalg.solve(R_np + BtPB, B_np.T @ P_np @ A_np)
        K = jnp.array(K_np)
    else:
        P = solve_continuous_are(A, B, Q, R)
        # K = R^{-1} B' P
        R_np = np.asarray(R)
        B_np = np.asarray(B)
        P_np = np.asarray(P)

        K_np = np.linalg.solve(R_np, B_np.T @ P_np)
        K = jnp.array(K_np)

    return K


class LQRController(ControlSchedule):
    """LQR feedback controller for stabilization.

    Implements the control law:
        u = u_ref - K @ (x - x_ref)

    where K is the LQR gain, x_ref is the reference state, and
    u_ref is the feedforward control at the reference.

    Convention notes:
        - K is (m x n) where m = control dim, n = state dim
        - Row convention: K[i,:] corresponds to control[i]
        - Sign convention: negative feedback (stabilizing)
        - The closed-loop system is: dx/dt = (A - B @ K) @ (x - x_ref)

    Attributes:
        K: Feedback gain matrix (m, n)
        x_ref: Reference state vector (n,)
        u_ref: Feedforward control vector (m,)

    Example:
        # Create from pre-computed gain
        controller = LQRController(K=K, x_ref=x_eq, u_ref=u_eq)

        # Create from system linearization
        controller = LQRController.from_linearization(
            system, x_eq, u_eq, Q, R
        )
    """

    K: Array  # Feedback gain (m x n)
    x_ref: Array  # Reference state (n,)
    u_ref: Array  # Feedforward control (m,)

    def __call__(self, t: float, state: Array) -> Array:
        """Compute LQR control at current state.

        Args:
            t: Current time (unused for fixed-gain LQR)
            state: Current state vector

        Returns:
            Control vector: u = u_ref - K @ (state - x_ref)
        """
        error = state - self.x_ref
        return self.u_ref - self.K @ error

    @classmethod
    def from_linearization(
        cls,
        system: "JaxDynamicSystem",
        x_eq: Array,
        u_eq: Array,
        Q: Array,
        R: Array,
        discrete: bool = False,
        dt: Optional[float] = None,
    ) -> "LQRController":
        """Create LQR controller from system linearization.

        Linearizes the system at (x_eq, u_eq), computes the LQR gain,
        and returns a controller that stabilizes to that equilibrium.

        Args:
            system: JaxDynamicSystem to linearize
            x_eq: Equilibrium state (reference point)
            u_eq: Equilibrium control (feedforward)
            Q: State cost matrix (n, n), positive semi-definite
            R: Control cost matrix (m, m), positive definite
            discrete: If True, design discrete-time LQR
            dt: Time step for discretization (required if discrete=True)

        Returns:
            LQRController instance

        Raises:
            ValueError: If discrete=True but dt is not provided

        Example:
            Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0]))
            R = jnp.array([[0.1]])
            controller = LQRController.from_linearization(
                cartpole, x_eq, u_eq, Q, R
            )
        """
        if discrete and dt is None:
            raise ValueError("dt must be provided for discrete-time LQR")

        # Linearize system
        A, B = linearize(system, x_eq, u_eq)

        if discrete:
            # Discretize then design discrete LQR
            Ad, Bd = discretize_zoh(A, B, dt)
            K = compute_lqr_gain(Ad, Bd, Q, R, discrete=True)
        else:
            # Design continuous-time LQR
            K = compute_lqr_gain(A, B, Q, R, discrete=False)

        return cls(K=K, x_ref=jnp.asarray(x_eq), u_ref=jnp.asarray(u_eq))

    @classmethod
    def from_matrices(
        cls,
        A: Array,
        B: Array,
        Q: Array,
        R: Array,
        x_ref: Array,
        u_ref: Array,
        discrete: bool = False,
    ) -> "LQRController":
        """Create LQR controller from pre-computed linear system matrices.

        Use this when you already have the linearization or want to
        use specific A, B matrices (e.g., from a benchmark).

        Args:
            A: State matrix (n, n)
            B: Input matrix (n, m)
            Q: State cost matrix (n, n)
            R: Control cost matrix (m, m)
            x_ref: Reference state
            u_ref: Reference control (feedforward)
            discrete: If True, A and B are discrete-time matrices

        Returns:
            LQRController instance
        """
        K = compute_lqr_gain(A, B, Q, R, discrete=discrete)
        return cls(K=K, x_ref=jnp.asarray(x_ref), u_ref=jnp.asarray(u_ref))

    def with_reference(
        self,
        x_ref: Optional[Array] = None,
        u_ref: Optional[Array] = None,
    ) -> "LQRController":
        """Return controller with updated setpoint.

        Creates a new controller with the same gain K but different
        reference state and/or feedforward control.

        Args:
            x_ref: New reference state (default: keep current)
            u_ref: New feedforward control (default: keep current)

        Returns:
            New LQRController with updated reference
        """
        new_x_ref = jnp.asarray(x_ref) if x_ref is not None else self.x_ref
        new_u_ref = jnp.asarray(u_ref) if u_ref is not None else self.u_ref
        return LQRController(K=self.K, x_ref=new_x_ref, u_ref=new_u_ref)

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.K.shape[1]

    @property
    def num_controls(self) -> int:
        """Number of control inputs."""
        return self.K.shape[0]


class TrajectoryLQRController(ControlSchedule):
    """LQR with time-varying reference trajectory.

    Implements the control law:
        u(t) = u_ref(t) - K @ (x - x_ref(t))

    Uses a fixed gain K computed at a single linearization point,
    but tracks a time-varying reference trajectory. Suitable for
    trajectories where the dynamics don't vary significantly.

    For aggressive maneuvers where dynamics change significantly
    along the trajectory, use TVLQRController instead.

    Attributes:
        K: Fixed feedback gain matrix (m, n)
        x_ref_fn: Function returning reference state at time t
        u_ref_fn: Function returning feedforward control at time t

    Example:
        # Create with interpolated trajectory
        controller = TrajectoryLQRController.from_trajectory(
            system, times, x_refs, u_refs, Q, R
        )
    """

    K: Array  # Fixed feedback gain (m x n)
    times: Array  # Reference trajectory times
    x_refs: Array  # Reference states (T x n)
    u_refs: Array  # Reference controls (T x m)

    def __call__(self, t: float, state: Array) -> Array:
        """Compute control at current state tracking trajectory.

        Args:
            t: Current time
            state: Current state vector

        Returns:
            Control vector: u = u_ref(t) - K @ (state - x_ref(t))
        """
        # Get reference at current time via linear interpolation
        x_ref = self._interpolate_state(t)
        u_ref = self._interpolate_control(t)

        error = state - x_ref
        return u_ref - self.K @ error

    def _interpolate_state(self, t: float) -> Array:
        """Linearly interpolate reference state at time t."""
        # Find interpolation indices
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)

        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        x0 = self.x_refs[idx]
        x1 = self.x_refs[idx + 1]

        alpha = (t - t0) / (t1 - t0 + 1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return x0 + alpha * (x1 - x0)

    def _interpolate_control(self, t: float) -> Array:
        """Linearly interpolate reference control at time t."""
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)

        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        u0 = self.u_refs[idx]
        u1 = self.u_refs[idx + 1]

        alpha = (t - t0) / (t1 - t0 + 1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return u0 + alpha * (u1 - u0)

    @classmethod
    def from_trajectory(
        cls,
        system: "JaxDynamicSystem",
        times: Array,
        x_refs: Array,
        u_refs: Array,
        Q: Array,
        R: Array,
        linearize_at: Optional[int] = None,
    ) -> "TrajectoryLQRController":
        """Create from reference trajectory arrays.

        Computes a single LQR gain at the specified linearization point
        (default: first point) and uses it throughout the trajectory.

        Args:
            system: JaxDynamicSystem to linearize
            times: Trajectory time points (T,)
            x_refs: Reference states (T, n)
            u_refs: Reference controls (T, m)
            Q: State cost matrix (n, n)
            R: Control cost matrix (m, m)
            linearize_at: Index in trajectory to linearize at (default: 0)

        Returns:
            TrajectoryLQRController instance
        """
        times = jnp.asarray(times)
        x_refs = jnp.asarray(x_refs)
        u_refs = jnp.asarray(u_refs)

        # Choose linearization point
        lin_idx = linearize_at if linearize_at is not None else 0
        x_lin = x_refs[lin_idx]
        u_lin = u_refs[lin_idx]

        # Compute LQR gain at linearization point
        A, B = linearize(system, x_lin, u_lin)
        K = compute_lqr_gain(A, B, Q, R, discrete=False)

        return cls(K=K, times=times, x_refs=x_refs, u_refs=u_refs)

    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        return float(self.times[-1] - self.times[0])

    @property
    def num_points(self) -> int:
        """Number of trajectory points."""
        return len(self.times)


class TVLQRController(ControlSchedule):
    """Time-varying LQR with gain recomputation along trajectory.

    Implements the control law:
        u(t) = u_ref(t) - K(t) @ (x - x_ref(t))

    Recomputes the LQR gain K(t) by relinearizing at each trajectory
    point. Required for aggressive maneuvers where dynamics vary
    significantly along the trajectory.

    Attributes:
        times: Trajectory time points
        x_refs: Reference states (T, n)
        u_refs: Reference controls (T, m)
        Ks: Pre-computed gains at each time point (T, m, n)

    Note:
        Gains are pre-computed at construction for efficiency.
        Gains are interpolated between trajectory points during simulation.
    """

    times: Array  # (T,)
    x_refs: Array  # (T, n)
    u_refs: Array  # (T, m)
    Ks: Array  # (T, m, n) - pre-computed gains

    def __call__(self, t: float, state: Array) -> Array:
        """Compute control with time-varying gain.

        Args:
            t: Current time
            state: Current state vector

        Returns:
            Control vector: u = u_ref(t) - K(t) @ (state - x_ref(t))
        """
        # Get reference and gain at current time
        x_ref = self._interpolate_state(t)
        u_ref = self._interpolate_control(t)
        K = self._interpolate_gain(t)

        error = state - x_ref
        return u_ref - K @ error

    def _interpolate_state(self, t: float) -> Array:
        """Linearly interpolate reference state at time t."""
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)

        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        x0 = self.x_refs[idx]
        x1 = self.x_refs[idx + 1]

        alpha = (t - t0) / (t1 - t0 + 1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return x0 + alpha * (x1 - x0)

    def _interpolate_control(self, t: float) -> Array:
        """Linearly interpolate reference control at time t."""
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)

        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        u0 = self.u_refs[idx]
        u1 = self.u_refs[idx + 1]

        alpha = (t - t0) / (t1 - t0 + 1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return u0 + alpha * (u1 - u0)

    def _interpolate_gain(self, t: float) -> Array:
        """Interpolate gain matrix at time t.

        Uses linear interpolation between cached gain matrices.
        """
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)

        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        K0 = self.Ks[idx]
        K1 = self.Ks[idx + 1]

        alpha = (t - t0) / (t1 - t0 + 1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return K0 + alpha * (K1 - K0)

    @classmethod
    def from_trajectory(
        cls,
        system: "JaxDynamicSystem",
        times: Array,
        x_refs: Array,
        u_refs: Array,
        Q: Array,
        R: Array,
    ) -> "TVLQRController":
        """Create TVLQR by relinearizing at each trajectory point.

        Pre-computes gains at each trajectory point for efficiency
        during simulation.

        Args:
            system: JaxDynamicSystem to linearize
            times: Trajectory time points (T,)
            x_refs: Reference states (T, n)
            u_refs: Reference controls (T, m)
            Q: State cost matrix (n, n)
            R: Control cost matrix (m, m)

        Returns:
            TVLQRController with pre-computed gains
        """
        times = jnp.asarray(times)
        x_refs = jnp.asarray(x_refs)
        u_refs = jnp.asarray(u_refs)

        n_points = len(times)
        n_states = x_refs.shape[1]
        n_controls = u_refs.shape[1]

        # Pre-compute gains at each trajectory point
        Ks = np.zeros((n_points, n_controls, n_states))

        for i in range(n_points):
            A, B = linearize(system, x_refs[i], u_refs[i])
            Ks[i] = np.asarray(compute_lqr_gain(A, B, Q, R, discrete=False))

        return cls(
            times=times,
            x_refs=x_refs,
            u_refs=u_refs,
            Ks=jnp.array(Ks),
        )

    def gain_at(self, idx: int) -> Array:
        """Get pre-computed gain at trajectory index."""
        return self.Ks[idx]

    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        return float(self.times[-1] - self.times[0])

    @property
    def num_points(self) -> int:
        """Number of trajectory points."""
        return len(self.times)


# Convenience function for quick LQR design
def lqr(
    system: "JaxDynamicSystem",
    x_eq: Array,
    u_eq: Array,
    Q: Array,
    R: Array,
    discrete: bool = False,
    dt: Optional[float] = None,
) -> LQRController:
    """Quick LQR controller design (convenience function).

    Equivalent to LQRController.from_linearization().

    Args:
        system: JaxDynamicSystem to stabilize
        x_eq: Equilibrium state
        u_eq: Equilibrium control
        Q: State cost matrix
        R: Control cost matrix
        discrete: If True, design discrete-time LQR
        dt: Time step for discretization (required if discrete=True)

    Returns:
        LQRController instance

    Example:
        controller = lqr(cartpole, x_eq, u_eq, Q, R)
    """
    return LQRController.from_linearization(
        system, x_eq, u_eq, Q, R, discrete=discrete, dt=dt
    )
