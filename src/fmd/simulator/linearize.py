"""Linearization utilities for dynamic systems.

This module provides functions for linearizing JAX dynamic systems around
equilibrium points, discretizing continuous-time linear systems, and
checking controllability.

All functions are designed to work with the JaxDynamicSystem interface
and can be used for LQR controller design and benchmark validation.

Example:
    from fmd.simulator import Cartpole, linearize, discretize_zoh
    from fmd.simulator.params import CARTPOLE_CLASSIC
    import jax.numpy as jnp

    cartpole = Cartpole(CARTPOLE_CLASSIC)
    x_eq = cartpole.upright_state()
    u_eq = jnp.array([0.0])

    # Get continuous-time linearization
    A, B = linearize(cartpole, x_eq, u_eq)

    # Discretize with zero-order hold
    Ad, Bd = discretize_zoh(A, B, dt=0.01)
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy import linalg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmd.simulator.base import JaxDynamicSystem


def linearize(
    system: "JaxDynamicSystem",
    x_eq: Array,
    u_eq: Array,
    t: float = 0.0,
) -> tuple[Array, Array]:
    """Compute A = df/dx, B = df/du at equilibrium using the system's Jacobian methods.

    Linearizes the system dynamics dx/dt = f(x, u, t) around the operating
    point (x_eq, u_eq) to obtain the linear system:
        dx/dt = A @ (x - x_eq) + B @ (u - u_eq)

    CRITICAL: This uses the system's get_state_jacobian() and get_control_jacobian()
    methods, which in turn use forward_dynamics(). This ensures the same dynamics
    function is used for both linearization and simulation. This includes any
    preprocessing, clamps, or saturation in forward_dynamics().

    Args:
        system: JaxDynamicSystem instance to linearize.
        x_eq: Equilibrium state vector (n,).
        u_eq: Equilibrium control vector (m,).
        t: Time at which to linearize (default 0.0). Most systems are
           time-invariant, but this is included for generality.

    Returns:
        tuple of (A, B) where:
            A: State Jacobian df/dx at (x_eq, u_eq), shape (n, n)
            B: Control Jacobian df/du at (x_eq, u_eq), shape (n, m)

    Example:
        >>> cartpole = Cartpole(CARTPOLE_CLASSIC)
        >>> A, B = linearize(cartpole, cartpole.upright_state(), jnp.array([0.0]))
        >>> A.shape
        (4, 4)
        >>> B.shape
        (4, 1)

    Note:
        The Jacobians are computed using JAX's autodiff (jax.jacobian),
        which provides machine-precision derivatives for pure JAX functions.
        Expected precision is ~1e-14 for well-conditioned systems.
    """
    # Convert inputs to JAX arrays if needed
    x_eq = jnp.asarray(x_eq)
    u_eq = jnp.asarray(u_eq)

    # Use the built-in Jacobian methods from JaxDynamicSystem
    A = system.get_state_jacobian(x_eq, u_eq, t)
    B = system.get_control_jacobian(x_eq, u_eq, t)

    return A, B


def discretize_zoh(
    A: Array,
    B: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Zero-order hold discretization using matrix exponential.

    Converts continuous-time linear system (A, B) to discrete-time (Ad, Bd)
    assuming the control is held constant over each time step:
        x[k+1] = Ad @ x[k] + Bd @ u[k]

    The exact discretization is:
        Ad = exp(A * dt)
        Bd = inv(A) @ (Ad - I) @ B  (or via matrix exponential for singular A)

    This uses scipy.linalg.expm which handles ill-conditioned matrices well.

    Args:
        A: Continuous-time state matrix (n, n).
        B: Continuous-time input matrix (n, m).
        dt: Time step (s).

    Returns:
        tuple of (Ad, Bd) where:
            Ad: Discrete-time state matrix (n, n)
            Bd: Discrete-time input matrix (n, m)

    Note:
        - Uses scipy.linalg.expm for numerical stability
        - For very small dt (<1e-6), Euler approximation may be more stable
        - Expected precision: ~1e-10 for well-conditioned systems
        - Converts between JAX and NumPy arrays as needed

    Example:
        >>> Ad, Bd = discretize_zoh(A, B, dt=0.01)
        >>> # For small dt, should be close to Euler:
        >>> # Ad ≈ I + A*dt, Bd ≈ B*dt
    """
    # Convert to numpy for scipy
    A_np = np.asarray(A)
    B_np = np.asarray(B)

    n = A_np.shape[0]
    m = B_np.shape[1]

    # Build augmented matrix [A, B; 0, 0] for exact discretization
    # exp([A, B; 0, 0] * dt) = [Ad, Bd_approx; 0, I]
    # where Bd_approx = int_0^dt exp(A*tau) @ B dtau
    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = A_np * dt
    augmented[:n, n:] = B_np * dt

    # Matrix exponential of augmented system
    exp_aug = linalg.expm(augmented)

    # Extract discrete matrices
    Ad_np = exp_aug[:n, :n]
    Bd_np = exp_aug[:n, n:]

    # Convert back to JAX arrays
    Ad = jnp.array(Ad_np)
    Bd = jnp.array(Bd_np)

    return Ad, Bd


def discretize_euler(
    A: Array,
    B: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Simple Euler discretization: Ad = I + A*dt, Bd = B*dt.

    This is the simplest discretization method and is appropriate for:
    - Very small dt where higher-order terms are negligible
    - Matching benchmarks that use Euler integration
    - Initial development/debugging

    For larger dt or stiff systems, use discretize_zoh() instead.

    Args:
        A: Continuous-time state matrix (n, n).
        B: Continuous-time input matrix (n, m).
        dt: Time step (s).

    Returns:
        tuple of (Ad, Bd) where:
            Ad: Discrete-time state matrix = I + A*dt
            Bd: Discrete-time input matrix = B*dt

    Example:
        >>> Ad, Bd = discretize_euler(A, B, dt=0.001)
        >>> # Equivalent to forward Euler integration
    """
    n = A.shape[0]
    I = jnp.eye(n)

    Ad = I + A * dt
    Bd = B * dt

    return Ad, Bd


def controllability_matrix(A: Array, B: Array) -> Array:
    """Compute controllability matrix [B, AB, A^2B, ..., A^{n-1}B].

    The controllability matrix C has shape (n, n*m) where n is the number
    of states and m is the number of controls. The system is controllable
    if and only if rank(C) = n.

    Args:
        A: State matrix (n, n).
        B: Input matrix (n, m).

    Returns:
        Controllability matrix with shape (n, n*m).
        Columns are [B, AB, A^2B, ..., A^{n-1}B].

    Example:
        >>> C = controllability_matrix(A, B)
        >>> rank = jnp.linalg.matrix_rank(C)
        >>> is_controllable = (rank == A.shape[0])
    """
    n = A.shape[0]

    # Build columns iteratively: B, AB, A^2B, ...
    columns = [B]
    A_power_B = B

    for _ in range(n - 1):
        A_power_B = A @ A_power_B
        columns.append(A_power_B)

    # Stack horizontally to form [B, AB, A^2B, ..., A^{n-1}B]
    C = jnp.hstack(columns)

    return C


def is_controllable(A: Array, B: Array, tol: float = 1e-10) -> bool:
    """Check if the linear system (A, B) is controllable.

    A system is controllable if all states can be driven to any desired
    value using the available controls. This is equivalent to the
    controllability matrix having full row rank.

    Args:
        A: State matrix (n, n).
        B: Input matrix (n, m).
        tol: Tolerance for rank determination. Singular values below
             this threshold relative to the largest are considered zero.

    Returns:
        True if the system is controllable (rank(C) = n), False otherwise.

    Note:
        Uses numpy's matrix_rank with the specified tolerance.
        For numerical robustness, the tolerance should be chosen based
        on the expected condition number of the controllability matrix.

    Example:
        >>> # Cartpole at upright equilibrium should be controllable
        >>> A, B = linearize(cartpole, x_eq, u_eq)
        >>> assert is_controllable(A, B)
    """
    C = controllability_matrix(A, B)

    # Use numpy for rank computation (more robust)
    C_np = np.asarray(C)
    n = A.shape[0]

    # Compute rank using SVD
    # Note: numpy.linalg.matrix_rank uses SVD internally
    rank = np.linalg.matrix_rank(C_np, tol=tol)

    return rank == n


def observability_matrix(A: Array, C: Array) -> Array:
    """Compute observability matrix [C; CA; CA^2; ...; CA^{n-1}].

    The observability matrix O has shape (n*p, n) where n is the number
    of states and p is the number of outputs. The system is observable
    if and only if rank(O) = n.

    Args:
        A: State matrix (n, n).
        C: Output matrix (p, n).

    Returns:
        Observability matrix with shape (n*p, n).
        Rows are [C; CA; CA^2; ...; CA^{n-1}].

    Example:
        >>> O = observability_matrix(A, C)
        >>> rank = jnp.linalg.matrix_rank(O)
        >>> is_observable = (rank == A.shape[0])
    """
    n = A.shape[0]

    # Build rows iteratively: C, CA, CA^2, ...
    rows = [C]
    CA_power = C

    for _ in range(n - 1):
        CA_power = CA_power @ A
        rows.append(CA_power)

    # Stack vertically to form [C; CA; CA^2; ...; CA^{n-1}]
    O = jnp.vstack(rows)

    return O


def is_observable(A: Array, C: Array, tol: float = 1e-10) -> bool:
    """Check if the linear system (A, C) is observable.

    A system is observable if the current state can be determined from
    past and current outputs. This is equivalent to the observability
    matrix having full column rank.

    Args:
        A: State matrix (n, n).
        C: Output matrix (p, n).
        tol: Tolerance for rank determination.

    Returns:
        True if the system is observable (rank(O) = n), False otherwise.

    Example:
        >>> # Check observability with full state output
        >>> C = jnp.eye(n)  # Full state measurement
        >>> assert is_observable(A, C)  # Always observable
    """
    O = observability_matrix(A, C)

    # Use numpy for rank computation
    O_np = np.asarray(O)
    n = A.shape[0]

    rank = np.linalg.matrix_rank(O_np, tol=tol)

    return rank == n
