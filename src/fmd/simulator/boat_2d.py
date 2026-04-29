"""JAX 2D planar boat dynamics with Coriolis coupling.

A physically-consistent 3-DOF maneuvering model (Fossen-style, without added mass).
Body-frame velocities with proper rotational coupling terms.

This is the JAX equivalent of fmd.simulator.boat_2d.Boat2D.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import Boat2DParams


# State vector indices
X = 0  # North position (m)
Y = 1  # East position (m)
PSI = 2  # Heading angle, wrapped to [-π, π] (rad)
U = 3  # Surge velocity, body +x (m/s)
V = 4  # Sway velocity, body +y (m/s)
R = 5  # Yaw rate, body +z rotation (rad/s)

# Control vector indices
THRUST = 0  # Forward thrust (N)
YAW_MOMENT = 1  # Yaw moment (N·m)


class Boat2DJax(JaxDynamicSystem):
    """2D planar boat with coupled rigid-body dynamics - JAX implementation.

    State vector (6 elements):
        [0] x   - North position (m)
        [1] y   - East position (m)
        [2] psi - Heading angle from North, CW positive (rad), wrapped to [-π, π]
        [3] u   - Surge velocity in body +x (m/s)
        [4] v   - Sway velocity in body +y (m/s)
        [5] r   - Yaw rate about body +z (rad/s)

    Control vector (2 elements):
        [0] thrust     - Forward thrust force (N)
        [1] yaw_moment - Yaw moment about z (N·m)

    Equations of motion (coupled planar rigid body):
        x_dot   = u·cos(psi) - v·sin(psi)
        y_dot   = u·sin(psi) + v·cos(psi)
        psi_dot = r
        u_dot   = thrust/m - (D_u/m)·u + r·v      # Coriolis: +rv
        v_dot   = -(D_v/m)·v - r·u                # Coriolis: -ru
        r_dot   = yaw_moment/Izz - (D_r/Izz)·r

    Attributes:
        mass: Vehicle mass (kg)
        izz: Yaw moment of inertia (kg*m^2)
        drag_surge: Surge damping coefficient (kg/s)
        drag_sway: Sway damping coefficient (kg/s)
        drag_yaw: Yaw damping coefficient (kg*m^2/s)

    Example:
        from fmd.simulator import Boat2DJax, simulate
        from fmd.simulator.params import BOAT2D_TEST_DEFAULT

        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)

        def control_fn(t, state):
            return jnp.array([50.0, 5.0])  # [thrust, yaw_moment]

        result = simulate(boat, boat.default_state(), dt=0.01, duration=20.0,
                         control=ConstantControl(jnp.array([50.0, 5.0])))

        # For gradient computation, use from_values:
        boat = Boat2DJax.from_values(mass=100.0, izz=50.0, drag_surge=10.0,
                                     drag_sway=20.0, drag_yaw=5.0)
    """

    mass: float
    izz: float
    drag_surge: float
    drag_sway: float
    drag_yaw: float

    # Static metadata
    state_names: tuple[str, ...] = eqx.field(
        static=True, default=("x", "y", "psi", "u", "v", "r")
    )
    control_names: tuple[str, ...] = eqx.field(
        static=True, default=("thrust", "yaw_moment")
    )

    def __init__(self, params: Boat2DParams):
        """Initialize boat model from parameters.

        Args:
            params: Boat2DParams instance with validated model parameters.

        Note:
            For JAX gradient computation, use from_values() instead to
            avoid non-JAX operations in params validation.
        """
        self.mass = params.mass
        self.izz = params.izz
        self.drag_surge = params.drag_surge
        self.drag_sway = params.drag_sway
        self.drag_yaw = params.drag_yaw

    @classmethod
    def from_values(
        cls,
        mass: float,
        izz: float,
        drag_surge: float,
        drag_sway: float,
        drag_yaw: float,
    ) -> "Boat2DJax":
        """Create boat directly from values (JAX-traceable).

        Use this constructor when differentiating through boat parameters,
        as it avoids the attrs validation in Boat2DParams.

        Args:
            mass: Vehicle mass (kg)
            izz: Yaw moment of inertia (kg*m^2)
            drag_surge: Surge damping coefficient (kg/s)
            drag_sway: Sway damping coefficient (kg/s)
            drag_yaw: Yaw damping coefficient (kg*m^2/s)

        Returns:
            Boat2DJax instance
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "mass", mass)
        object.__setattr__(obj, "izz", izz)
        object.__setattr__(obj, "drag_surge", drag_surge)
        object.__setattr__(obj, "drag_sway", drag_sway)
        object.__setattr__(obj, "drag_yaw", drag_yaw)
        object.__setattr__(obj, "state_names", ("x", "y", "psi", "u", "v", "r"))
        object.__setattr__(obj, "control_names", ("thrust", "yaw_moment"))
        return obj

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative with Coriolis coupling.

        Args:
            state: Current state [x, y, psi, u, v, r]
            control: Control input [thrust, yaw_moment]
            t: Current time (unused)

        Returns:
            State derivative [x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot]
        """
        psi = state[PSI]
        u = state[U]
        v = state[V]
        r = state[R]

        # Handle empty or partial control
        thrust = jnp.where(control.shape[0] > THRUST, control[THRUST], 0.0)
        yaw_moment = jnp.where(control.shape[0] > YAW_MOMENT, control[YAW_MOMENT], 0.0)

        cos_psi = jnp.cos(psi)
        sin_psi = jnp.sin(psi)

        # Kinematics (body → NED)
        x_dot = u * cos_psi - v * sin_psi
        y_dot = u * sin_psi + v * cos_psi
        psi_dot = r

        # Dynamics with Coriolis coupling
        u_dot = thrust / self.mass - (self.drag_surge / self.mass) * u + r * v
        v_dot = -(self.drag_sway / self.mass) * v - r * u
        r_dot = yaw_moment / self.izz - (self.drag_yaw / self.izz) * r

        return jnp.array([x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot])

    def default_state(self) -> Array:
        """Return default initial state (at origin, at rest)."""
        return jnp.zeros(6)

    def default_control(self) -> Array:
        """Return default control (no thrust, no moment)."""
        return jnp.zeros(2)

    def post_step(self, state: Array) -> Array:
        """Post-process state after integration step.

        Wraps heading angle to [-pi, pi] for numerical stability.

        Args:
            state: State vector after integration step

        Returns:
            State with wrapped heading angle
        """
        # Wrap psi to [-pi, pi]
        psi_wrapped = jnp.arctan2(jnp.sin(state[PSI]), jnp.cos(state[PSI]))
        return state.at[PSI].set(psi_wrapped)

    # Analytical solution methods

    def steady_state_surge(self, thrust: float) -> float:
        """Steady-state surge velocity for straight-line motion (r=0).

        u_ss = thrust / D_u

        Args:
            thrust: Forward thrust force (N)

        Returns:
            Steady-state surge velocity (m/s)
        """
        return thrust / self.drag_surge

    def steady_state_yaw_rate(self, yaw_moment: float) -> float:
        """Steady-state yaw rate.

        r_ss = yaw_moment / D_r

        Args:
            yaw_moment: Yaw moment (N·m)

        Returns:
            Steady-state yaw rate (rad/s)
        """
        return yaw_moment / self.drag_yaw

    def surge_time_constant(self) -> float:
        """Time constant for surge dynamics.

        tau = m / D_u

        Returns:
            Time constant (s)
        """
        return self.mass / self.drag_surge

    def yaw_time_constant(self) -> float:
        """Time constant for yaw dynamics.

        tau_r = Izz / D_r

        Returns:
            Time constant (s)
        """
        return self.izz / self.drag_yaw

    def steady_turn_sway(self, u: float, r: float) -> float:
        """Steady-state sway velocity during a turn.

        v_ss = -m·r·u / D_v

        This is the sideslip induced by turning with forward velocity.

        Args:
            u: Surge velocity (m/s)
            r: Yaw rate (rad/s)

        Returns:
            Steady-state sway velocity (m/s)
        """
        return -self.mass * r * u / self.drag_sway

    def speed_over_ground(self, state: Array) -> float:
        """Compute speed over ground from state.

        Args:
            state: Current state vector

        Returns:
            Speed over ground (m/s)
        """
        psi = state[PSI]
        u = state[U]
        v = state[V]

        cos_psi = jnp.cos(psi)
        sin_psi = jnp.sin(psi)

        x_dot = u * cos_psi - v * sin_psi
        y_dot = u * sin_psi + v * cos_psi

        return jnp.sqrt(x_dot**2 + y_dot**2)

    def course_over_ground(self, state: Array) -> float:
        """Compute course over ground from state.

        Args:
            state: Current state vector

        Returns:
            Course over ground (rad)
        """
        psi = state[PSI]
        u = state[U]
        v = state[V]

        cos_psi = jnp.cos(psi)
        sin_psi = jnp.sin(psi)

        x_dot = u * cos_psi - v * sin_psi
        y_dot = u * sin_psi + v * cos_psi

        return jnp.arctan2(y_dot, x_dot)
