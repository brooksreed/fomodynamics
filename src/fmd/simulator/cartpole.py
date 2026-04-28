"""JAX Cartpole (inverted pendulum) dynamics.

A cart moving on a frictionless track with a pole attached at the pivot.
Based on Barto, Sutton, and Anderson (1983) equations of motion.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import CartpoleParams
from fmd.simulator.params.base import STANDARD_GRAVITY


# State vector indices
X = 0  # Cart position (m)
X_DOT = 1  # Cart velocity (m/s)
THETA = 2  # Pole angle from vertical (rad), positive = clockwise
THETA_DOT = 3  # Pole angular velocity (rad/s)

# Control vector indices
FORCE = 0  # Horizontal force on cart (N)


class CartpoleJax(JaxDynamicSystem):
    """Cartpole (inverted pendulum on cart) dynamics - JAX implementation.

    State vector (4 elements):
        [0] x         - Cart position (m)
        [1] x_dot     - Cart velocity (m/s)
        [2] theta     - Pole angle from vertical (rad), positive = clockwise
        [3] theta_dot - Pole angular velocity (rad/s)

    Control vector (1 element):
        [0] F - Horizontal force on cart (N), positive = right

    Equations of motion (Barto, Sutton, Anderson 1983):
        theta_ddot = (g*sin(θ) + cos(θ)*(-F - m_p*l*θ̇²*sin(θ))/(m_c + m_p)) /
                     (l*(4/3 - m_p*cos²(θ)/(m_c + m_p)))

        x_ddot = (F + m_p*l*(θ̇²*sin(θ) - θ̈*cos(θ))) / (m_c + m_p)

    Attributes:
        mass_cart: Cart mass (kg)
        mass_pole: Pole mass (kg)
        pole_length: Half-length to pole center of mass (m)
        g: Gravitational acceleration (m/s^2)

    Example:
        from fmd.simulator import Cartpole, simulate, ConstantControl
        from fmd.simulator.params import CARTPOLE_CLASSIC
        import jax.numpy as jnp

        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])  # Small tilt
        control = ConstantControl(jnp.array([0.0]))  # No force
        result = simulate(cartpole, initial, dt=0.01, duration=5.0, control=control)

        # For gradient computation, use from_values:
        cartpole = Cartpole.from_values(mass_cart=1.0, mass_pole=0.1, pole_length=0.5)
    """

    mass_cart: float
    mass_pole: float
    pole_length: float
    g: float

    # Static metadata
    state_names: tuple[str, ...] = eqx.field(
        static=True, default=("x", "x_dot", "theta", "theta_dot")
    )
    control_names: tuple[str, ...] = eqx.field(static=True, default=("F",))

    def __init__(self, params: CartpoleParams):
        """Initialize cartpole model from parameters.

        Args:
            params: CartpoleParams instance with validated model parameters.

        Note:
            For JAX gradient computation, use from_values() instead to
            avoid non-JAX operations in params validation.
        """
        self.mass_cart = params.mass_cart
        self.mass_pole = params.mass_pole
        self.pole_length = params.pole_length
        self.g = params.g

    @classmethod
    def from_values(
        cls,
        mass_cart: float,
        mass_pole: float,
        pole_length: float,
        g: float = STANDARD_GRAVITY,
    ) -> "CartpoleJax":
        """Create cartpole directly from values (JAX-traceable).

        Use this constructor when differentiating through cartpole
        parameters, as it avoids the attrs validation in CartpoleParams.

        Args:
            mass_cart: Cart mass (kg)
            mass_pole: Pole mass (kg)
            pole_length: Half-length to pole center of mass (m)
            g: Gravitational acceleration (m/s^2)

        Returns:
            CartpoleJax instance
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "mass_cart", mass_cart)
        object.__setattr__(obj, "mass_pole", mass_pole)
        object.__setattr__(obj, "pole_length", pole_length)
        object.__setattr__(obj, "g", g)
        object.__setattr__(obj, "state_names", ("x", "x_dot", "theta", "theta_dot"))
        object.__setattr__(obj, "control_names", ("F",))
        return obj

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative.

        Args:
            state: Current state [x, x_dot, theta, theta_dot]
            control: Control input [F]
            t: Current time (unused)

        Returns:
            State derivative [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x_dot = state[X_DOT]
        theta = state[THETA]
        theta_dot = state[THETA_DOT]

        # Handle empty or partial control
        F = jnp.where(control.shape[0] > FORCE, control[FORCE], 0.0)

        # Convenience aliases
        m_c = self.mass_cart
        m_p = self.mass_pole
        l = self.pole_length
        g = self.g

        cos_th = jnp.cos(theta)
        sin_th = jnp.sin(theta)
        total_mass = m_c + m_p

        # Barto, Sutton, Anderson (1983) equations
        # theta_ddot numerator
        temp = (-F - m_p * l * theta_dot**2 * sin_th) / total_mass
        theta_ddot_num = g * sin_th + cos_th * temp

        # theta_ddot denominator
        theta_ddot_den = l * (4.0 / 3.0 - m_p * cos_th**2 / total_mass)

        theta_ddot = theta_ddot_num / theta_ddot_den

        # x_ddot from theta_ddot
        x_ddot = (F + m_p * l * (theta_dot**2 * sin_th - theta_ddot * cos_th)) / total_mass

        return jnp.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def post_step(self, state: Array) -> Array:
        """Apply post-step corrections (wrap theta to [-pi, pi]).

        This prevents numerical issues and confusing angle values
        during long simulations where the pole makes multiple rotations.

        Args:
            state: State after integration step

        Returns:
            Corrected state with wrapped theta
        """
        theta = state[THETA]
        theta_wrapped = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
        return state.at[THETA].set(theta_wrapped)

    def default_state(self) -> Array:
        """Return default initial state (upright equilibrium)."""
        return jnp.zeros(4)

    def default_control(self) -> Array:
        """Return default control (no force)."""
        return jnp.zeros(1)

    def upright_state(self) -> Array:
        """Return state at upright equilibrium (unstable).

        The pole is vertical (theta=0) and everything is at rest.
        """
        return jnp.zeros(4)

    def hanging_state(self) -> Array:
        """Return state at hanging equilibrium (stable).

        The pole is inverted (theta=pi) and everything is at rest.
        """
        return jnp.array([0.0, 0.0, jnp.pi, 0.0])

    def energy(self, state: Array) -> float:
        """Compute total mechanical energy.

        E = KE_cart + KE_pole + PE_pole

        The potential energy reference is at the pivot point (cart height).
        At theta=0 (upright), PE = m_p * g * l.
        At theta=pi (hanging), PE = -m_p * g * l.

        Args:
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            Total mechanical energy (J)
        """
        x_dot = state[X_DOT]
        theta = state[THETA]
        theta_dot = state[THETA_DOT]

        l = self.pole_length
        m_c = self.mass_cart
        m_p = self.mass_pole

        # Cart kinetic energy: (1/2) * m_c * x_dot^2
        KE_cart = 0.5 * m_c * x_dot**2

        # Pole center of mass velocity in world frame
        # x_pole = x + l * sin(theta)
        # y_pole = l * cos(theta)
        # x_pole_dot = x_dot + l * cos(theta) * theta_dot
        # y_pole_dot = -l * sin(theta) * theta_dot
        cos_th = jnp.cos(theta)
        sin_th = jnp.sin(theta)
        x_pole_dot = x_dot + l * cos_th * theta_dot
        y_pole_dot = -l * sin_th * theta_dot
        v_pole_sq = x_pole_dot**2 + y_pole_dot**2

        # Pole translational KE: (1/2) * m_p * v^2
        KE_pole_trans = 0.5 * m_p * v_pole_sq

        # Pole rotational KE about its center of mass: (1/2) * I * theta_dot^2
        # For a rod about its center: I = (1/12) * m * L^2 = (1/12) * m * (2l)^2 = (1/3) * m * l^2
        I_pole = (1.0 / 3.0) * m_p * l**2
        KE_pole_rot = 0.5 * I_pole * theta_dot**2

        # Total kinetic energy
        KE = KE_cart + KE_pole_trans + KE_pole_rot

        # Potential energy (reference at pivot, positive when pole is up)
        PE = m_p * self.g * l * cos_th

        return KE + PE

    def linearized_frequency(self) -> float:
        """Natural frequency of linearized system at upright equilibrium (rad/s).

        For small angles: omega = sqrt(g / l)

        This is the frequency of unstable oscillation about the upright position.
        """
        return jnp.sqrt(self.g / self.pole_length)

    def linearized_period(self) -> float:
        """Period of linearized system at upright equilibrium (s).

        T = 2 * pi * sqrt(l / g)
        """
        return 2 * jnp.pi * jnp.sqrt(self.pole_length / self.g)

    @property
    def total_mass(self) -> float:
        """Total system mass (kg)."""
        return self.mass_cart + self.mass_pole

    def pole_tip_position(self, state: Array) -> tuple[float, float]:
        """Compute Cartesian position of the pole tip.

        Args:
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            Tuple of (x_tip, y_tip) in world frame.
            y_tip is height above the track.
        """
        x = state[X]
        theta = state[THETA]

        # Pole extends 2*l from pivot (full length, not half-length)
        full_length = 2 * self.pole_length
        x_tip = x + full_length * jnp.sin(theta)
        y_tip = full_length * jnp.cos(theta)

        return x_tip, y_tip

    def pole_com_position(self, state: Array) -> tuple[float, float]:
        """Compute Cartesian position of the pole center of mass.

        Args:
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            Tuple of (x_com, y_com) in world frame.
        """
        x = state[X]
        theta = state[THETA]

        # COM is at half-length from pivot
        x_com = x + self.pole_length * jnp.sin(theta)
        y_com = self.pole_length * jnp.cos(theta)

        return x_com, y_com

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Position indices for symplectic integration.

        For cartpole: x (cart position) and theta (pole angle).
        """
        return (X, THETA)

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Velocity indices for symplectic integration.

        For cartpole: x_dot (cart velocity) and theta_dot (pole angular velocity).
        """
        return (X_DOT, THETA_DOT)
