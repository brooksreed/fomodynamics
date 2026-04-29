"""CasADi Cartpole (inverted pendulum) dynamics.

A cart moving on a frictionless track with a pole attached at the pivot.
Based on Barto, Sutton, and Anderson (1983) equations of motion.

This is the CasADi implementation that exactly matches CartpoleJax for
MPC applications and equivalence testing.
"""

from __future__ import annotations

import casadi as cs

from fmd.simulator.casadi.base import CasadiDynamicSystem
from fmd.simulator.params import CartpoleParams


class CartpoleCasadiExact(CasadiDynamicSystem):
    """Cartpole (inverted pendulum on cart) dynamics - CasADi implementation.

    Exactly matches CartpoleJax for equivalence testing.

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

    Note: theta_ddot is computed BEFORE x_ddot (they are coupled).

    Attributes:
        mass_cart: Cart mass (kg)
        mass_pole: Pole mass (kg)
        pole_length: Half-length to pole center of mass (m)
        g: Gravitational acceleration (m/s^2)
    """

    state_names: tuple[str, ...] = ("x", "x_dot", "theta", "theta_dot")
    control_names: tuple[str, ...] = ("F",)

    def __init__(self, params: CartpoleParams):
        """Initialize cartpole model from parameters.

        Args:
            params: CartpoleParams instance with validated model parameters.
        """
        self.mass_cart = params.mass_cart
        self.mass_pole = params.mass_pole
        self.pole_length = params.pole_length
        self.g = params.g

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute state derivative: dx/dt = f(x, u, t).

        Args:
            x: Current state [x, x_dot, theta, theta_dot] as CasADi symbolic
            u: Control input [F] as CasADi symbolic
            t: Current time (unused for this time-invariant system)

        Returns:
            State derivative [x_dot, x_ddot, theta_dot, theta_ddot] as cs.SX
        """
        # Extract state components
        # x[0] is cart position (not used in dynamics)
        x_dot = x[1]
        theta = x[2]
        theta_dot = x[3]

        # Extract control
        F = u[0]

        # Convenience aliases
        m_c = self.mass_cart
        m_p = self.mass_pole
        l = self.pole_length
        g = self.g

        cos_th = cs.cos(theta)
        sin_th = cs.sin(theta)
        total_mass = m_c + m_p

        # Barto, Sutton, Anderson (1983) equations
        # theta_ddot numerator
        temp = (-F - m_p * l * theta_dot**2 * sin_th) / total_mass
        theta_ddot_num = g * sin_th + cos_th * temp

        # theta_ddot denominator
        theta_ddot_den = l * (4.0 / 3.0 - m_p * cos_th**2 / total_mass)

        theta_ddot = theta_ddot_num / theta_ddot_den

        # x_ddot from theta_ddot (must compute theta_ddot first!)
        x_ddot = (F + m_p * l * (theta_dot**2 * sin_th - theta_ddot * cos_th)) / total_mass

        return cs.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)

    def post_step(self, x: cs.SX) -> cs.SX:
        """Apply post-step corrections (wrap theta to [-pi, pi]).

        This matches CartpoleJax.post_step exactly using arctan2.

        Note:
            For MPC, consider using identity post_step and constraints
            instead, as arctan2 introduces non-smoothness.

        Args:
            x: State after integration step

        Returns:
            Corrected state with wrapped theta
        """
        theta = x[2]
        theta_wrapped = cs.arctan2(cs.sin(theta), cs.cos(theta))

        # Reconstruct state with wrapped theta
        return cs.vertcat(x[0], x[1], theta_wrapped, x[3])
