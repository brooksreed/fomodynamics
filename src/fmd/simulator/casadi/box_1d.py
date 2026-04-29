"""CasADi Box1D dynamics models.

These models provide CasADi-compatible versions of the Box1D models for
MPC applications. Each has a JAX counterpart with identical physics.
"""

from __future__ import annotations

import casadi as cs

from fmd.simulator.casadi.base import CasadiDynamicSystem
from fmd.simulator.params.box_1d import Box1DParams, Box1DFrictionParams


class Box1DCasadiExact(CasadiDynamicSystem):
    """1D box with linear drag - exact CasADi implementation.

    Matches Box1DJax exactly. States: [x, x_dot], Control: [F].
    Dynamics: x_ddot = (F - k * x_dot) / m
    """
    state_names: tuple[str, ...] = ("x", "x_dot")
    control_names: tuple[str, ...] = ("F",)

    def __init__(self, params: Box1DParams):
        self.mass = params.mass
        self.drag_coefficient = params.drag_coefficient

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        # State is (x, x_dot); acceleration is independent of x (position).
        vel = x[1]
        F = u[0]
        acc = (F - self.drag_coefficient * vel) / self.mass
        return cs.vertcat(vel, acc)


class Box1DFrictionCasadiExact(CasadiDynamicSystem):
    """1D box with Coulomb friction - exact CasADi implementation.

    Uses cs.sign() - matches Box1DFrictionJax exactly.
    Dynamics: x_ddot = (F - sign(x_dot) * mu * m * g - k * x_dot) / m
    """
    state_names: tuple[str, ...] = ("x", "x_dot")
    control_names: tuple[str, ...] = ("F",)

    def __init__(self, params: Box1DFrictionParams):
        self.mass = params.mass
        self.drag_coefficient = params.drag_coefficient
        self.friction_coefficient = params.friction_coefficient
        self.g = params.g

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        # State is (x, x_dot); acceleration is independent of x (position).
        vel = x[1]
        F = u[0]
        friction_force = cs.sign(vel) * self.friction_coefficient * self.mass * self.g
        acc = (F - friction_force - self.drag_coefficient * vel) / self.mass
        return cs.vertcat(vel, acc)


class Box1DFrictionCasadiSmooth(CasadiDynamicSystem):
    """1D box with Coulomb friction - smooth CasADi implementation.

    Uses cs.tanh(vel / epsilon) instead of cs.sign() for solver-friendly
    approximation. The smoothing is controlled by smoothing_epsilon.

    This version is better for MPC because:
    - IPOPT converges more reliably with smooth dynamics
    - Gradients are well-defined everywhere
    - The approximation is very close to sign() away from zero

    Note: This model intentionally diverges from the JAX implementation
    near v=0. Use Box1DFrictionCasadiExact for equivalence tests.
    """
    state_names: tuple[str, ...] = ("x", "x_dot")
    control_names: tuple[str, ...] = ("F",)

    def __init__(self, params: Box1DFrictionParams, smoothing_epsilon: float = 0.01):
        self.mass = params.mass
        self.drag_coefficient = params.drag_coefficient
        self.friction_coefficient = params.friction_coefficient
        self.g = params.g
        self.smoothing_epsilon = smoothing_epsilon

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        # State is (x, x_dot); acceleration is independent of x (position).
        vel = x[1]
        F = u[0]
        # Smooth approximation: tanh(v/eps) -> sign(v) as eps -> 0
        smooth_sign = cs.tanh(vel / self.smoothing_epsilon)
        friction_force = smooth_sign * self.friction_coefficient * self.mass * self.g
        acc = (F - friction_force - self.drag_coefficient * vel) / self.mass
        return cs.vertcat(vel, acc)
