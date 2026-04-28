"""Planar Quadrotor (2D) parameter class.

Immutable, validated parameters for the 2D Planar Quadrotor dynamics model.
Based on Safe Control Gym (UTIAS) and Crazyflie 2.1 specifications.
"""

from __future__ import annotations

import attrs
import numpy as np

from fmd.simulator.params.base import is_finite, positive, STANDARD_GRAVITY


@attrs.define(frozen=True, slots=True)
class PlanarQuadrotorParams:
    """Immutable parameters for 2D Planar Quadrotor dynamics model.

    All values must be finite (no NaN/Inf) and positive.

    The Planar Quadrotor operates in the x-z plane with pitch rotation.
    It has two rotors providing individual thrust forces T1 (right) and T2 (left).

    State vector: [x, z, theta, x_dot, z_dot, theta_dot]
    - x: horizontal position (m)
    - z: vertical position (m), positive = up
    - theta: pitch angle (rad), positive = nose up
    - x_dot, z_dot: velocities (m/s)
    - theta_dot: pitch rate (rad/s)

    Control: [T1, T2] - right/left rotor thrusts (N)

    Dynamics:
        Total thrust: T = T1 + T2
        Moment: M = (T1 - T2) * arm_length

        x_ddot = -(T/m) * sin(theta)
        z_ddot = (T/m) * cos(theta) - g
        theta_ddot = M / I

    Attributes:
        mass: Total vehicle mass (kg). Must be positive.
        arm_length: Distance from rotor to center of mass (m). Must be positive.
        inertia_pitch: Moment of inertia about pitch axis (kg*m^2). Must be positive.
        g: Gravitational acceleration (m/s^2). Must be positive.

    References:
        Safe Control Gym (UTIAS): https://github.com/utiasDSL/safe-control-gym
        Crazyflie 2.1: https://www.bitcraze.io/products/crazyflie-2-1/

    Example:
        >>> params = PlanarQuadrotorParams(
        ...     mass=0.030,
        ...     arm_length=0.0397,
        ...     inertia_pitch=1.4e-5,
        ... )
        >>> from fmd.simulator import PlanarQuadrotor
        >>> quad = PlanarQuadrotor(params)
    """

    mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Total vehicle mass"},
    )
    arm_length: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Rotor arm length (rotor to CG)"},
    )
    inertia_pitch: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg*m^2", "description": "Pitch moment of inertia"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )

    @property
    def hover_thrust_total(self) -> float:
        """Total thrust required for hover (N).

        At hover: T = m * g
        """
        return self.mass * self.g

    @property
    def hover_thrust_per_rotor(self) -> float:
        """Thrust per rotor for hover (N).

        At hover: T1 = T2 = (m * g) / 2
        """
        return self.hover_thrust_total / 2.0

    @property
    def thrust_to_weight_at_hover(self) -> float:
        """Thrust-to-weight ratio at hover (dimensionless).

        Always 1.0 at hover by definition.
        """
        return 1.0

    @property
    def moment_arm(self) -> float:
        """Effective moment arm for differential thrust (m).

        The moment produced by thrust difference T1-T2 is:
        M = (T1 - T2) * arm_length

        This is the same as arm_length for a symmetric quadrotor.
        """
        return self.arm_length

    def moment_from_thrust_difference(self, delta_T: float) -> float:
        """Compute pitch moment from thrust difference.

        Args:
            delta_T: Thrust difference T1 - T2 (N)

        Returns:
            Pitch moment (N*m)
        """
        return delta_T * self.arm_length

    def thrust_difference_for_moment(self, moment: float) -> float:
        """Compute thrust difference needed for a given moment.

        Args:
            moment: Desired pitch moment (N*m)

        Returns:
            Required thrust difference T1 - T2 (N)
        """
        return moment / self.arm_length

    def angular_acceleration_for_moment(self, moment: float) -> float:
        """Compute angular acceleration from applied moment.

        Args:
            moment: Applied pitch moment (N*m)

        Returns:
            Angular acceleration (rad/s^2)
        """
        return moment / self.inertia_pitch

    def with_mass(self, mass: float) -> PlanarQuadrotorParams:
        """Return new params with updated mass.

        Args:
            mass: New mass (kg).

        Returns:
            New PlanarQuadrotorParams instance.
        """
        return attrs.evolve(self, mass=mass)

    def with_arm_length(self, arm_length: float) -> PlanarQuadrotorParams:
        """Return new params with updated arm length.

        Args:
            arm_length: New arm length (m).

        Returns:
            New PlanarQuadrotorParams instance.
        """
        return attrs.evolve(self, arm_length=arm_length)

    def with_inertia(self, inertia_pitch: float) -> PlanarQuadrotorParams:
        """Return new params with updated pitch inertia.

        Args:
            inertia_pitch: New pitch inertia (kg*m^2).

        Returns:
            New PlanarQuadrotorParams instance.
        """
        return attrs.evolve(self, inertia_pitch=inertia_pitch)

    def scaled(self, scale: float) -> PlanarQuadrotorParams:
        """Return params scaled geometrically.

        Useful for comparing 1:10 scale models to full size.

        Scaling rules (assuming constant density):
        - Length scales linearly: L' = L * s
        - Mass scales cubically: m' = m * s^3
        - Inertia scales to 5th power: I' = I * s^5

        Args:
            scale: Geometric scale factor (e.g., 10 for 10x larger)

        Returns:
            New PlanarQuadrotorParams with scaled parameters.
        """
        return attrs.evolve(
            self,
            mass=self.mass * scale**3,
            arm_length=self.arm_length * scale,
            inertia_pitch=self.inertia_pitch * scale**5,
        )
