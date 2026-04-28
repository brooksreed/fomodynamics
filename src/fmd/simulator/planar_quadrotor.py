"""JAX Planar Quadrotor (2D) dynamics.

A quadrotor restricted to the x-z plane with pitch rotation.
Two rotors provide thrust forces T1 (right) and T2 (left).
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import PlanarQuadrotorParams
from fmd.simulator.params.base import STANDARD_GRAVITY


# State vector indices
X = 0  # Horizontal position (m)
Z = 1  # Vertical position (m), positive = up
THETA = 2  # Pitch angle (rad), positive = nose up
X_DOT = 3  # Horizontal velocity (m/s)
Z_DOT = 4  # Vertical velocity (m/s)
THETA_DOT = 5  # Pitch rate (rad/s)

# Control vector indices
T1 = 0  # Right rotor thrust (N)
T2 = 1  # Left rotor thrust (N)


class PlanarQuadrotorJax(JaxDynamicSystem):
    """Planar quadrotor (2D) dynamics - JAX implementation.

    State vector (6 elements):
        [0] x         - Horizontal position (m)
        [1] z         - Vertical position (m), positive = up
        [2] theta     - Pitch angle (rad), positive = nose up
        [3] x_dot     - Horizontal velocity (m/s)
        [4] z_dot     - Vertical velocity (m/s)
        [5] theta_dot - Pitch rate (rad/s)

    Control vector (2 elements):
        [0] T1 - Right rotor thrust (N)
        [1] T2 - Left rotor thrust (N)

    Dynamics:
        Total thrust: T = T1 + T2
        Moment: M = (T1 - T2) * arm_length

        x_ddot = -(T/m) * sin(theta)
        z_ddot = (T/m) * cos(theta) - g
        theta_ddot = M / I

    Attributes:
        mass: Vehicle mass (kg)
        arm_length: Distance from rotor to center of mass (m)
        inertia_pitch: Pitch moment of inertia (kg*m^2)
        g: Gravitational acceleration (m/s^2)

    Example:
        from fmd.simulator import PlanarQuadrotor, simulate, ConstantControl
        from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT
        import jax.numpy as jnp

        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        initial = quad.default_state()
        control = ConstantControl(quad.hover_control())
        result = simulate(quad, initial, dt=0.001, duration=5.0, control=control)

        # For gradient computation, use from_values:
        quad = PlanarQuadrotor.from_values(mass=1.0, arm_length=0.25, inertia_pitch=0.01)
    """

    mass: float
    arm_length: float
    inertia_pitch: float
    g: float

    # Static metadata
    state_names: tuple[str, ...] = eqx.field(
        static=True, default=("x", "z", "theta", "x_dot", "z_dot", "theta_dot")
    )
    control_names: tuple[str, ...] = eqx.field(static=True, default=("T1", "T2"))

    def __init__(self, params: PlanarQuadrotorParams):
        """Initialize planar quadrotor model from parameters.

        Args:
            params: PlanarQuadrotorParams instance with validated model parameters.

        Note:
            For JAX gradient computation, use from_values() instead to
            avoid non-JAX operations in params validation.
        """
        self.mass = params.mass
        self.arm_length = params.arm_length
        self.inertia_pitch = params.inertia_pitch
        self.g = params.g

    @classmethod
    def from_values(
        cls,
        mass: float,
        arm_length: float,
        inertia_pitch: float,
        g: float = STANDARD_GRAVITY,
    ) -> "PlanarQuadrotorJax":
        """Create planar quadrotor directly from values (JAX-traceable).

        Use this constructor when differentiating through quadrotor
        parameters, as it avoids the attrs validation in PlanarQuadrotorParams.

        Args:
            mass: Vehicle mass (kg)
            arm_length: Distance from rotor to center of mass (m)
            inertia_pitch: Pitch moment of inertia (kg*m^2)
            g: Gravitational acceleration (m/s^2)

        Returns:
            PlanarQuadrotorJax instance
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "mass", mass)
        object.__setattr__(obj, "arm_length", arm_length)
        object.__setattr__(obj, "inertia_pitch", inertia_pitch)
        object.__setattr__(obj, "g", g)
        object.__setattr__(obj, "state_names", ("x", "z", "theta", "x_dot", "z_dot", "theta_dot"))
        object.__setattr__(obj, "control_names", ("T1", "T2"))
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
            state: Current state [x, z, theta, x_dot, z_dot, theta_dot]
            control: Control input [T1, T2]
            t: Current time (unused)

        Returns:
            State derivative [x_dot, z_dot, theta_dot, x_ddot, z_ddot, theta_ddot]
        """
        # Extract velocities
        x_dot = state[X_DOT]
        z_dot = state[Z_DOT]
        theta_dot = state[THETA_DOT]
        theta = state[THETA]

        # Extract thrusts (default to hover if control is wrong shape)
        T1_val = jnp.where(control.shape[0] > 0, control[T1], self.hover_thrust_per_rotor())
        T2_val = jnp.where(control.shape[0] > 1, control[T2], self.hover_thrust_per_rotor())

        # Total thrust and moment
        T_total = T1_val + T2_val
        moment = (T1_val - T2_val) * self.arm_length

        # Trigonometric functions
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        # Accelerations
        # Thrust acts along body -Z axis (up in body frame)
        # In world frame: ax = -T*sin(theta), az = T*cos(theta)
        x_ddot = -(T_total / self.mass) * sin_theta
        z_ddot = (T_total / self.mass) * cos_theta - self.g
        theta_ddot = moment / self.inertia_pitch

        return jnp.array([x_dot, z_dot, theta_dot, x_ddot, z_ddot, theta_ddot])

    def default_state(self) -> Array:
        """Return default initial state (hover at origin)."""
        return jnp.zeros(6)

    def default_control(self) -> Array:
        """Return default control (hover thrust)."""
        return self.hover_control()

    def hover_control(self) -> Array:
        """Return control for hover (equal thrust per rotor).

        At hover: T1 = T2 = m*g/2
        """
        T_hover = self.hover_thrust_per_rotor()
        return jnp.array([T_hover, T_hover])

    def hover_thrust_per_rotor(self) -> float:
        """Thrust per rotor for hover (N)."""
        return (self.mass * self.g) / 2.0

    def hover_thrust_total(self) -> float:
        """Total thrust for hover (N)."""
        return self.mass * self.g

    def energy(self, state: Array) -> float:
        """Compute total mechanical energy.

        E = KE + PE
        KE = 0.5 * m * (x_dot^2 + z_dot^2) + 0.5 * I * theta_dot^2
        PE = m * g * z

        Args:
            state: Current state [x, z, theta, x_dot, z_dot, theta_dot]

        Returns:
            Total mechanical energy (J)
        """
        z = state[Z]
        x_dot = state[X_DOT]
        z_dot = state[Z_DOT]
        theta_dot = state[THETA_DOT]

        # Kinetic energy (translational + rotational)
        KE_trans = 0.5 * self.mass * (x_dot**2 + z_dot**2)
        KE_rot = 0.5 * self.inertia_pitch * theta_dot**2
        KE = KE_trans + KE_rot

        # Potential energy (reference at z=0)
        PE = self.mass * self.g * z

        return KE + PE

    def kinetic_energy(self, state: Array) -> float:
        """Compute kinetic energy only.

        Args:
            state: Current state

        Returns:
            Kinetic energy (J)
        """
        x_dot = state[X_DOT]
        z_dot = state[Z_DOT]
        theta_dot = state[THETA_DOT]

        KE_trans = 0.5 * self.mass * (x_dot**2 + z_dot**2)
        KE_rot = 0.5 * self.inertia_pitch * theta_dot**2

        return KE_trans + KE_rot

    def potential_energy(self, state: Array) -> float:
        """Compute potential energy only.

        Args:
            state: Current state

        Returns:
            Potential energy (J)
        """
        z = state[Z]
        return self.mass * self.g * z

    def power_thrust(self, state: Array, control: Array) -> float:
        """Compute power delivered by thrust.

        Power = T * v (dot product of thrust force and velocity)

        This is the rate of energy addition by the actuators.

        Args:
            state: Current state
            control: Current control [T1, T2]

        Returns:
            Power (W)
        """
        theta = state[THETA]
        x_dot = state[X_DOT]
        z_dot = state[Z_DOT]

        T_total = control[T1] + control[T2]
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        # Thrust force in world frame
        Fx = -T_total * sin_theta
        Fz = T_total * cos_theta

        # Power = F dot v
        return Fx * x_dot + Fz * z_dot

    def power_gravity(self, state: Array) -> float:
        """Compute power removed by gravity.

        Power = -m * g * z_dot

        Negative when descending (gravity does positive work).

        Args:
            state: Current state

        Returns:
            Power (W), negative for positive work done by gravity
        """
        z_dot = state[Z_DOT]
        return -self.mass * self.g * z_dot

    def create_state(
        self,
        x: float = 0.0,
        z: float = 0.0,
        theta: float = 0.0,
        x_dot: float = 0.0,
        z_dot: float = 0.0,
        theta_dot: float = 0.0,
    ) -> Array:
        """Create a state vector with specified values.

        Args:
            x: Horizontal position (m)
            z: Vertical position (m)
            theta: Pitch angle (rad)
            x_dot: Horizontal velocity (m/s)
            z_dot: Vertical velocity (m/s)
            theta_dot: Pitch rate (rad/s)

        Returns:
            State vector
        """
        return jnp.array([x, z, theta, x_dot, z_dot, theta_dot])

    def create_control(
        self,
        T1_val: float | None = None,
        T2_val: float | None = None,
    ) -> Array:
        """Create a control vector with specified thrusts.

        If thrusts are not specified, defaults to hover thrust.

        Args:
            T1_val: Right rotor thrust (N)
            T2_val: Left rotor thrust (N)

        Returns:
            Control vector [T1, T2]
        """
        T_hover = self.hover_thrust_per_rotor()
        T1_out = T_hover if T1_val is None else T1_val
        T2_out = T_hover if T2_val is None else T2_val
        return jnp.array([T1_out, T2_out])

    def speed(self, state: Array) -> float:
        """Compute speed (magnitude of velocity).

        Args:
            state: Current state

        Returns:
            Speed (m/s)
        """
        x_dot = state[X_DOT]
        z_dot = state[Z_DOT]
        return jnp.sqrt(x_dot**2 + z_dot**2)

    def flight_path_angle(self, state: Array) -> float:
        """Compute flight path angle (angle of velocity vector from horizontal).

        gamma = atan2(z_dot, x_dot)

        Args:
            state: Current state

        Returns:
            Flight path angle (rad), positive = climbing
        """
        x_dot = state[X_DOT]
        z_dot = state[Z_DOT]
        return jnp.arctan2(z_dot, x_dot)

    def angle_of_attack(self, state: Array) -> float:
        """Compute angle of attack (pitch minus flight path angle).

        alpha = theta - gamma

        This is the angle between the body x-axis and the velocity vector.

        Args:
            state: Current state

        Returns:
            Angle of attack (rad)
        """
        theta = state[THETA]
        gamma = self.flight_path_angle(state)
        return theta - gamma

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Position indices for symplectic integration.

        For planar quadrotor: x (horizontal), z (vertical), theta (pitch angle).
        """
        return (X, Z, THETA)

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Velocity indices for symplectic integration.

        For planar quadrotor: x_dot, z_dot, theta_dot.
        """
        return (X_DOT, Z_DOT, THETA_DOT)
