"""Moth 3DOF measurement models for state estimation.

Provides sensor-realistic measurement models for the Moth3D longitudinal
dynamics model. Each variant corresponds to a different sensor suite:

- **full_state**: Observe all 5 states directly (testing baseline)
- **speed_pitch_height**: forward speed + pitch + ride height (3 measurements)
- **speed_pitch_rate_height**: forward speed + pitch + pitch rate + ride height (4 measurements)
- **speed_pitch_rate_height_accel**: Above + vertical acceleration (5 measurements)

State convention: [pos_d, theta, w, q, u] with indices POS_D=0, THETA=1,
W=2, Q=3, U=4.

Example:
    from fmd.estimation import create_moth_measurement
    from fmd.simulator.params import MOTH_BIEKER_V3
    import jax.numpy as jnp

    model = create_moth_measurement(
        "speed_pitch_height",
        bowsprit_position=MOTH_BIEKER_V3.bowsprit_position,
        R=jnp.diag(jnp.array([0.1, 0.01, 0.05])) ** 2,
    )
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from fmd.estimation.measurement import (
    LinearMeasurementModel,
    MeasurementModel,
)

# Lazy imports to avoid circular dependency (estimation -> simulator -> estimation)
# These are resolved at first use rather than import time.
_wand_module = None


def _get_wand_module():
    global _wand_module
    if _wand_module is None:
        from fmd.simulator.components import moth_wand
        _wand_module = moth_wand
    return _wand_module


def _wand_angle_from_state(pos_d, theta, wand_pivot_position, wand_length, heel_angle):
    return _get_wand_module().wand_angle_from_state(
        pos_d, theta, wand_pivot_position, wand_length, heel_angle
    )


def _default_wand_length():
    return _get_wand_module().DEFAULT_WAND_LENGTH

# Moth3D state indices (mirror moth_3d.py)
POS_D = 0
THETA = 1
W = 2
Q = 3
U = 4

# Default gravity
_DEFAULT_G = 9.80665


def bow_ride_height(
    pos_d: Array,
    theta: Array,
    bowsprit_position: Array,
    heel_angle: float = 0.0,
) -> Array:
    """Compute bow ride height above water from state and bowsprit geometry.

    h = -pos_d + bowsprit_x * sin(theta) - bowsprit_z * cos(heel) * cos(theta)

    This is the negated z-row of R = Ry(theta) @ Rx(heel) applied to the
    body-frame bowsprit point, matching the canonical ``compute_foil_ned_depth``
    formula in ``moth_forces.py``.

    Positive return value means above water. Nonlinear in theta due to
    sin/cos terms.

    Args:
        pos_d: Vertical position (positive down, NED convention)
        theta: Pitch angle (positive nose-up)
        bowsprit_position: [x, y, z] position of bowsprit tip relative to hull CG
        heel_angle: Static heel angle (rad), default 0.0

    Returns:
        Scalar ride height (positive = above water)
    """
    bx = bowsprit_position[0]
    bz = bowsprit_position[2]
    return -pos_d + bx * jnp.sin(theta) - bz * jnp.cos(heel_angle) * jnp.cos(theta)


class SpeedPitchHeightMeasurement(MeasurementModel):
    """Speed + pitch + ride height measurement model.

    3 outputs: forward speed, pitch angle, and bowsprit ride height.
    Nonlinear due to ride height depending on sin/cos of pitch.

    Note: forward_speed is body-frame surge velocity x[U], not GPS SOG.
    The difference is u*(1-cos(theta)), <0.5% at typical foiling pitch.

    Attributes:
        output_names: ("forward_speed", "pitch", "ride_height")
        bowsprit_position: Bowsprit tip [x, y, z] relative to hull CG
        R: 3x3 measurement noise covariance
        heel_angle: Static heel angle (rad), default 0.0
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    bowsprit_position: Array
    R: Array
    heel_angle: float = eqx.field(static=True, default=0.0)

    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute measurement: [forward_speed, pitch, ride_height]."""
        forward_speed = x[U]
        pitch = x[THETA]
        rh = bow_ride_height(x[POS_D], x[THETA], self.bowsprit_position, self.heel_angle)
        return jnp.array([forward_speed, pitch, rh])

    @property
    def state_index_map(self) -> dict[str, int | None]:
        return {"forward_speed": U, "pitch": THETA, "ride_height": None}


class SpeedPitchRateHeightMeasurement(MeasurementModel):
    """Speed + pitch + pitch rate + ride height measurement model.

    4 outputs from an IMU suite with ride-height sensor.
    Nonlinear due to ride height.

    Attributes:
        output_names: ("forward_speed", "pitch", "pitch_rate", "ride_height")
        bowsprit_position: Bowsprit tip [x, y, z] relative to hull CG
        R: 4x4 measurement noise covariance
        heel_angle: Static heel angle (rad), default 0.0
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    bowsprit_position: Array
    R: Array
    heel_angle: float = eqx.field(static=True, default=0.0)

    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute measurement: [forward_speed, pitch, pitch_rate, ride_height]."""
        forward_speed = x[U]
        pitch = x[THETA]
        pitch_rate = x[Q]
        rh = bow_ride_height(x[POS_D], x[THETA], self.bowsprit_position, self.heel_angle)
        return jnp.array([forward_speed, pitch, pitch_rate, rh])

    @property
    def state_index_map(self) -> dict[str, int | None]:
        return {"forward_speed": U, "pitch": THETA, "pitch_rate": Q, "ride_height": None}


class SpeedPitchRateHeightAccelMeasurement(MeasurementModel):
    """Speed + pitch + pitch rate + ride height + vertical acceleration.

    5 outputs: base suite plus derived vertical acceleration.
    The vertical acceleration is a **near-trim pseudo-measurement** computed as:
        a_z = -g * cos(theta) + q * u

    This combines gravitational projection and centripetal acceleration.
    This is an approximation that omits:
      - w_dot (heave acceleration): zero at trim, small during transients
      - All aerodynamic/hydrodynamic force contributions to body-frame
        accelerations
    The approximation is valid near trim where these omitted terms are
    small compared to the gravitational and centripetal terms. During
    large transients (e.g. aggressive maneuvers, ventilation events),
    the omitted terms become significant and this pseudo-measurement
    introduces bias into the EKF.

    Attributes:
        output_names: ("forward_speed", "pitch", "pitch_rate", "ride_height", "vertical_accel")
        bowsprit_position: Bowsprit tip [x, y, z] relative to hull CG
        R: 5x5 measurement noise covariance
        g: Gravitational acceleration (m/s^2)
        heel_angle: Static heel angle (rad), default 0.0
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    bowsprit_position: Array
    R: Array
    g: float = eqx.field(static=True, default=_DEFAULT_G)
    heel_angle: float = eqx.field(static=True, default=0.0)

    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute measurement: [forward_speed, pitch, pitch_rate, ride_height, vertical_accel]."""
        forward_speed = x[U]
        pitch = x[THETA]
        pitch_rate = x[Q]
        rh = bow_ride_height(x[POS_D], x[THETA], self.bowsprit_position, self.heel_angle)
        vertical_accel = -self.g * jnp.cos(x[THETA]) + x[Q] * x[U]
        return jnp.array([forward_speed, pitch, pitch_rate, rh, vertical_accel])

    @property
    def state_index_map(self) -> dict[str, int | None]:
        return {
            "forward_speed": U, "pitch": THETA, "pitch_rate": Q,
            "ride_height": None, "vertical_accel": None,
        }


class WandAngleMeasurement(MeasurementModel):
    """Wand angle measurement model (calm-water, for EKF internal use).

    1 output: wand angle computed from pos_d and theta via geometry.
    Nonlinear due to arccos and trig functions.

    Attributes:
        output_names: ("wand_angle",)
        wand_pivot_position: Wand pivot [x, y, z] relative to hull CG
        wand_length: Physical wand length from pivot to float (m)
        R: 1x1 measurement noise covariance
        heel_angle: Static heel angle (rad), default 0.0
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    wand_pivot_position: Array
    wand_length: float = eqx.field(static=True)
    R: Array
    heel_angle: float = eqx.field(static=True, default=0.0)

    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute measurement: [wand_angle]."""
        wa = _wand_angle_from_state(
            x[POS_D], x[THETA], self.wand_pivot_position,
            self.wand_length, self.heel_angle,
        )
        return jnp.array([wa])

    @property
    def state_index_map(self) -> dict[str, int | None]:
        return {"wand_angle": None}


class SpeedPitchWandMeasurement(MeasurementModel):
    """Speed + pitch + wand angle measurement model (calm-water, for EKF internal use).

    3 outputs: forward speed, pitch angle, and wand angle.
    Nonlinear due to wand angle depending on arccos and trig of pitch.

    Attributes:
        output_names: ("forward_speed", "pitch", "wand_angle")
        wand_pivot_position: Wand pivot [x, y, z] relative to hull CG
        wand_length: Physical wand length from pivot to float (m)
        R: 3x3 measurement noise covariance
        heel_angle: Static heel angle (rad), default 0.0
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    wand_pivot_position: Array
    wand_length: float = eqx.field(static=True)
    R: Array
    heel_angle: float = eqx.field(static=True, default=0.0)

    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute measurement: [forward_speed, pitch, wand_angle]."""
        forward_speed = x[U]
        pitch = x[THETA]
        wa = _wand_angle_from_state(
            x[POS_D], x[THETA], self.wand_pivot_position,
            self.wand_length, self.heel_angle,
        )
        return jnp.array([forward_speed, pitch, wa])

    @property
    def state_index_map(self) -> dict[str, int | None]:
        return {"forward_speed": U, "pitch": THETA, "wand_angle": None}


def create_moth_measurement(
    variant: str,
    bowsprit_position: Array | None = None,
    R: Array | None = None,
    g: float = _DEFAULT_G,
    num_states: int = 5,
    heel_angle: float = 0.0,
    wand_pivot_position: Array | None = None,
    wand_length: float | None = None,
) -> MeasurementModel:
    """Factory function for Moth measurement model variants.

    Args:
        variant: One of "full_state", "speed_pitch_height",
            "speed_pitch_rate_height", "speed_pitch_rate_height_accel",
            "wand_only", "speed_pitch_wand"
        bowsprit_position: Bowsprit tip [x,y,z] relative to hull CG.
            Required for all variants except "full_state" and wand variants.
        R: Measurement noise covariance matrix. Shape depends on variant:
            - full_state: (5, 5)
            - speed_pitch_height: (3, 3)
            - speed_pitch_rate_height: (4, 4)
            - speed_pitch_rate_height_accel: (5, 5)
            - wand_only: (1, 1)
            - speed_pitch_wand: (3, 3)
            If None, uses identity scaled by 0.01. **This default is a
            placeholder** that applies the same sigma=0.1 to all channels.
            In practice, speed (m/s), pitch (rad), and ride height (m) have
            very different noise characteristics and R should be tuned
            per-channel based on sensor specifications.
        g: Gravitational acceleration (only used by speed_pitch_rate_height_accel)
        num_states: Number of states (default 5 for Moth3D)
        heel_angle: Static heel angle in radians (default 0.0)
        wand_pivot_position: Wand pivot [x,y,z] relative to hull CG.
            Required for wand variants ("wand_only", "speed_pitch_wand").
        wand_length: Physical wand length from pivot to float (m).
            Default is DEFAULT_WAND_LENGTH (1.175m) from moth_wand module.

    Returns:
        MeasurementModel instance for the requested variant

    Raises:
        ValueError: If variant is unknown or required args are missing
    """
    if wand_length is None:
        wand_length = _default_wand_length()

    _EXPECTED_SIZES = {
        "full_state": num_states,
        "speed_pitch_height": 3,
        "speed_pitch_rate_height": 4,
        "speed_pitch_rate_height_accel": 5,
        "wand_only": 1,
        "speed_pitch_wand": 3,
    }

    if variant not in _EXPECTED_SIZES:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from: {list(_EXPECTED_SIZES.keys())}"
        )

    expected_size = _EXPECTED_SIZES[variant]

    if R is None:
        R = jnp.eye(expected_size) * 0.01

    if R.shape != (expected_size, expected_size):
        raise ValueError(
            f"R shape {R.shape} must be ({expected_size}, {expected_size}) "
            f"for variant '{variant}'"
        )

    if variant == "full_state":
        # Use LinearMeasurementModel with H=I for full state observation
        state_names = ("pos_d", "theta", "w", "q", "u")
        return LinearMeasurementModel(
            output_names=state_names,
            H=jnp.eye(num_states),
            R=R,
        )

    # Wand variants require wand_pivot_position
    if variant in ("wand_only", "speed_pitch_wand"):
        if wand_pivot_position is None:
            raise ValueError(
                f"wand_pivot_position is required for variant '{variant}'"
            )
        wp = jnp.asarray(wand_pivot_position)
        if variant == "wand_only":
            return WandAngleMeasurement(
                output_names=("wand_angle",),
                wand_pivot_position=wp,
                wand_length=wand_length,
                R=R,
                heel_angle=heel_angle,
            )
        else:  # speed_pitch_wand
            return SpeedPitchWandMeasurement(
                output_names=("forward_speed", "pitch", "wand_angle"),
                wand_pivot_position=wp,
                wand_length=wand_length,
                R=R,
                heel_angle=heel_angle,
            )

    # All other variants require bowsprit_position
    if bowsprit_position is None:
        raise ValueError(
            f"bowsprit_position is required for variant '{variant}'"
        )
    bp = jnp.asarray(bowsprit_position)

    if variant == "speed_pitch_height":
        return SpeedPitchHeightMeasurement(
            output_names=("forward_speed", "pitch", "ride_height"),
            bowsprit_position=bp,
            R=R,
            heel_angle=heel_angle,
        )
    elif variant == "speed_pitch_rate_height":
        return SpeedPitchRateHeightMeasurement(
            output_names=("forward_speed", "pitch", "pitch_rate", "ride_height"),
            bowsprit_position=bp,
            R=R,
            heel_angle=heel_angle,
        )
    elif variant == "speed_pitch_rate_height_accel":
        return SpeedPitchRateHeightAccelMeasurement(
            output_names=("forward_speed", "pitch", "pitch_rate", "ride_height", "vertical_accel"),
            bowsprit_position=bp,
            R=R,
            g=g,
            heel_angle=heel_angle,
        )

    # Should never reach here due to check above
    raise ValueError(f"Unknown variant '{variant}'")
