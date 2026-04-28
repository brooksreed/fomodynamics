"""Moth wand linkage geometry module.

Standalone geometry module that models the mechanical wand system used for
passive ride height control on Moth sailboats. A wand arm pivots at the
bowsprit, touches the water surface, and its angle mechanically drives the
main foil flap through a kinematic linkage.

Currently models rigid body geometry only. Future integration as a
measurement module in the Moth3D simulator will add dynamic effects
(e.g., wand inertia, water interaction).

Kinematic chain:
    wand angle -> [wand lever] -> pullrod displacement -> [gearing] ->
    aft pullrod -> [bellcrank] -> pushrod displacement -> [flap lever] ->
    flap angle

Conventions:
    - Wand angle (theta_w): 0 = vertical (straight down), pi/2 = horizontal.
      Increases as boat gets lower.
    - Fastpoint: Peak gain angle (default 30 deg from vertical).
    - Flap angle: Positive = trailing edge down = more lift.
      Negative = trailing edge up = less lift.
    - Physical mapping: boat HIGH -> wand vertical (small theta_w) -> flap UP
      (negative). Boat LOW -> wand horizontal (large theta_w) -> flap DOWN
      (positive).
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np

from fmd.simulator.components.moth_forces import compute_foil_ned_depth

if TYPE_CHECKING:
    from fmd.simulator.waves import WaveField

# Default linkage dimensions (SI units: meters, radians).
# Based on Bieker Moth V3 measurements.
DEFAULT_WAND_LEVER = 0.020          # L_w: wand lever arm length (m)
DEFAULT_FASTPOINT = np.radians(45.0)  # fastpoint angle (rad)
DEFAULT_BELLCRANK_INPUT = 0.030     # L_p: bellcrank input arm (m)
DEFAULT_BELLCRANK_OUTPUT = 0.030    # L_v: bellcrank output arm (m)
DEFAULT_BELLCRANK_ANGLE = np.pi / 2  # alpha: angle between bellcrank arms (rad)
DEFAULT_FLAP_LEVER = 0.030          # L_f: flap lever arm length (m)
DEFAULT_PULLROD_OFFSET = 0.0        # ride height adjuster offset (m)
DEFAULT_WAND_LENGTH = 1.175         # physical wand length, pivot to float (m)
                                    # midpoint of 0.95-1.40m adjustable range

# Gearing rod geometry. The rod is 170mm long; the hull pullrod (input)
# attaches to the full length. The bellcrank pullrod (output) taps off at
# a point along the rod, so the output is a fraction of the input.
DEFAULT_GEARING_ROD_LENGTH = 0.170  # total rod length (m)
DEFAULT_GEARING_ROD_TAP = 0.130    # output tap position along rod (m)
DEFAULT_GEARING_RATIO = DEFAULT_GEARING_ROD_TAP / DEFAULT_GEARING_ROD_LENGTH


def gearing_ratio_from_rod(rod_length: float, tap_position: float) -> float:
    """Compute gearing ratio from physical rod dimensions.

    The gearing rod reduces wand lever displacement before the bellcrank.
    The input (hull pullrod) connects to the full rod length; the output
    (bellcrank pullrod) taps off at ``tap_position`` along the rod.

    Args:
        rod_length: Total rod length (m).
        tap_position: Output tap position along the rod (m).

    Returns:
        Gearing ratio (dimensionless, < 1 for typical Moth setups).
    """
    return tap_position / rod_length


class WandLinkageState(NamedTuple):
    """All intermediate values through the wand linkage kinematic chain.

    Attributes:
        wand_angle: Input wand angle (rad).
        pullrod_dx: Pullrod displacement from wand lever (m).
        aft_pullrod_dx: Aft pullrod displacement after gearing (m).
        bellcrank_phi: Bellcrank rotation angle (rad).
        pushrod_dy: Pushrod displacement from bellcrank (m).
        flap_angle: Output flap angle (rad).
    """
    wand_angle: Array
    pullrod_dx: Array
    aft_pullrod_dx: Array
    bellcrank_phi: Array
    pushrod_dy: Array
    flap_angle: Array


_SAFE_TRIG_EPS = 1e-7
"""Epsilon for safe arccos/arcsin boundary clamping."""


def _safe_arccos(x: Array) -> Array:
    """Compute arccos with well-defined gradients at domain boundaries.

    Standard ``jnp.arccos(jnp.clip(x, 0, 1))`` produces NaN gradients
    at the clip boundaries because clip has zero gradient while arccos
    has infinite gradient, giving ``0 * inf = NaN``.

    This function clamps the input slightly inside [0, 1] before
    applying arccos, then uses ``jnp.where`` to return the boundary
    values (0 or pi/2) when the input is outside [0, 1]. The
    ``jnp.where`` ensures the NaN-producing arccos branch is never
    selected while maintaining valid gradients everywhere.

    Args:
        x: Input value (h / wand_length ratio).

    Returns:
        arccos(x) clamped to [0, pi/2] with finite gradients.
    """
    eps = _SAFE_TRIG_EPS
    x_safe = jnp.clip(x, eps, 1.0 - eps)
    result = jnp.arccos(x_safe)

    # At boundaries, return the exact boundary angle
    result = jnp.where(x >= 1.0 - eps, 0.0, result)        # wand vertical
    result = jnp.where(x <= eps, jnp.pi / 2, result)       # wand horizontal
    return result


def _safe_arcsin(x: Array) -> Array:
    """Compute arcsin with well-defined gradients at domain boundaries.

    Same pattern as :func:`_safe_arccos` but for arcsin on [-1, 1].
    Prevents NaN gradients from ``jnp.arcsin(jnp.clip(x, -1, 1))``
    at the clip boundaries.

    Args:
        x: Input value, nominally in [-1, 1].

    Returns:
        arcsin(x) clamped to [-pi/2, pi/2] with finite gradients.
    """
    eps = _SAFE_TRIG_EPS
    x_safe = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    result = jnp.arcsin(x_safe)

    result = jnp.where(x >= 1.0 - eps, jnp.pi / 2, result)
    result = jnp.where(x <= -1.0 + eps, -jnp.pi / 2, result)
    return result


def wand_angle_from_state(
    pos_d: Array,
    theta: Array,
    wand_pivot_position: Array,
    wand_length: float = DEFAULT_WAND_LENGTH,
    heel_angle: float = 0.0,
) -> Array:
    """Compute wand angle from Moth3D state variables.

    Uses ``compute_foil_ned_depth`` to find the NED depth of the wand pivot,
    then converts to wand angle via arccos geometry.

    Uses :func:`_safe_arccos` to avoid NaN gradients at the clip
    boundaries (ratio = 0 or 1), which can occur when the EKF estimated
    state drifts to an extreme geometry. See ``_safe_arccos`` docstring.

    Args:
        pos_d: CG vertical position in NED (m), positive = deeper.
        theta: Pitch angle (rad).
        wand_pivot_position: Wand pivot [x, y, z] relative to CG (m),
            body frame (FRD).
        wand_length: Physical wand length from pivot to float (m).
        heel_angle: Static heel angle (rad).

    Returns:
        Wand angle (rad). 0 = vertical, pi/2 = horizontal.
    """
    pivot_x = wand_pivot_position[0]
    pivot_z = wand_pivot_position[2]

    # Height of pivot above water = -NED_depth (positive when above surface)
    pivot_depth = compute_foil_ned_depth(pos_d, pivot_x, pivot_z, theta, heel_angle)
    h = -pivot_depth  # height above water (positive = above)

    ratio = h / wand_length
    return _safe_arccos(ratio)


def wand_angle_from_state_waves(
    pos_d: Array,
    theta: Array,
    fwd_speed: Array,
    t: float,
    wave_field: Optional[WaveField],
    wand_pivot_position: Array,
    wand_length: float,
    heel_angle: float = 0.0,
    n_iterations: int = 5,
) -> Array:
    """Compute wand angle with wave-aware fixed-point iteration.

    When ``wave_field is None``, delegates directly to
    :func:`wand_angle_from_state` (no iteration, guaranteed calm-water
    equivalence).

    When a wave field is provided, the wand tip contacts the wave surface
    at a position that depends on the wand angle itself. This creates a
    coupled geometry problem solved via fixed-point iteration:

    .. code-block:: text

        Constraint: D_pivot + L*cos(theta_w) = -eta(N_tip, t)

        Per iteration:
          1. N_tip = N_pivot + L*sin(theta_w)
          2. eta = wave_field.elevation(N_tip, 0, t)
          3. cos(theta_w) = (-eta - D_pivot) / L
          4. theta_w = _safe_arccos(cos_val)

    The calm-water solution is used as the initial guess.

    Args:
        pos_d: CG vertical position in NED (m), positive = deeper.
        theta: Pitch angle (rad).
        fwd_speed: Forward speed (m/s), used for NED position estimate.
        t: Simulation time (s).
        wave_field: Optional WaveField for wave surface queries.
        wand_pivot_position: Wand pivot [x, y, z] relative to CG (m),
            body frame (FRD).
        wand_length: Physical wand length from pivot to float (m).
        heel_angle: Static heel angle (rad).
        n_iterations: Number of fixed-point iterations (default 5,
            validated by convergence study to achieve <1e-6 rad for all
            tested wave conditions including short steep waves).

    Returns:
        Wand angle (rad). 0 = vertical, pi/2 = horizontal.
    """
    if wave_field is None:
        return wand_angle_from_state(
            pos_d, theta, wand_pivot_position, wand_length, heel_angle
        )

    pivot_x = wand_pivot_position[0]
    pivot_z = wand_pivot_position[2]

    # Pivot NED depth (positive = submerged)
    pivot_depth = compute_foil_ned_depth(
        pos_d, pivot_x, pivot_z, theta, heel_angle
    )

    # Pivot NED-north position (same pattern as moth_3d wave queries)
    u_safe = jnp.maximum(fwd_speed, 0.1)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    n_pivot = u_safe * t + pivot_x * cos_theta + pivot_z * sin_theta

    # Start from calm-water solution
    theta_w_init = wand_angle_from_state(
        pos_d, theta, wand_pivot_position, wand_length, heel_angle
    )

    def iteration_body(_, theta_w):
        # Tip NED-north position
        n_tip = n_pivot + wand_length * jnp.sin(theta_w)
        # Wave elevation at tip
        eta = wave_field.elevation(n_tip, 0.0, t)
        # Recompute angle from constraint: D_pivot + L*cos(theta_w) = -eta
        cos_val = (-eta - pivot_depth) / wand_length
        # Use _safe_arccos for gradient-safe boundary handling
        return _safe_arccos(cos_val)

    theta_w = jax.lax.fori_loop(0, n_iterations, iteration_body, theta_w_init)
    return theta_w


class WandLinkage(eqx.Module):
    """Wand-to-flap kinematic linkage as an Equinox module.

    Models the mechanical linkage that converts wand angle to flap angle
    through a series of lever arms and a bellcrank.

    Attributes:
        wand_lever: L_w, wand lever arm length (m).
        fastpoint: Fastpoint angle (rad), peak gain angle.
        bellcrank_input: L_p, bellcrank input arm (m).
        bellcrank_output: L_v, bellcrank output arm (m).
        bellcrank_angle: Alpha, angle between bellcrank arms (rad).
            At pi/2 (default), the bellcrank is a pure linear scaling
            by L_v/L_p. At other angles, the bellcrank introduces
            nonlinearities and the small-displacement gain scales
            as sin(alpha).
        flap_lever: L_f, flap lever arm length (m).
        gearing_ratio: Ratio between forward and aft pullrod displacement.
        pullrod_offset: Ride height adjuster offset (m).
    """

    wand_lever: float
    fastpoint: float
    bellcrank_input: float
    bellcrank_output: float
    bellcrank_angle: float
    flap_lever: float
    gearing_ratio: float
    pullrod_offset: float

    def compute(self, wand_angle: Array) -> Array:
        """Compute flap angle from wand angle.

        Args:
            wand_angle: Wand angle (rad). 0 = vertical, pi/2 = horizontal.

        Returns:
            Flap angle (rad). Positive = trailing edge down (more lift).
        """
        return self.compute_detailed(wand_angle).flap_angle

    def compute_detailed(self, wand_angle: Array) -> WandLinkageState:
        """Compute flap angle with all intermediate values.

        Args:
            wand_angle: Wand angle (rad). 0 = vertical, pi/2 = horizontal.

        Returns:
            WandLinkageState with all intermediate values through the chain.
        """
        # Stage 1: Wand angle -> pullrod displacement
        pullrod_dx = (
            self.wand_lever * jnp.sin(wand_angle - self.fastpoint)
            + self.pullrod_offset
        )

        # Stage 2: Gearing
        aft_pullrod_dx = pullrod_dx * self.gearing_ratio

        # Stage 3: Bellcrank rotation
        bellcrank_phi = _safe_arcsin(aft_pullrod_dx / self.bellcrank_input)

        # Stage 4: Pushrod displacement
        # General bellcrank geometry: output arm tip y-displacement as
        # the bellcrank rotates by phi. At alpha=pi/2 this reduces to
        # L_v * sin(phi). At other angles it introduces nonlinearities.
        pushrod_dy = self.bellcrank_output * (
            jnp.cos(self.bellcrank_angle - bellcrank_phi)
            - jnp.cos(self.bellcrank_angle)
        )

        # Stage 5: Flap angle
        flap_angle = _safe_arcsin(pushrod_dy / self.flap_lever)

        return WandLinkageState(
            wand_angle=wand_angle,
            pullrod_dx=pullrod_dx,
            aft_pullrod_dx=aft_pullrod_dx,
            bellcrank_phi=bellcrank_phi,
            pushrod_dy=pushrod_dy,
            flap_angle=flap_angle,
        )

    def gain(self, wand_angle: Array) -> Array:
        """Compute differential gain d(flap)/d(wand) at a wand angle.

        Uses JAX automatic differentiation.

        Args:
            wand_angle: Wand angle (rad).

        Returns:
            Differential gain (rad/rad).
        """
        return jax.grad(lambda wa: self.compute(wa))(wand_angle)


def create_wand_linkage(**overrides) -> WandLinkage:
    """Create a WandLinkage with default Moth parameters.

    Any parameter can be overridden by keyword argument.

    Args:
        **overrides: Keyword arguments matching WandLinkage fields.

    Returns:
        WandLinkage instance.
    """
    defaults = dict(
        wand_lever=DEFAULT_WAND_LEVER,
        fastpoint=float(DEFAULT_FASTPOINT),
        bellcrank_input=DEFAULT_BELLCRANK_INPUT,
        bellcrank_output=DEFAULT_BELLCRANK_OUTPUT,
        bellcrank_angle=DEFAULT_BELLCRANK_ANGLE,
        flap_lever=DEFAULT_FLAP_LEVER,
        gearing_ratio=DEFAULT_GEARING_RATIO,
        pullrod_offset=DEFAULT_PULLROD_OFFSET,
    )
    defaults.update(overrides)
    return WandLinkage(**defaults)
