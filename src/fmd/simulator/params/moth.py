"""Moth sailboat parameters.

Defines immutable parameter class for Moth simulation with full 6DOF
geometry for future expansion.

Coordinate Systems:
    Hull datum: x positive aft from bow, y positive starboard, z positive up
                from hull bottom. This is the geometric source of truth for
                structural positions.
    Body FRD:   x positive forward, y positive starboard, z positive down.
                Origin at boat CG (non-sailor hull/rig/foil CG).
    System CG:  Combined boat + sailor CG, offset from body origin by
                combined_cg_offset.

Structural positions are stored in hull-datum coordinates and exposed as
body-frame @properties via hull_datum_to_body().
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import attrs

from fmd.simulator.params.base import (
    STANDARD_GRAVITY,
    WATER_DENSITY_SALT,
    is_finite,
    is_finite_array,
    is_valid_inertia,
    is_3vector,
    positive,
    non_negative,
    to_float_array,
)


@attrs.define(frozen=True, slots=True, eq=False)
class MothParams:
    """Immutable parameters for Moth sailboat dynamics.

    Includes full 6DOF geometry for future expansion, even though v1
    only uses pitch + heave dynamics (longitudinal model).

    Frame Convention:
        - Body frame (FRD): +x forward, +y starboard, +z down
        - Hull datum: +x aft from bow, +y starboard, +z up from hull bottom
        - All stored positions are in hull-datum coordinates
        - Body-frame positions are computed @properties via hull_datum_to_body()
        - NED world frame at theta=0

    Parameter Groups:
        - Hull: Mass properties, dimensions, and geometry reference
        - Sailor: Mass and position (affects combined CG)
        - Main foil: T-foil geometry and lift/drag coefficients
        - Rudder: T-foil with elevator for pitch control
        - Wand: Height sensor mechanical system
        - Sail: Aerodynamic force application point
        - Environment: Gravity, water density

    Note:
        All values are approximate and marked as WIP. Actual measurements
        from the specific boat should be used for accurate simulation.

    Example:
        >>> from fmd.simulator.params import MothParams, MOTH_BIEKER_V3
        >>> params = MOTH_BIEKER_V3
        >>> params.total_mass
        125.0
    """

    # =========================================================================
    # Hull Parameters
    # =========================================================================

    hull_mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Non-sailor boat mass (hull + rig + foils + rigging)"},
    )
    hull_inertia: NDArray = attrs.field(
        converter=to_float_array,
        validator=[is_finite_array, is_valid_inertia],
        metadata={
            "unit": "kg*m^2",
            "description": "Hull moments of inertia [Ixx, Iyy, Izz] or 3x3 tensor",
        },
    )
    hull_length: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Hull length overall (LOA)"},
    )
    hull_beam: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Hull maximum beam"},
    )

    # =========================================================================
    # Sailor Parameters
    # =========================================================================

    sailor_mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Sailor mass"},
    )
    sailor_position: NDArray = attrs.field(
        converter=to_float_array,
        validator=[is_finite_array, is_3vector],
        metadata={
            "unit": "m",
            "description": "Sailor CG position [x, y, z] relative to hull CG (body FRD)",
        },
    )

    # =========================================================================
    # Main Foil Parameters
    # =========================================================================

    main_foil_span: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Main T-foil wingspan"},
    )
    main_foil_chord: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Main foil mean chord"},
    )
    main_foil_area: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m^2", "description": "Main foil planform area"},
    )
    main_foil_cl_alpha: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "1/rad", "description": "Lift curve slope dCL/dalpha"},
    )
    main_foil_cl0: float = attrs.field(
        default=0.0,
        kw_only=True,
        validator=[is_finite],
        metadata={"unit": "-", "description": "Zero-AoA lift coefficient"},
    )
    main_foil_cd0: float = attrs.field(
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={
            "unit": "-",
            "description": (
                "Total zero-lift drag coefficient. If cd0_section and "
                "cd0_parasitic are both provided, this should equal their sum."
            ),
        },
    )
    main_foil_cd0_section: float = attrs.field(
        default=0.0,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "-", "description": "Section/base drag coefficient (airfoil profile)"},
    )
    main_foil_cd0_parasitic: float = attrs.field(
        default=0.0,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={
            "unit": "-",
            "description": "Parasitic drag coefficient (junction/strut/roughness proxy)",
        },
    )
    main_foil_oswald: float = attrs.field(
        default=0.85,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "-", "description": "Oswald efficiency factor for induced drag"},
    )
    main_foil_flap_effectiveness: float = attrs.field(
        default=0.5,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "-", "description": "Flap lift effectiveness (dCL/d_flap / CL_alpha)"},
    )
    main_foil_cd_flap: float = attrs.field(
        default=0.0,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={
            "unit": "-",
            "description": (
                "Flap deflection drag coefficient. Adds Cd_flap * delta_flap^2 "
                "to the main foil drag coefficient. Based on McCormick plain flap "
                "formula: dCd = 1.7 * (cf/c)^1.38 * (Sf/S) * sin^2(delta). "
                "Default 0.0 preserves backward compatibility."
            ),
        },
    )
    main_strut_chord: float = attrs.field(
        default=0.09,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Main foil strut chord (streamwise)"},
    )
    main_strut_thickness: float = attrs.field(
        default=0.013,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Main foil strut thickness facing the flow"},
    )
    main_strut_cd_pressure: float = attrs.field(
        default=0.01,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "-", "description": "Main strut pressure drag Cd based on frontal area"},
    )
    main_strut_cf_skin: float = attrs.field(
        default=0.003,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "-", "description": "Main strut skin friction Cf based on wetted area"},
    )

    # =========================================================================
    # Rudder Parameters
    # =========================================================================

    rudder_span: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Rudder T-foil wingspan"},
    )
    rudder_chord: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Rudder mean chord"},
    )
    rudder_area: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m^2", "description": "Rudder planform area"},
    )
    rudder_elevator_min: float = attrs.field(
        kw_only=True,
        validator=[is_finite],
        metadata={"unit": "rad", "description": "Minimum (most negative) elevator deflection"},
    )
    rudder_elevator_max: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "rad", "description": "Maximum elevator deflection magnitude"},
    )
    rudder_cl_alpha: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "1/rad", "description": "Rudder lift curve slope"},
    )
    rudder_cd0: float = attrs.field(
        default=0.0,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "-", "description": "Rudder zero-lift drag coefficient"},
    )
    rudder_oswald: float = attrs.field(
        default=0.85,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "-", "description": "Rudder Oswald efficiency factor for induced drag"},
    )
    rudder_strut_chord: float = attrs.field(
        default=0.07,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Rudder strut chord (streamwise)"},
    )
    rudder_strut_thickness: float = attrs.field(
        default=0.010,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Rudder strut thickness facing the flow"},
    )
    rudder_strut_cd_pressure: float = attrs.field(
        default=0.01,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "-", "description": "Rudder strut pressure drag Cd based on frontal area"},
    )
    rudder_strut_cf_skin: float = attrs.field(
        default=0.003,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "-", "description": "Rudder strut skin friction Cf based on wetted area"},
    )

    # =========================================================================
    # Wand Parameters
    # =========================================================================

    wand_length: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Wand length from pivot to float"},
    )
    wand_gearing_ratio: float = attrs.field(
        default=1.0,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "-", "description": "Wand angle to flap angle ratio (placeholder)"},
    )

    # =========================================================================
    # Sail Parameters
    # =========================================================================

    sail_area: float = attrs.field(
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m^2", "description": "Sail area"},
    )

    # =========================================================================
    # Environmental Parameters (with defaults)
    # =========================================================================

    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )
    rho_water: float = attrs.field(
        default=WATER_DENSITY_SALT,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "kg/m^3", "description": "Water density"},
    )

    # =========================================================================
    # V1 Simplified Force Model Parameters
    # =========================================================================

    sail_thrust_coeff: float = attrs.field(
        default=50.0,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "N", "description": "Constant sail forward thrust (v1 simplified)"},
    )
    sail_thrust_slope: float = attrs.field(
        default=0.0,
        kw_only=True,
        validator=[is_finite],
        metadata={
            "unit": "N/(m/s)",
            "description": (
                "Speed-dependent sail thrust slope. "
                "F_sail(u) = sail_thrust_coeff + sail_thrust_slope * u. "
                "Default 0.0 preserves constant-thrust v1 model."
            ),
        },
    )
    sail_thrust_speeds: tuple[float, ...] = attrs.field(
        default=(),
        kw_only=True,
        metadata={
            "unit": "m/s",
            "description": (
                "Calibration speed points for thrust lookup table. "
                "When non-empty, MothSailForce uses jnp.interp to "
                "interpolate thrust as a function of forward speed. "
                "Must be monotonically increasing and same length as "
                "sail_thrust_values."
            ),
        },
    )
    sail_thrust_values: tuple[float, ...] = attrs.field(
        default=(),
        kw_only=True,
        metadata={
            "unit": "N",
            "description": (
                "Thrust values at each calibration speed. "
                "Must be same length as sail_thrust_speeds."
            ),
        },
    )
    hull_drag_coeff: float = attrs.field(
        default=500.0,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "N/m", "description": "Hull contact drag penalty per meter of immersion"},
    )
    hull_buoyancy_coeff: float = attrs.field(
        default=5000.0,
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={"unit": "N/m", "description": "Hull buoyancy restoring force per meter of immersion"},
    )

    # =========================================================================
    # Added Mass Coefficients (Hydrodynamic)
    # =========================================================================

    added_mass_heave: float = attrs.field(
        default=10.0,  # kg - 70% of ~14 kg research value
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={
            "unit": "kg",
            "description": (
                "Added mass for heave motion from water acceleration. "
                "Research value ~14 kg (thin airfoil theory); using 70% for "
                "3D effects and free surface proximity. See damping_mechanisms_research.md."
            ),
        },
    )

    added_inertia_pitch: float = attrs.field(
        default=8.75,  # kg*m^2 - 70% of ~12.5 kg*m^2 research value
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={
            "unit": "kg*m^2",
            "description": (
                "Added pitch inertia from water acceleration at foils. "
                "Research value ~12.5 kg*m^2; using 70% for 3D effects. "
                "See damping_mechanisms_research.md."
            ),
        },
    )

    added_mass_surge: float = attrs.field(
        default=5.0,  # kg - estimated from foil/strut frontal area
        kw_only=True,
        validator=[is_finite, non_negative],
        metadata={
            "unit": "kg",
            "description": (
                "Added mass for surge motion from water acceleration. "
                "Smaller than heave added mass due to streamlined frontal area."
            ),
        },
    )

    # =========================================================================
    # Hull Geometry (hull-datum reference frame)
    # =========================================================================

    hull_depth: float = attrs.field(
        default=0.45,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Overall hull height, bottom to deck"},
    )
    hull_cg_above_bottom: float = attrs.field(
        default=1.0,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={
            "unit": "m",
            "description": (
                "Boat CG (body frame origin) height above hull bottom. "
                "Includes contribution from mast, foils, and wing racks."
            ),
        },
    )
    hull_cg_from_bow: float = attrs.field(
        default=1.9,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Boat CG distance aft of bow"},
    )
    main_foil_strut_depth: float = attrs.field(
        default=1.0,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Main foil strut depth below hull bottom"},
    )
    rudder_strut_depth: float = attrs.field(
        default=0.95,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Rudder strut depth below hull bottom"},
    )
    wing_rack_span: float = attrs.field(
        default=2.25,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Wing rack tip-to-tip span"},
    )
    wing_dihedral: float = attrs.field(
        default=0.5236,  # 30 degrees
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "rad", "description": "Wing rack dihedral angle"},
    )

    # =========================================================================
    # Hull-Datum Structural Positions
    # =========================================================================

    main_foil_from_bow: float = attrs.field(
        default=1.6,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Main foil strut position, aft from bow"},
    )
    wing_rack_from_bow: float = attrs.field(
        default=2.0,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Wing rack attachment position, aft from bow"},
    )
    rudder_from_bow: float = attrs.field(
        default=3.855,
        kw_only=True,
        validator=[is_finite, positive],
        metadata={
            "unit": "m",
            "description": "Rudder strut position, aft from bow (hull + gantry)",
        },
    )
    sail_ce_hull_datum: NDArray = attrs.field(
        factory=lambda: np.array([1.9, 0.0, 2.95]),
        kw_only=True,
        converter=to_float_array,
        validator=[is_finite_array, is_3vector],
        metadata={
            "unit": "m",
            "description": (
                "Sail CE in hull-datum coords [x_aft_from_bow, y, z_above_bottom]. "
                "z ~= deck height + distance up the mast to CE."
            ),
        },
    )
    bowsprit_hull_datum: NDArray = attrs.field(
        factory=lambda: np.array([0.0, 0.0, 0.45]),
        kw_only=True,
        converter=to_float_array,
        validator=[is_finite_array, is_3vector],
        metadata={
            "unit": "m",
            "description": "Bowsprit tip in hull-datum [x_aft_from_bow, y, z_above_bottom]",
        },
    )
    wand_pivot_hull_datum: NDArray = attrs.field(
        factory=lambda: np.array([0.0, 0.0, 0.35]),
        kw_only=True,
        converter=to_float_array,
        validator=[is_finite_array, is_3vector],
        metadata={
            "unit": "m",
            "description": "Wand pivot in hull-datum [x_aft_from_bow, y, z_above_bottom]",
        },
    )

    def __attrs_post_init__(self):
        """Validate cross-field constraints."""
        if len(self.sail_thrust_speeds) != len(self.sail_thrust_values):
            raise ValueError(
                f"sail_thrust_speeds ({len(self.sail_thrust_speeds)}) and "
                f"sail_thrust_values ({len(self.sail_thrust_values)}) must have same length"
            )
        if len(self.sail_thrust_speeds) > 0:
            if not all(np.isfinite(s) for s in self.sail_thrust_speeds):
                raise ValueError(
                    "sail_thrust_speeds must contain only finite values, "
                    f"got {self.sail_thrust_speeds}"
                )
            if not all(np.isfinite(v) for v in self.sail_thrust_values):
                raise ValueError(
                    "sail_thrust_values must contain only finite values, "
                    f"got {self.sail_thrust_values}"
                )
            if not all(s >= 0.0 for s in self.sail_thrust_speeds):
                raise ValueError(
                    "sail_thrust_speeds must be non-negative, "
                    f"got {self.sail_thrust_speeds}"
                )
        if len(self.sail_thrust_speeds) > 1:
            for i in range(1, len(self.sail_thrust_speeds)):
                if self.sail_thrust_speeds[i] <= self.sail_thrust_speeds[i - 1]:
                    raise ValueError(
                        "sail_thrust_speeds must be monotonically increasing, "
                        f"got {self.sail_thrust_speeds}"
                    )

    # =========================================================================
    # Frame Conversion Helpers
    # =========================================================================

    def hull_datum_to_body(self, pos_datum: NDArray) -> NDArray:
        """Convert hull-datum position to body FRD position.

        Hull datum: x aft from bow, y starboard, z up from hull bottom.
        Body FRD: x forward from CG, y starboard, z down from CG.
        """
        return np.array([
            self.hull_cg_from_bow - pos_datum[0],   # aft-from-bow -> forward-from-CG
            pos_datum[1],                             # y unchanged
            self.hull_cg_above_bottom - pos_datum[2], # up-from-bottom -> down-from-CG
        ])

    def body_to_hull_datum(self, pos_body: NDArray) -> NDArray:
        """Convert body FRD position to hull-datum position.

        Body FRD: x forward from CG, y starboard, z down from CG.
        Hull datum: x aft from bow, y starboard, z up from hull bottom.
        """
        return np.array([
            self.hull_cg_from_bow - pos_body[0],
            pos_body[1],
            self.hull_cg_above_bottom - pos_body[2],
        ])

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def total_mass(self) -> float:
        """Total system mass (hull + sailor) in kg."""
        return self.hull_mass + self.sailor_mass

    @property
    def hull_inertia_matrix(self) -> NDArray:
        """Full 3x3 hull inertia matrix."""
        if self.hull_inertia.shape == (3,):
            return np.diag(self.hull_inertia)
        return self.hull_inertia

    @property
    def main_foil_aspect_ratio(self) -> float:
        """Main foil aspect ratio (span^2 / area)."""
        return self.main_foil_span ** 2 / self.main_foil_area

    @property
    def rudder_aspect_ratio(self) -> float:
        """Rudder aspect ratio (span^2 / area)."""
        return self.rudder_span ** 2 / self.rudder_area

    @property
    def combined_cg_offset(self) -> NDArray:
        """Combined CG offset from hull CG due to sailor mass.

        Returns position relative to hull CG where the combined
        system CG is located.
        """
        return (self.sailor_mass * self.sailor_position) / self.total_mass

    @property
    def composite_pitch_inertia(self) -> float:
        """Composite pitch inertia about system CG (reduced-mass parallel axis theorem)."""
        x_s = self.sailor_position[0]
        z_s = self.sailor_position[2]
        reduced_mass = self.hull_mass * self.sailor_mass / self.total_mass
        return float(self.hull_inertia_matrix[1, 1] + reduced_mass * (x_s**2 + z_s**2))

    # =========================================================================
    # Derived Body-Frame Positions (from hull-datum)
    # =========================================================================

    @property
    def hull_contact_depth(self) -> float:
        """System-CG to hull-bottom distance in body frame (m).

        This is a computed property: hull_cg_above_bottom - combined_cg_offset[2].
        Since combined_cg_offset[2] is negative when the sailor is above boat CG,
        this adds to hull_cg_above_bottom.

        Note: This gives the static default-sailor value. Runtime code with
        sailor_position_schedule(t) should derive from the current CG offset.
        """
        return self.hull_cg_above_bottom - self.combined_cg_offset[2]

    @property
    def main_foil_position(self) -> NDArray:
        """Main foil AC position in body FRD, derived from hull-datum geometry.

        Hull-datum z is negative (below hull bottom), so strut_depth is negated.
        """
        return self.hull_datum_to_body(
            np.array([self.main_foil_from_bow, 0.0, -self.main_foil_strut_depth])
        )

    @property
    def rudder_position(self) -> NDArray:
        """Rudder AC position in body FRD, derived from hull-datum geometry.

        Hull-datum z is negative (below hull bottom), so strut_depth is negated.
        """
        return self.hull_datum_to_body(
            np.array([self.rudder_from_bow, 0.0, -self.rudder_strut_depth])
        )

    @property
    def sail_ce_position(self) -> NDArray:
        """Sail CE position in body FRD, derived from hull-datum."""
        return self.hull_datum_to_body(self.sail_ce_hull_datum)

    @property
    def bowsprit_position(self) -> NDArray:
        """Bowsprit tip position in body FRD, derived from hull-datum."""
        return self.hull_datum_to_body(self.bowsprit_hull_datum)

    @property
    def wand_pivot_position(self) -> NDArray:
        """Wand pivot position in body FRD, derived from hull-datum."""
        return self.hull_datum_to_body(self.wand_pivot_hull_datum)

    # =========================================================================
    # Helper Methods (evolve pattern)
    # =========================================================================

    def with_sailor_mass(self, mass: float) -> MothParams:
        """Return new params with updated sailor mass."""
        return attrs.evolve(self, sailor_mass=mass)

    def with_sailor_position(self, position: NDArray | list) -> MothParams:
        """Return new params with updated sailor position."""
        return attrs.evolve(self, sailor_position=np.asarray(position))

    def with_wand_gearing(self, ratio: float) -> MothParams:
        """Return new params with updated wand gearing ratio."""
        return attrs.evolve(self, wand_gearing_ratio=ratio)

    # =========================================================================
    # Custom Equality and Hash (for array fields)
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        """Compare equality with proper numpy array handling."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name))
            if isinstance(getattr(self, f.name), np.ndarray)
            else getattr(self, f.name) == getattr(other, f.name)
            for f in attrs.fields(type(self))
        )

    def __hash__(self) -> int:
        """Hash based on all stored fields."""
        parts = []
        for f in attrs.fields(type(self)):
            v = getattr(self, f.name)
            parts.append(v.tobytes() if isinstance(v, np.ndarray) else v)
        return hash(tuple(parts))
