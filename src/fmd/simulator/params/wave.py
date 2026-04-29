"""Wave field parameter class.

Immutable, validated parameters for ocean wave modeling.
Supports regular (Airy), JONSWAP, and Pierson-Moskowitz spectra
with optional directional spreading.
"""

from __future__ import annotations

import attrs
import numpy as np


from fmd.simulator.params.base import (
    STANDARD_GRAVITY,
    WATER_DENSITY_SALT,
    is_finite,
    non_negative,
    positive,
)


def _validate_gamma(instance, attribute, value):
    """Validate JONSWAP peakedness parameter gamma > 1."""
    if value <= 1.0:
        raise ValueError(f"{attribute.name} must be > 1.0, got {value}")


def _validate_spectrum_type(instance, attribute, value):
    """Validate spectrum type string."""
    valid = {"jonswap", "pierson_moskowitz", "regular"}
    if value not in valid:
        raise ValueError(f"{attribute.name} must be one of {valid}, got {value!r}")


def _validate_num_components(instance, attribute, value):
    """Validate num_components in [1, 200]."""
    if not (1 <= value <= 200):
        raise ValueError(f"{attribute.name} must be in [1, 200], got {value}")


def _validate_stokes_order(instance, attribute, value):
    """Validate stokes_order is 1 or 2."""
    if value not in (1, 2):
        raise ValueError(f"{attribute.name} must be 1 or 2, got {value}")


@attrs.define(frozen=True, slots=True, eq=False)
class WaveParams:
    """Immutable parameters for ocean wave field generation.

    Supports three spectrum types:
    - "regular": Single-component Airy wave (use .regular() factory)
    - "jonswap": JONSWAP spectrum (wind-sea, fetch-limited)
    - "pierson_moskowitz": Pierson-Moskowitz spectrum (fully-developed sea)

    Directional spreading uses cos^2s model. Set spreading_exponent=0
    and num_directions=1 for long-crested (unidirectional) waves.

    Attributes:
        significant_wave_height: Hs (m), 4*sqrt(m0) of the spectrum.
        peak_period: Tp (s), period of peak spectral energy.
        spectrum_type: One of "jonswap", "pierson_moskowitz", "regular".
        gamma: JONSWAP peakedness parameter (>1, default 3.3).
        mean_direction: Mean wave propagation direction (rad, NED convention).
        spreading_exponent: cos^2s spreading parameter (0 = long-crested).
        num_directions: Number of directional bins (1 = long-crested).
        num_components: Number of frequency components (1-200).
        water_depth: Water depth (m), use float('inf') for deep water.
        seed: Random seed for deterministic phase generation.
        g: Gravitational acceleration (m/s^2).
        water_density: Water density (kg/m^3).
    """

    significant_wave_height: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Significant wave height Hs"},
    )
    peak_period: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "s", "description": "Peak spectral period Tp"},
    )
    spectrum_type: str = attrs.field(
        default="jonswap",
        validator=[_validate_spectrum_type],
        metadata={"description": "Spectrum type: jonswap, pierson_moskowitz, regular"},
    )
    gamma: float = attrs.field(
        default=3.3,
        validator=[is_finite, _validate_gamma],
        metadata={"description": "JONSWAP peakedness parameter"},
    )
    mean_direction: float = attrs.field(
        default=0.0,
        validator=[is_finite],
        metadata={"unit": "rad", "description": "Mean wave direction (NED)"},
    )
    spreading_exponent: float = attrs.field(
        default=0.0,
        validator=[is_finite, non_negative],
        metadata={"description": "cos^2s directional spreading exponent"},
    )
    num_directions: int = attrs.field(
        default=1,
        validator=[],
        metadata={"description": "Number of directional bins"},
    )
    num_components: int = attrs.field(
        default=30,
        validator=[_validate_num_components],
        metadata={"description": "Number of frequency components"},
    )
    water_depth: float = attrs.field(
        default=float("inf"),
        validator=[positive],
        metadata={"unit": "m", "description": "Water depth"},
    )
    seed: int = attrs.field(
        default=42,
        metadata={"description": "Random seed for phase generation"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )
    water_density: float = attrs.field(
        default=WATER_DENSITY_SALT,
        validator=[is_finite, positive],
        metadata={"unit": "kg/m^3", "description": "Water density"},
    )
    stokes_order: int = attrs.field(
        default=1,
        validator=[_validate_stokes_order],
        metadata={"description": "Wave theory order: 1 (Airy) or 2 (Stokes 2nd-order)"},
    )

    @classmethod
    def regular(
        cls,
        amplitude: float,
        period: float,
        direction: float = 0.0,
        water_depth: float = float("inf"),
        g: float = STANDARD_GRAVITY,
        water_density: float = WATER_DENSITY_SALT,
        stokes_order: int = 1,
    ) -> WaveParams:
        """Create parameters for a single regular (Airy) wave.

        Args:
            amplitude: Wave amplitude (m), half of wave height.
            period: Wave period (s).
            direction: Wave propagation direction (rad, NED).
            water_depth: Water depth (m), inf for deep water.
            g: Gravitational acceleration (m/s^2).
            water_density: Water density (kg/m^3).
            stokes_order: Wave theory order (1=Airy, 2=Stokes 2nd-order).

        Returns:
            WaveParams configured for a single regular wave component.
        """
        return cls(
            significant_wave_height=2.0 * amplitude,
            peak_period=period,
            spectrum_type="regular",
            gamma=3.3,
            mean_direction=direction,
            spreading_exponent=0.0,
            num_directions=1,
            num_components=1,
            water_depth=water_depth,
            seed=0,
            g=g,
            water_density=water_density,
            stokes_order=stokes_order,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(
            getattr(self, f.name) == getattr(other, f.name)
            for f in attrs.fields(type(self))
        )

    def __hash__(self) -> int:
        return hash(tuple(
            getattr(self, f.name) for f in attrs.fields(type(self))
        ))
