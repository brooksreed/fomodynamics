"""Environment module for fomodynamics simulator.

Bundles environmental conditions (waves, future: wind, current)
into a single Equinox module that can be threaded through the
dynamics chain.

The Environment is an optional argument (env=None) throughout
the simulation pipeline. When None, all models behave as if
in calm conditions (backwards compatible).
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

import equinox as eqx
from typing import Optional, Union

from fmd.simulator.waves import WaveField
from fmd.simulator.params.wave import WaveParams


class Environment(eqx.Module):
    """Environmental conditions for simulation.

    Bundles wave field (and future: wind, current) into a single
    object threaded through the dynamics chain.

    Attributes:
        wave_field: Optional WaveField for ocean wave effects.
    """

    wave_field: Optional[WaveField] = None

    @classmethod
    def calm(cls) -> Environment:
        """Create calm environment (no waves, no wind, no current)."""
        return cls(wave_field=None)

    @classmethod
    def with_waves(cls, waves: Union[WaveParams, WaveField]) -> Environment:
        """Create environment with wave field.

        Args:
            waves: WaveParams (will be converted to WaveField) or WaveField.

        Returns:
            Environment with wave field set.
        """
        if isinstance(waves, WaveParams):
            wave_field = WaveField.from_params(waves)
        else:
            wave_field = waves
        return cls(wave_field=wave_field)
