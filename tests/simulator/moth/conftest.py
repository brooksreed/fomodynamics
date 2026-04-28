"""Shared fixtures for moth test subdirectory.

Provides moth-specific JIT pre-warm fixture for canonical combos.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _prewarm_moth_jit_cache():
    """Pre-warm JAX JIT cache for moth-specific canonical combos.

    Runs a trivial Moth3D simulation for each moth canonical combo
    (dt=0.005) at session start. Moth3D has a different pytree structure
    than SimplePendulum, so it needs its own pre-warm pass.
    """
    try:
        from fmd.simulator import Moth3D, simulate
        from fmd.simulator.params import MOTH_BIEKER_V3
    except ImportError:
        return

    moth = Moth3D(MOTH_BIEKER_V3)
    state0 = moth.default_state()

    # Pre-warm the 3 moth combos: dt=0.005 x {1.0, 2.0, 5.0}
    for duration in (1.0, 2.0, 5.0):
        simulate(moth, state0, dt=0.005, duration=duration)
