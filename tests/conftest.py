"""Root pytest configuration for BLUR tests.

This module:
1. Configures JAX with GPU auto-detection and memory management
2. Provides fixtures available to all tests
3. Registers custom markers
4. Pre-warms the JIT cache for canonical (dt, duration) combos

The JAX configuration runs BEFORE any test collection, ensuring:
- Float64 is enabled
- GPU memory is limited (if GPU available)
- Tests automatically use GPU when CUDA JAX is installed
"""

import pytest

# Canonical (dt, duration) combos used across the test suite.
# New tests should use one of these unless a specific dt/duration is
# required for physics reasons (see docs/public/dev/testing.md for exceptions).
CANONICAL_SIM_COMBOS = (
    (0.01, 1.0),    # Primary standard
    (0.01, 0.5),    # Short-sim standard
    (0.001, 1.0),   # High-accuracy standard
    (0.01, 2.0),    # Medium-duration
    (0.01, 5.0),    # Long-sim standard
    (0.001, 2.0),   # High-accuracy medium
    (0.001, 5.0),   # High-accuracy long
    (0.005, 2.0),   # Moth standard
    (0.005, 1.0),   # Moth short-sim
    (0.005, 5.0),   # Moth long-sim
)


def pytest_configure(config):
    """Configure JAX and register custom markers.

    This runs before test collection, ensuring JAX is configured
    before any test modules are imported.
    """
    import os
    import tempfile

    # Import and configure JAX with compilation cache
    from fmd.core.jax_config import configure_jax

    cache_dir = os.environ.get(
        "FMD_JAX_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "fmd_jax_cache"),
    )
    device_info = configure_jax(cache_dir=cache_dir)

    # Store device info for use in fixtures
    config._jax_device_info = device_info

    # Log configuration (visible with -v flag)
    backend = device_info["backend"]
    mem_frac = device_info["memory_fraction"]

    if device_info["gpu_available"]:
        print(f"\n[JAX] GPU detected: {device_info['device']}")
        print(f"[JAX] Memory fraction: {mem_frac:.0%}")
        print("[JAX] Note: --forked is incompatible with GPU (os.fork + CUDA conflict)")
    else:
        print(f"\n[JAX] Running on CPU: {device_info['device']}")
        print("[JAX] Tip: Add --forked for memory isolation on CPU")

    # Register markers
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (skipped if GPU not available)",
    )
    config.addinivalue_line(
        "markers",
        "cpu_only: marks tests that should only run on CPU",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "notebooks: marks notebook tests",
    )
    config.addinivalue_line(
        "markers",
        "notebooks_smoke: marks notebook smoke tests",
    )
    config.addinivalue_line(
        "markers",
        "notebooks_validation: marks notebook validation tests",
    )
    config.addinivalue_line(
        "markers",
        "tier1: marks tier 1 tests (fast, core functionality)",
    )
    config.addinivalue_line(
        "markers",
        "tier2: marks tier 2 tests (slower, integration)",
    )
    config.addinivalue_line(
        "markers",
        "tier3: marks tier 3 tests (slowest, full validation)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU-only tests if GPU is not available."""
    device_info = getattr(config, "_jax_device_info", None)
    gpu_available = device_info and device_info.get("gpu_available", False)

    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_cpu_only = pytest.mark.skip(reason="Test requires CPU-only environment")

    for item in items:
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
        if "cpu_only" in item.keywords and gpu_available:
            item.add_marker(skip_cpu_only)


@pytest.fixture(scope="session")
def jax_device_info(request):
    """Fixture providing JAX device information.

    Returns:
        DeviceInfo dict with backend, device, gpu_available, memory_fraction
    """
    return getattr(request.config, "_jax_device_info", None)


@pytest.fixture(scope="session")
def has_gpu(jax_device_info):
    """Fixture indicating if GPU is available."""
    return jax_device_info and jax_device_info.get("gpu_available", False)


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility."""
    import numpy as np
    return np.random.default_rng(42)


def pytest_addoption(parser):
    """Add --save-artifacts option for validation report generation."""
    parser.addoption(
        "--save-artifacts",
        action="store_true",
        default=False,
        help="Save test artifacts (NPZ) for validation report generation",
    )


class _NoopArtifactSaver:
    """Module-level fallback stub used when the private validation tree is unavailable.

    Allows tests/fmd/ to collect cleanly in isolated installs where
    tests/private/validation/ is not present (e.g. the Phase 7 staging dir).
    """

    def __init__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


@pytest.fixture
def artifact_saver(request):
    """Fixture providing an ArtifactSaver instance.

    Activated by --save-artifacts flag or BLUR_SAVE_ARTIFACTS=1 env var.

    Note: ArtifactSaver lives in tests.private.validation (it's part of the
    private validation reporting infrastructure). Importing it lazily here
    keeps the public fmd test tree from depending on the private one at
    collection time.
    """
    import os

    active = request.config.getoption("--save-artifacts") or os.environ.get(
        "BLUR_SAVE_ARTIFACTS", ""
    ) == "1"
    test_file = request.fspath.purebasename  # e.g. "test_moth_estimation"

    if active:
        # User asked for artifact saving — fail loudly if infra is missing.
        from tests.private.validation.artifact_saver import ArtifactSaver
        return ArtifactSaver(active=active, _test_file=test_file)

    try:
        from tests.private.validation.artifact_saver import ArtifactSaver
    except Exception:
        return _NoopArtifactSaver()
    return ArtifactSaver(active=False, _test_file=test_file)


@pytest.fixture(scope="session", autouse=True)
def _prewarm_jit_cache():
    """Pre-warm JAX JIT cache for canonical (dt, duration) combos.

    Runs a trivial SimplePendulum simulation for each non-moth canonical
    combo at session start. This ensures that individual tests hit the
    warm JIT cache instead of paying cold-compile cost.
    """
    try:
        import jax.numpy as jnp
        from fmd.simulator import SimplePendulum, simulate
        from fmd.simulator.params import PENDULUM_1M
    except ImportError:
        return

    pendulum = SimplePendulum(PENDULUM_1M)
    initial_state = jnp.array([0.1, 0.0])

    # Pre-warm the 7 non-moth combos (dt=0.01 and dt=0.001)
    for dt, duration in CANONICAL_SIM_COMBOS:
        if dt in (0.01, 0.001):
            simulate(pendulum, initial_state, dt=dt, duration=duration)
