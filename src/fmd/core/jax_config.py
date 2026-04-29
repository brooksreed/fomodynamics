"""JAX configuration for the fomodynamics library.

This module handles:
1. Float64 precision (configurable, default on for numerical accuracy)
2. GPU/CPU auto-detection
3. Memory management for GPU (especially important for Windows/WSL)
4. Dtype constants for consistent precision across the library
5. Persistent compilation cache (optional, for faster cold starts)

Import this module before any JAX operations to ensure proper configuration.

Environment variable overrides:
- JAX_PLATFORMS: Force specific platform ("cpu", "cuda", "gpu")
- XLA_PYTHON_CLIENT_MEM_FRACTION: GPU memory fraction (0.0-1.0)
- XLA_PYTHON_CLIENT_PREALLOCATE: Set to "false" to disable preallocation
- FMD_USE_FLOAT32: Set to "1" to use float32 instead of float64
- FMD_JAX_CACHE_DIR: Directory for persistent compilation cache
  (reduces cold-start compilation by ~50%). Defaults to
  ~/.cache/<package_root>/jax where <package_root> derives from this
  module's __name__ (e.g. "fmd"). Set to empty string to disable.

Usage:
    from fmd.core.jax_config import configure_jax, get_device_info, FMD_DTYPE

    # Auto-configure (called automatically on import)
    configure_jax()

    # Check what's available
    info = get_device_info()
    print(f"Backend: {info['backend']}, Device: {info['device']}")

    # Use consistent dtype
    import jax.numpy as jnp
    arr = jnp.zeros(10, dtype=FMD_DTYPE)
"""

import os
from pathlib import Path
from typing import TypedDict, Any

import numpy as np

# Constants
DEFAULT_GPU_MEMORY_FRACTION = 0.65  # 65% of GPU memory, leaves headroom for OS/display

# Derive the package-root cache directory from this module's __name__.
# __name__ here is e.g. "fmd.core.jax_config" -> package root "fmd". This
# means the cache path follows the package wherever it lives, so renaming
# the package doesn't strand the existing cache nor collide with another
# package's cache.
_PACKAGE_ROOT = __name__.split(".", 1)[0]
DEFAULT_JAX_CACHE_DIR = str(Path.home() / ".cache" / _PACKAGE_ROOT / "jax")

# Precision configuration: check env var before JAX import
_USE_FLOAT32 = os.environ.get("FMD_USE_FLOAT32", "0") == "1"

# NumPy dtype (available immediately, no JAX import needed)
FMD_NP_DTYPE: Any = np.float32 if _USE_FLOAT32 else np.float64
"""NumPy dtype for fmd arrays. Controlled by FMD_USE_FLOAT32 env var."""


class DeviceInfo(TypedDict):
    """Information about the current JAX device configuration."""

    backend: str  # "cpu", "cuda", or "gpu"
    device: str  # Device name (e.g., "CpuDevice(id=0)")
    gpu_available: bool  # Whether CUDA GPU is being used
    memory_fraction: float | None  # GPU memory fraction if GPU, else None


def _detect_cuda_available() -> bool:
    """Check if CUDA-enabled JAX is installed without importing JAX.

    Returns True if jax-cuda plugin is available.
    """
    import importlib.util

    # Check for CUDA 12 plugin (modern JAX)
    if importlib.util.find_spec("jax_cuda12_plugin") is not None:
        return True

    # Check for CUDA 13 plugin
    if importlib.util.find_spec("jax_cuda13_plugin") is not None:
        return True

    return False


def _configure_memory_for_gpu() -> float:
    """Configure GPU memory settings if not already set by user.

    This sets XLA_PYTHON_CLIENT_MEM_FRACTION to limit GPU memory usage,
    leaving headroom for the operating system and display (important on Windows).

    Returns the memory fraction that was set or found.
    """
    env_var = "XLA_PYTHON_CLIENT_MEM_FRACTION"

    if env_var in os.environ:
        # User has explicitly set memory fraction, respect it
        return float(os.environ[env_var])

    # Set default memory fraction for GPU
    os.environ[env_var] = str(DEFAULT_GPU_MEMORY_FRACTION)
    return DEFAULT_GPU_MEMORY_FRACTION


def configure_jax(
    enable_x64: bool | None = None,
    gpu_memory_fraction: float | None = None,
    cache_dir: str | None = None,
) -> DeviceInfo:
    """Configure JAX for fomodynamics library usage.

    This function:
    1. Configures float precision (float64 by default, float32 if FMD_USE_FLOAT32=1)
    2. Auto-detects GPU availability
    3. Configures GPU memory limits if GPU is available
    4. Enables persistent compilation cache if configured

    Args:
        enable_x64: Enable float64 precision. If None (default), uses FMD_USE_FLOAT32
                    env var (float32 if set to "1", else float64).
        gpu_memory_fraction: Override GPU memory fraction (None = use default 0.65)
        cache_dir: Directory for persistent compilation cache. If None, uses
                   FMD_JAX_CACHE_DIR env var, falling back to the
                   package-derived cache dir (default ~/.cache/fmd/jax).
                   Set to empty string to disable.

    Returns:
        DeviceInfo dict with backend, device, gpu_available, memory_fraction

    Note:
        This function must be called BEFORE importing jax.numpy or creating
        any JAX arrays. It's called automatically when importing fmd.core.jax_config.
    """
    global FMD_DTYPE

    # Detect CUDA and configure memory BEFORE importing JAX
    cuda_available = _detect_cuda_available()
    memory_fraction = None

    if cuda_available:
        if gpu_memory_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_memory_fraction)
            memory_fraction = gpu_memory_fraction
        else:
            memory_fraction = _configure_memory_for_gpu()

    # Now import JAX (this is when it initializes the backend)
    import jax
    import jax.numpy as jnp

    # Determine x64 setting: explicit arg > env var > default (True)
    if enable_x64 is None:
        enable_x64 = not _USE_FLOAT32

    # Enable float64 precision (must be before any array operations)
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    # Configure persistent compilation cache (enabled by default)
    _cache_dir = cache_dir if cache_dir is not None else os.environ.get(
        "FMD_JAX_CACHE_DIR", DEFAULT_JAX_CACHE_DIR
    )
    if _cache_dir:
        Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    # Set JAX dtype based on x64 setting
    # Note: FMD_DTYPE is a module-level global set once at init time.
    # Not thread-safe if multiple threads call init_jax() with different settings.
    FMD_DTYPE = jnp.float64 if enable_x64 else jnp.float32

    # Get actual device info after configuration
    backend = jax.default_backend()
    devices = jax.devices()
    device_str = str(devices[0]) if devices else "unknown"

    # Check if we're actually using GPU (not just that CUDA is installed)
    gpu_in_use = backend in ("cuda", "gpu")

    return DeviceInfo(
        backend=backend,
        device=device_str,
        gpu_available=gpu_in_use,
        memory_fraction=memory_fraction if gpu_in_use else None,
    )


def get_device_info() -> DeviceInfo:
    """Get current JAX device configuration information.

    Returns:
        DeviceInfo dict with backend, device, gpu_available, memory_fraction
    """
    import jax

    backend = jax.default_backend()
    devices = jax.devices()
    device_str = str(devices[0]) if devices else "unknown"
    gpu_available = backend in ("cuda", "gpu")

    mem_frac_str = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    memory_fraction = float(mem_frac_str) if mem_frac_str else None

    return DeviceInfo(
        backend=backend,
        device=device_str,
        gpu_available=gpu_available,
        memory_fraction=memory_fraction if gpu_available else None,
    )


def is_gpu_available() -> bool:
    """Check if JAX is running on GPU.

    Returns:
        True if GPU backend is active, False otherwise
    """
    import jax

    return jax.default_backend() in ("cuda", "gpu")


# JAX dtype placeholder (set by configure_jax)
# This will be jnp.float64 or jnp.float32 after configuration
FMD_DTYPE: Any = None
"""JAX dtype for fmd arrays. Set after configure_jax() is called."""

# Auto-configure on import (enables float64 by default, sets up GPU memory if available)
_device_info = configure_jax()


def get_fmd_dtype() -> Any:
    """Get the current fmd JAX dtype.

    Returns:
        jnp.float64 or jnp.float32 depending on configuration.
    """
    return FMD_DTYPE


def get_fmd_np_dtype() -> Any:
    """Get the current fmd NumPy dtype.

    Returns:
        np.float64 or np.float32 depending on configuration.
    """
    return FMD_NP_DTYPE
