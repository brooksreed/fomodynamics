"""JAX configuration for fmd.control.

This module ensures JAX is configured correctly for fmd.control.
Configuration is now centralized in fmd.core.jax_config.

Usage:
    import fmd.control._config  # Side-effect import enables float64 and GPU

    # Use consistent dtype
    from fmd.control._config import FMD_DTYPE
    arr = jnp.zeros(10, dtype=FMD_DTYPE)
"""

# Import from central config (this enables float64 and configures GPU)
from fmd.core.jax_config import (
    configure_jax,
    get_device_info,
    is_gpu_available,
    DeviceInfo,
    DEFAULT_GPU_MEMORY_FRACTION,
    FMD_DTYPE,
    FMD_NP_DTYPE,
    get_fmd_dtype,
    get_fmd_np_dtype,
)

# Re-export for any code that imports from here
__all__ = [
    "configure_jax",
    "get_device_info",
    "is_gpu_available",
    "DeviceInfo",
    "DEFAULT_GPU_MEMORY_FRACTION",
    "FMD_DTYPE",
    "FMD_NP_DTYPE",
    "get_fmd_dtype",
    "get_fmd_np_dtype",
]
