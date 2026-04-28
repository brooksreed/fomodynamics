"""Base parameter infrastructure for BLUR simulator.

This module provides:
- Environmental constants (gravity, density)
- Validators for attrs parameter classes
- Utility functions for parameter validation
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fmd.core.jax_config import FMD_NP_DTYPE


# ============================================================================
# Environmental Constants
# ============================================================================

STANDARD_GRAVITY: float = 9.80665
"""Standard gravitational acceleration (m/s^2)."""

WATER_DENSITY_FRESH: float = 1000.0
"""Density of fresh water at 4C (kg/m^3)."""

WATER_DENSITY_SALT: float = 1025.0
"""Approximate density of seawater (kg/m^3)."""

AIR_DENSITY_SL: float = 1.225
"""Air density at sea level, standard conditions (kg/m^3)."""


# ============================================================================
# Scalar Validators
# ============================================================================


def is_finite(instance: Any, attribute: Any, value: float) -> None:
    """Validate that a scalar value is finite (not NaN or Inf).

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The value to validate.

    Raises:
        ValueError: If value is NaN or Inf.
    """
    if not np.isfinite(value):
        raise ValueError(
            f"{attribute.name} must be finite, got {value}"
        )


def positive(instance: Any, attribute: Any, value: float) -> None:
    """Validate that a scalar value is strictly positive.

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The value to validate.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(
            f"{attribute.name} must be positive, got {value}"
        )


def non_negative(instance: Any, attribute: Any, value: float) -> None:
    """Validate that a scalar value is non-negative.

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The value to validate.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(
            f"{attribute.name} must be non-negative, got {value}"
        )


# ============================================================================
# Array Validators
# ============================================================================


def is_finite_array(instance: Any, attribute: Any, value: NDArray) -> None:
    """Validate that all elements of an array are finite.

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The array to validate.

    Raises:
        ValueError: If any element is NaN or Inf.
    """
    if not np.all(np.isfinite(value)):
        raise ValueError(
            f"{attribute.name} must have all finite elements, got {value}"
        )


def is_3vector(instance: Any, attribute: Any, value: NDArray) -> None:
    """Validate that value is a 3-element vector.

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The array to validate.

    Raises:
        ValueError: If array shape is not (3,).
    """
    if value.shape != (3,):
        raise ValueError(
            f"{attribute.name} must be a 3-element vector, got shape {value.shape}"
        )


def positive_array(instance: Any, attribute: Any, value: NDArray) -> None:
    """Validate that all elements of an array are strictly positive.

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The array to validate.

    Raises:
        ValueError: If any element is not positive.
    """
    if np.any(value <= 0):
        raise ValueError(
            f"{attribute.name} must have all positive elements, got {value}"
        )


def is_valid_inertia(instance: Any, attribute: Any, value: NDArray) -> None:
    """Validate an inertia tensor.

    Accepts either:
    - 3-element diagonal [Ixx, Iyy, Izz]: all elements must be positive
    - 3x3 tensor: must be symmetric, positive semi-definite, with positive diagonal

    Args:
        instance: The attrs instance being validated.
        attribute: The attrs Attribute being validated.
        value: The inertia array to validate.

    Raises:
        ValueError: If inertia is invalid.
    """
    if value.shape == (3,):
        # Diagonal form: all elements must be positive
        if np.any(value <= 0):
            raise ValueError(
                f"{attribute.name} diagonal elements must all be positive, got {value}"
            )
    elif value.shape == (3, 3):
        # Full 3x3 tensor
        # Check symmetry
        if not np.allclose(value, value.T, rtol=1e-10, atol=1e-10):
            raise ValueError(
                f"{attribute.name} must be symmetric, got:\n{value}"
            )

        # Check positive diagonal
        diag = np.diag(value)
        if np.any(diag <= 0):
            raise ValueError(
                f"{attribute.name} must have positive diagonal elements, got {diag}"
            )

        # Check positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(value)
        if np.any(eigenvalues < -1e-10):  # Small tolerance for numerical errors
            raise ValueError(
                f"{attribute.name} must be positive semi-definite, "
                f"got eigenvalues {eigenvalues}"
            )
    else:
        raise ValueError(
            f"{attribute.name} must be a 3-element vector or 3x3 matrix, "
            f"got shape {value.shape}"
        )


# ============================================================================
# Converters
# ============================================================================


def to_float_array(value: Any) -> NDArray:
    """Convert value to a numpy array with fmd's configured dtype.

    Args:
        value: Input value (list, tuple, or array).

    Returns:
        Numpy array with FMD_NP_DTYPE (float64 by default, float32 if configured).
    """
    return np.asarray(value, dtype=FMD_NP_DTYPE)
