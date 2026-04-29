"""Mathematical operations with proper handling of circular quantities.

Provides angle-aware math that handles wraparound correctly:
- Subtraction: 359° - 1° = -2° (not 358°)
- Interpolation: 350° to 10° goes through 0° (not 180°)
- Mean: average of [350°, 10°] is 0° (not 180°)
- Filtering: handles discontinuities at wrap boundaries
"""

import numpy as np
from numpy.typing import ArrayLike


def wrap_angle(
    angle: ArrayLike,
    low: float = -np.pi,
    high: float = np.pi,
) -> np.ndarray:
    """Wrap angle(s) to the range [low, high).
    
    Args:
        angle: Angle(s) in radians
        low: Lower bound of range (inclusive)
        high: Upper bound of range (exclusive)
        
    Returns:
        Wrapped angle(s) in [low, high)
        
    Examples:
        >>> wrap_angle(2 * np.pi)  # 360° -> 0°
        0.0
        >>> wrap_angle(-np.pi - 0.1)  # Just below -180° -> just below 180°
        3.04159...
    """
    angle = np.asarray(angle)
    range_size = high - low
    return low + (angle - low) % range_size


def unwrap_angle(angle: ArrayLike) -> np.ndarray:
    """Unwrap angle sequence to remove discontinuities.
    
    Adjusts angles so that absolute jumps between consecutive values
    are minimized. Useful before interpolation or filtering.
    
    Args:
        angle: Angle sequence in radians
        
    Returns:
        Unwrapped angles (may exceed [-π, π])
    """
    return np.unwrap(np.asarray(angle))


def circular_subtract(
    a: ArrayLike,
    b: ArrayLike,
    wrap_range: tuple[float, float] = (-np.pi, np.pi),
) -> np.ndarray:
    """Subtract angles with proper wraparound handling.
    
    Returns the shortest angular difference from b to a.
    
    Args:
        a: First angle(s) in radians
        b: Second angle(s) in radians  
        wrap_range: Range to wrap result to
        
    Returns:
        Angular difference a - b, wrapped to wrap_range
        
    Examples:
        >>> circular_subtract(np.radians(1), np.radians(359))  # 1° - 359° = 2°
        0.0349...  # ≈ 2° in radians
        >>> circular_subtract(np.radians(359), np.radians(1))  # 359° - 1° = -2°
        -0.0349...  # ≈ -2° in radians
    """
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    return wrap_angle(diff, wrap_range[0], wrap_range[1])


def circular_mean(
    angles: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """Compute the circular (directional) mean of angles.

    Uses the unit vector method: average the unit vectors, then
    take the angle of the result.

    Args:
        angles: Angles in radians
        weights: Optional weights for weighted mean

    Returns:
        Circular mean angle in radians (range [-pi, pi] from atan2).
        Returns NaN when all input angles are NaN or when weights sum
        to zero or negative.

    Note:
        When the resultant vector length is near zero (R < 1e-12), the mean
        direction is undefined and this function returns 0.0 by contract.
        This occurs when angles cancel out, such as:
        - Opposite angles: [0, pi] or [pi/2, -pi/2]
        - Uniformly distributed angles: [0, pi/2, pi, 3*pi/2]
        - Any set where the vector sum is near the origin

        The 0.0 return value is a deterministic convention, not a mathematically
        meaningful result. Callers needing to detect this case should check the
        resultant length separately using circular statistics.

    Examples:
        >>> circular_mean([np.radians(350), np.radians(10)])  # Mean of 350° and 10°
        0.0  # 0°, not 180°
        >>> circular_mean([0, np.pi])  # Opposite angles cancel out
        0.0  # Undefined direction, returns 0.0 by contract
    """
    angles = np.asarray(angles, dtype=float)

    # Drop NaNs
    valid = np.isfinite(angles)
    if not np.any(valid):
        return float("nan")

    angles_v = angles[valid]

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != angles.shape:
            raise ValueError("weights must have the same shape as angles")
        w = w[valid]
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            return float("nan")
        w = w / w_sum
        x = float(np.sum(w * np.cos(angles_v)))
        y = float(np.sum(w * np.sin(angles_v)))
    else:
        x = float(np.mean(np.cos(angles_v)))
        y = float(np.mean(np.sin(angles_v)))

    r = float(np.hypot(x, y))
    if r < 1e-12:
        # Hard contract: undefined mean direction -> deterministic 0.0
        return 0.0

    return float(np.arctan2(y, x))


def angle_difference_to_vector(
    heading: ArrayLike,
    target: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vector components from heading to target.
    
    Useful for computing cross-track and along-track components.
    
    Args:
        heading: Current heading in radians
        target: Target direction in radians
        
    Returns:
        (lateral, longitudinal) components where:
        - lateral > 0 means target is to the right
        - longitudinal > 0 means target is ahead
    """
    heading = np.asarray(heading)
    target = np.asarray(target)
    diff = circular_subtract(target, heading)
    return (np.sin(diff), np.cos(diff))

