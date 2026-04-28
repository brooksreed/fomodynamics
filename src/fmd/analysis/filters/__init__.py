"""Signal filtering and smoothing library.

This module provides filters for processing time-series data:

- **Acausal filters**: Use past and future data (offline smoothing)
- **Causal filters**: Use only past data (online compatible)

Quick start:
    >>> from fmd.analysis.filters import MovingAverage, ExponentialMovingAverage
    >>>
    >>> # Offline smoothing (acausal)
    >>> ma = MovingAverage(half_window=2)  # window of 5 points
    >>> result = ma.apply(data)
    >>> smoothed = result.data
    >>>
    >>> # Online filtering (causal)
    >>> ema = ExponentialMovingAverage(alpha=0.1)
    >>> result = ema.apply(data)
    >>> filtered = result.data

For convenience, commonly used filters are exposed at the top level.
For the full collection, import from the submodules:

    >>> from fmd.analysis.filters.acausal import MovingAverage
    >>> from fmd.analysis.filters.causal import ExponentialMovingAverage
"""

# Base classes
from .base import Filter, FilterResult, FilterType, CausalFilter, AcausalFilter

# Acausal (offline) filters
from .acausal import MovingAverage

# Causal (online) filters
from .causal import ExponentialMovingAverage


def list_filters() -> dict[str, list[str]]:
    """List all available filters by type.

    Returns:
        Dict with keys 'acausal' and 'causal', each containing
        a list of filter class names.
    """
    return {
        "acausal": [
            "MovingAverage",
            # Future filters here
        ],
        "causal": [
            "ExponentialMovingAverage",
            # Future filters here
        ],
    }


__all__ = [
    # Base
    "Filter",
    "FilterResult",
    "FilterType",
    "CausalFilter",
    "AcausalFilter",
    # Acausal filters
    "MovingAverage",
    # Causal filters
    "ExponentialMovingAverage",
    # Utilities
    "list_filters",
]
