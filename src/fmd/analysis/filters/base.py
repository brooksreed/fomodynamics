"""Base classes and protocols for filters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class FilterType(Enum):
    """Type of filter based on data access pattern."""
    ACAUSAL = "acausal"  # Uses past and future data (offline only)
    CAUSAL = "causal"    # Uses only past data (online compatible)


@dataclass
class FilterResult:
    """Result of applying a filter to data.

    Attributes:
        data: Filtered data (same length as input)
        filter_name: Name of the filter used
        filter_type: Whether filter is causal or acausal
        params: Parameters used for filtering
    """
    data: np.ndarray | pd.Series
    filter_name: str
    filter_type: FilterType
    params: dict


class Filter(ABC):
    """Abstract base class for all filters.

    Subclasses must implement:
    - name: Human-readable name
    - filter_type: CAUSAL or ACAUSAL
    - apply(): Apply filter to data
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the filter."""
        pass

    @property
    @abstractmethod
    def filter_type(self) -> FilterType:
        """Whether this filter is causal or acausal."""
        pass

    @abstractmethod
    def apply(self, data: np.ndarray | pd.Series) -> FilterResult:
        """Apply the filter to data.

        Args:
            data: 1D array or Series to filter

        Returns:
            FilterResult with filtered data and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CausalFilter(Filter):
    """Base class for causal (online-compatible) filters.

    Causal filters only use past and current data, making them
    suitable for real-time / streaming applications.
    """

    @property
    def filter_type(self) -> FilterType:
        return FilterType.CAUSAL


class AcausalFilter(Filter):
    """Base class for acausal (offline) smoothing filters.

    Acausal filters use both past and future data, providing
    better smoothing at the cost of requiring all data upfront.
    """

    @property
    def filter_type(self) -> FilterType:
        return FilterType.ACAUSAL
