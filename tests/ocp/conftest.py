"""Test configuration for OCP tests.

This module provides fixtures and constants for testing OCP solvers.
"""

import pytest

# Skip entire module if casadi not installed
casadi = pytest.importorskip("casadi")
