"""Test configuration for CasADi simulator tests.

This module provides fixtures and tolerance constants for testing
CasADi dynamics models and verifying JAX/CasADi equivalence.
"""

import pytest

# Skip entire module if casadi not installed
casadi = pytest.importorskip("casadi")

# -----------------------------------------------------------------------------
# Tolerance Constants for Equivalence Testing
# -----------------------------------------------------------------------------
#
# These tolerances are calibrated for float64 precision.
#
# Note: CasADi evaluates numerically in float64 (double). If fmd switches JAX
# simulation to float32 in the future (FMD_USE_FLOAT32=1), strict equivalence
# tests against CasADi will likely fail due to dtype mismatch and reduced
# mantissa bits. Options then:
# - Force JAX to run these tests in float64, or
# - Relax tolerances substantially (e.g., DERIV_RTOL=1e-5, TRAJ_RTOL=1e-4), or
# - Skip equivalence tests in float32 mode.
# -----------------------------------------------------------------------------

# Level 0: Derivative equivalence (pure math, identical formulas)
# JAX and CasADi should produce numerically identical results for forward_dynamics
# when given the same inputs. Any difference is a bug in transcription.
DERIV_RTOL = 1e-12
DERIV_ATOL = 1e-14

# Level 1: Jacobian equivalence (autodiff vs symbolic)
# JAX uses reverse-mode autodiff, CasADi uses symbolic differentiation.
# Both should produce the same Jacobian matrices, though minor floating-point
# differences may occur due to different evaluation order.
JAC_RTOL = 1e-10
JAC_ATOL = 1e-12

# Level 2: Trajectory equivalence (RK4 accumulation)
# Multi-step RK4 integration should match between JAX and CasADi.
# Slightly relaxed tolerances due to floating-point accumulation.
TRAJ_RTOL = 1e-10
TRAJ_ATOL = 1e-12
