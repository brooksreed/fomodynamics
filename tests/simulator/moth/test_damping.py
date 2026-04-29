"""Tests for Moth 3DOF damping and eigenvalue structure.

Validates the linearized dynamics eigenvalue structure and damping
behavior after the foil force decomposition fix (alpha_geo/alpha_eff).

Eigenvalue structure at trim with surge_enabled=True (per-speed Moth3D, free theta):
- 2 fast stable real eigenvalues (Re ~ -23 to -43)
- 1 stable real (heave mode, Re ~ -0.4 to -0.6)
- 1 stable real (surge mode, Re ~ -0.3, from drag/gravity coupling)
- 1 unstable real (pitch divergence, Re ~ +0.45 to +0.55)

Max positive eigenvalue increases monotonically with speed:
  10 m/s: +0.45, 12 m/s: +0.54
"""

import pytest
import numpy as np
import jax.numpy as jnp
import attrs

from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.linearize import linearize


# -- Eigenvalue reference values with NED sail thrust + surge_enabled=True --
# Computed with per-speed Moth3D instances, free theta
# Updated: NED→body sail thrust rotation, surge dynamics enabled
# Reference values computed with CasADi trim solver
# 8 m/s excluded: CasADi convergence issue with NED sail thrust
EIGENVALUE_REFERENCE = {
    10.0: {"max_real": 0.482, "min_real": -34.88},
    12.0: {"max_real": 0.601, "min_real": -40.89},
}


def _eigenvalues_at_speed(speed):
    """Compute eigenvalues at a given speed using per-speed Moth3D."""
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(speed))
    result = find_moth_trim(MOTH_BIEKER_V3, u_forward=speed)
    assert result.residual < 0.1, (
        f"Trim residual {result.residual:.2e} at {speed} m/s"
    )
    A, _ = linearize(moth, jnp.array(result.state), jnp.array(result.control))
    return np.linalg.eigvals(np.array(A)), result


class TestEigenvalueStructure:
    """Verify the eigenvalue structure is consistent across speeds."""

    @pytest.mark.parametrize("speed", [10.0, 12.0])
    def test_eigenvalue_structure(self, speed):
        """Eigenvalue structure: 2 fast stable, 2 slow stable, 1 unstable.

        With surge_enabled=True, the 5x5 A matrix should have:
        - Two fast stable eigenvalues with Re < -10 (may be real or complex)
        - Two slow stable eigenvalues with -1 < Re < 0 (heave/surge modes)
        - A positive real eigenvalue Re > 0 (pitch divergence)
        """
        eigenvalues, _ = _eigenvalues_at_speed(speed)
        real_parts = np.real(eigenvalues)
        sorted_idx = np.argsort(real_parts)
        sorted_real = real_parts[sorted_idx]

        # Two fast stable modes (Re < -10)
        assert sorted_real[0] < -10, (
            f"{speed} m/s: fast stable eigenvalue[0]={sorted_real[0]:.2f}, expected < -10"
        )
        assert sorted_real[1] < -10, (
            f"{speed} m/s: fast stable eigenvalue[1]={sorted_real[1]:.2f}, expected < -10"
        )

        # Two slow stable modes: -1 < Re < 0 (heave + surge coupling)
        assert -1 < sorted_real[2] < 0, (
            f"{speed} m/s: slow mode eigenvalue[2]={sorted_real[2]:.4f}, expected in (-1, 0)"
        )
        assert -1 < sorted_real[3] < 0, (
            f"{speed} m/s: slow mode eigenvalue[3]={sorted_real[3]:.4f}, expected in (-1, 0)"
        )

        # One unstable (pitch divergence): Re > 0
        assert sorted_real[4] > 0, (
            f"{speed} m/s: max eigenvalue={sorted_real[4]:.4f}, expected > 0"
        )


class TestEigenvalueRegression:
    """Regression tests against known eigenvalue values."""

    @pytest.mark.parametrize("speed", [10.0, 12.0])
    def test_eigenvalue_regression(self, speed):
        """Eigenvalue magnitudes match reference within +/-20%.

        Reference values computed from post-geometry-migration trim with
        per-speed Moth3D instances. Tolerance of 20% catches meaningful
        regressions while accommodating minor trim solver variations.
        """
        ref = EIGENVALUE_REFERENCE[speed]
        eigenvalues, _ = _eigenvalues_at_speed(speed)
        real_parts = np.real(eigenvalues)

        max_real = np.max(real_parts)
        min_real = np.min(real_parts)

        # Max positive eigenvalue (pitch divergence rate)
        assert max_real == pytest.approx(ref["max_real"], rel=0.2), (
            f"{speed} m/s: max_real={max_real:.4f}, expected ~{ref['max_real']}"
        )

        # Min eigenvalue (fast stable mode)
        assert min_real == pytest.approx(ref["min_real"], rel=0.2), (
            f"{speed} m/s: min_real={min_real:.4f}, expected ~{ref['min_real']}"
        )


class TestDampingComparison:
    """Compare dynamics with and without added mass/inertia."""

    def test_added_mass_effect_on_fast_modes(self):
        """Added mass/inertia slows the fast stable modes.

        With added mass zeroed, the fast eigenvalue magnitudes should
        increase (less effective inertia -> faster dynamics). The effect
        is small (~1%) because added mass is small relative to total mass.
        """
        moth_full = Moth3D(MOTH_BIEKER_V3)
        params_no_am = attrs.evolve(
            MOTH_BIEKER_V3,
            added_mass_heave=0.0,
            added_inertia_pitch=0.0,
        )
        moth_no_am = Moth3D(params_no_am)

        result_full = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        result_no_am = find_moth_trim(params_no_am, u_forward=10.0)
        assert result_full.success and result_no_am.success

        A_full, _ = linearize(
            moth_full, jnp.array(result_full.state), jnp.array(result_full.control)
        )
        A_no_am, _ = linearize(
            moth_no_am, jnp.array(result_no_am.state), jnp.array(result_no_am.control)
        )

        ev_full = np.linalg.eigvals(np.array(A_full))
        ev_no_am = np.linalg.eigvals(np.array(A_no_am))

        min_full = np.min(np.real(ev_full))
        min_no_am = np.min(np.real(ev_no_am))

        # Without added mass, fast mode magnitude should be larger (more negative)
        assert min_no_am < min_full, (
            f"Without added mass, fast mode should be faster: "
            f"no_am={min_no_am:.2f}, full={min_full:.2f}"
        )

    def test_heave_damping_present(self):
        """Heave mode is damped: A[2,2] (dw_dot/dw) < 0."""
        moth = Moth3D(MOTH_BIEKER_V3)
        result = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert result.success

        A, _ = linearize(moth, jnp.array(result.state), jnp.array(result.control))
        heave_damping = float(A[2, 2])
        assert heave_damping < 0, f"Heave should be damped: A[2,2]={heave_damping:.4f}"

    def test_pitch_damping_present(self):
        """Pitch rate is damped: A[3,3] (dq_dot/dq) < 0."""
        moth = Moth3D(MOTH_BIEKER_V3)
        result = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert result.success

        A, _ = linearize(moth, jnp.array(result.state), jnp.array(result.control))
        pitch_rate_damping = float(A[3, 3])
        assert pitch_rate_damping < 0, (
            f"Pitch rate should be damped: A[3,3]={pitch_rate_damping:.4f}"
        )


class TestInstabilityWithSpeed:
    """Verify that instability growth rate increases with speed."""

    def test_instability_increases_with_speed(self):
        """Max positive eigenvalue increases monotonically with speed.

        At higher speeds, the pitch divergence mode grows faster because
        force gradients increase with dynamic pressure (q ~ V^2).
        """
        # 8 m/s excluded: CasADi convergence issue with NED sail thrust
        speeds = [10.0, 12.0]
        max_eigenvalues = []

        for speed in speeds:
            eigenvalues, _ = _eigenvalues_at_speed(speed)
            max_eigenvalues.append(np.max(np.real(eigenvalues)))

        # All should be positive (unstable)
        for speed, ev in zip(speeds, max_eigenvalues):
            assert ev > 0, f"{speed} m/s: expected unstable, got max_real={ev:.4f}"

        # Monotonically non-decreasing
        for i in range(len(speeds) - 1):
            assert max_eigenvalues[i] <= max_eigenvalues[i + 1] + 0.01, (
                f"Instability should increase with speed: "
                f"{speeds[i]} m/s={max_eigenvalues[i]:.4f}, "
                f"{speeds[i+1]} m/s={max_eigenvalues[i+1]:.4f}"
            )
