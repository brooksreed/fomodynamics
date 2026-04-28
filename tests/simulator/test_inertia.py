"""Tests for inertia estimation module."""

import numpy as np
import pytest

from fmd.simulator.params.inertia import (
    ComponentSpec,
    estimate_composite_inertia,
    estimate_moth_inertia,
)
from fmd.simulator.params import MOTH_BIEKER_V3


class TestEstimateCompositeInertia:
    """Tests for estimate_composite_inertia()."""

    def test_two_point_masses_known_result(self):
        """Two point masses on x-axis: Iyy = m1*d1^2 + m2*d2^2 about CG."""
        m1, m2 = 3.0, 5.0
        x1, x2 = -2.0, 4.0
        # CG = (m1*x1 + m2*x2) / (m1+m2) = (-6 + 20) / 8 = 1.75
        cg_x = (m1 * x1 + m2 * x2) / (m1 + m2)
        d1 = x1 - cg_x
        d2 = x2 - cg_x

        components = [
            ComponentSpec("m1", m1, [x1, 0.0, 0.0]),
            ComponentSpec("m2", m2, [x2, 0.0, 0.0]),
        ]
        inertia, cg, total_mass = estimate_composite_inertia(components)

        assert total_mass == pytest.approx(m1 + m2)
        np.testing.assert_allclose(cg, [cg_x, 0.0, 0.0], atol=1e-15)
        # Iyy (rotation about y) uses distances in x and z
        expected_iyy = m1 * d1**2 + m2 * d2**2
        assert inertia[1] == pytest.approx(expected_iyy, rel=1e-12)
        # Izz same as Iyy for points on x-axis (distances in x and y)
        assert inertia[2] == pytest.approx(expected_iyy, rel=1e-12)
        # Ixx = 0 (all mass on x-axis, no y or z offset from CG)
        assert inertia[0] == pytest.approx(0.0, abs=1e-15)

    def test_single_component_at_origin(self):
        """Single component at origin: inertia equals local_inertia."""
        local_i = np.array([1.0, 2.0, 3.0])
        components = [
            ComponentSpec("sole", 5.0, [0.0, 0.0, 0.0], local_inertia=local_i),
        ]
        inertia, cg, total_mass = estimate_composite_inertia(components)

        assert total_mass == pytest.approx(5.0)
        np.testing.assert_allclose(cg, [0.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(inertia, local_i, atol=1e-15)

    def test_single_point_mass_offset(self):
        """Single point mass offset from origin: inertia = m*d^2 per axis."""
        m = 4.0
        pos = np.array([1.0, 2.0, 3.0])
        components = [ComponentSpec("pt", m, pos)]
        inertia, cg, total_mass = estimate_composite_inertia(components)

        assert total_mass == pytest.approx(m)
        np.testing.assert_allclose(cg, pos, atol=1e-15)
        # About CG (which is at pos), all distances are zero
        np.testing.assert_allclose(inertia, [0.0, 0.0, 0.0], atol=1e-15)

    def test_reference_cg_vs_estimated_cg(self):
        """Inertia about prescribed CG differs from inertia about estimated CG."""
        components = [
            ComponentSpec("a", 2.0, [0.0, 0.0, 0.0]),
            ComponentSpec("b", 2.0, [4.0, 0.0, 0.0]),
        ]
        # About estimated CG (at x=2.0): Iyy = 2*2^2 + 2*2^2 = 16
        inertia_est, cg_est, _ = estimate_composite_inertia(components)
        assert cg_est[0] == pytest.approx(2.0)
        assert inertia_est[1] == pytest.approx(16.0, rel=1e-12)

        # About reference CG at x=0: Iyy = 2*0^2 + 2*4^2 = 32
        ref_cg = np.array([0.0, 0.0, 0.0])
        inertia_ref, cg_ref, _ = estimate_composite_inertia(components, reference_cg=ref_cg)
        # CG is still the mass-weighted value, not the reference
        assert cg_ref[0] == pytest.approx(2.0)
        assert inertia_ref[1] == pytest.approx(32.0, rel=1e-12)

    def test_zero_total_mass_raises(self):
        """Empty component list raises ValueError."""
        with pytest.raises(ValueError, match="Total mass must be positive"):
            estimate_composite_inertia([])

    def test_local_inertia_adds_to_pat(self):
        """Local inertia adds to parallel axis contribution."""
        local_i = np.array([1.0, 2.0, 3.0])
        m = 5.0
        offset = np.array([1.0, 0.0, 0.0])
        components = [
            ComponentSpec("with_local", m, offset, local_inertia=local_i),
        ]
        inertia, _, _ = estimate_composite_inertia(components)
        # About CG (= offset), PAT = 0, so inertia = local_inertia
        np.testing.assert_allclose(inertia, local_i, atol=1e-15)


class TestEstimateMothInertia:
    """Tests for estimate_moth_inertia()."""

    def test_total_mass_is_50kg(self):
        """Moth non-sailor boat mass should be 50 kg."""
        _, _, total_mass, _ = estimate_moth_inertia()
        assert total_mass == pytest.approx(50.0)

    def test_output_matches_preset_inertia(self):
        """Estimated inertia should match MOTH_BIEKER_V3 hull_inertia."""
        inertia, _, _, _ = estimate_moth_inertia()
        np.testing.assert_allclose(
            inertia, MOTH_BIEKER_V3.hull_inertia, rtol=0.01,
        )

    def test_cg_position_is_3vector(self):
        """CG position should be a 3-element vector in hull-datum."""
        _, cg, _, _ = estimate_moth_inertia()
        assert cg.shape == (3,)

    def test_cg_y_is_zero(self):
        """CG should be on centerline (y=0) for symmetric boat."""
        _, cg, _, _ = estimate_moth_inertia()
        assert cg[1] == pytest.approx(0.0, abs=1e-15)

    def test_returns_components(self):
        """Function returns the component list for inspection."""
        _, _, _, components = estimate_moth_inertia()
        assert len(components) > 0
        assert all(isinstance(c, ComponentSpec) for c in components)

    def test_all_inertia_positive(self):
        """All inertia components should be positive."""
        inertia, _, _, _ = estimate_moth_inertia()
        assert all(i > 0 for i in inertia)

    def test_iyy_gt_ixx(self):
        """Pitch inertia (Iyy) should exceed roll inertia (Ixx) for long boat."""
        inertia, _, _, _ = estimate_moth_inertia()
        assert inertia[1] > inertia[0]

    def test_cg_matches_preset_hull_cg(self):
        """Estimated CG is near preset hull_cg, but preset may differ intentionally.

        The preset hull_cg_from_bow (2.1m) was intentionally shifted aft from the
        component estimate (~1.82m) for pitch balance (Session A-pt-II CG alignment).
        This test verifies the estimate is physically reasonable, not that it matches
        the preset exactly.
        """
        _, cg, _, _ = estimate_moth_inertia()
        # Component CG is ~1.82m from bow, ~0.82m above bottom
        assert cg[0] == pytest.approx(1.82, abs=0.05)
        assert cg[2] == pytest.approx(0.82, abs=0.05)
