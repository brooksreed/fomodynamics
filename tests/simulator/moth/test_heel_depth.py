"""Tests for heel-corrected foil depth computation.

Validates that compute_foil_ned_depth() matches the full rotation matrix
used by the viz geometry code, and that all callers are consistent.
"""

import numpy as np
import pytest

from fmd.simulator.components.moth_forces import compute_foil_ned_depth
from fmd.simulator.moth_scenarios import compute_tip_at_surface_pos_d
from fmd.simulator.params import MOTH_BIEKER_V3


class TestComputeFoilNedDepth:
    """Unit tests for the canonical compute_foil_ned_depth function."""

    def test_zero_heel_identity(self):
        """At heel=0, depth = pos_d + z*cos(theta) - x*sin(theta)."""
        pos_d = -0.3
        x, z = 0.6, 0.6
        theta = np.deg2rad(5.0)
        heel = 0.0

        result = float(compute_foil_ned_depth(pos_d, x, z, theta, heel))
        expected = pos_d + z * np.cos(theta) - x * np.sin(theta)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_heel_sweep_monotonic(self):
        """Depth decreases monotonically with increasing heel (same pos_d/theta)."""
        pos_d = -0.3
        x, z = 0.6, 0.6
        theta = np.deg2rad(3.0)
        heel_angles = np.deg2rad(np.arange(0, 46, 5))

        depths = [float(compute_foil_ned_depth(pos_d, x, z, theta, h))
                  for h in heel_angles]

        for i in range(len(depths) - 1):
            assert depths[i] > depths[i + 1], (
                f"Depth not monotonically decreasing: "
                f"heel={np.degrees(heel_angles[i]):.0f}deg -> {depths[i]:.6f}, "
                f"heel={np.degrees(heel_angles[i+1]):.0f}deg -> {depths[i+1]:.6f}"
            )

    def test_foil_depth_matches_rotation_matrix(self):
        """compute_foil_ned_depth matches the full R = Ry(theta) @ Rx(heel) z-row."""
        test_cases = [
            (0.0, 0.0),         # no pitch, no heel
            (0.0, 30.0),        # heel only
            (5.0, 30.0),        # typical foiling
            (-3.0, 15.0),       # nose-down + moderate heel
            (10.0, 45.0),       # large pitch + large heel
        ]
        pos_d = -0.35
        x, z = 0.6, 0.6  # typical main foil position

        for theta_deg, heel_deg in test_cases:
            theta = np.deg2rad(theta_deg)
            heel = np.deg2rad(heel_deg)

            # Full rotation matrix
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            cos_h, sin_h = np.cos(heel), np.sin(heel)
            R = np.array([
                [cos_t,  sin_t * sin_h,  sin_t * cos_h],
                [0.0,    cos_h,          -sin_h],
                [-sin_t, cos_t * sin_h,  cos_t * cos_h],
            ])
            # World z for point [x, 0, z]
            body_pt = np.array([x, 0.0, z])
            world_z = R[2] @ body_pt + pos_d  # z-row of rotation + CG offset

            # Canonical function
            canonical = float(compute_foil_ned_depth(pos_d, x, z, theta, heel))

            np.testing.assert_allclose(
                canonical, world_z, atol=1e-12,
                err_msg=f"Mismatch at theta={theta_deg}deg, heel={heel_deg}deg"
            )

    def test_compute_tip_at_surface_self_consistent(self):
        """compute_tip_at_surface_pos_d returns pos_d where tip depth = 0."""
        params = MOTH_BIEKER_V3
        heel = np.deg2rad(30.0)
        theta = 0.0

        pos_d = compute_tip_at_surface_pos_d(params, heel, theta)

        # Compute foil center depth at this pos_d
        total_mass = params.hull_mass + params.sailor_mass
        cg_offset = params.sailor_mass * params.sailor_position / total_mass
        foil_pos = params.main_foil_position - cg_offset

        foil_center_depth = float(compute_foil_ned_depth(
            pos_d, foil_pos[0], foil_pos[2], theta, heel
        ))

        # Tip depth = center depth - half_span_rise
        half_span_rise = (params.main_foil_span / 2.0) * np.sin(heel)
        tip_depth = foil_center_depth - half_span_rise

        np.testing.assert_allclose(tip_depth, 0.0, atol=1e-10,
                                   err_msg="Tip should be at surface (depth=0)")

    def test_physics_vs_viz_depth_at_trim(self):
        """Foil depth from compute_foil_ned_depth matches viz geometry rotation."""
        params = MOTH_BIEKER_V3
        heel = np.deg2rad(30.0)
        pos_d = -0.35
        theta = np.deg2rad(3.0)

        total_mass = params.hull_mass + params.sailor_mass
        cg_offset = params.sailor_mass * params.sailor_position / total_mass
        foil_pos = params.main_foil_position - cg_offset

        # Physics model: canonical function
        physics_depth = float(compute_foil_ned_depth(
            pos_d, foil_pos[0], foil_pos[2], theta, heel
        ))

        # Viz geometry: full rotation matrix (same as compute_surface_waterline)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_h, sin_h = np.cos(heel), np.sin(heel)
        R = np.array([
            [cos_t,  sin_t * sin_h,  sin_t * cos_h],
            [0.0,    cos_h,          -sin_h],
            [-sin_t, cos_t * sin_h,  cos_t * cos_h],
        ])
        body_pt = np.array([foil_pos[0], 0.0, foil_pos[2]])
        viz_depth = R[2] @ body_pt + pos_d

        np.testing.assert_allclose(physics_depth, viz_depth, atol=1e-12,
                                   err_msg="Physics and viz depth disagree")

    def test_eta_subtracts_from_depth(self):
        """Wave elevation reduces effective depth (foil shallower on crest)."""
        pos_d = -0.3
        x, z = 0.6, 0.6
        theta = 0.0
        heel = np.deg2rad(30.0)

        depth_no_wave = float(compute_foil_ned_depth(pos_d, x, z, theta, heel))
        depth_crest = float(compute_foil_ned_depth(pos_d, x, z, theta, heel, eta=0.1))

        assert depth_crest < depth_no_wave, "Crest should make foil shallower"
        np.testing.assert_allclose(depth_no_wave - depth_crest, 0.1, atol=1e-12)
