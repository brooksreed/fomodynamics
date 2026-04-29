"""Tests for Moth wireframe geometry generation and surface waterline."""

import numpy as np
import pytest

from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.analysis.viz3d.geometry import build_moth_wireframe, compute_surface_waterline


@pytest.fixture
def wireframe():
    return build_moth_wireframe(MOTH_BIEKER_V3)


class TestBuildMothWireframe:
    def test_returns_all_keys(self, wireframe):
        expected_keys = {
            "hull_profile", "hull_deck",
            "main_foil", "main_foil_strut",
            "rudder", "rudder_strut",
            "wing_rack_port", "wing_rack_stbd",
            "sail_mast",
        }
        assert set(wireframe.keys()) == expected_keys

    def test_vertex_shapes_are_3d(self, wireframe):
        for name, verts in wireframe.items():
            assert verts.ndim == 2, f"{name} should be 2D array"
            assert verts.shape[1] == 3, f"{name} should have 3 columns (xyz)"
            assert verts.shape[0] >= 2, f"{name} should have at least 2 vertices"

    def test_main_foil_span_matches_params(self, wireframe):
        """Main foil span in y should match params."""
        foil = wireframe["main_foil"]
        y_span = foil[:, 1].max() - foil[:, 1].min()
        np.testing.assert_allclose(
            y_span, MOTH_BIEKER_V3.main_foil_span, rtol=1e-10
        )

    def test_rudder_span_matches_params(self, wireframe):
        rudder = wireframe["rudder"]
        y_span = rudder[:, 1].max() - rudder[:, 1].min()
        np.testing.assert_allclose(
            y_span, MOTH_BIEKER_V3.rudder_span, rtol=1e-10
        )

    def test_main_foil_is_forward_and_below(self, wireframe):
        """Main foil should be at positive x (forward) and positive z (below)."""
        foil = wireframe["main_foil"]
        # Center x of the rectangle should be forward
        x_center = (foil[:4, 0].max() + foil[:4, 0].min()) / 2
        assert x_center > 0, "Main foil should be forward (x > 0)"
        assert foil[0, 2] > 0, "Main foil should be below hull (z > 0 in FRD)"

    def test_rudder_is_aft_and_below(self, wireframe):
        """Rudder should be at negative x (aft) and positive z (below)."""
        rudder = wireframe["rudder"]
        x_center = (rudder[:4, 0].max() + rudder[:4, 0].min()) / 2
        assert x_center < 0, "Rudder should be aft (x < 0)"
        assert rudder[0, 2] > 0, "Rudder should be below hull (z > 0 in FRD)"

    def test_sail_mast_goes_upward(self, wireframe):
        """Sail mast tip should be above deck (z < 0 in FRD = up)."""
        mast = wireframe["sail_mast"]
        assert mast[-1, 2] < 0, "Sail CE should be above hull (z < 0 in FRD)"

    def test_hull_profile_closed_loop(self, wireframe):
        """Hull profile should be a closed loop."""
        hull = wireframe["hull_profile"]
        np.testing.assert_allclose(hull[0], hull[-1], atol=1e-10)

    def test_hull_deck_closed_loop(self, wireframe):
        """Hull deck should be a closed loop."""
        deck = wireframe["hull_deck"]
        np.testing.assert_allclose(deck[0], deck[-1], atol=1e-10)

    def test_struts_connect_hull_to_foils(self, wireframe):
        """Strut bottom Z should match foil Z, top Z at hull bottom."""
        hull_bottom_z = MOTH_BIEKER_V3.hull_cg_above_bottom

        # Main strut: vertices 1 and 2 are the bottom edge, 0 and 3 are top
        strut = wireframe["main_foil_strut"]
        foil = wireframe["main_foil"]
        assert strut[1, 2] == pytest.approx(foil[0, 2])
        assert strut[2, 2] == pytest.approx(foil[0, 2])
        assert strut[0, 2] == pytest.approx(hull_bottom_z)
        assert strut[3, 2] == pytest.approx(hull_bottom_z)

        # Rudder strut
        r_strut = wireframe["rudder_strut"]
        r_foil = wireframe["rudder"]
        assert r_strut[1, 2] == pytest.approx(r_foil[0, 2])
        assert r_strut[2, 2] == pytest.approx(r_foil[0, 2])
        assert r_strut[0, 2] == pytest.approx(hull_bottom_z)
        assert r_strut[3, 2] == pytest.approx(hull_bottom_z)

    def test_foil_rectangles_are_closed_loops(self, wireframe):
        """All 4 foil/strut surfaces should be (5, 3) with first == last."""
        for name in ["main_foil", "main_foil_strut", "rudder", "rudder_strut"]:
            verts = wireframe[name]
            assert verts.shape == (5, 3), f"{name} should be (5, 3)"
            np.testing.assert_allclose(
                verts[0], verts[-1], atol=1e-14,
                err_msg=f"{name} should be a closed loop"
            )

    def test_wing_racks_shape_and_dihedral(self, wireframe):
        """Wing racks should be (2, 3) lines angling upward from centerline."""
        for name in ["wing_rack_port", "wing_rack_stbd"]:
            verts = wireframe[name]
            assert verts.shape == (2, 3), f"{name} should be (2, 3)"
            # Base at y=0
            assert verts[0, 1] == pytest.approx(0.0)
            # Tip is above base (more negative z in FRD)
            assert verts[1, 2] < verts[0, 2], f"{name} tip should be above base"
        # Port tip at negative y, starboard at positive y
        assert wireframe["wing_rack_port"][1, 1] < 0
        assert wireframe["wing_rack_stbd"][1, 1] > 0

    def test_main_foil_chord_matches_params(self, wireframe):
        """Main foil X extent should match main_foil_chord."""
        foil = wireframe["main_foil"]
        x_extent = foil[:4, 0].max() - foil[:4, 0].min()
        np.testing.assert_allclose(
            x_extent, MOTH_BIEKER_V3.main_foil_chord, rtol=1e-10
        )

    def test_main_strut_chord_matches_params(self, wireframe):
        """Main strut X extent should match main_strut_chord."""
        strut = wireframe["main_foil_strut"]
        x_extent = strut[:4, 0].max() - strut[:4, 0].min()
        np.testing.assert_allclose(
            x_extent, MOTH_BIEKER_V3.main_strut_chord, rtol=1e-10
        )

    def test_rudder_chord_matches_params(self, wireframe):
        """Rudder X extent should match rudder_chord."""
        rudder = wireframe["rudder"]
        x_extent = rudder[:4, 0].max() - rudder[:4, 0].min()
        np.testing.assert_allclose(
            x_extent, MOTH_BIEKER_V3.rudder_chord, rtol=1e-10
        )

    def test_rudder_strut_chord_matches_params(self, wireframe):
        """Rudder strut X extent should match rudder_strut_chord."""
        strut = wireframe["rudder_strut"]
        x_extent = strut[:4, 0].max() - strut[:4, 0].min()
        np.testing.assert_allclose(
            x_extent, MOTH_BIEKER_V3.rudder_strut_chord, rtol=1e-10
        )

    def test_hull_depth_matches_params(self, wireframe):
        """Hull profile Z extent should match hull_depth."""
        hull = wireframe["hull_profile"]
        z_extent = hull[:, 2].max() - hull[:, 2].min()
        np.testing.assert_allclose(
            z_extent, MOTH_BIEKER_V3.hull_depth, rtol=1e-10
        )

    def test_hull_bottom_at_cg_above_bottom(self, wireframe):
        """Hull bottom should be at z = hull_cg_above_bottom in body FRD."""
        hull = wireframe["hull_profile"]
        np.testing.assert_allclose(
            hull[:, 2].max(), MOTH_BIEKER_V3.hull_cg_above_bottom, rtol=1e-10
        )


class TestComputeSurfaceWaterline:
    """Tests for surface waterline intersection computation."""

    def test_vertical_strut_intersects_water(self):
        """Strut spanning the waterline returns (2, 3)."""
        # Strut rectangle in X-Z plane, spanning from z=0.1 to z=0.6
        # Boat at pos_d=0.3 => top at world D=0.1+0.3=0.4, bottom at 0.6+0.3=0.9
        # Wait, we need the strut to straddle D=0.
        # pos_d=-0.3 => top at world D=0.1-0.3=-0.2 (above), bottom at 0.6-0.3=0.3 (below)
        verts = np.array([
            [0.05, 0.0, 0.1],   # fwd top
            [0.05, 0.0, 0.6],   # fwd bottom
            [-0.05, 0.0, 0.6],  # aft bottom
            [-0.05, 0.0, 0.1],  # aft top
        ])
        result = compute_surface_waterline(verts, pos_d=-0.3, theta=0.0)
        assert result is not None
        assert result.shape == (2, 3)

    def test_strut_waterline_is_chordwise(self):
        """Intersection points on a strut should have same Y and Z, different X."""
        verts = np.array([
            [0.05, 0.0, 0.1],
            [0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.1],
        ])
        result = compute_surface_waterline(verts, pos_d=-0.3, theta=0.0)
        assert result is not None
        # Both points should be at y=0
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-14)
        # Z should be the same for both intersection points
        np.testing.assert_allclose(result[0, 2], result[1, 2], atol=1e-14)
        # X should differ (chord extent)
        assert abs(result[0, 0] - result[1, 0]) > 0.01

    def test_horizontal_foil_at_water_returns_line(self):
        """A horizontal foil near the water surface intersects with slight pitch."""
        # Flat foil at z=0.5 with pos_d=-0.5 puts all vertices at world D=0
        # exactly (no sign change). Adding slight pitch tilts the foil so
        # forward vertices go deeper and aft vertices go shallower, creating
        # two edge crossings.
        verts = np.array([
            [0.06, -0.5, 0.5],
            [0.06,  0.5, 0.5],
            [-0.06, 0.5, 0.5],
            [-0.06, -0.5, 0.5],
        ])
        result = compute_surface_waterline(
            verts, pos_d=-0.5, theta=0.01  # slight nose-down pitch
        )
        assert result is not None
        assert result.shape == (2, 3)

    def test_returns_none_fully_submerged(self):
        """All vertices below water returns None."""
        verts = np.array([
            [0.05, -0.5, 0.5],
            [0.05,  0.5, 0.5],
            [-0.05, 0.5, 0.5],
            [-0.05, -0.5, 0.5],
        ])
        # pos_d=1.0 puts everything at world D >= 1.5 (well below water D=0)
        result = compute_surface_waterline(verts, pos_d=1.0, theta=0.0)
        assert result is None

    def test_returns_none_fully_above(self):
        """All vertices above water returns None."""
        verts = np.array([
            [0.05, -0.5, 0.5],
            [0.05,  0.5, 0.5],
            [-0.05, 0.5, 0.5],
            [-0.05, -0.5, 0.5],
        ])
        # pos_d=-5.0 puts everything at world D=-4.5 (well above water D=0)
        result = compute_surface_waterline(verts, pos_d=-5.0, theta=0.0)
        assert result is None

    def test_no_margin_threshold(self):
        """Water just barely crossing should return intersection (no 0.5m cutoff)."""
        # Strut from z=0.1 to z=0.6.
        # pos_d=-0.09 => top at world D=0.01 (just below), bottom at 0.51
        # Top is just below water -> all below -> None
        # pos_d=-0.11 => top at world D=-0.01 (just above), bottom at 0.49
        # Top above, bottom below -> intersection!
        verts = np.array([
            [0.05, 0.0, 0.1],
            [0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.1],
        ])
        result = compute_surface_waterline(verts, pos_d=-0.11, theta=0.0)
        assert result is not None, "Should detect even marginal water crossing"

    def test_analytical_level_boat(self):
        """Known geometry: verify exact intersection coordinates for level boat."""
        # Strut: fwd-top(0.05, 0, 0.1), fwd-bot(0.05, 0, 0.6),
        #        aft-bot(-0.05, 0, 0.6), aft-top(-0.05, 0, 0.1)
        # pos_d=-0.3, theta=0 => world z = body_z + pos_d = body_z - 0.3
        # Top edge: world z = 0.1 - 0.3 = -0.2 (above)
        # Bottom edge: world z = 0.6 - 0.3 = 0.3 (below)
        # Water at D=0. Intersection on edges 0-1 and 2-3:
        # Edge 0-1: t = d0/(d0-d1) = -0.2/(-0.2-0.3) = -0.2/-0.5 = 0.4
        #   -> body z = 0.1 + 0.4*(0.6-0.1) = 0.1 + 0.2 = 0.3
        #   -> body x = 0.05, body y = 0.0
        # Edge 2-3: t = d2/(d2-d3) = 0.3/(0.3-(-0.2)) = 0.3/0.5 = 0.6
        #   -> body z = 0.6 + 0.6*(0.1-0.6) = 0.6 - 0.3 = 0.3
        #   -> body x = -0.05 + 0.6*(−0.05−(−0.05)) = -0.05
        #   Actually: body = verts[2] + 0.6*(verts[3]-verts[2])
        #            = [-0.05,0,0.6] + 0.6*([-0.05,0,0.1]-[-0.05,0,0.6])
        #            = [-0.05,0,0.6] + 0.6*[0,0,-0.5] = [-0.05,0,0.3]
        verts = np.array([
            [0.05, 0.0, 0.1],
            [0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.1],
        ])
        result = compute_surface_waterline(verts, pos_d=-0.3, theta=0.0)
        assert result is not None
        # Sort by x to get consistent order
        result_sorted = result[result[:, 0].argsort()]
        np.testing.assert_allclose(result_sorted[0], [-0.05, 0.0, 0.3], atol=1e-14)
        np.testing.assert_allclose(result_sorted[1], [0.05, 0.0, 0.3], atol=1e-14)

    def test_pitched_boat(self):
        """Non-zero pitch changes the intersection location."""
        verts = np.array([
            [0.05, 0.0, 0.1],
            [0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.1],
        ])
        result_level = compute_surface_waterline(verts, pos_d=-0.3, theta=0.0)
        result_pitched = compute_surface_waterline(
            verts, pos_d=-0.3, theta=np.deg2rad(5.0)
        )
        assert result_level is not None
        assert result_pitched is not None
        # Pitched result should differ from level
        assert not np.allclose(result_level, result_pitched)

    def test_wave_elevation_fn(self):
        """Wave elevation function affects the intersection."""
        verts = np.array([
            [0.05, 0.0, 0.1],
            [0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.6],
            [-0.05, 0.0, 0.1],
        ])

        def wave_fn(n, e, t):
            return 0.1  # water raised by 0.1m

        result_flat = compute_surface_waterline(verts, pos_d=-0.3, theta=0.0)
        result_wave = compute_surface_waterline(
            verts, pos_d=-0.3, theta=0.0, water_elevation_fn=wave_fn
        )
        assert result_flat is not None
        assert result_wave is not None
        # Wave should shift the intersection
        assert not np.allclose(result_flat, result_wave)

    def test_heeled_boat(self):
        """Non-zero heel tilts a horizontal foil, creating a chordwise waterline."""
        # Foil in X-Y plane at z=0.5, pos_d=-0.5 puts it at water level.
        # With 30 deg heel, port rises and stbd sinks, creating a crossing.
        verts = np.array([
            [0.06, -0.5, 0.5],
            [0.06,  0.5, 0.5],
            [-0.06, 0.5, 0.5],
            [-0.06, -0.5, 0.5],
        ])
        result = compute_surface_waterline(
            verts, pos_d=-0.5, theta=0.0, heel_angle=np.deg2rad(30.0)
        )
        assert result is not None
        assert result.shape == (2, 3)
        # Intersection is chordwise: X differs, Y same
        np.testing.assert_allclose(result[0, 1], result[1, 1], atol=1e-10)
        assert abs(result[0, 0] - result[1, 0]) > 0.01

    def test_invalid_vertices_shape_raises(self):
        """Should raise ValueError for non-(4,3) input."""
        verts = np.array([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(ValueError, match="must be \\(4, 3\\)"):
            compute_surface_waterline(verts, pos_d=0.0, theta=0.0)
