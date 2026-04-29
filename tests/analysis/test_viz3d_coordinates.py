"""Tests for coordinate conversion between fomodynamics and Rerun frames."""

import numpy as np
import pytest

from fmd.analysis.viz3d.coordinates import (
    ned_to_rerun,
    frd_to_rerun,
    pitch_to_rerun_quat,
    moth_3dof_to_rerun_quat,
    fmd_quat_to_rerun,
)


class TestNedToRerun:
    def test_basic_mapping(self):
        """NED [1, 2, 3] -> Rerun [2, 1, -3]."""
        result = ned_to_rerun(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [2.0, 1.0, -3.0])

    def test_north_maps_to_y(self):
        """Pure north [1, 0, 0] -> Rerun [0, 1, 0] (Y axis)."""
        result = ned_to_rerun(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0])

    def test_east_maps_to_x(self):
        """Pure east [0, 1, 0] -> Rerun [1, 0, 0] (X axis)."""
        result = ned_to_rerun(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_down_maps_to_neg_z(self):
        """Pure down [0, 0, 1] -> Rerun [0, 0, -1] (negative Z = down)."""
        result = ned_to_rerun(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0])

    def test_positive_depth_moves_down(self):
        """Increasing pos_d (deeper) should decrease Z in Rerun."""
        shallow = ned_to_rerun(np.array([0.0, 0.0, 0.3]))
        deep = ned_to_rerun(np.array([0.0, 0.0, 0.5]))
        assert deep[2] < shallow[2], "Deeper boat should have lower Z in viewer"

    def test_batch_conversion(self):
        """Should work with (N, 3) arrays."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ned_to_rerun(points)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [2.0, 1.0, -3.0])
        np.testing.assert_allclose(result[1], [5.0, 4.0, -6.0])


class TestFrdToRerun:
    def test_basic_mapping(self):
        """FRD [1, 2, 3] -> Rerun [2, 1, -3]."""
        result = frd_to_rerun(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [2.0, 1.0, -3.0])

    def test_forward_maps_to_y(self):
        """Body forward [1, 0, 0] -> Rerun [0, 1, 0]."""
        result = frd_to_rerun(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0])

    def test_starboard_maps_to_x(self):
        """Body starboard [0, 1, 0] -> Rerun [1, 0, 0]."""
        result = frd_to_rerun(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_down_maps_to_neg_z(self):
        """Body down [0, 0, 1] -> Rerun [0, 0, -1]."""
        result = frd_to_rerun(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0])

    def test_lift_force_points_up_in_viewer(self):
        """Foil lift (Fz < 0 in body = upward) should point +Z in Rerun."""
        # Lift in body frame: [0, 0, -500] (upward)
        lift_frd = np.array([0.0, 0.0, -500.0])
        lift_rr = frd_to_rerun(lift_frd)
        assert lift_rr[2] > 0, "Lift force should point upward (+Z) in Rerun"

    def test_gravity_in_body_frame_points_down_in_viewer(self):
        """At zero pitch, gravity [0, 0, +mg] in body -> [0, 0, -mg] in Rerun."""
        mg = 92.0 * 9.81  # total_mass * g
        gravity_frd = np.array([0.0, 0.0, mg])
        gravity_rr = frd_to_rerun(gravity_frd)
        assert gravity_rr[2] < 0, "Gravity should point downward (-Z) in Rerun"

    def test_batch_conversion(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = frd_to_rerun(points)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [2.0, 1.0, -3.0])


class TestPitchToRerunQuat:
    def test_zero_pitch_is_identity(self):
        """Zero pitch should produce identity quaternion."""
        q = pitch_to_rerun_quat(0.0)
        # xyzw identity: [0, 0, 0, 1]
        np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-15)

    def test_unit_quaternion(self):
        """All pitch angles should produce unit quaternions."""
        thetas = np.linspace(-np.pi / 2, np.pi / 2, 20)
        quats = pitch_to_rerun_quat(thetas)
        norms = np.linalg.norm(quats, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-15)

    def test_positive_pitch_nose_up(self):
        """Positive pitch (nose-up) should rotate bow upward in Rerun.

        In FRD, bow is at +x. In Rerun, this maps to +Y.
        Positive pitch rotates about Rerun +X axis.
        With standard right-hand rule: +rotation about X lifts +Y toward +Z.
        So positive pitch should make the bow (+Y_rerun) rise (+Z_rerun).
        """
        theta = np.radians(10.0)
        q = pitch_to_rerun_quat(theta)
        # For small angle rotation about X: qx = sin(θ/2) > 0
        assert q[0] > 0, "Positive pitch should have positive qx"
        assert q[3] > 0, "qw should be positive for small angles"

    def test_batch_conversion(self):
        thetas = np.array([0.0, np.pi / 4, -np.pi / 4])
        quats = pitch_to_rerun_quat(thetas)
        assert quats.shape == (3, 4)

    def test_symmetry(self):
        """Opposite pitch angles should produce opposite rotation."""
        q_pos = pitch_to_rerun_quat(0.1)
        q_neg = pitch_to_rerun_quat(-0.1)
        # qx should be opposite, qw should be same
        np.testing.assert_allclose(q_pos[0], -q_neg[0], atol=1e-15)
        np.testing.assert_allclose(q_pos[3], q_neg[3], atol=1e-15)


class TestSignConventionIntegration:
    """Cross-module sign convention checks per plan Step 3b."""

    def test_bow_rises_with_positive_pitch(self):
        """At θ=+10°, bow (+x FRD) should move upward (+Z Rerun).

        The bow in FRD is at [L/2, 0, 0]. After FRD->Rerun it's at [0, L/2, 0].
        Pitch rotation about Rerun X axis lifts +Y toward +Z.
        """
        from fmd.analysis.viz3d.geometry import build_moth_wireframe
        from fmd.simulator.params import MOTH_BIEKER_V3

        wireframe = build_moth_wireframe(MOTH_BIEKER_V3)
        bow_frd = wireframe["hull_profile"][0]  # bow vertex
        bow_rerun = frd_to_rerun(bow_frd)

        # Apply pitch rotation manually
        theta = np.radians(10.0)
        # Rotation about X axis by theta
        c, s = np.cos(theta), np.sin(theta)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        bow_rotated = Rx @ bow_rerun

        # Bow should be higher after positive pitch
        assert bow_rotated[2] > bow_rerun[2], \
            "Positive pitch should raise the bow in viewer"

    def test_deeper_boat_moves_down_in_viewer(self):
        """Increasing pos_d should move boat down (-Z) in viewer."""
        shallow = ned_to_rerun(np.array([0.0, 0.0, 0.3]))
        deep = ned_to_rerun(np.array([0.0, 0.0, 0.5]))
        assert deep[2] < shallow[2]

    def test_gravity_always_points_down_regardless_of_pitch(self):
        """Body-frame gravity at any pitch, transformed to Rerun, should have -Z component."""
        for theta_deg in [0, 10, -10, 30, -30]:
            theta = np.radians(theta_deg)
            mg = 92.0 * 9.81
            # Body-frame gravity decomposition (same as moth_3d.py)
            grav_fx = -mg * np.sin(theta)
            grav_fz = mg * np.cos(theta)
            grav_frd = np.array([grav_fx, 0.0, grav_fz])
            grav_rr = frd_to_rerun(grav_frd)

            # In Rerun frame, we need to rotate by pitch to get world frame
            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            grav_world_rr = Rx @ grav_rr

            # Should always point down (-Z) in world
            assert grav_world_rr[2] < 0, \
                f"Gravity should point down at theta={theta_deg}°, got Z={grav_world_rr[2]}"
            # Magnitude should be mg
            np.testing.assert_allclose(
                np.linalg.norm(grav_world_rr), mg, rtol=1e-10
            )


class TestBlurQuatToRerun:
    """Tests for fomodynamics quaternion to Rerun conversion with frame transformation.

    The conversion applies a frame transformation (180° about [1,1,0]/√2)
    that maps FRD/NED axes to Rerun's Z-up frame:
    - FRD X (forward) → Rerun Y (north)
    - FRD Y (starboard) → Rerun X (east)
    - FRD Z (down) → Rerun -Z (down)
    """

    def test_identity_quaternion(self):
        """fomodynamics identity [1, 0, 0, 0] -> Rerun identity [0, 0, 0, 1]."""
        fmd = np.array([1.0, 0.0, 0.0, 0.0])
        rerun = fmd_quat_to_rerun(fmd)
        np.testing.assert_allclose(rerun, [0.0, 0.0, 0.0, 1.0])

    def test_x_rotation(self):
        """90° rotation about FRD X (forward) -> rotation about Rerun Y (north)."""
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        fmd = np.array([c, s, 0.0, 0.0])
        rerun = fmd_quat_to_rerun(fmd)
        # FRD X -> Rerun Y, so rotation about X becomes rotation about Y
        np.testing.assert_allclose(rerun, [0.0, s, 0.0, c])

    def test_y_rotation(self):
        """90° rotation about FRD Y (starboard) -> rotation about Rerun X (east)."""
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        fmd = np.array([c, 0.0, s, 0.0])
        rerun = fmd_quat_to_rerun(fmd)
        # FRD Y -> Rerun X, so rotation about Y becomes rotation about X
        np.testing.assert_allclose(rerun, [s, 0.0, 0.0, c])

    def test_z_rotation(self):
        """90° rotation about FRD Z (down) -> rotation about Rerun -Z."""
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        fmd = np.array([c, 0.0, 0.0, s])
        rerun = fmd_quat_to_rerun(fmd)
        # FRD Z -> Rerun -Z, so rotation about Z becomes rotation about -Z
        np.testing.assert_allclose(rerun, [0.0, 0.0, -s, c])

    def test_arbitrary_quaternion(self):
        """Test with arbitrary unit quaternion."""
        fmd = np.array([0.5, 0.5, 0.5, 0.5])  # normalized
        rerun = fmd_quat_to_rerun(fmd)
        # Frame transformation swaps axes and negates Z component
        np.testing.assert_allclose(rerun, [0.5, 0.5, -0.5, 0.5])

    def test_batch_conversion(self):
        """Should work with (N, 4) arrays."""
        fmd = np.array([
            [1.0, 0.0, 0.0, 0.0],  # identity
            [0.5, 0.5, 0.5, 0.5],  # arbitrary
        ])
        rerun = fmd_quat_to_rerun(fmd)
        assert rerun.shape == (2, 4)
        np.testing.assert_allclose(rerun[0], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rerun[1], [0.5, 0.5, -0.5, 0.5])

    def test_preserves_norm(self):
        """Conversion should preserve quaternion norm."""
        fmd = np.array([0.7071, 0.7071, 0.0, 0.0])
        rerun = fmd_quat_to_rerun(fmd)
        np.testing.assert_allclose(np.linalg.norm(rerun), np.linalg.norm(fmd), atol=1e-10)

    def test_batch_preserves_norms(self):
        """Batch conversion should preserve all norms."""
        fmd = np.random.randn(10, 4)
        fmd = fmd / np.linalg.norm(fmd, axis=1, keepdims=True)  # normalize
        rerun = fmd_quat_to_rerun(fmd)
        norms_fmd = np.linalg.norm(fmd, axis=1)
        norms_rerun = np.linalg.norm(rerun, axis=1)
        np.testing.assert_allclose(norms_rerun, norms_fmd, atol=1e-10)


class TestMoth3dofToRerunQuat:
    """Tests for combined pitch + heel quaternion conversion."""

    def test_zero_heel_matches_pitch_only(self):
        """With heel_angle=0, output should match pitch_to_rerun_quat."""
        thetas = np.linspace(-np.pi / 4, np.pi / 4, 10)
        for theta in thetas:
            q_new = moth_3dof_to_rerun_quat(theta, heel_angle=0.0)
            q_old = pitch_to_rerun_quat(theta)
            np.testing.assert_allclose(q_new, q_old, atol=1e-14)

    def test_all_outputs_are_unit_quaternions(self):
        """All outputs should be unit quaternions."""
        thetas = np.linspace(-np.pi / 4, np.pi / 4, 20)
        for heel in [0.0, np.deg2rad(15), np.deg2rad(30), np.deg2rad(-15)]:
            quats = moth_3dof_to_rerun_quat(thetas, heel_angle=heel)
            norms = np.linalg.norm(quats, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_nonzero_heel_differs_from_pitch_only(self):
        """Non-zero heel should produce a different quaternion."""
        theta = np.deg2rad(5.0)
        q_no_heel = moth_3dof_to_rerun_quat(theta, heel_angle=0.0)
        q_with_heel = moth_3dof_to_rerun_quat(theta, heel_angle=np.deg2rad(30.0))
        assert not np.allclose(q_no_heel, q_with_heel), \
            "30° heel should change the quaternion"

    def test_batch_conversion(self):
        """Should work with arrays of pitch angles."""
        thetas = np.array([0.0, np.pi / 6, -np.pi / 6])
        quats = moth_3dof_to_rerun_quat(thetas, heel_angle=np.deg2rad(30.0))
        assert quats.shape == (3, 4)
        norms = np.linalg.norm(quats, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_zero_pitch_zero_heel_is_identity(self):
        """Zero pitch and zero heel should produce identity quaternion."""
        q = moth_3dof_to_rerun_quat(0.0, heel_angle=0.0)
        np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-15)
