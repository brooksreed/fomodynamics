"""Tests for physical constraints (KeepOutZone, HalfSpaceConstraint)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fmd.simulator.constraints import (
    KeepOutZone,
    HalfSpaceConstraint,
    Capability,
    ConstraintCategory,
    ConstraintTestHelper,
)


class TestKeepOutZone:
    """Tests for KeepOutZone (spherical exclusion)."""

    def test_creation(self):
        """Valid KeepOutZone should be created."""
        c = KeepOutZone(
            name="zone1",
            center=jnp.array([10.0, 20.0, 30.0]),
            radius=5.0,
        )
        assert c.name == "zone1"
        assert c.radius == 5.0
        assert c.category == ConstraintCategory.SAFETY
        np.testing.assert_array_equal(c.center, [10.0, 20.0, 30.0])

    def test_invalid_radius_raises(self):
        """Non-positive radius should raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            KeepOutZone("bad", center=jnp.array([0.0, 0.0, 0.0]), radius=0.0)

        with pytest.raises(ValueError, match="radius must be positive"):
            KeepOutZone("bad", center=jnp.array([0.0, 0.0, 0.0]), radius=-1.0)

    def test_center_pos_indices_mismatch_raises(self):
        """Mismatched center and pos_indices should raise ValueError."""
        with pytest.raises(ValueError, match="center has .* elements but pos_indices"):
            KeepOutZone(
                "bad",
                center=jnp.array([0.0, 0.0, 0.0]),
                radius=1.0,
                pos_indices=(0, 1),  # 2 indices but 3-element center
            )

    def test_capabilities(self):
        """Should have HAS_SYMBOLIC_FORM but not HARD_CLIP."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        assert c.has_capability(Capability.HAS_SYMBOLIC_FORM)
        assert not c.has_capability(Capability.HARD_CLIP)

    def test_value_outside_zone(self):
        """Point outside zone should satisfy constraint (value <= 0)."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        # Point at (2, 0, 0) is distance 2 from origin, outside radius 1
        x = jnp.array([2.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val < 0  # radius - dist = 1 - 2 = -1

    def test_value_on_boundary(self):
        """Point on boundary should have value = 0."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        x = jnp.array([1.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert jnp.abs(val) < 1e-10

    def test_value_inside_zone(self):
        """Point inside zone should violate constraint (value > 0)."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        x = jnp.array([0.5, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val > 0  # radius - dist = 1 - 0.5 = 0.5

    def test_value_at_center(self):
        """Point at center should have maximum violation."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=5.0)
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val == 5.0  # radius - 0 = 5

    def test_custom_pos_indices(self):
        """Should use custom position indices."""
        # State where position is at indices 3, 4, 5
        c = KeepOutZone(
            "test",
            center=jnp.array([0.0, 0.0, 0.0]),
            radius=1.0,
            pos_indices=(3, 4, 5),
        )
        x = jnp.array([99.0, 99.0, 99.0, 2.0, 0.0, 0.0])  # pos at indices 3-5
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val < 0  # Outside zone

    def test_2d_zone(self):
        """Should work with 2D positions."""
        c = KeepOutZone(
            "test_2d",
            center=jnp.array([0.0, 0.0]),
            radius=1.0,
            pos_indices=(0, 1),
        )
        x = jnp.array([2.0, 0.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val < 0

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        ConstraintTestHelper.check_jit_compatible(c)

    def test_vmap_compatible(self):
        """Should be vmap compatible."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        ConstraintTestHelper.check_vmap_compatible(c)

    def test_differentiable(self):
        """Should be differentiable."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        x = jnp.zeros(13)
        x = x.at[0].set(2.0)  # Set pos_n = 2
        u = jnp.zeros(4)
        grad_x, grad_u = ConstraintTestHelper.check_differentiable(c, 0.0, x, u)
        # Gradient should point away from center
        assert grad_x[0] != 0

    def test_sign_convention(self):
        """Should follow <= 0 satisfied (outside), > 0 violated (inside)."""
        c = KeepOutZone("test", center=jnp.array([0.0, 0.0, 0.0]), radius=1.0)
        u = jnp.array([])

        ConstraintTestHelper.check_sign_convention(
            c,
            feasible_points=[
                (0.0, jnp.array([2.0, 0.0, 0.0]), u),
                (0.0, jnp.array([1.0, 0.0, 0.0]), u),  # On boundary
                (0.0, jnp.array([0.0, 1.5, 0.0]), u),
            ],
            infeasible_points=[
                (0.0, jnp.array([0.5, 0.0, 0.0]), u),
                (0.0, jnp.array([0.0, 0.0, 0.0]), u),  # At center
            ],
        )


class TestHalfSpaceConstraint:
    """Tests for HalfSpaceConstraint (plane constraint)."""

    def test_creation(self):
        """Valid HalfSpaceConstraint should be created."""
        c = HalfSpaceConstraint(
            name="plane",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
        )
        assert c.name == "plane"
        assert c.offset == 0.0
        assert c.category == ConstraintCategory.PHYSICAL

    def test_normal_normalized(self):
        """Normal should be normalized."""
        c = HalfSpaceConstraint(
            name="plane",
            normal=jnp.array([0.0, 0.0, 2.0]),  # Not unit length
            offset=0.0,
        )
        np.testing.assert_allclose(jnp.linalg.norm(c.normal), 1.0)
        np.testing.assert_allclose(c.normal, [0.0, 0.0, 1.0])

    def test_zero_normal_raises(self):
        """Zero normal vector should raise ValueError."""
        with pytest.raises(ValueError, match="normal vector cannot be zero"):
            HalfSpaceConstraint("bad", normal=jnp.array([0.0, 0.0, 0.0]), offset=0.0)

    def test_normal_pos_indices_mismatch_raises(self):
        """Mismatched normal and pos_indices should raise ValueError."""
        with pytest.raises(ValueError, match="normal has .* elements but pos_indices"):
            HalfSpaceConstraint(
                "bad",
                normal=jnp.array([0.0, 0.0, 1.0]),
                offset=0.0,
                pos_indices=(0, 1),
            )

    def test_capabilities(self):
        """Should have HAS_SYMBOLIC_FORM and HARD_CLIP."""
        c = HalfSpaceConstraint("test", normal=jnp.array([0.0, 0.0, 1.0]), offset=0.0)
        assert c.has_capability(Capability.HAS_SYMBOLIC_FORM)
        assert c.has_capability(Capability.HARD_CLIP)

    def test_ground_plane_above(self):
        """Point above ground (pos_d < 0 in NED) should satisfy constraint."""
        # Ground at pos_d=0, normal pointing down (+D), constraint: pos_d <= 0
        c = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
        )
        x = jnp.array([0.0, 0.0, -1.0])  # 1m above ground in NED
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val < 0  # normal.dot(pos) - offset = -1 - 0 = -1

    def test_ground_plane_below(self):
        """Point below ground (pos_d > 0 in NED) should violate constraint."""
        c = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
        )
        x = jnp.array([0.0, 0.0, 1.0])  # 1m below ground in NED
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val > 0  # normal.dot(pos) - offset = 1 - 0 = 1

    def test_altitude_ceiling(self):
        """Altitude ceiling constraint should work."""
        # Stay below 100m altitude: pos_d >= -100 (in NED, up is negative)
        # Equivalently: -pos_d <= 100, or normal=[-1] on pos_d, offset=100
        c = HalfSpaceConstraint(
            name="ceiling",
            normal=jnp.array([0.0, 0.0, -1.0]),
            offset=100.0,
        )
        # At 50m altitude: pos_d = -50
        x_ok = jnp.array([0.0, 0.0, -50.0])
        val_ok = c.value(0.0, x_ok, jnp.array([]))
        assert val_ok < 0  # -(-50) - 100 = 50 - 100 = -50

        # At 150m altitude: pos_d = -150
        x_bad = jnp.array([0.0, 0.0, -150.0])
        val_bad = c.value(0.0, x_bad, jnp.array([]))
        assert val_bad > 0  # -(-150) - 100 = 150 - 100 = 50

    def test_arbitrary_plane(self):
        """Should work with arbitrary plane orientations."""
        # Plane: x + y + z = 1
        c = HalfSpaceConstraint(
            name="diagonal",
            normal=jnp.array([1.0, 1.0, 1.0]),
            offset=1.0 * jnp.sqrt(3.0),  # Adjusted for normalized normal
        )
        # Point at origin should be on negative side
        x_origin = jnp.array([0.0, 0.0, 0.0])
        val_origin = c.value(0.0, x_origin, jnp.array([]))
        assert val_origin < 0

    def test_custom_pos_indices(self):
        """Should use custom position indices."""
        c = HalfSpaceConstraint(
            name="plane",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
            pos_indices=(3, 4, 5),
        )
        x = jnp.array([99.0, 99.0, 99.0, 0.0, 0.0, -1.0])
        u = jnp.array([])
        val = c.value(0.0, x, u)
        assert val < 0

    def test_2d_line(self):
        """Should work with 2D positions (line constraint)."""
        # Line: x = 1
        c = HalfSpaceConstraint(
            name="line",
            normal=jnp.array([1.0, 0.0]),
            offset=1.0,
            pos_indices=(0, 1),
        )
        x_left = jnp.array([0.0, 0.0])
        x_right = jnp.array([2.0, 0.0])
        u = jnp.array([])

        assert c.value(0.0, x_left, u) < 0  # x=0 is left of line
        assert c.value(0.0, x_right, u) > 0  # x=2 is right of line

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        c = HalfSpaceConstraint("test", normal=jnp.array([0.0, 0.0, 1.0]), offset=0.0)
        ConstraintTestHelper.check_jit_compatible(c)

    def test_vmap_compatible(self):
        """Should be vmap compatible."""
        c = HalfSpaceConstraint("test", normal=jnp.array([0.0, 0.0, 1.0]), offset=0.0)
        ConstraintTestHelper.check_vmap_compatible(c)

    def test_differentiable(self):
        """Should be differentiable."""
        c = HalfSpaceConstraint("test", normal=jnp.array([0.0, 0.0, 1.0]), offset=0.0)
        x = jnp.zeros(13)
        x = x.at[2].set(-1.0)  # pos_d = -1
        u = jnp.zeros(4)
        grad_x, grad_u = ConstraintTestHelper.check_differentiable(c, 0.0, x, u)
        # Gradient should be in direction of normal
        assert grad_x[2] != 0

    def test_sign_convention(self):
        """Should follow <= 0 satisfied, > 0 violated convention."""
        # Ground plane: pos_d <= 0
        c = HalfSpaceConstraint("ground", normal=jnp.array([0.0, 0.0, 1.0]), offset=0.0)
        u = jnp.array([])

        ConstraintTestHelper.check_sign_convention(
            c,
            feasible_points=[
                (0.0, jnp.array([0.0, 0.0, -1.0]), u),  # Above ground
                (0.0, jnp.array([0.0, 0.0, 0.0]), u),  # On ground
            ],
            infeasible_points=[
                (0.0, jnp.array([0.0, 0.0, 1.0]), u),  # Below ground
            ],
        )
    def test_clip_feasible_unchanged(self):
        """Feasible state should remain unchanged after clip."""
        # Ground plane constraint: pos_d <= 0
        c = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
        )
        x = jnp.array([1.0, 2.0, -1.0])  # Above ground (feasible)
        u = jnp.array([0.5])
        
        x_clipped, u_clipped = c.clip(0.0, x, u)
        
        np.testing.assert_allclose(x_clipped, x)
        np.testing.assert_allclose(u_clipped, u)

    def test_clip_violated_projected(self):
        """Violated state should be projected to boundary."""
        # Ground plane constraint: pos_d <= 0
        c = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
        )
        x = jnp.array([1.0, 2.0, 2.0])  # Below ground (pos_d=2, violated)
        u = jnp.array([0.5])
        
        x_clipped, u_clipped = c.clip(0.0, x, u)
        
        # pos_d should be clipped to 0
        np.testing.assert_allclose(x_clipped[0], 1.0)  # pos_n unchanged
        np.testing.assert_allclose(x_clipped[1], 2.0)  # pos_e unchanged
        np.testing.assert_allclose(x_clipped[2], 0.0)  # pos_d clipped to boundary
        np.testing.assert_allclose(u_clipped, u)  # control unchanged

    def test_clip_on_boundary_unchanged(self):
        """State exactly on boundary should remain unchanged."""
        c = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=0.0,
        )
        x = jnp.array([1.0, 2.0, 0.0])  # On boundary (pos_d=0)
        u = jnp.array([])
        
        x_clipped, u_clipped = c.clip(0.0, x, u)
        
        np.testing.assert_allclose(x_clipped, x)

    def test_clip_arbitrary_plane(self):
        """Clip should work for arbitrary plane orientations."""
        # Plane at x=5, forbidding x>5
        c = HalfSpaceConstraint(
            name="wall",
            normal=jnp.array([1.0, 0.0, 0.0]),
            offset=5.0,
        )
        x = jnp.array([7.0, 2.0, 3.0])  # x=7 violates x<=5
        u = jnp.array([])
        
        x_clipped, u_clipped = c.clip(0.0, x, u)
        
        np.testing.assert_allclose(x_clipped[0], 5.0)  # x clipped to 5
        np.testing.assert_allclose(x_clipped[1], 2.0)  # y unchanged
        np.testing.assert_allclose(x_clipped[2], 3.0)  # z unchanged

    def test_clip_diagonal_plane(self):
        """Clip should work for diagonal planes."""
        # Plane: x + y <= sqrt(2) (45 degree plane)
        c = HalfSpaceConstraint(
            name="diagonal",
            normal=jnp.array([1.0, 1.0]),
            offset=jnp.sqrt(2.0),
            pos_indices=(0, 1),
        )
        # Point at (2, 2) violates x + y <= sqrt(2) * sqrt(2) = 2
        # Actually after normalization: (1/sqrt(2), 1/sqrt(2)).dot((2,2)) - sqrt(2)
        # = 2*sqrt(2) - sqrt(2) = sqrt(2) violation
        x = jnp.array([2.0, 2.0])
        u = jnp.array([])
        
        x_clipped, u_clipped = c.clip(0.0, x, u)
        
        # Should be projected back along normal direction
        # The clipped value should satisfy the constraint
        val = c.value(0.0, x_clipped, u)
        np.testing.assert_allclose(val, 0.0, atol=1e-6)

    def test_clip_jit_compatible(self):
        """Clip should be JIT compatible."""
        import equinox as eqx
        c = HalfSpaceConstraint("test", normal=jnp.array([0.0, 0.0, 1.0]), offset=0.0)
        clip_jit = eqx.filter_jit(c.clip)
        x = jnp.array([0.0, 0.0, 2.0])  # Violated
        u = jnp.array([0.1])
        
        x_clipped, u_clipped = clip_jit(0.0, x, u)
        np.testing.assert_allclose(x_clipped[2], 0.0)
