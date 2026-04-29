"""Parametric wireframe geometry for Moth visualization.

Generates body-frame wireframe vertices from MothParams geometry.
All vertices are in body frame FRD [x_fwd, y_stbd, z_down].

No rerun dependency — pure numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_moth_wireframe(params) -> dict[str, NDArray]:
    """Generate body-frame wireframe vertices from MothParams.

    Returns a dict of named line strips, each an (N, 3) array of
    FRD body-frame coordinates [x_fwd, y_stbd, z_down].

    Args:
        params: MothParams instance with hull/foil geometry.

    Returns:
        Dict mapping component names to (N, 3) vertex arrays.
    """
    L = params.hull_length
    B = params.hull_beam

    # Hull geometry in body FRD frame:
    # CG is at body origin (0, 0, 0).
    # Hull bottom is at z = hull_cg_above_bottom (positive = below CG).
    # Hull deck is at z = hull_cg_above_bottom - hull_depth.
    hull_bottom_z = params.hull_cg_above_bottom
    deck_z = hull_bottom_z - params.hull_depth

    # Hull side profile (sagittal plane, y=0)
    # Bow → bottom → transom → deck → bow
    bow_x = L / 2
    stern_x = -L / 2
    hull_profile = np.array([
        [bow_x, 0.0, deck_z],                   # bow deck
        [bow_x * 0.8, 0.0, hull_bottom_z],      # bow bottom curve
        [stern_x, 0.0, hull_bottom_z],           # stern bottom
        [stern_x, 0.0, deck_z],                  # transom deck
        [bow_x, 0.0, deck_z],                    # close loop
    ])

    # Hull deck plan (top view, z=deck_z)
    # Bow → port → stern → starboard → bow
    hull_deck = np.array([
        [bow_x, 0.0, deck_z],
        [bow_x * 0.5, -B / 2, deck_z],
        [stern_x * 0.8, -B / 2, deck_z],
        [stern_x, 0.0, deck_z],
        [stern_x * 0.8, B / 2, deck_z],
        [bow_x * 0.5, B / 2, deck_z],
        [bow_x, 0.0, deck_z],
    ])

    # Main foil (horizontal T-foil) — chord-width rectangle in X-Y plane
    foil_pos = params.main_foil_position  # [x, y, z] in FRD
    half_span = params.main_foil_span / 2
    half_fc = params.main_foil_chord / 2
    main_foil = np.array([
        [foil_pos[0] + half_fc, -half_span, foil_pos[2]],  # LE port
        [foil_pos[0] + half_fc,  half_span, foil_pos[2]],  # LE stbd
        [foil_pos[0] - half_fc,  half_span, foil_pos[2]],  # TE stbd
        [foil_pos[0] - half_fc, -half_span, foil_pos[2]],  # TE port
        [foil_pos[0] + half_fc, -half_span, foil_pos[2]],  # close
    ])

    # Main foil strut (vertical) — chord-width rectangle in X-Z plane
    # Strut extends from hull bottom down to the foil
    half_sc = params.main_strut_chord / 2
    main_foil_strut = np.array([
        [foil_pos[0] + half_sc, 0.0, hull_bottom_z],    # fwd top (hull bottom)
        [foil_pos[0] + half_sc, 0.0, foil_pos[2]],      # fwd bottom (foil)
        [foil_pos[0] - half_sc, 0.0, foil_pos[2]],      # aft bottom
        [foil_pos[0] - half_sc, 0.0, hull_bottom_z],     # aft top
        [foil_pos[0] + half_sc, 0.0, hull_bottom_z],     # close
    ])

    # Rudder (horizontal T-foil at stern) — chord-width rectangle in X-Y plane
    rudder_pos = params.rudder_position
    half_rudder = params.rudder_span / 2
    half_rc = params.rudder_chord / 2
    rudder = np.array([
        [rudder_pos[0] + half_rc, -half_rudder, rudder_pos[2]],  # LE port
        [rudder_pos[0] + half_rc,  half_rudder, rudder_pos[2]],  # LE stbd
        [rudder_pos[0] - half_rc,  half_rudder, rudder_pos[2]],  # TE stbd
        [rudder_pos[0] - half_rc, -half_rudder, rudder_pos[2]],  # TE port
        [rudder_pos[0] + half_rc, -half_rudder, rudder_pos[2]],  # close
    ])

    # Rudder strut — chord-width rectangle in X-Z plane
    half_rsc = params.rudder_strut_chord / 2
    rudder_strut = np.array([
        [rudder_pos[0] + half_rsc, 0.0, hull_bottom_z],      # fwd top
        [rudder_pos[0] + half_rsc, 0.0, rudder_pos[2]],      # fwd bottom
        [rudder_pos[0] - half_rsc, 0.0, rudder_pos[2]],      # aft bottom
        [rudder_pos[0] - half_rsc, 0.0, hull_bottom_z],       # aft top
        [rudder_pos[0] + half_rsc, 0.0, hull_bottom_z],       # close
    ])

    # Wing racks with dihedral — two lines from hull centerline angling upward
    # Wing racks attach at approximately CG height and angle upward (dihedral).
    # In FRD, "up" is negative z.
    half_rack = params.wing_rack_span / 2
    dihedral = params.wing_dihedral
    rack_tip_y = half_rack * np.cos(dihedral)
    rack_tip_z = -half_rack * np.sin(dihedral)  # negative z = upward in FRD
    rack_x = params.hull_datum_to_body(np.array([params.wing_rack_from_bow, 0.0, 0.0]))[0]
    rack_z_base = deck_z  # racks attach at deck level
    wing_rack_port = np.array([
        [rack_x, 0.0, rack_z_base],
        [rack_x, -rack_tip_y, rack_z_base + rack_tip_z],
    ])
    wing_rack_stbd = np.array([
        [rack_x, 0.0, rack_z_base],
        [rack_x, rack_tip_y, rack_z_base + rack_tip_z],
    ])

    # Sail/mast line (deck to sail CE)
    sail_pos = params.sail_ce_position  # z is negative (above) in FRD
    sail_mast = np.array([
        [0.0, 0.0, deck_z],  # mast base at deck
        [sail_pos[0], sail_pos[1], sail_pos[2]],  # sail CE
    ])

    return {
        "hull_profile": hull_profile,
        "hull_deck": hull_deck,
        "main_foil": main_foil,
        "main_foil_strut": main_foil_strut,
        "rudder": rudder,
        "rudder_strut": rudder_strut,
        "wing_rack_port": wing_rack_port,
        "wing_rack_stbd": wing_rack_stbd,
        "sail_mast": sail_mast,
    }


def compute_surface_waterline(
    vertices_body: NDArray,
    pos_d: float,
    theta: float,
    heel_angle: float = 0.0,
    *,
    water_elevation_fn=None,
    boat_n: float = 0.0,
    boat_e: float = 0.0,
    t: float = 0.0,
) -> NDArray | None:
    """Compute where the water surface intersects a body-frame rectangle.

    Uses an edge-intersection algorithm: for each of the 4 edges of the
    rectangle, check if the endpoints straddle the water surface. If so,
    interpolate to find the intersection point. Returns the pair of
    intersection points as a body-frame line segment.

    Args:
        vertices_body: (4, 3) array of rectangle corners in FRD body frame.
        pos_d: Vertical position in NED (positive = deeper).
        theta: Pitch angle in radians.
        heel_angle: Heel (roll) angle in radians. Default 0.
        water_elevation_fn: Optional callable(n, e, t) -> eta for wave fields.
        boat_n: Boat north position in world frame.
        boat_e: Boat east position in world frame.
        t: Simulation time (for wave queries).

    Returns:
        (2, 3) array of FRD body-frame intersection points, or None if
        the water surface does not cross the rectangle. Returns None for
        degenerate cases (vertex exactly on water, or more/fewer than 2
        edge intersections).
    """
    vertices_body = np.asarray(vertices_body)
    if vertices_body.shape != (4, 3):
        raise ValueError(f"vertices_body must be (4, 3), got {vertices_body.shape}")

    # Build body->world rotation matrix R from theta (pitch) and heel (roll)
    # R = Ry(theta) @ Rx(heel)  (NED convention: pitch then roll)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_h = np.cos(heel_angle)
    sin_h = np.sin(heel_angle)

    R = np.array([
        [cos_t,  sin_t * sin_h,  sin_t * cos_h],
        [0.0,    cos_h,          -sin_h],
        [-sin_t, cos_t * sin_h,  cos_t * cos_h],
    ])

    # Transform vertices to world NED
    world = (R @ vertices_body.T).T
    world[:, 2] += pos_d  # add boat depth

    # Compute signed distance to water for each vertex
    d = np.empty(4)
    for k in range(4):
        if water_elevation_fn is not None:
            n_world = boat_n + world[k, 0]
            e_world = boat_e + world[k, 1]
            eta = water_elevation_fn(n_world, e_world, t)
            water_level = -eta
        else:
            water_level = 0.0
        d[k] = world[k, 2] - water_level

    # Find edge intersections (edges: 0-1, 1-2, 2-3, 3-0)
    intersections = []
    for i_edge in range(4):
        j_edge = (i_edge + 1) % 4
        if d[i_edge] * d[j_edge] < 0:  # endpoints straddle water
            t_param = d[i_edge] / (d[i_edge] - d[j_edge])
            pt = vertices_body[i_edge] + t_param * (vertices_body[j_edge] - vertices_body[i_edge])
            intersections.append(pt)

    if len(intersections) == 2:
        return np.array(intersections)
    return None
