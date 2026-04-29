#!/usr/bin/env python3
"""Bieker Moth V3 — engineering geometry reference drawing.

Plots front and side profiles of the Moth with all key dimensions,
derived from the MOTH_BIEKER_V3 preset. Uses hull-datum plotting frame
(x = distance aft from bow, y = height above hull bottom).

Shows both the boat CG (body frame origin) and system CG (boat + sailor).

Usage:
    uv run python examples/moth_geometry.py
    uv run python examples/moth_geometry.py --output-dir results/custom/
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Rectangle

from fmd.simulator.params.presets import MOTH_BIEKER_V3

# --- Dimensions from preset ---
p = MOTH_BIEKER_V3

hull_length = p.hull_length
hull_width = p.hull_beam
hull_depth = p.hull_depth
beam_total = p.wing_rack_span

# Strut geometry from preset
strut_length = p.main_foil_strut_depth
main_foil_pos = p.main_foil_from_bow
rudder_strut_depth_val = p.rudder_strut_depth
main_foil_span = p.main_foil_span
main_foil_chord = p.main_foil_chord
main_strut_chord = p.main_strut_chord
rudder_span = p.rudder_span
rudder_chord = p.rudder_chord
rudder_strut_chord = p.rudder_strut_chord

# CG from preset
cg_height = p.hull_cg_above_bottom
cg_from_bow = p.hull_cg_from_bow

# Wing geometry from preset
dihedral_rad = p.wing_dihedral
dihedral_deg = np.degrees(dihedral_rad)

# System CG (boat + sailor)
system_cg_offset = p.combined_cg_offset  # body FRD
# Convert system CG offset to hull-datum: datum_z = hull_cg_above_bottom - body_z
system_cg_height = cg_height - system_cg_offset[2]
system_cg_from_bow = cg_from_bow - system_cg_offset[0]

# --- Dimensions NOT in preset (sail geometry, appendages) ---
bowsprit_length = 0.5  # extends forward of bow
rudder_gantry_length = 0.5  # extends aft of transom
mast_pos = 1.2  # from bow (not in preset — preset only has sail CE)
mast_height = 5.185  # full mast height (not in preset)
boom_length = 1.8  # approximate boom/foot length
head_width = 0.4  # square-top head width

# --- Derived ---
rack_edge = beam_total / 2
wing_tip_height = hull_depth + (rack_edge - hull_width / 2) * np.tan(dihedral_rad)
rudder_strut_pos = p.rudder_from_bow

# Hull side profile — pointed bow at mid-height, flat bottom, square transom
bow_tip_y = hull_depth * 0.5
hull_side_x = [
    0,                     # bow tip
    hull_length * 0.12,    # upper bow curve
    hull_length * 0.30,    # deck start
    hull_length,           # transom top
    hull_length,           # transom bottom
    hull_length * 0.30,    # flat bottom start
    hull_length * 0.12,    # lower bow curve
    0,                     # bow tip (close)
]
hull_side_y = [
    bow_tip_y,             # bow tip
    hull_depth * 0.88,     # upper bow
    hull_depth,            # deck level
    hull_depth,            # transom top
    0,                     # transom bottom
    0,                     # flat bottom
    hull_depth * 0.08,     # lower bow
    bow_tip_y,             # bow tip (close)
]


def main(output_dir: str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==========================================
    # FRONT PROFILE (ax1)
    # ==========================================
    # Displacement waterline
    waterline_y = 0.05
    ax1.axhline(y=waterline_y, color="blue", linestyle="--", alpha=0.6, label="Waterline")

    # Hull cross-section
    ax1.plot(
        [-hull_width / 2, hull_width / 2, hull_width / 2, -hull_width / 2, -hull_width / 2],
        [hull_depth, hull_depth, 0, 0, hull_depth],
        "k-",
        lw=2,
        label="Hull",
    )

    # Wing racks (with dihedral)
    ax1.plot(
        [hull_width / 2, rack_edge],
        [hull_depth, wing_tip_height],
        "k-",
        lw=2,
        label="Wings",
    )
    ax1.plot(
        [-hull_width / 2, -rack_edge],
        [hull_depth, wing_tip_height],
        "k-",
        lw=2,
    )

    # Dihedral angle annotation (on starboard wing)
    arc_radius = 0.35
    arc = Arc(
        (hull_width / 2, hull_depth),
        arc_radius * 2,
        arc_radius * 2,
        angle=0,
        theta1=0,
        theta2=dihedral_deg,
        color="red",
        lw=1.5,
    )
    ax1.add_patch(arc)
    arc_label_angle = np.radians(dihedral_deg / 2)
    ax1.annotate(
        f"{dihedral_deg:.0f}\u00b0",
        xy=(
            hull_width / 2 + arc_radius * 1.15 * np.cos(arc_label_angle),
            hull_depth + arc_radius * 1.15 * np.sin(arc_label_angle),
        ),
        fontsize=10,
        color="red",
        ha="left",
        va="bottom",
    )

    # Main foil strut (edge-on in front view)
    ax1.plot([0, 0], [0, -strut_length], "k-", lw=2, label="Foil strut")

    # Main foil
    foil_thickness_front = 0.02
    main_foil_rect = Rectangle(
        (-main_foil_span / 2, -strut_length - foil_thickness_front / 2),
        main_foil_span,
        foil_thickness_front,
        linewidth=1.5,
        edgecolor="k",
        facecolor="k",
    )
    ax1.add_patch(main_foil_rect)
    ax1.text(0, -strut_length - 0.06, "Main foil", fontsize=7, ha="center", va="top")

    # Boat CG marker (body frame origin)
    ax1.plot(
        0,
        cg_height,
        "o",
        color="royalblue",
        markersize=10,
        markeredgewidth=1.5,
        markerfacecolor="none",
        label=f"Boat CG ({cg_height:.2f} m)",
    )

    # System CG marker (boat + sailor)
    ax1.plot(
        0,
        system_cg_height,
        "r+",
        markersize=14,
        markeredgewidth=2.5,
        label=f"System CG ({system_cg_height:.2f} m)",
    )

    ax1.set_title("Front Profile")
    ax1.set_xlabel("Width (m)")
    ax1.set_ylabel("Height relative to hull bottom (m)")
    ax1.axis("equal")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="upper right", fontsize=8)

    # ==========================================
    # SIDE PROFILE (ax2)
    # ==========================================
    waterline_y = 0.05
    ax2.axhline(y=waterline_y, color="blue", linestyle="--", alpha=0.6, label="Waterline")

    # Hull side profile
    ax2.plot(hull_side_x, hull_side_y, "k-", lw=2, label="Hull")

    # Bowsprit
    ax2.plot(
        [0, -bowsprit_length],
        [bow_tip_y, bow_tip_y],
        "k-",
        lw=1.5,
        label=f"Bowsprit ({bowsprit_length} m)",
    )

    # Rudder gantry
    gantry_y = 0
    ax2.plot(
        [hull_length, hull_length + rudder_gantry_length],
        [gantry_y, gantry_y],
        "k-",
        lw=1.5,
        label=f"Rudder gantry ({rudder_gantry_length} m)",
    )

    # Main foil strut (from hull bottom to strut depth below hull bottom)
    main_strut_bottom = -strut_length
    main_strut_rect = Rectangle(
        (main_foil_pos - main_strut_chord / 2, main_strut_bottom),
        main_strut_chord,
        strut_length,  # from hull bottom (y=0) to -strut_length
        linewidth=1.5,
        edgecolor="k",
        facecolor="none",
    )
    ax2.add_patch(main_strut_rect)

    # Main foil (side view)
    foil_thickness_side = 0.015
    main_foil_rect_side = Rectangle(
        (main_foil_pos - main_foil_chord / 2, main_strut_bottom - foil_thickness_side / 2),
        main_foil_chord,
        foil_thickness_side,
        linewidth=1.5,
        edgecolor="k",
        facecolor="k",
        label="Main foil",
    )
    ax2.add_patch(main_foil_rect_side)

    # Rudder strut (from gantry to strut depth below hull bottom)
    rudder_strut_bottom = -rudder_strut_depth_val
    rudder_strut_rect = Rectangle(
        (rudder_strut_pos - rudder_strut_chord / 2, rudder_strut_bottom),
        rudder_strut_chord,
        gantry_y - rudder_strut_bottom,
        linewidth=1.5,
        edgecolor="k",
        facecolor="none",
    )
    ax2.add_patch(rudder_strut_rect)

    # Rudder foil (side view)
    rudder_foil_rect_side = Rectangle(
        (rudder_strut_pos - rudder_chord / 2, rudder_strut_bottom - foil_thickness_side / 2),
        rudder_chord,
        foil_thickness_side,
        linewidth=1.5,
        edgecolor="k",
        facecolor="k",
        label="Rudder",
    )
    ax2.add_patch(rudder_foil_rect_side)

    # Mast
    ax2.plot(
        [mast_pos, mast_pos],
        [hull_depth, hull_depth + mast_height],
        "g-",
        lw=2,
        label="Mast",
    )

    # Sail plan
    mast_top = hull_depth + mast_height
    mast_base = hull_depth
    n_leech = 20
    t = np.linspace(0, 1, n_leech)
    leech_top_x = mast_pos + head_width
    leech_top_y = mast_top
    clew_x = mast_pos + boom_length
    clew_y = mast_base
    roach = 0.25
    leech_x = leech_top_x + t * (clew_x - leech_top_x) + roach * np.sin(np.pi * t)
    leech_y = leech_top_y + t * (clew_y - leech_top_y)

    sail_x = np.concatenate([
        [mast_pos], [mast_pos], [leech_top_x],
        leech_x, [clew_x], [mast_pos],
    ])
    sail_y = np.concatenate([
        [mast_base], [mast_top], [mast_top],
        leech_y, [clew_y], [mast_base],
    ])
    ax2.plot(sail_x, sail_y, "g--", lw=1.5, label="Sail plan")

    # Boat CG marker (body frame origin)
    ax2.plot(
        cg_from_bow,
        cg_height,
        "o",
        color="royalblue",
        markersize=10,
        markeredgewidth=1.5,
        markerfacecolor="none",
        label=f"Boat CG ({cg_height:.2f} m)",
    )

    # System CG marker (boat + sailor)
    ax2.plot(
        system_cg_from_bow,
        system_cg_height,
        "r+",
        markersize=14,
        markeredgewidth=2.5,
        label=f"System CG ({system_cg_height:.2f} m)",
    )

    ax2.set_title("Side Profile")
    ax2.set_xlabel("Length from Bow (m)")
    ax2.axis("equal")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle("Bieker Moth V3 — Engineering Geometry", fontsize=16)
    plt.tight_layout()

    # Save output
    if output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join("results", "moth-geometry", ts)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "moth_geometry.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print coordinate verification
    print("\n--- Coordinate Verification ---")
    print(f"Boat CG:    ({cg_from_bow:.2f}, {cg_height:.2f})")
    print(f"System CG:  ({system_cg_from_bow:.2f}, {system_cg_height:.2f})")
    print(f"Main foil:  ({main_foil_pos:.2f}, {-strut_length:.2f}) below hull bottom")
    print(f"Rudder:     ({rudder_strut_pos:.3f}, {-rudder_strut_depth_val:.2f}) below hull bottom")
    print(f"Wing tip height: {wing_tip_height:.3f} m")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Bieker Moth V3 geometry")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/moth-geometry/<timestamp>/)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
