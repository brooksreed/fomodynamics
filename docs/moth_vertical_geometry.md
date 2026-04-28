# Moth Vertical Geometry Reference

This document describes the vertical layout of the Moth 3DOF model, including
the hull-datum coordinate system and how body-frame positions are derived.

## Coordinate Systems

### Hull Datum
- **Origin**: Hull bottom, at bow
- **x**: Positive aft from bow (0 = bow)
- **y**: Positive starboard (same as body FRD)
- **z**: Positive up from hull bottom (0 = hull bottom)
- **Use**: Geometric source of truth for structural positions

### Body Frame (FRD)
- **Origin**: Boat CG (non-sailor hull/rig/foil CG)
- **x**: Positive forward
- **y**: Positive starboard
- **z**: Positive down
- **Use**: Dynamics reference frame

### System CG
- Boat CG + sailor mass contribution
- Offset from body origin by `combined_cg_offset`
- With default sailor (75 kg at [-0.30, 0, -0.2] body FRD):
  `combined_cg_offset = [-0.18, 0, -0.12]`

## Vertical Layout (MOTH_BIEKER_V3)

```
Body FRD z (down +)           Hull datum z (up +)

  z = -1.18  Sail CE          2.00m --- Sail CE (~1.55m above deck)
         |                        |
         |                        |
  z = -0.12  System CG        0.94m --- System CG (hull_contact_depth)
  z =  0.00  Boat CG (origin) 0.82m --- Boat CG (hull_cg_above_bottom)
         |                        |
  z = +0.37  Deck level        0.45m --- Deck (hull_depth)
         |                        |
  z = +0.82  Hull bottom       0.00m --- Hull bottom (hull datum origin)
         |                        |
  z = +1.77  Rudder foil      -0.95m --- Rudder strut depth
  z = +1.85  Main foil        -1.03m --- Main strut depth
```

## Key Dimensions

| Quantity | Value | Formula |
|----------|-------|---------|
| `hull_depth` | 0.45 m | Hull bottom to deck |
| `hull_cg_above_bottom` | 0.82 m | Hull bottom to boat CG |
| `hull_cg_from_bow` | 1.99 m | Bow to boat CG (aft, measured) |
| `main_foil_strut_depth` | 1.03 m | Hull bottom to main foil (measured) |
| `rudder_strut_depth` | 0.95 m | Hull bottom to rudder foil |
| `hull_contact_depth` | 0.94 m | System CG to hull bottom (computed) |
| `wing_rack_span` | 2.25 m | Tip-to-tip span |
| `wing_dihedral` | 30 deg | Wing rack dihedral angle |

## Frame Conversion

Hull-datum to body FRD:
```python
body_x = hull_cg_from_bow - datum_x    # aft-from-bow -> forward-from-CG
body_y = datum_y                         # unchanged
body_z = hull_cg_above_bottom - datum_z  # up-from-bottom -> down-from-CG
```

Body FRD to hull-datum (inverse):
```python
datum_x = hull_cg_from_bow - body_x
datum_y = body_y
datum_z = hull_cg_above_bottom - body_z
```

## Derived Body-Frame Positions

These are `@property` methods on `MothParams`, derived from hull-datum fields:

| Position | Hull-Datum Source | Body FRD Result |
|----------|------------------|-----------------|
| `main_foil_position` | `main_foil_from_bow=1.57`, `strut_depth=1.03` | [0.42, 0, 1.85] |
| `rudder_position` | `rudder_from_bow=3.855`, `strut_depth=0.95` | [-1.865, 0, 1.77] |
| `sail_ce_position` | `sail_ce_hull_datum=[2.5, 0, 2.0]` | [-0.51, 0, -1.18] |
| `bowsprit_position` | `bowsprit_hull_datum=[0, 0, 0.45]` | [1.99, 0, 0.37] |
| `wand_pivot_position` | `wand_pivot_hull_datum=[0, 0, 0.35]` | [1.99, 0, 0.47] |

## Hull Contact Depth

`hull_contact_depth` is a computed property:

```
hull_contact_depth = hull_cg_above_bottom - combined_cg_offset[2]
                   = 0.82 - (-0.12)
                   = 0.94 m  (unchanged; sailor z unchanged)
```

This is the distance from the **system CG** down to the hull bottom in body
frame z. The hull enters the water when `pos_d > -hull_contact_depth` (i.e.,
when the system CG is less than 0.94m above the waterline).

**Important**: This is the static default-sailor value. Runtime code with
`sailor_position_schedule(t)` should derive from the current CG offset,
not from `params.hull_contact_depth`.

## Operating Regime

With the hull-datum geometry, the foiling operating regime shifts significantly:

| Quantity | Value | Notes |
|----------|-------|-------|
| Hull enters water | `pos_d > -0.94` | `max(0, pos_d + hull_contact_depth)` |
| Normal foiling trim | `pos_d ~ -1.3` | Hull clear, foils submerged |
| Leeward tip at surface (30 deg heel) | `pos_d ~ -1.58` | Main foil tip ventilation onset |
| Main foil center at surface | `pos_d ~ -1.82` | Main foil fully ventilated |

Ventilation geometry at 30 deg heel:
- Main foil: `max_submergence = (0.95/2) * sin(30) = 0.2375 m`
- Rudder: `max_submergence = (0.68/2) * sin(30) = 0.17 m`
