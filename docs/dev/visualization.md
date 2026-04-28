# Visualization Reference

Plotting system architecture and Rerun 3D visualization.

**Related docs:**
- [../frame_conventions.md](../frame_conventions.md) — NED/FRD coordinate frame details

---

## Plotting System

`fmd.analysis.plots` is a 3-layer system for matplotlib-based visualization:

| Layer | Module | Takes | Use Case |
|-------|--------|-------|----------|
| L0 (Primitives) | `_primitives.py` | axes + arrays | Custom layouts, reusable across any data |
| L1 (Simulation) | `_simulation.py` | arrays + metadata | Tests and analysis scripts with model-agnostic data |
| L2 (Convenience) | `_convenience.py` | Result objects | Quick plotting of SimulationResult, LQGResult, etc. |

Additional modules:
- `_datastream.py`: Original `plot_time_series`/`plot_polar` for DataStream/DataFrame
- `_style.py`: Shared style (`BLUR_STYLE`, `style_axis`, `get_colors`, `savefig_and_close`)
- `_windowing.py`: Adaptive axis windowing for sweep/divergence plots

**Key constants** for Moth 3DOF: `MOTH_3DOF_STATE_LABELS`, `MOTH_3DOF_STATE_TRANSFORMS`, `MOTH_3DOF_CONTROL_LABELS`, `MOTH_3DOF_CONTROL_TRANSFORMS`.

**Import paths**: Both `from fmd.analysis.plots import plot_time_series` and `from fmd.analysis import plot_time_series` work.

## 3D Visualization (Rerun)

The `fmd.analysis.viz3d` module provides Rerun-based 3D visualization (included in default `uv sync`).

### Frame Conventions

BLUR and Rerun use different coordinate systems that require transformation:

| Frame | X | Y | Z | Handedness |
|-------|---|---|---|------------|
| BLUR Body (FRD) | Forward | Starboard (Right) | Down | Right-handed |
| BLUR World (NED) | North | East | Down | Right-handed |
| Rerun Display | East | North | Up | Right-handed, Z-up |

**Transformation (NED/FRD → Rerun):**
- Swap X and Y axes
- Negate Z axis
- This is a 180° rotation about the [1,1,0]/√2 axis

**Key functions in `fmd.analysis.viz3d.coordinates`:**
- `ned_to_rerun(pos)` — Transform positions: [N,E,D] → [E,N,-D]
- `frd_to_rerun(vec)` — Transform body vectors: [F,R,D] → [R,F,-D]
- `blur_quat_to_rerun(quat)` — Transform quaternions with frame conjugation

### Quaternion Transformation

BLUR quaternions represent rotation from body (FRD) to world (NED). For Rerun visualization, the quaternion must be conjugated by the frame transformation:

```python
q_rerun = q_frame ⊗ q_blur ⊗ q_frame⁻¹
```

Where `q_frame = [0, 1/√2, 1/√2, 0]` (180° about [1,1,0]/√2).

This ensures that:
- Roll (rotation about FRD X/forward) appears as rotation about Rerun Y (north)
- Pitch (rotation about FRD Y/starboard) appears as rotation about Rerun X (east)
- Yaw (rotation about FRD Z/down) appears as rotation about Rerun -Z
