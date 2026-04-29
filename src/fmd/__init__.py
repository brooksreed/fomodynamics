"""fomodynamics — Foiling Moth Dynamics.

A library for simulation, control, and analysis of 3D vehicles, with a
focus on hydrofoiling moth sailboats. Provides JAX-based 6-DOF rigid body
dynamics, classical control (LQR), CasADi-based trim solving and MPC
infrastructure, telemetry analysis, and 3D visualization.

Subpackages:
    fmd.core: Shared math, units, and abstractions
    fmd.simulator: 6-DOF rigid body dynamics simulation
    fmd.analysis: Telemetry data loading, filtering, and plotting
    fmd.control: Control design (LQR)
    fmd.estimation: State estimation (Kalman filters, measurement models)
    fmd.ocp: Optimal control problem formulation (multiple shooting)
"""

__version__ = "0.1.0"
