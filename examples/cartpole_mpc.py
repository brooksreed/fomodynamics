"""Cartpole Waiter's Task — direct multiple-shooting MPC demo.

Problem: move a cartpole from rest at x=0 to x=3 m in 4 seconds while
keeping the pole within ±5° of vertical (the "Waiter's Task" — balance a
tray while walking across the room).

    State:   [x, x_dot, theta, theta_dot]   (m, m/s, rad, rad/s)
    Control: F (horizontal force on cart, N)

    Constraints:
        |F|     ≤ 10 N      (actuator limit)
        |theta| ≤ 5°        (pole-tilt constraint)
        terminal: x=3, x_dot=theta=theta_dot=0

This example uses fmd.ocp.MultipleShootingOCP with a CasADi cartpole
mirror and solves the NLP with IPOPT.

Usage:
    uv run python examples/cartpole_mpc.py
"""

import numpy as np

from fmd.ocp import MultipleShootingOCP, OCPResult, compute_control_smoothness
from fmd.simulator.casadi import CartpoleCasadiExact
from fmd.simulator.params import CARTPOLE_CLASSIC

# ── Problem parameters ─────────────────────────────────────────────────────────
N = 200          # discretisation steps
T = 4.0          # fixed time horizon (s)
F_MAX = 10.0     # control bound (N)
THETA_MAX = np.deg2rad(5.0)  # ±5° in radians
X_TARGET = 3.0   # target cart position (m)


def solve_waiters_task() -> OCPResult:
    model = CartpoleCasadiExact(CARTPOLE_CLASSIC)
    ocp = MultipleShootingOCP(model, N=N, T_fixed=T)

    ocp.set_initial_state([0.0, 0.0, 0.0, 0.0])
    ocp.set_terminal_state([X_TARGET, 0.0, 0.0, 0.0])
    ocp.set_control_bounds(-F_MAX, F_MAX)
    ocp.set_state_bounds(index=2, lower=-THETA_MAX, upper=THETA_MAX)

    # Strong terminal cost drives the solver to the target
    ocp.set_terminal_cost(1000 * np.eye(4))

    return ocp.solve()


def report(result: OCPResult) -> None:
    print("=" * 60)
    print("Cartpole Waiter's Task — MPC solution")
    print("=" * 60)

    status = "CONVERGED" if result.converged else "FAILED"
    print(f"  Status:           {status}")
    print(f"  Objective cost:   {result.cost:.4f}")
    print(f"  Solver iters:     {result.solver_stats.get('iter_count', '?')}")

    if not result.converged:
        print(f"  Return status:    {result.solver_stats.get('return_status', '?')}")

    # Trajectory summary
    x_traj = result.states[:, 0]
    theta_traj = result.states[:, 2]
    u_traj = result.controls[:, 0]

    print()
    print("  Trajectory summary:")
    print(f"    Cart position:    {x_traj[0]:.3f} → {x_traj[-1]:.3f} m  (target {X_TARGET} m)")
    print(f"    Max |theta|:      {np.rad2deg(np.max(np.abs(theta_traj))):.3f}°  (limit 5°)")
    print(f"    Max |force|:      {np.max(np.abs(u_traj)):.3f} N  (limit {F_MAX} N)")

    # Control smoothness
    smoothness = compute_control_smoothness(result)
    print(f"    Control rate:     {smoothness['mean_du_dt']:.4f} N/s  (mean |du/dt|)")

    print()
    print("  Constraint check:")
    theta_ok = np.max(np.abs(theta_traj)) <= THETA_MAX * 1.002  # 0.2% numerical tol
    force_ok = np.max(np.abs(u_traj)) <= F_MAX * 1.001
    print(f"    Pole constraint:  {'PASS' if theta_ok else 'FAIL'}")
    print(f"    Force constraint: {'PASS' if force_ok else 'FAIL'}")
    print("=" * 60)


def main() -> None:
    print("Solving Waiter's Task OCP (N=200, T=4s, IPOPT)...")
    result = solve_waiters_task()
    report(result)

    if not result.converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
