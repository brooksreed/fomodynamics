# BLUR Timestep Guide

This document covers timestep selection, RK4 stability, and the timestep architecture across BLUR's Moth 3DOF simulation and control stack.

**Related Documentation:**
- [control_guide.md](control_guide.md) - LQR design, integrator selection, tuning
- [simulator_models.md](simulator_models.md) - Model state vectors and dynamics

---

## 1. RK4 Stability Theory

The RK4 integrator is stable when all eigenvalues of the system's Jacobian satisfy:

```
|lambda * dt| < 2.785   (real-axis bound)
```

For a system with fastest eigenvalue `lambda_max`:

```
dt_max = 2.785 / |lambda_max|
```

This is a **necessary** condition for open-loop RK4 stability. Closed-loop feedback (LQR) modifies the eigenvalues, but the RK4 stability of the **open-loop** plant still constrains the integration timestep because the plant dynamics are evaluated during each RK4 sub-step.

---

## 2. Moth Eigenvalue-Timestep Table

The Moth 3DOF model has speed-dependent dynamics. The fast pitch mode eigenvalue increases with forward speed, tightening the RK4 stability constraint.

| Speed (m/s) | Fast eigenvalue (rad/s) | Max stable dt (ms) | |lambda * 5ms| | Margin at 5ms |
|-------------|------------------------|---------------------|----------------|---------------|
| 6 | ~100 | 27.9 | 0.50 | 82% |
| 8 | ~155 | 18.0 | 0.78 | 72% |
| 10 | ~280 | 9.6 | 1.40 | 50% |
| 12 | ~355 | 7.8 | 1.78 | 36% |
| 14 | ~415 | 6.7 | 2.08 | 25% |
| 17 | ~500 | 5.6 | 2.50 | 10% |
| 20 | ~587 | 4.7 | 2.94 | **UNSTABLE** |

**Key insight**: At 5ms, the default timestep provides adequate stability margin through 17 m/s. At 20 m/s, 5ms exceeds the RK4 stability limit -- use dt=0.002 or smaller for speeds above 17 m/s.

---

## 3. Timestep Architecture

| Component | Control dt | Integration dt | Notes |
|-----------|-----------|----------------|-------|
| Open-loop sim | N/A | 5ms | `simulate(dt=0.005)` |
| LQR design | 5ms | N/A (ZOH) | `discretize_zoh(A, B, 0.005)` -- exact, not RK4 |
| LQR sim | 5ms | 5ms | `simulate(dt=0.005)` |
| EKF | 5ms | 5ms | `EKF(dt=0.005)` |
| LQG | 5ms | 5ms | `simulate_lqg(dt=0.005)` |
| High-speed (>17 m/s) | varies | <4.7ms | Use smaller dt directly |

The default timestep is `MOTH_DEFAULT_DT = 0.005` (5ms), exported from `fmd.simulator`.

**Non-Moth models** (cartpole, pendulum, planar quadrotor, boat_2d, etc.) typically use `dt=0.01` (10ms). These models have much slower eigenvalues, so 10ms provides ample RK4 stability margin.

### Why 5ms?

- At 10 m/s (default operating speed), the fast pitch eigenvalue is ~280 rad/s, giving a max stable RK4 dt of 9.6ms. 5ms provides 50% margin.
- At 17 m/s, margin drops to ~10% but remains stable.
- The EKF was already running at 5ms (halved from 10ms during Phase 3 controller retuning).
- MPC uses 20ms control intervals with M=4 internal sub-steps, giving 5ms integration dt.
- 5ms is a round number that divides evenly into common control intervals (20ms, 50ms, 100ms).

---

## 4. High-Speed Timestep Guidance

At speeds above 17 m/s, 5ms exceeds the RK4 stability limit. Options:

1. **Use a smaller dt directly**: `simulate(dt=0.002)` for speeds up to ~27 m/s. This is the simplest approach for open-loop or LQR simulations.

2. **MPC handles this internally**: MPC uses its own CasADi inline RK4 with M sub-steps. Increasing M or reducing dt_mpc achieves smaller effective integration steps.

3. **LQR design at smaller dt**: Design gains with `design_moth_lqr(dt=0.002)` for high-speed operation. The `validate_simulation_dt()` helper will warn if simulation dt doesn't match the design dt.

---

## 5. Trim Solver and Timestep Independence

The trim solver (`find_moth_trim`) finds continuous-time equilibria where the state derivative is zero (`f(x, u) = 0`). This is **dt-independent** -- trim points are properties of the continuous dynamics, not the discretization.

The discretization step (`discretize_zoh`) is applied afterward for control design only. ZOH discretization is exact (matrix exponential), not an approximation like RK4.

---

## 6. Runtime dt Validation

The `MothGainScheduledController` stores its `design_dt` (the timestep used during LQR design). Before running a simulation, call `validate_simulation_dt()` to check for mismatches:

```python
from fmd.simulator import validate_simulation_dt

controller = MothGainScheduledController.from_gain_schedule(schedule)
validate_simulation_dt(controller, sim_dt=0.005)  # OK
validate_simulation_dt(controller, sim_dt=0.02)   # Warning!
```

The function issues a `UserWarning` if the relative difference exceeds the tolerance (default 20%). It does not raise -- the mismatch may be intentional (e.g., testing robustness).

---

## 7. EKF Multirate Design (Future)

The current EKF implementation is fixed-rate: `step()` = predict + update at every timestep, one measurement per step.

A future multirate design would split the operations:

- **`predict()`** at the fast rate (e.g., 5ms) -- propagates state and covariance
- **`update()`** only when measurements arrive -- incorporates sensor data

This enables:
- Running the EKF prediction faster than sensor rates
- Supporting multiple sensors at different rates
- Decoupling the EKF rate from the control rate

**Not currently implemented.** The EKF has no support for:
- Multirate measurements
- Timestamped measurements
- Asynchronous measurement arrival
- Multiple measurements per step

The `dt` parameter in `ExtendedKalmanFilter` is for the prediction step only.

---

## 8. MPC Timing Architecture

The MPC timing has three distinct dt concepts:

```
dt_mpc = 20ms     MPC control interval (how often MPC re-plans)
M = 4             RK4 sub-steps per interval in the NLP
dt_sub = 5ms      Actual integration dt inside the NLP (dt_mpc / M)
dt_sim = 5ms      True plant simulation dt
```

### Constraints

- `dt_sub` must be within the RK4 stability limit for the operating speed
- `dt_mpc` must be a multiple of `dt_sim` for clean ZOH alignment
- Control is applied via zero-order hold between MPC solves

### How M sub-steps work

M sub-steps add dynamics accuracy **without adding NLP decision variables**. The MPC's decision variables are the control inputs at each `dt_mpc` interval. Within each interval, M RK4 sub-steps integrate the dynamics constraint more accurately. The CasADi NLP handles this inline.

### Warm-start and IRK

Warm-starting the NLP from the previous solution and implicit Runge-Kutta (IRK) integration are separate future work items. See the MPC-next-steps planning note.
