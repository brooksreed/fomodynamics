# Simulator Models Documentation

This document describes the physics models available in `fmd.simulator`. All models are implemented in JAX for JIT compilation and automatic differentiation support.

**Related documentation:**
- [Simulator Architecture](simulator_architecture.md) - Core architecture, equations of motion, state vectors
- [JAX Simulator Guide](jax_simulator_guide.md) - JIT compilation, autodiff, control interface

---

## Table of Contents

1. [Boat2D - Planar Marine Vehicle](#1-boat2d---planar-marine-vehicle)
2. [Cartpole - Inverted Pendulum](#2-cartpole---inverted-pendulum)
3. [PlanarQuadrotor - 2D Quadrotor](#3-planarquadrotor---2d-quadrotor)
4. [SimplePendulum](#4-simplependulum)
5. [Box1D / Box1DFriction](#5-box1d--box1dfriction)
6. [Moth3D - Hydrofoiling Sailboat](#6-moth3d---hydrofoiling-sailboat)
7. [Validation Summary](#7-validation-summary)

---

## 1. Boat2D - Planar Marine Vehicle

### System Overview

**Subject:** Small Inflatable Motorboat (e.g., Zodiac style)
**Configuration:** Single operator seated near center, outboard motor on transom.

| Parameter | Value | Notes |
| --- | --- | --- |
| **Total Mass** | **150 kg** | Boat (~75kg) + Person (~75kg) |
| **Length** | **3.0 m** | ~10 ft |
| **Beam** | **1.5 m** | ~5 ft |
| **Center of Gravity** | Aft of center | Due to engine mass |

### Conventions (Planar 3-DOF)

- **Inertial frame:** NED (North-East-Down) for position $(x, y)$.
- **Body frame:** $u$ forward (+x), $v$ right (+y), yaw rate $r$ about +z (down).
- **Heading:** $\psi$ measured from North, **clockwise positive**.

### State Vector (6 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | X | North position | m |
| 1 | Y | East position | m |
| 2 | ψ | Heading angle | rad |
| 3 | u | Surge velocity (forward) | m/s |
| 4 | v | Sway velocity (right) | m/s |
| 5 | r | Yaw rate | rad/s |

### Control Vector (2 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | T | Thrust force | N |
| 1 | τ | Yaw moment | N·m |

### Equations of Motion

The Boat2D model includes Coriolis coupling between surge and sway:

$$\dot{u} = \frac{T}{m} - \frac{D_u}{m} u + rv$$
$$\dot{v} = -\frac{D_v}{m} v - ru$$
$$\dot{r} = \frac{\tau}{I_{zz}} - \frac{D_r}{I_{zz}} r$$

**Kinematics (body to world):**
$$\dot{X} = u \cos\psi - v \sin\psi$$
$$\dot{Y} = u \sin\psi + v \cos\psi$$
$$\dot{\psi} = r$$

### Damping Coefficients

| Axis | Coeff | Unit | **Low Speed** (~2 m/s) | **High Speed** (~8 m/s) |
| --- | --- | --- | --- | --- |
| **Surge** | $D_u$ | N·s/m | **160** | **650** |
| **Sway** | $D_v$ | N·s/m | **300** | **1,200** |
| **Yaw** | $D_r$ | N·m·s | **350** | **1,400** |

### Usage

```python
from fmd.simulator import Boat2D, simulate, ConstantControl
from fmd.simulator.params import BOAT2D_TEST_DEFAULT
import jax.numpy as jnp

boat = Boat2D(BOAT2D_TEST_DEFAULT)
initial = boat.default_state()
control = ConstantControl(jnp.array([50.0, 5.0]))  # [thrust, yaw_moment]
result = simulate(boat, initial, dt=0.01, duration=20.0, control=control)
```

---

## 2. Cartpole - Inverted Pendulum

### System Overview

Classic control benchmark: a pole balanced on a cart that moves along a frictionless track. Based on Barto, Sutton, and Anderson (1983).

### State Vector (4 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x | Cart position | m |
| 1 | ẋ | Cart velocity | m/s |
| 2 | θ | Pole angle from vertical | rad |
| 3 | θ̇ | Pole angular velocity | rad/s |

**Sign Convention:** θ = 0 is upright (unstable equilibrium), θ = π is hanging (stable equilibrium). Positive θ is clockwise.

### Control Vector (1 element)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | F | Horizontal force on cart | N |

### Equations of Motion

The Barto-Sutton-Anderson equations with coupled cart-pole dynamics:

$$\ddot{\theta} = \frac{g \sin\theta + \cos\theta \cdot \text{temp}}{l \left(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p}\right)}$$

where:
$$\text{temp} = \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}$$

$$\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta)}{m_c + m_p}$$

### Validation

**Reference:** [OpenAI Gym CartPole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

The implementation matches the OpenAI Gym equations exactly after algebraic rearrangement. Golden values have been manually verified:
- At θ=0.1, F=0: θ̈ = 1.5748530266, ẍ = -0.0712266147 ✓
- At θ=0, F=10: θ̈ = -14.6341463415, ẍ = 9.7560975610 ✓

### Parameters

| Parameter | Symbol | CARTPOLE_CLASSIC |
|-----------|--------|------------------|
| Cart mass | m_c | 1.0 kg |
| Pole mass | m_p | 0.1 kg |
| Pole half-length | l | 0.5 m |
| Gravity | g | 9.80665 m/s² |

### Usage

```python
from fmd.simulator import Cartpole, simulate, ConstantControl
from fmd.simulator.params import CARTPOLE_CLASSIC
import jax.numpy as jnp

cartpole = Cartpole(CARTPOLE_CLASSIC)
initial = jnp.array([0.0, 0.0, 0.1, 0.0])  # Small tilt
result = simulate(cartpole, initial, dt=0.01, duration=5.0)
```

### Analytical Properties

- **Linearized period:** $T = 2\pi\sqrt{l/g}$
- **Energy:** $E = KE_{cart} + KE_{pole} + PE_{pole}$
- **Upright equilibrium:** Unstable (pole falls with any perturbation)
- **Hanging equilibrium:** Stable (pole oscillates around θ=π)

---

## 3. PlanarQuadrotor - 2D Quadrotor

### System Overview

A quadrotor restricted to the x-z plane with pitch rotation. Two rotors provide thrust forces T1 (right) and T2 (left).

### State Vector (6 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x | Horizontal position | m |
| 1 | z | Vertical position (up positive) | m |
| 2 | θ | Pitch angle (nose up positive) | rad |
| 3 | ẋ | Horizontal velocity | m/s |
| 4 | ż | Vertical velocity | m/s |
| 5 | θ̇ | Pitch rate | rad/s |

### Control Vector (2 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | T1 | Right rotor thrust | N |
| 1 | T2 | Left rotor thrust | N |

### Equations of Motion

$$\ddot{x} = -\frac{T}{m} \sin\theta$$
$$\ddot{z} = \frac{T}{m} \cos\theta - g$$
$$\ddot{\theta} = \frac{M}{I}$$

where:
- Total thrust: $T = T_1 + T_2$
- Moment: $M = (T_1 - T_2) \cdot L_{arm}$

### Validation

**Reference:** [Cookie Robotics 2D Quadrotor](https://cookierobotics.com/052/)

The equations match the standard 2D quadrotor model exactly:
- ÿ = -(f/m)·sin(Φ) ✓
- z̈ = (f/m)·cos(Φ) - g ✓
- Φ̈ = τ/I ✓

### Parameters

| Parameter | Symbol | PLANAR_QUAD_TEST_DEFAULT |
|-----------|--------|--------------------------|
| Mass | m | 1.0 kg |
| Arm length | L | 0.25 m |
| Pitch inertia | I | 0.01 kg·m² |
| Gravity | g | 9.80665 m/s² |

### Usage

```python
from fmd.simulator import PlanarQuadrotor, simulate, ConstantControl
from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT
import jax.numpy as jnp

quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
control = ConstantControl(quad.hover_control())
result = simulate(quad, quad.default_state(), dt=0.001, duration=5.0, control=control)
```

### Analytical Properties

- **Hover thrust:** $T_{hover} = mg$ (split equally between rotors)
- **Freefall:** $\ddot{z} = -g$ when $T = 0$
- **Energy:** $E = \frac{1}{2}m(\dot{x}^2 + \dot{z}^2) + \frac{1}{2}I\dot{\theta}^2 + mgz$

---

## 4. SimplePendulum

A simple pendulum with state [θ, θ̇] and no control input. Useful for testing and as a minimal example.

$$\ddot{\theta} = -\frac{g}{l} \sin\theta$$

---

## 5. Box1D / Box1DFriction

Minimal 1D systems for testing and control algorithm development.

**Box1D:** Double integrator (position + velocity). Control = force. No friction.
- State: [x, v] (2 states), Control: [F] (1 control)

**Box1DFriction:** Same as Box1D with Coulomb + viscous friction.
- State: [x, v] (2 states), Control: [F] (1 control)
- Smooth Coulomb friction approximation via `tanh(v/epsilon)`
- **Parameters:** `Box1DParams` / `Box1DFrictionParams`

---

## 6. Moth3D - Hydrofoiling Sailboat

Moth 3DOF longitudinal dynamics (pitch + heave + surge) for a Bieker Moth V3 hydrofoiling sailboat.

- **5 states:** [pos_d, theta, w, q, u] — vertical position, pitch, body velocities
- **2 controls:** [main_flap_angle, rudder_elevator_angle]
- Component-based force model: main T-foil, rudder elevator, sail (NED→body thrust), hull drag
- Per-timestep CG recomputation from sailor position schedule
- Surge dynamics enabled by default (surge_enabled=True); disable with surge_enabled=False
- Ventilation modeling (smooth or binary mode)
- Wave disturbance support via Environment
- **Parameters:** `MothParams` (`MOTH_BIEKER_V3` preset)
- **Usage:** `from fmd.simulator import Moth3D, simulate`

See [moth_simulation_vision.md](moth_simulation_vision.md) for the full design vision.

---

## 7. Validation Summary

| Model | Reference | Validation Method |
|-------|-----------|-------------------|
| **Boat2D** | Standard marine vehicle dynamics | Coriolis coupling verified, steady-state analysis |
| **Cartpole** | OpenAI Gym, Barto-Sutton-Anderson 1983 | Equations match after rearrangement, golden values verified |
| **PlanarQuadrotor** | Cookie Robotics, MIT lecture notes | Exact equation match |
| **SimplePendulum** | Classical mechanics | Period matches 2π√(l/g) |
| **Box1D** | Classical mechanics (double integrator) | Analytical solution verification |
| **Moth3D** | Literature-based foil model (Bieker V3) | Trim equilibrium, component force verification |

### Test Coverage

| Model | Test Count | Categories |
|-------|------------|------------|
| Cartpole | 53 | Basics, Derivatives, Physics, Simulation, JIT, Grad, Vmap, Golden |
| PlanarQuadrotor | 49 | Basics, Derivatives, Physics, Simulation, Utilities, JIT, Grad, Vmap, Golden |

All models include:
- Construction and attribute tests
- State derivative verification
- Physics validation (energy conservation, equilibrium stability)
- JIT compilation tests
- Automatic differentiation tests
- Golden value regression tests
