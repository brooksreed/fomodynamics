"""Ocean wave field model for BLUR simulator.

Provides a JIT-compatible WaveField Equinox module that evaluates
superposed linear (Airy) wave components at arbitrary (x, y, z, t).

Supports regular waves, JONSWAP and Pierson-Moskowitz spectra,
and optional cos^2s directional spreading.

All evaluation methods are pure JAX and compatible with jit/vmap/grad.

Frame convention:
    - NED world frame: x=North, y=East, z=Down (positive into water)
    - Wave elevation eta is positive upward (surface displaced up = crest)
    - Orbital velocities in NED frame
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from fmd.simulator.params.wave import WaveParams


# ============================================================================
# Spectrum functions (pure JAX, not methods)
# ============================================================================


def jonswap_spectrum(omega: Array, omega_p: float, hs: float, gamma: float) -> Array:
    """JONSWAP spectral density S(omega).

    Args:
        omega: Angular frequencies (rad/s), shape (N,).
        omega_p: Peak angular frequency (rad/s).
        hs: Significant wave height (m).
        gamma: JONSWAP peakedness parameter (>1).

    Returns:
        Spectral density S(omega) in m^2*s/rad, shape (N,).
    """
    # Pierson-Moskowitz base spectrum
    alpha_pm = (5.0 / 16.0) * hs**2 * omega_p**4
    s_pm = alpha_pm * omega**(-5) * jnp.exp(-1.25 * (omega_p / omega) ** 4)

    # JONSWAP peak enhancement
    sigma = jnp.where(omega <= omega_p, 0.07, 0.09)
    r = jnp.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    gamma_factor = gamma**r

    # Normalization so Hs = 4*sqrt(m0)
    # The raw JONSWAP needs a correction factor
    # C_gamma ≈ 1 - 0.287*ln(gamma)
    c_gamma = 1.0 - 0.287 * jnp.log(gamma)

    return c_gamma * s_pm * gamma_factor


def pierson_moskowitz_spectrum(omega: Array, omega_p: float, hs: float) -> Array:
    """Pierson-Moskowitz spectral density S(omega).

    Equivalent to JONSWAP with gamma=1 (fully developed sea).

    Args:
        omega: Angular frequencies (rad/s), shape (N,).
        omega_p: Peak angular frequency (rad/s).
        hs: Significant wave height (m).

    Returns:
        Spectral density S(omega) in m^2*s/rad, shape (N,).
    """
    alpha_pm = (5.0 / 16.0) * hs**2 * omega_p**4
    return alpha_pm * omega**(-5) * jnp.exp(-1.25 * (omega_p / omega) ** 4)


def dispersion_relation(omega: Array, depth: float, g: float) -> Array:
    """Solve the linear dispersion relation for wavenumber k.

    omega^2 = g * k * tanh(k * depth)

    Uses deep water approximation for infinite depth,
    Newton iteration for finite depth.

    Args:
        omega: Angular frequencies (rad/s), shape (N,).
        depth: Water depth (m), inf for deep water.
        g: Gravitational acceleration (m/s^2).

    Returns:
        Wavenumber k (rad/m), shape (N,).
    """
    k_deep = omega**2 / g

    if depth == float("inf"):
        return k_deep

    # Newton iteration for finite depth
    # f(k) = omega^2 - g*k*tanh(k*h) = 0
    # f'(k) = -g*(tanh(k*h) + k*h*sech^2(k*h))
    def newton_body(i, k):
        kh = k * depth
        tanh_kh = jnp.tanh(kh)
        f = omega**2 - g * k * tanh_kh
        sech2_kh = 1.0 / jnp.cosh(kh) ** 2
        fp = -g * (tanh_kh + k * depth * sech2_kh)
        # Clamp update to avoid overshoot
        dk = -f / fp
        return jnp.maximum(k + dk, 1e-10)

    k = jax.lax.fori_loop(0, 10, newton_body, k_deep)
    return k


def cos2s_spreading(theta: Array, theta_mean: float, s: float) -> Array:
    """Directional spreading function D(theta) = cos^2s((theta - theta_mean)/2).

    Normalized so that integral over [-pi, pi] = 1.

    Args:
        theta: Directions (rad), shape (M,).
        theta_mean: Mean wave direction (rad).
        s: Spreading exponent (0 = uniform, higher = more focused).

    Returns:
        Directional weights D(theta), shape (M,), normalized.
    """
    half_diff = (theta - theta_mean) / 2.0
    d = jnp.cos(half_diff) ** (2.0 * s)
    # Normalize
    d_sum = jnp.sum(d)
    return jnp.where(d_sum > 0, d / d_sum, jnp.ones_like(d) / theta.shape[0])


# ============================================================================
# WaveField Equinox module
# ============================================================================


class WaveField(eqx.Module):
    """JIT-compatible ocean wave field.

    Pre-computes spectral decomposition at construction time, then
    evaluates superposition of linear wave components at arbitrary
    (x, y, z, t) via pure JAX operations.

    All arrays are stored as JAX arrays for JIT compatibility.

    Attributes:
        amplitudes: Component amplitudes, shape (N, M).
        frequencies: Angular frequencies, shape (N,).
        wavenumbers: Wavenumbers, shape (N,).
        directions: Wave directions, shape (M,).
        phases: Random phases, shape (N, M).
        g: Gravitational acceleration (m/s^2).
        water_density: Water density (kg/m^3).
        depth: Water depth (m).
    """

    amplitudes: Array      # (N, M) - component amplitudes
    frequencies: Array     # (N,)   - angular frequencies omega
    wavenumbers: Array     # (N,)   - wavenumbers k
    directions: Array      # (M,)   - propagation directions
    phases: Array          # (N, M) - random phases
    g: float = eqx.field(static=True)
    water_density: float = eqx.field(static=True)
    depth: float = eqx.field(static=True)
    stokes_order: int = eqx.field(static=True, default=1)

    @classmethod
    def from_params(cls, params: WaveParams) -> WaveField:
        """Construct a WaveField from WaveParams.

        Discretizes the spectrum, solves dispersion, generates random
        phases, and computes directional weights.

        Args:
            params: WaveParams instance.

        Returns:
            WaveField ready for evaluation.
        """
        import numpy as np

        omega_p = 2.0 * np.pi / params.peak_period
        n_freq = params.num_components
        n_dir = params.num_directions
        rng = np.random.default_rng(params.seed)

        if params.spectrum_type == "regular":
            # Single component: amplitude = Hs/2
            amplitude = params.significant_wave_height / 2.0
            omega = np.array([omega_p])
            k = np.array([omega_p**2 / params.g]) if params.water_depth == float("inf") else None
            if k is None:
                k = _solve_dispersion_numpy(omega, params.water_depth, params.g)
            amplitudes_2d = np.array([[amplitude]])
            phases = np.array([[0.0]])
            directions = np.array([params.mean_direction])

        else:
            # Spectral discretization with equal-energy spacing
            # Frequency range: 0.5*omega_p to 3.0*omega_p
            omega_min = 0.5 * omega_p
            omega_max = 3.0 * omega_p
            omega = np.linspace(omega_min, omega_max, n_freq)
            d_omega = omega[1] - omega[0] if n_freq > 1 else 1.0

            # Small random perturbation to avoid periodicity
            if n_freq > 1:
                perturbation = rng.uniform(-0.2, 0.2, size=n_freq) * d_omega
                perturbation[0] = max(perturbation[0], 0.0)  # Keep first positive
                omega = omega + perturbation
                omega = np.sort(omega)
                omega = np.maximum(omega, 0.01)

            # Compute spectrum
            omega_jax = jnp.array(omega)
            if params.spectrum_type == "jonswap":
                S = np.array(jonswap_spectrum(omega_jax, omega_p, params.significant_wave_height, params.gamma))
            else:  # pierson_moskowitz
                S = np.array(pierson_moskowitz_spectrum(omega_jax, omega_p, params.significant_wave_height))

            # Component amplitudes from spectrum: a_i = sqrt(2 * S(omega_i) * d_omega)
            d_omega_arr = np.diff(omega, prepend=omega[0] - (omega[1] - omega[0]) if n_freq > 1 else omega[0])
            d_omega_arr = np.maximum(np.abs(d_omega_arr), 1e-6)
            component_amplitudes = np.sqrt(2.0 * S * d_omega_arr)

            # Directional spreading
            if n_dir == 1:
                directions = np.array([params.mean_direction])
                dir_weights = np.array([1.0])
            else:
                directions = np.linspace(
                    params.mean_direction - np.pi,
                    params.mean_direction + np.pi,
                    n_dir,
                    endpoint=False,
                )
                if params.spreading_exponent > 0:
                    dir_weights = np.array(cos2s_spreading(
                        jnp.array(directions),
                        params.mean_direction,
                        params.spreading_exponent,
                    ))
                else:
                    dir_weights = np.ones(n_dir) / n_dir

            # 2D amplitude array: a_ij = a_i * sqrt(D_j)
            amplitudes_2d = component_amplitudes[:, None] * np.sqrt(dir_weights[None, :])

            # Random phases
            phases = rng.uniform(0, 2 * np.pi, size=(n_freq, n_dir))

            # Solve dispersion
            k = _solve_dispersion_numpy(omega, params.water_depth, params.g)

        return cls(
            amplitudes=jnp.array(amplitudes_2d),
            frequencies=jnp.array(omega),
            wavenumbers=jnp.array(k),
            directions=jnp.array(directions),
            phases=jnp.array(phases),
            g=params.g,
            water_density=params.water_density,
            depth=params.water_depth,
            stokes_order=params.stokes_order,
        )

    @classmethod
    def regular(
        cls,
        amplitude: float,
        period: float,
        direction: float = 0.0,
        water_depth: float = float("inf"),
        g: float = 9.80665,
        water_density: float = 1025.0,
    ) -> WaveField:
        """Create a single regular (Airy) wave field.

        Args:
            amplitude: Wave amplitude (m).
            period: Wave period (s).
            direction: Propagation direction (rad, NED).
            water_depth: Water depth (m), inf for deep water.
            g: Gravitational acceleration (m/s^2).
            water_density: Water density (kg/m^3).

        Returns:
            WaveField with single component.
        """
        params = WaveParams.regular(amplitude, period, direction, water_depth, g, water_density)
        return cls.from_params(params)

    def elevation(self, x: float, y: float, t: float) -> Array:
        """Compute sea surface elevation at (x, y, t).

        eta = sum_i sum_j a_ij * cos(k_i*(x*cos(theta_j) + y*sin(theta_j)) - omega_i*t + phi_ij)

        Positive eta means the surface is displaced upward (crest).

        Args:
            x: North position (m).
            y: East position (m).
            t: Time (s).

        Returns:
            Surface elevation eta (m), scalar.
        """
        # Phase argument: k_i * (x*cos(theta_j) + y*sin(theta_j)) - omega_i*t + phi_ij
        cos_dir = jnp.cos(self.directions)  # (M,)
        sin_dir = jnp.sin(self.directions)  # (M,)

        # Spatial phase: k_i * (x*cos(theta_j) + y*sin(theta_j))
        spatial = self.wavenumbers[:, None] * (x * cos_dir[None, :] + y * sin_dir[None, :])  # (N, M)

        # Temporal phase
        temporal = self.frequencies[:, None] * t  # (N, 1)

        # Total phase
        phase = spatial - temporal + self.phases  # (N, M)

        # Superposition (1st order Airy)
        eta = jnp.sum(self.amplitudes * jnp.cos(phase))

        # 2nd-order Stokes self-interaction correction (deep water)
        if self.stokes_order >= 2:
            # eta_2 = sum_ij (a_ij^2 * k_i / 2) * cos(2 * phase_ij)
            eta_2 = jnp.sum(
                self.amplitudes**2 * self.wavenumbers[:, None] / 2.0 * jnp.cos(2.0 * phase)
            )
            eta = eta + eta_2

        return eta

    def orbital_velocity(self, x: float, y: float, z: float, t: float) -> Array:
        """Compute orbital velocity at (x, y, z, t) in NED frame.

        In NED: z is positive downward (depth below still water level).
        Uses linear wave theory with exponential depth decay.

        Args:
            x: North position (m).
            y: East position (m).
            z: Down position (m), positive into water.
            t: Time (s).

        Returns:
            Velocity [u_n, u_e, u_d] in NED (m/s), shape (3,).
        """
        cos_dir = jnp.cos(self.directions)  # (M,)
        sin_dir = jnp.sin(self.directions)  # (M,)

        spatial = self.wavenumbers[:, None] * (x * cos_dir[None, :] + y * sin_dir[None, :])
        temporal = self.frequencies[:, None] * t
        phase = spatial - temporal + self.phases

        # Depth decay factor: exp(-k*z) for NED (z positive down)
        # Clamp z >= 0 to prevent exponential blowup above surface (EDGE-2)
        z_safe = jnp.maximum(z, 0.0)
        depth_factor = jnp.exp(-self.wavenumbers * z_safe)  # (N,)

        # Horizontal velocity magnitude: a * omega * exp(-k*z) * cos(phase)
        vel_mag = self.amplitudes * self.frequencies[:, None] * depth_factor[:, None] * jnp.cos(phase)

        # Project onto NED components
        u_n = jnp.sum(vel_mag * cos_dir[None, :])
        u_e = jnp.sum(vel_mag * sin_dir[None, :])

        # Vertical velocity: a * omega * exp(-k*z) * sin(phase)
        # In NED, positive w is downward. Orbital motion has upward velocity
        # at the crest (phase=0), so w_ned = -a*omega*exp(-k*z)*sin(phase)
        u_d = -jnp.sum(
            self.amplitudes * self.frequencies[:, None] * depth_factor[:, None] * jnp.sin(phase)
        )

        # 2nd-order Stokes velocity corrections (deep water)
        if self.stokes_order >= 2:
            depth_factor_2 = jnp.exp(-2.0 * self.wavenumbers * z_safe)  # (N,)
            # Horizontal magnitude: a^2 * omega * k * cos(2*phase) * exp(-2*k*z)
            vel_mag_2 = (
                self.amplitudes**2 * self.frequencies[:, None]
                * self.wavenumbers[:, None] * jnp.cos(2.0 * phase)
                * depth_factor_2[:, None]
            )
            u_n = u_n + jnp.sum(vel_mag_2 * cos_dir[None, :])
            u_e = u_e + jnp.sum(vel_mag_2 * sin_dir[None, :])
            # Vertical: -a^2 * omega * k * sin(2*phase) * exp(-2*k*z)
            u_d = u_d - jnp.sum(
                self.amplitudes**2 * self.frequencies[:, None]
                * self.wavenumbers[:, None] * jnp.sin(2.0 * phase)
                * depth_factor_2[:, None]
            )

        return jnp.array([u_n, u_e, u_d])

    def orbital_acceleration(self, x: float, y: float, z: float, t: float) -> Array:
        """Compute orbital acceleration at (x, y, z, t) in NED frame.

        Args:
            x: North position (m).
            y: East position (m).
            z: Down position (m), positive into water.
            t: Time (s).

        Returns:
            Acceleration [a_n, a_e, a_d] in NED (m/s^2), shape (3,).
        """
        cos_dir = jnp.cos(self.directions)
        sin_dir = jnp.sin(self.directions)

        spatial = self.wavenumbers[:, None] * (x * cos_dir[None, :] + y * sin_dir[None, :])
        temporal = self.frequencies[:, None] * t
        phase = spatial - temporal + self.phases

        # Clamp z >= 0 to prevent exponential blowup above surface (EDGE-2)
        z_safe = jnp.maximum(z, 0.0)
        depth_factor = jnp.exp(-self.wavenumbers * z_safe)

        # Acceleration = a * omega^2 * exp(-k*z) * sin(phase) for horizontal
        acc_mag = self.amplitudes * self.frequencies[:, None] ** 2 * depth_factor[:, None] * jnp.sin(phase)

        a_n = jnp.sum(acc_mag * cos_dir[None, :])
        a_e = jnp.sum(acc_mag * sin_dir[None, :])

        # Vertical: -a * omega^2 * exp(-k*z) * cos(phase) → positive down in NED
        a_d = jnp.sum(
            self.amplitudes * self.frequencies[:, None] ** 2 * depth_factor[:, None] * jnp.cos(phase)
        )

        # 2nd-order Stokes acceleration corrections (deep water)
        if self.stokes_order >= 2:
            depth_factor_2 = jnp.exp(-2.0 * self.wavenumbers * z_safe)
            # Time derivative of 2nd-order velocity:
            # d/dt [a^2 * omega * k * cos(2*phase)] = 2 * a^2 * omega^2 * k * sin(2*phase)
            acc_mag_2 = (
                2.0 * self.amplitudes**2 * self.frequencies[:, None]**2
                * self.wavenumbers[:, None] * jnp.sin(2.0 * phase)
                * depth_factor_2[:, None]
            )
            a_n = a_n + jnp.sum(acc_mag_2 * cos_dir[None, :])
            a_e = a_e + jnp.sum(acc_mag_2 * sin_dir[None, :])
            # Vertical: d/dt [-a^2 * omega * k * sin(2*phase) * exp(-2kz)]
            # = -a^2 * omega * k * 2*(-omega)*cos(2*phase) * exp(-2kz)
            # = +2 * a^2 * omega^2 * k * cos(2*phase) * exp(-2kz)
            # NED sign: 1st-order a_d uses +cos(phase), so 2nd-order uses +cos(2*phase)
            a_d = a_d + jnp.sum(
                2.0 * self.amplitudes**2 * self.frequencies[:, None]**2
                * self.wavenumbers[:, None] * jnp.cos(2.0 * phase)
                * depth_factor_2[:, None]
            )

        return jnp.array([a_n, a_e, a_d])

    def pressure(self, x: float, y: float, z: float, t: float) -> Array:
        """Compute dynamic wave pressure at (x, y, z, t).

        Undisturbed pressure field for Froude-Krylov integration:
        p = rho * g * a * exp(-k*z) * cos(phase)

        Args:
            x: North position (m).
            y: East position (m).
            z: Down position (m), positive into water.
            t: Time (s).

        Returns:
            Dynamic wave pressure (Pa), scalar.
        """
        cos_dir = jnp.cos(self.directions)
        sin_dir = jnp.sin(self.directions)

        spatial = self.wavenumbers[:, None] * (x * cos_dir[None, :] + y * sin_dir[None, :])
        temporal = self.frequencies[:, None] * t
        phase = spatial - temporal + self.phases

        # Clamp z >= 0 to prevent exponential blowup above surface (EDGE-2)
        z_safe = jnp.maximum(z, 0.0)
        depth_factor = jnp.exp(-self.wavenumbers * z_safe)

        p = self.water_density * self.g * jnp.sum(
            self.amplitudes * depth_factor[:, None] * jnp.cos(phase)
        )
        return p


# ============================================================================
# Internal helpers
# ============================================================================


def _solve_dispersion_numpy(omega, depth, g):
    """Solve dispersion relation using numpy (for construction time).

    Args:
        omega: Angular frequencies, numpy array.
        depth: Water depth (m).
        g: Gravitational acceleration (m/s^2).

    Returns:
        Wavenumber array, numpy.
    """
    import numpy as np

    k = omega**2 / g  # Deep water initial guess

    if depth == float("inf"):
        return k

    # Newton iteration
    for _ in range(10):
        kh = k * depth
        tanh_kh = np.tanh(kh)
        f = omega**2 - g * k * tanh_kh
        sech2_kh = 1.0 / np.cosh(kh) ** 2
        fp = -g * (tanh_kh + k * depth * sech2_kh)
        dk = -f / fp
        k = np.maximum(k + dk, 1e-10)

    return k
