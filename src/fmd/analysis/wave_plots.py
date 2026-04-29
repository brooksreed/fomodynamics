"""Wave field plotting utilities.

Provides visualization tools for wave elevation time series,
encounter spectra, and spatial wave fields.
"""

from __future__ import annotations

import numpy as np

from fmd.analysis.plots._style import style_axis


def plot_wave_elevation_timeseries(
    times: np.ndarray,
    elevation: np.ndarray,
    ax=None,
    title: str = "Wave Elevation",
    ylabel: str = "Elevation (m)",
    **kwargs,
):
    """Plot wave surface elevation vs time.

    Args:
        times: Time array (s)
        elevation: Surface elevation array (m)
        ax: Optional matplotlib axes. If None, creates new figure.
        title: Plot title
        ylabel: Y-axis label
        **kwargs: Passed to ax.plot()

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(times, elevation, **kwargs)
    style_axis(ax, xlabel="Time (s)", ylabel=ylabel, title=title)

    return ax


def plot_wave_encounter_spectrum(
    times: np.ndarray,
    elevation: np.ndarray,
    ax=None,
    title: str = "Wave Encounter Spectrum",
    **kwargs,
):
    """Plot FFT-based encounter frequency spectrum.

    Args:
        times: Time array (s), assumed uniform spacing
        elevation: Surface elevation array (m)
        ax: Optional matplotlib axes
        title: Plot title
        **kwargs: Passed to ax.plot()

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    n = len(elevation)
    dt = times[1] - times[0]

    # Compute one-sided power spectral density via FFT
    fft_vals = np.fft.rfft(elevation)
    freqs = np.fft.rfftfreq(n, d=dt)

    # PSD: |X(f)|^2 * 2*dt/N (one-sided, factor of 2 for negative freqs)
    psd = (2.0 * dt / n) * np.abs(fft_vals) ** 2
    # DC component should not be doubled
    psd[0] /= 2.0
    if n % 2 == 0:
        # Nyquist component should not be doubled
        psd[-1] /= 2.0

    ax.plot(freqs, psd, **kwargs)
    style_axis(ax, xlabel="Frequency (Hz)", ylabel="PSD (m$^2$/Hz)", title=title)

    return ax


def plot_waterfall(
    wave_field,
    x_range: tuple[float, float] = (0.0, 100.0),
    t_range: tuple[float, float] = (0.0, 20.0),
    nx: int = 200,
    nt: int = 100,
    y: float = 0.0,
    ax=None,
    title: str = "Wave Propagation",
    **kwargs,
):
    """Plot wave elevation as a function of position and time (waterfall/contour plot).

    Args:
        wave_field: WaveField instance with an elevation(x, y, t) method
        x_range: (x_min, x_max) in meters
        t_range: (t_min, t_max) in seconds
        nx: Number of spatial points
        nt: Number of time points
        y: Fixed y-coordinate (m)
        ax: Optional matplotlib axes
        title: Plot title
        **kwargs: Passed to ax.pcolormesh()

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    xs = np.linspace(x_range[0], x_range[1], nx)
    ts = np.linspace(t_range[0], t_range[1], nt)

    # Evaluate wave field on the x-t grid (vectorized)
    import jax
    import jax.numpy as jnp

    X_grid, T_grid = np.meshgrid(xs, ts)
    x_flat = jnp.asarray(X_grid.ravel())
    t_flat = jnp.asarray(T_grid.ravel())
    eta_flat = jax.vmap(lambda xi, ti: wave_field.elevation(xi, y, ti))(x_flat, t_flat)
    eta = np.asarray(eta_flat).reshape(nt, nx)

    X, T = np.meshgrid(xs, ts)
    mesh = ax.pcolormesh(X, T, eta, shading="auto", **kwargs)
    ax.figure.colorbar(mesh, ax=ax, label="Elevation (m)")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)

    return ax


def plot_wave_field_snapshot(
    wave_field,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    t: float = 0.0,
    nx: int = 100,
    ny: int = 100,
    ax=None,
    title: str = "Wave Field Snapshot",
    **kwargs,
):
    """Plot 2D snapshot of wave surface elevation.

    Args:
        wave_field: WaveField instance with an elevation(x, y, t) method
        x_range: (x_min, x_max) in meters
        y_range: (y_min, y_max) in meters
        t: Time instant (s)
        nx: Number of x-points
        ny: Number of y-points
        ax: Optional matplotlib axes
        title: Plot title
        **kwargs: Passed to ax.pcolormesh()

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)

    # Evaluate wave field on the x-y grid (vectorized)
    import jax
    import jax.numpy as jnp

    X_grid, Y_grid = np.meshgrid(xs, ys)
    x_flat = jnp.asarray(X_grid.ravel())
    y_flat = jnp.asarray(Y_grid.ravel())
    eta_flat = jax.vmap(lambda xi, yi: wave_field.elevation(xi, yi, t))(x_flat, y_flat)
    eta = np.asarray(eta_flat).reshape(ny, nx)

    X, Y = np.meshgrid(xs, ys)
    mesh = ax.pcolormesh(X, Y, eta, shading="auto", **kwargs)
    ax.figure.colorbar(mesh, ax=ax, label="Elevation (m)")
    ax.set_xlabel("x / North (m)")
    ax.set_ylabel("y / East (m)")
    ax.set_title(title)
    ax.set_aspect("equal")

    return ax
