"""Shared matplotlib style helpers for fomodynamics plots."""

from contextlib import contextmanager

import matplotlib.pyplot as plt

FMD_STYLE = {
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "lines.linewidth": 1.0,
}


@contextmanager
def apply_fmd_style():
    """Context manager that applies fomodynamics's default rcParams."""
    with plt.rc_context(FMD_STYLE):
        yield


def style_axis(ax, xlabel=None, ylabel=None, title=None, grid=True):
    """Apply common axis styling.

    Args:
        ax: matplotlib Axes
        xlabel: x-axis label
        ylabel: y-axis label
        title: axis title
        grid: whether to show grid (default True)
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)


def get_colors(n):
    """Return n distinct colors from a qualitative colormap."""
    if n <= 10:
        cmap = plt.colormaps.get_cmap("tab10").resampled(10)
    else:
        cmap = plt.colormaps.get_cmap("tab20").resampled(20)
    return [cmap(i) for i in range(n)]


def savefig_and_close(fig, path, dpi=150):
    """Save figure to path and close it.

    Args:
        fig: matplotlib Figure
        path: output file path
        dpi: resolution (default 150)
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
