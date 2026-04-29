"""Import guard for optional rerun-sdk dependency."""


def require_rerun():
    """Check that rerun-sdk is installed, raise helpful error if not."""
    try:
        import rerun  # noqa: F401
    except ImportError:
        raise ImportError(
            "rerun-sdk is required for 3D visualization. "
            "Install with: uv sync"
        ) from None
