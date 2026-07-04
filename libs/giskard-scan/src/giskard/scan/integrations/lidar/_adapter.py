"""Adapter that runs the private lidar scanner through Giskard scan."""

import logging
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def lidar_available() -> bool:
    """Return True if the private lidar dependency is importable."""
    return find_spec("lidar") is not None


def _require_lidar() -> None:
    if not lidar_available():
        raise ImportError(
            "lidar is not installed. It is a private Giskard package and is not "
            "available on PyPI. Install it from git: "
            "pip install git+https://github.com/Giskard-AI/lidar.git@v0.2.7"
        )


class LidarScanAdapter:
    """Build and run a Giskard suite from a lidar scan. Filled in by later tasks."""
