"""Exporting the old Pipeline class for backwards compatibility."""

from .workflow import ChatWorkflow as Pipeline

__all__ = ["Pipeline"]
