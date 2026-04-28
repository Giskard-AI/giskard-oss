"""Shared utilities for giskard-llm providers."""

from .arguments import deserialize_arguments, serialize_arguments
from .compact import compact

__all__ = [
    "compact",
    "deserialize_arguments",
    "serialize_arguments",
]
