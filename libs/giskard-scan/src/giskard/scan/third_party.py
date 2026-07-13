"""Alias for the third-party scan entry point.

Re-exports ``third_party_scan`` so ``giskard.scan.third_party`` keeps working as
an import path. The canonical import is ``from giskard.scan import
third_party_scan``.
"""

from .integrations import third_party_scan

__all__ = ["third_party_scan"]
