"""Test configuration with auto-skip for missing or present SDKs."""

import importlib
import os

import pytest

_PROVIDER_PACKAGES = {
    "openai": "openai",
    "bare": "openai",
    "google": "google.genai",
    "gemini": "google.genai",
    "anthropic": "anthropic",
    "azure": "openai",
    "azure_ai": "openai",
}

# When TEST_PROVIDER is set, keep only tests whose pytest provider mark is in this set.
# This lets CI slice the suite with one credential without repeating -m expressions.
# - ``google`` includes ``gemini`` (same ``google.genai`` SDK, different routing prefix).
_TEST_PROVIDER_GROUPS: dict[str, frozenset[str]] = {
    "openai": frozenset({"openai"}),
    "google": frozenset({"google", "gemini"}),
    "anthropic": frozenset({"anthropic"}),
}

_ANY_PROVIDER_PACKAGES = ["openai", "google.genai", "anthropic"]


def _parse_test_provider() -> tuple[frozenset[str], str] | None:
    """Return allowed marks and display name, or ``None`` if filtering is disabled."""

    raw = os.environ.get("TEST_PROVIDER", "").strip()
    if not raw or raw.lower() == "all":
        return None
    key = raw.lower()
    return _TEST_PROVIDER_GROUPS.get(key, frozenset({key})), raw


def _is_installed(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip tests based on SDK availability.

    **TEST_PROVIDER** (optional): if set to a nonempty value other than ``all``,
    skip tests that carry a provider mark (``openai``, ``google``, …) not allowed
    for that value. Example: ``TEST_PROVIDER=openai`` keeps ``openai`` and ``bare``.

    **Otherwise**

    - Tests marked with a provider name skip when that SDK is missing.
    - Tests marked ``no_providers`` skip when any provider SDK is installed.
    """
    provider_filter = _parse_test_provider()
    if provider_filter is not None:
        allowed, tp_disp = provider_filter
        for item in items:
            present = {name for name in _PROVIDER_PACKAGES if name in item.keywords}
            if present and present.isdisjoint(allowed):
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"Not selected by TEST_PROVIDER={tp_disp!r} "
                            f"(allowed marks: {sorted(allowed)})"
                        )
                    )
                )

    installed_cache: dict[str, bool] = {}
    for item in items:
        for mark_name, package in _PROVIDER_PACKAGES.items():
            if mark_name in item.keywords:
                if package not in installed_cache:
                    installed_cache[package] = _is_installed(package)
                if not installed_cache[package]:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Provider SDK '{package}' not installed"
                        )
                    )

    if "no_providers" not in {kw for item in items for kw in item.keywords}:
        return

    any_installed = any(_is_installed(p) for p in _ANY_PROVIDER_PACKAGES)
    if any_installed:
        for item in items:
            if "no_providers" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(
                        reason="no_providers tests require no provider SDKs installed"
                    )
                )
