from typing import Any

from giskard.checks import SuiteResult, Target


async def third_party_scan(
    target: Target,  # pyright: ignore[reportMissingTypeArgument]
    tool: str,
    **kwargs: Any,  # pyright: ignore[reportExplicitAny]
) -> SuiteResult:
    """Run an external security scanner against a Giskard target.

    Args:
        target: The callable to test (sync or async, any return type).
        tool: Name of the scanner to use, e.g. ``"garak"``.
        **kwargs: Tool-specific options forwarded to the adapter.
            For garak: ``probes`` (``list[str]``) restricts which probes run.
            If ``probes`` is omitted, all probes that need no extra credentials
            are discovered and run automatically.

    Example::

        # Run all garak probes
        result = await third_party_scan(target=my_agent, tool="garak")

        # Run specific probes only
        result = await third_party_scan(
            target=my_agent, tool="garak", probes=["probes.dan.Dan_11_0"]
        )
    """
    from . import _registry  # noqa: F401 — triggers __init__.py auto-registration
    from ._registry import available, get

    adapter_class = get(tool)
    if adapter_class is None:
        raise ValueError(f"Unknown tool {tool!r}. Available: {available()}")
    return await adapter_class().run(target, **kwargs)
