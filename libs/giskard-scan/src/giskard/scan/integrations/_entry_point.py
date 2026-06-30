from typing import Any

from giskard.checks import SuiteResult, Target


async def third_party_scan(
    target: Target,  # pyright: ignore[reportMissingTypeArgument]
    tool: str,
    **kwargs: Any,  # pyright: ignore[reportExplicitAny]
) -> SuiteResult:
    from . import _registry  # noqa: F401 — triggers __init__.py auto-registration
    from ._registry import available, get

    adapter_class = get(tool)
    if adapter_class is None:
        raise ValueError(f"Unknown tool {tool!r}. Available: {available()}")
    return await adapter_class().run(target, **kwargs)
