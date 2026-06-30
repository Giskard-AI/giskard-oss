from typing import Any

from giskard.checks import SuiteResult, Target

from ._registry import available, get


async def third_party_scan(
    target: Target,  # pyright: ignore[reportMissingTypeArgument]
    tool: str,
    **kwargs: Any,  # pyright: ignore[reportExplicitAny]
) -> SuiteResult:
    adapter_class = get(tool)
    if adapter_class is None:
        raise ValueError(f"Unknown tool {tool!r}. Available: {available()}")
    return await adapter_class().run(target, **kwargs)
