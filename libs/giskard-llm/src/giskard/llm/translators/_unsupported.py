import logging
from collections.abc import Iterable
from typing import Any

from ..errors import UnsupportedContentError

logger = logging.getLogger(__name__)


def handle_unsupported_content(
    provider: str, content_type: str, *, ignore_unsupported_content: bool
) -> None:
    """Raise for unsupported response content, or warn when the caller opted out.

    Parameters
    ----------
    provider : str
        Provider name reported in the error / warning.
    content_type : str
        Provider-native type of the unsupported item.
    ignore_unsupported_content : bool
        If ``True``, log a warning (the caller then drops the item) instead of raising.

    Raises
    ------
    UnsupportedContentError
        If ``ignore_unsupported_content`` is ``False``.
    """
    if not ignore_unsupported_content:
        raise UnsupportedContentError(provider, content_type)
    logger.warning(
        "%s provider: dropping unsupported content type '%s' from response",
        provider,
        content_type,
    )


def keep_supported(
    items: Iterable[Any],
    supported_types: frozenset[str],
    provider: str,
    *,
    ignore_unsupported_content: bool,
    label_prefix: str = "",
) -> list[Any]:
    """Keep dict items whose ``type`` is supported; handle the rest as unsupported.

    Parameters
    ----------
    items : Iterable[Any]
        Raw (``model_dump``-ed) items, each expected to be a dict with a ``type`` key.
    supported_types : frozenset[str]
        ``type`` values giskard-llm can represent.
    provider : str
        Provider name reported in the error / warning.
    ignore_unsupported_content : bool
        If ``True``, drop unsupported items with a warning instead of raising.
    label_prefix : str
        Prefix for the reported content type (e.g. ``"reasoning_details:"``).

    Returns
    -------
    list[Any]
        The supported items, in order.
    """
    kept: list[Any] = []
    for item in items:
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type in supported_types:
            kept.append(item)
        else:
            handle_unsupported_content(
                provider,
                f"{label_prefix}{item_type}",
                ignore_unsupported_content=ignore_unsupported_content,
            )
    return kept
