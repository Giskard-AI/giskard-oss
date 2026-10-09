import logging

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
