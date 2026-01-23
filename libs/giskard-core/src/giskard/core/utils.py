"""Utility constants and helpers for the Giskard library ecosystem."""


class NotProvided:
    """Sentinel class to indicate that a value was not provided."""

    pass


NOT_PROVIDED = NotProvided()


def provide_not_none[T](value: T | None) -> T | NotProvided:
    return value if value is not None else NOT_PROVIDED
