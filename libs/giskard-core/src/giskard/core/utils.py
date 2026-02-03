"""Utility constants and helpers for the Giskard library ecosystem."""

from typing import Literal

from pydantic import BaseModel


class NotProvided(BaseModel):
    """Sentinel class to indicate that a value was not provided."""

    __type__: Literal["not_provided"] = "not_provided"


NOT_PROVIDED = NotProvided()


def provide_not_none[T](value: T | None) -> T | NotProvided:
    return value if value is not None else NOT_PROVIDED
