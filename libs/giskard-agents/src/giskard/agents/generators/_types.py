"""Shared types for the generators package.

Kept in a separate module to break the circular dependency between
``base`` (BaseGenerator) and ``middleware`` (CompletionMiddleware).
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..chat import Message
from ..tools import Tool


class Response(BaseModel):
    message: Message
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "null"] | None
    )


class GenerationParams(BaseModel):
    """Parameters for generating a completion.

    Attributes
    ----------
    tools : list[Tool], optional
        List of tools available to the model.
    timeout : float | int | None, optional
        Maximum time in seconds to wait for the completion request.
    """

    temperature: float = Field(default=1.0)
    max_tokens: int | None = Field(default=None)
    response_format: type[BaseModel] | None = Field(default=None)
    tools: list[Tool] = Field(default_factory=list)
    timeout: float | int | None = Field(default=None)
