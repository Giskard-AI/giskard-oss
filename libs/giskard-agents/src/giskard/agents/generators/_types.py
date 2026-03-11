"""Shared types for the generators package.

Kept in a separate module to break the circular dependency between
``base`` (BaseGenerator) and ``middleware`` (CompletionMiddleware).
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..chat import Message
from ..tools import Tool

type FinishReason = (
    Literal["stop", "length", "tool_calls", "content_filter", "null"] | None
)


class Response(BaseModel):
    message: Message
    finish_reason: FinishReason


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

    def merge(self, overrides: "GenerationParams | None") -> "GenerationParams":
        """Return a copy with *overrides*' explicitly-set fields applied on top.

        Scalar fields from *overrides* replace the base values (only if set).
        Tools are concatenated.

        Parameters
        ----------
        overrides : GenerationParams or None
            Per-call overrides. Only explicitly-set fields take effect.
            If None, returns an unmodified copy.

        Returns
        -------
        GenerationParams
            A new instance with merged values.
        """
        if overrides is None:
            return self.model_copy()
        updates = overrides.model_dump(exclude={"tools"}, exclude_unset=True)
        merged = self.model_copy(update=updates)
        merged.tools = self.tools + overrides.tools
        return merged
