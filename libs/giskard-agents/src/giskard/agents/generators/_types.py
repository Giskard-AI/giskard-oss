"""Shared types for the generators package.

Kept in a separate module to break the circular dependency between
``base`` (BaseGenerator) and ``middleware`` (CompletionMiddleware).
"""

from typing import Any, ClassVar

from giskard.llm.types import NormalizedReasoningEffort
from pydantic import BaseModel, ConfigDict, Field

from ..tools import Tool


class GenerationParams(BaseModel):
    """Parameters for generating a completion.

    Attributes
    ----------
    tools : list[Tool], optional
        List of tools available to the model.
    timeout : float | int | None, optional
        Maximum time in seconds to wait for the completion request.
    reasoning_effort : NormalizedReasoningEffort | None, optional
        Canonical reasoning effort level (``none``..``max``), case-normalized.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    temperature: float = Field(default=1.0)
    max_tokens: int | None = Field(default=None)
    response_format: type[Any] | None = Field(default=None)
    tools: list[Tool] = Field(default_factory=list)
    timeout: float | int | None = Field(default=None)
    reasoning_effort: NormalizedReasoningEffort | None = Field(default=None)

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
