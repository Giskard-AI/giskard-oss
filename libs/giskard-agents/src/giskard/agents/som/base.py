"""Provider-independent interfaces for System One Models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from giskard.core import Discriminated, discriminated_base
from giskard.llm.types import ChatMessage, Usage
from pydantic import BaseModel, ConfigDict, Field


class SOMResponse(BaseModel):
    """Probability that the supplied question is true for the input messages."""

    probability: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    model: str = Field(min_length=1)
    usage: Usage | None = None


@discriminated_base
class BaseSOM(Discriminated, ABC):
    """A System One Model that predicts probabilities instead of generating text.

    Unknown configuration fields are rejected so serialized models cannot
    silently lose settings. See ``Discriminated`` for the subclass rule.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)

    @abstractmethod
    async def predict(
        self,
        messages: Sequence[ChatMessage],
        question: str,
        *,
        timeout: float | int | None = None,
    ) -> SOMResponse:
        """Predict the probability of ``question`` using the supplied context."""
        ...
