from typing import Any

from pydantic import BaseModel, Field

from ..core.interaction_result import InteractionResult


class Context(BaseModel):
    """Context for generating interactions.

    Contains the history of previous interactions and metadata that can be used
    by interaction generators to create context-aware test cases.

    Attributes
    ----------
    previous_interactions:
        List of previous interaction results in this test scenario.
    """

    previous_interactions: list[InteractionResult[Any, Any]] = Field(
        default_factory=list
    )
