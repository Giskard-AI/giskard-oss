from typing import Any

from pydantic import BaseModel, Field

from ..core.interaction import Interaction


class Context(BaseModel):
    """Context for generating interactions.

    Contains the history of previous interactions and metadata that can be used
    by interaction generators to create context-aware test cases.

    Attributes
    ----------
    previous_interactions:
        List of previous interactions in this test scenario.
    """

    previous_interactions: list[Interaction[Any, Any]] = Field(default_factory=list)
