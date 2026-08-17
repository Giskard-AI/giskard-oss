from pydantic import AliasChoices, Field

from ._base import _BaseModel


class Usage(_BaseModel):
    """Token counts reported by a completion or response call.

    Provider field names (``prompt_tokens`` / ``completion_tokens``) are accepted
    as validation aliases.
    """

    input_tokens: int = Field(
        default=0, validation_alias=AliasChoices("prompt_tokens", "input_tokens")
    )
    output_tokens: int = Field(
        default=0, validation_alias=AliasChoices("completion_tokens", "output_tokens")
    )
    total_tokens: int = 0
