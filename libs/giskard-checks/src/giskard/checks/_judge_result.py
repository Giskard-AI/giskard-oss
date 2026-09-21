"""Result schema shared by LLM and SOM judges."""

from pydantic import BaseModel, Field, field_validator


class LLMCheckResult(BaseModel):
    """Default result model for LLM-based checks."""

    reason: str = Field(
        ...,
        min_length=1,
        description="Explanation for the pass or fail verdict",
    )
    passed: bool = Field(..., description="Whether the check passed or failed")

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_reason(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
