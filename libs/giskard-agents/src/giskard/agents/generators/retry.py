from pydantic import BaseModel, Field


class RetryPolicy(BaseModel):
    """Adds a retry policy to the generator."""

    max_retries: int = Field(default=3)
    base_delay: float = Field(default=1.0)
