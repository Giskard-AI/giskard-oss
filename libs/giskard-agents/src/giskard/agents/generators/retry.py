import sys

from pydantic import BaseModel, Field

MAX_WAIT_SECONDS = sys.maxsize / 2


class RetryPolicy(BaseModel):
    """Adds a retry policy to the generator.

    Attributes
    ----------
    max_retries : int
        Maximum number of retry attempts.
    base_delay : float
        Base delay in seconds for exponential backoff.
    max_delay : float | None
        Maximum delay in seconds between retries. If None, defaults to MAX_WAIT_SECONDS.
    """

    max_retries: int = Field(default=3)
    base_delay: float = Field(default=1.0)
    max_delay: float | None = Field(default=None)
