from pydantic import BaseModel


class Error(BaseModel):
    """A basic serializable error."""

    message: str

    def __str__(self) -> str:
        return "ERROR: " + self.message
