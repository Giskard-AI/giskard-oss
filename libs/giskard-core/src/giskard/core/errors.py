from pydantic import BaseModel


class Error(BaseModel):
    """A basic serializable error.

    This class provides a minimal, serializable error representation that
    can be used across the Giskard ecosystem for consistent error handling
    and serialization.

    Examples
    --------
    >>> error = Error(message="Something went wrong")
    >>> str(error)
    'ERROR: Something went wrong'
    >>> error.model_dump()
    {'message': 'Something went wrong'}
    """

    message: str

    def __str__(self) -> str:
        return "ERROR: " + self.message
