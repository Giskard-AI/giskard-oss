from typing import Any

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    """Context object that tools can use to store and retrieve information across turns."""

    data: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the context.

        Parameters
        ----------
        key : str
            The key to store the value under.
        value : Any
            The value to store.
        """
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the context.

        Parameters
        ----------
        key : str
            The key to retrieve the value for.
        default : Any, optional
            The default value to return if the key is not found.

        Returns
        -------
        Any
            The stored value or the default.
        """
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the context.

        Parameters
        ----------
        key : str
            The key to check for.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return key in self.data

    def clear(self) -> None:
        """Clear all data from the context."""
        self.data.clear()
