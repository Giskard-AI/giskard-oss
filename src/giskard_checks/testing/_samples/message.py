"""Minimal Message model to replace counterpoint dependency."""

from pydantic import BaseModel


class Message(BaseModel):
    """Minimal message model for chat interactions.

    This replaces the counterpoint.Message dependency with a simple Pydantic model
    that provides the same interface for testing purposes.
    """

    role: str
    content: str
