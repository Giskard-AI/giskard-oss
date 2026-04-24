from typing import Literal

from .types import (
    AssistantMessage,
    ChatMessage,
    DeveloperMessage,
    SystemMessage,
    UserMessage,
)


def user(content: str) -> UserMessage:
    return UserMessage(content=content)


def assistant(content: str) -> AssistantMessage:
    return AssistantMessage(content=content)


def system(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def developer(content: str) -> DeveloperMessage:
    return DeveloperMessage(content=content)


def message(
    content: str, role: Literal["user", "assistant", "system", "developer"]
) -> ChatMessage:
    match role:
        case "user":
            return user(content)
        case "assistant":
            return assistant(content)
        case "system":
            return system(content)
        case "developer":
            return developer(content)
        case _:
            raise ValueError(f"Unknown role: {role!r}")
