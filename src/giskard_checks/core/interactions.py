# interactlab/core/interaction.py
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(BaseModel, Generic[InputT, OutputT]):
    input: InputT
    output: OutputT | None = None
    metadata: dict[str, Any] | None = None
