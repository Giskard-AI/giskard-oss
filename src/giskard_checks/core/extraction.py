from __future__ import annotations

from typing import Any, ClassVar

from jsonpath_ng import parse
from pydantic import BaseModel, ConfigDict, Field

from giskard_checks.core.interactions import Interaction


class Extractor(BaseModel):
    """Base extractor strategy.

    Implementations return a list of extracted items from an `Interaction`.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    def extract(self, interaction: Interaction[Any, Any]) -> list[Any]:
        raise NotImplementedError


class JsonPathExtractor(Extractor):
    """Extract values using a JSONPath expression evaluated on `interaction.model_dump()`.

    The extractor unwraps objects that expose a `value` attribute (e.g., jsonpath-ng DatumInContext)
    and returns a plain list of Python values.
    """

    key: str = Field(..., description="JSONPath selecting values to extract")

    def extract(self, interaction: Interaction[Any, Any]) -> list[Any]:
        expr = parse(self.key)
        matches = expr.find(interaction.model_dump())
        results: list[Any] = []
        for item in matches:
            value = getattr(item, "value", item)
            results.append(value)
        return results
