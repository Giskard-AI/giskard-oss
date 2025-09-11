from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

from pydantic import Field

from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.extraction import Extractor, JsonPathExtractor
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class ExtractionCheck(Check[InteractionT], ABC):
    """Abstract base class for checks that extract values from interactions and evaluate them.

    This class provides common functionality for value extraction and evaluation patterns
    used by checks like EqualityCheck, StringMatchingCheck, and future LLM-based checks
    like ConformityCheck, RuleBasedCheck, etc.
    """

    # Common fields for value extraction
    extractor: Extractor | None = Field(
        default=None, description="Optional extractor for selecting values"
    )
    # Backcompat convenience to select values via JSONPath when no extractor is provided
    key: str | None = Field(default=None, description="JSON path to the key to check")
    # Evaluation mode - determines how multiple values are evaluated
    evaluation_mode: Literal["any", "all"] = Field(
        default="any",
        description="How to evaluate multiple values: 'any' (at least one must pass) or 'all' (all must pass)",
    )

    def _extract_values(self, interaction: InteractionT) -> list[Any]:
        """Extract values using configured extractor or fall back to JSONPath.

        This method handles the common pattern of:
        1. Using a custom extractor if provided
        2. Falling back to JSONPath extraction with a specific key
        3. Defaulting to extracting the 'output' field
        """
        if self.extractor is not None:
            return self.extractor.extract(interaction)
        elif self.key:
            return JsonPathExtractor(key=self.key).extract(interaction)
        else:
            # Default to selecting the output field via JSONPath
            return JsonPathExtractor(key="output").extract(interaction)

    @abstractmethod
    def _evaluate_values(self, values: list[Any]) -> bool:
        """Evaluate the extracted values according to the check's specific logic.

        Subclasses must implement this method to define their evaluation criteria.
        The method receives a list of extracted values and should return True if
        the check passes, False otherwise.

        Parameters
        ----------
        values : list[Any]
            The extracted values to evaluate

        Returns
        -------
        bool
            True if the check passes, False otherwise
        """
        pass

    @abstractmethod
    def _create_success_result(self, values: list[Any]) -> CheckResult:
        """Create a success result with appropriate message and details.

        Subclasses should implement this to provide meaningful success messages
        and relevant details for their specific check type.
        """
        pass

    @abstractmethod
    def _create_failure_result(self, values: list[Any]) -> CheckResult:
        """Create a failure result with appropriate message and details.

        Subclasses should implement this to provide meaningful failure messages
        and relevant details for their specific check type.
        """
        pass

    async def run(self, interaction: InteractionT) -> CheckResult:
        """Execute the check by extracting values and evaluating them.

        This method implements the common pattern:
        1. Extract values from the interaction
        2. Evaluate the values according to the check's logic
        3. Return appropriate success or failure result
        """
        values = self._extract_values(interaction)

        # Evaluate values based on evaluation mode
        if self.evaluation_mode == "all":
            # For "all" mode, we need all values to pass the evaluation
            passed = all(self._evaluate_values([value]) for value in values)
        else:  # evaluation_mode == "any"
            # For "any" mode, we need at least one value to pass
            passed = any(self._evaluate_values([value]) for value in values)

        if passed:
            return self._create_success_result(values)
        else:
            return self._create_failure_result(values)
