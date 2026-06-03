from typing import ClassVar

from .base import DatasetScenarioGenerator


class PromptInjectionScenarioGenerator(DatasetScenarioGenerator):
    """Dataset-backed generator for prompt-injection test scenarios.

    Loads scenarios from the bundled ``prompt_injection.jsonl`` dataset and
    tags them as OWASP LLM Top-10 2025 – LLM01 (prompt injection) threats.
    Sampling behaviour is inherited from
    :class:`~giskard.checks.scenarios_generator.base.DatasetScenarioGenerator`.
    """

    tags: ClassVar[list[str]] = [
        "gsk:threat-type='prompt-injection'",
        "owasp:llm-top-10-2025='LLM01'",
    ]
    dataset_name: str = "prompt_injection"
