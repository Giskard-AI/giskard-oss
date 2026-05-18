from typing import ClassVar

from .base import DatasetScenarioGenerator


class PromptInjectionScenarioGenerator(DatasetScenarioGenerator):
    tags: ClassVar[list[str]] = [
        "gsk:threat-type='prompt-injection'",
        "owasp:llm-top-10-2025='LLM01'",
    ]
    dataset_name: str = "prompt_injection"
