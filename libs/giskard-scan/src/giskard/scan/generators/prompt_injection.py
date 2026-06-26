from pydantic import Field

from .base import LocalDatasetScenarioGenerator


class PromptInjectionScenarioGenerator(LocalDatasetScenarioGenerator):
    """Dataset-backed generator for prompt-injection test scenarios.

    Loads scenarios from the bundled ``prompt_injection.jsonl`` dataset and
    tags them as OWASP LLM Top-10 2025 – LLM01 (prompt injection) threats.
    Sampling behaviour is inherited from LocalDatasetScenarioGenerator.
    """

    dataset_name: str = "prompt_injection"
    tags: list[str] = Field(
        default_factory=lambda: [
            "threat-type:prompt-injection",
            "owasp:llm-top-10-2025:LLM01",
        ]
    )
