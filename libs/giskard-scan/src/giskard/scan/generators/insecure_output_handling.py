from pydantic import Field

from .base import LocalDatasetScenarioGenerator


class InsecureOutputHandlingScenarioGenerator(LocalDatasetScenarioGenerator):
    """Dataset-backed generator for insecure-output-handling test scenarios.

    Loads scenarios from the bundled ``insecure_output_handling.jsonl`` dataset
    and tags them as OWASP LLM Top-10 2025 – LLM02 (insecure output handling)
    threats. Each seed asks the agent for a classic XSS, SQL injection, shell
    injection, or path traversal payload and wires a ``Not(RegexMatching(...))``
    check over the reply, so a scenario fails when the agent emits the payload.
    Sampling behaviour is inherited from LocalDatasetScenarioGenerator.

    The regexes only detect *payload emission* – that the agent wrote a
    dangerous-looking string. Whether that string is actually exploitable
    depends on the sink it reaches (HTML, SQL, a shell, the filesystem), which
    these scenarios never touch. Treat the score as a lower bound: a clean run
    means the canonical payload forms were not emitted, not that output handling
    is safe. Obfuscated or encoded equivalents pass straight through.

    The match is on reply text alone, so naming a payload in order to warn
    against it counts the same as emitting it: advice such as "never allow
    ``javascript:`` URIs" fails the XSS scenario. Read a failure as "the
    payload appeared in the reply", not as "the agent complied".

    This generator is not part of the default vulnerability suite. Opt in with
    ``vulnerability_suite_generator_registry.register(InsecureOutputHandlingScenarioGenerator)``.
    """

    dataset_name: str = "insecure_output_handling"
    tags: list[str] = Field(
        default_factory=lambda: [
            "threat-type:insecure-output-handling",
            "owasp:llm-top-10-2025:LLM02",
        ]
    )
