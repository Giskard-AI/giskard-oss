# Add PIIDetection Built-in Check (Closes #2374)

## Summary

Adds a built-in `PIIDetection` check that flags personally identifiable information (PII)
in agent responses using a **hybrid regex-pattern + LLM-judgment** approach. Structured PII
(emails, phone numbers, SSNs, credit cards, IP addresses) is caught deterministically by
regex; contextual PII (names, addresses, medical, financial) is caught by an LLM judge. The
check follows the existing judge architecture (same shape as `Toxicity`) and integrates with
the Scenario/Suite framework.

## What's included

### Core implementation
`libs/giskard-checks/src/giskard/checks/judges/pii_detection.py`

- LLM-based judge extending `BaseLLMCheck`, registered as `"pii_detection"`.
- **9 categories**: email, phone, ssn, credit_card, ip_address, name, address, medical, financial.
- **Three detection modes** (`mode`, default `"hybrid"`):
  - `"pattern"` — fast, deterministic regex only.
  - `"llm"` — LLM judgment for structured and contextual PII.
  - `"hybrid"` — patterns first; if high/critical-severity structured PII is found the check
    fails immediately **without an LLM call**, otherwise the LLM evaluates contextual PII and
    its findings are merged with any pattern matches.
- **Structured LLM output** (`PIIJudgeResult`): the judge returns `passed`, `reason`,
  `categories_detected`, `confidence`, and `severity` directly — no parsing of free text.
- Result `details` expose `severity` (low/medium/high/critical), `confidence` (0–1),
  `detected_via` (pattern/llm/hybrid), and `categories_detected`.
- Configurable `categories`, custom `output_key` extraction, and full Pydantic
  serialization round-trip via the discriminated `Check` union.
- Compiled regex patterns are cached at module level across instances.

### Prompt template
`libs/giskard-checks/src/giskard/checks/prompts/judges/pii_detection.j2`

- Per-category guidance with false-positive caveats (test/example data, placeholders).
- Instructs the judge to emit the structured `PIIJudgeResult` fields (categories, severity,
  confidence) alongside the verdict.

### Tests
`libs/giskard-checks/tests/builtin/test_pii_detection.py` — 35 async tests

- Core: clean content passes, PII fails, trace extraction (incl. custom `output_key`),
  full-trace context for multi-turn conversations.
- Per-category coverage for all 9 categories and category filtering.
- Hybrid/pattern modes: pattern-only detection without LLM calls, hybrid early-exit on
  high-severity PII (asserts the LLM is not called), severity/confidence values, pattern cache.
- Structured output: `test_llm_structured_fields_propagate` verifies LLM-reported
  categories/severity/confidence flow into the result.
- Edge cases: direct output overrides trace, `None` reason handling, serialization round-trip.

### Exports
`judges/__init__.py` and `checks/__init__.py` export `PIIDetection`.

## Validation

- `ruff format` + `ruff check` (repo config): clean.
- `basedpyright --level error` (judges + tests): 0 errors.
- `vermin --target=3.12-`: compatible.
- PII tests: **35 passed**.
- Full `giskard-checks` unit suite: **619 passed, 4 skipped** (no regressions).

## Usage

```python
from giskard.checks import PIIDetection, Scenario
from giskard.agents.generators import Generator

# Default: hybrid mode, all categories
scenario = (
    Scenario(name="pii_safety")
    .interact(inputs="What's your contact info?", outputs="My email is john@example.com")
    .check(PIIDetection())
)

# Fast, deterministic-only
check = PIIDetection(
    output="Call me at 555-123-4567 or info@example.com",
    categories=["email", "phone"],
    mode="pattern",
)

# Contextual PII via LLM
check = PIIDetection(
    categories=["name", "address", "medical"],
    mode="llm",
    generator=Generator(model="openai/gpt-4o"),
)
```

## Acceptance criteria

- [x] Detects structured PII via regex (emails, phone numbers, SSNs)
- [x] Detects contextual PII via LLM (names, addresses)
- [x] Configurable category filtering
- [x] Hybrid mode combines both approaches
- [x] Tests cover: clean output passes, PII present fails, category filtering

## Related issue

Closes #2374
