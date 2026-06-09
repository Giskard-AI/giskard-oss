## Description

Adds a built-in `PIIDetection` check that flags personally identifiable information (PII)
in agent responses using a **hybrid regex-pattern + LLM-judgment** approach.

- Structured PII (emails, phone numbers, SSNs, credit cards, IP addresses) is caught
  deterministically by compiled regex patterns — no LLM call needed.
- Contextual PII (names, addresses, medical details, financial details) is caught by an LLM
  judge that returns a structured `PIIJudgeResult` (fields: `passed`, `reason`,
  `categories_detected`, `confidence`, `severity`) — no fragile free-text parsing.
- In `"hybrid"` mode (the default) patterns run first; if high/critical-severity structured
  PII is found the check fails immediately without an LLM call, otherwise the LLM evaluates
  contextual PII and its findings are merged with any pattern matches.
- Follows the existing judge architecture (same shape as `Toxicity`, extends `BaseLLMCheck`)
  and integrates with the Scenario/Suite framework via the discriminated `Check` union.
- Result `details` expose `severity` (low/medium/high/critical), `confidence` (0–1),
  `detected_via` (pattern/llm/hybrid), and `categories_detected`.
- Compiled regex patterns are cached at module level across instances.

**Files changed:**
- `libs/giskard-checks/src/giskard/checks/judges/pii_detection.py` — core implementation
- `libs/giskard-checks/src/giskard/checks/prompts/judges/pii_detection.j2` — judge prompt
- `libs/giskard-checks/tests/builtin/test_pii_detection.py` — 35 async tests
- `libs/giskard-checks/src/giskard/checks/judges/__init__.py` — export
- `libs/giskard-checks/src/giskard/checks/__init__.py` — export

**Validation:**
- `ruff format` + `ruff check`: clean
- `basedpyright --level error` (judges + tests): 0 errors
- `vermin --target=3.12-`: compatible
- PII tests: **35 passed**
- Full `giskard-checks` unit suite: **619 passed, 4 skipped** (no regressions)

## Related Issue

Closes #2374

## Type of Change

- [ ] 📚 Examples / docs / tutorials / dependencies update
- [ ] 🔧 Bug fix (non-breaking change which fixes an issue)
- [ ] 🥂 Improvement (non-breaking change which improves an existing feature)
- [x] 🚀 New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 🔐 Security fix

## Checklist

- [x] I've read the [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) document.
- [x] I've read the [`CONTRIBUTING.md`](../CONTRIBUTING.md) guide.
- [x] I've written tests for all new methods and classes that I created.
- [x] I've written the docstring in NumPy format for all the methods and classes that I created or modified.
- [ ] I've updated the `uv.lock` running `uv lock` (only applicable when `pyproject.toml` has been modified)
