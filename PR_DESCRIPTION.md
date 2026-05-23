# Add PIIDetection Built-in LLM Judge Check (Closes #2374)

## Summary

This PR implements a complete `PIIDetection` check for detecting personally identifiable information (PII) in AI agent responses. The check follows the existing `Toxicity` check architecture and integrates seamlessly with the Giskard checks framework.

## What's Included

### 🎯 Core Implementation

**PIIDetection Check Class** (`libs/giskard-checks/src/giskard/checks/judges/pii_detection.py`)
- LLM-based judge check extending `BaseLLMCheck`
- Registration name: `"pii_detection"`
- Full type safety with Literal types for categories and modes
- Complete docstring with usage examples

**Features:**
- **9 PII Categories**: email, phone, SSN, credit card, IP address, name, address, medical info, financial info
- **3 Detection Modes**: 
  - `"pattern"` - Regex-based detection for structured PII
  - `"llm"` (default) - LLM-based contextual detection
  - `"hybrid"` - Combined approach
- **Configurable Categories**: Filter to specific PII types via `categories` parameter
- **Custom Extraction**: Support for custom output paths via `output_key` parameter
- **Serialization**: Full Pydantic support for `Check.model_validate()` round-trip

### 🎨 Prompt Template

**Jinja2 Template** (`libs/giskard-checks/src/giskard/checks/prompts/judges/pii_detection.j2`)
- Comprehensive explanations of all PII categories with examples
- Clear guidance for structured and contextual PII detection
- Detection mode-aware instructions
- Standard JSON output format (`passed` + `reason`)
- Follows existing judge prompt conventions

### ✅ Comprehensive Test Suite

**22 Test Cases** (`libs/giskard-checks/tests/builtin/test_pii_detection.py`)

Core functionality:
- Clean content passes the check
- PII-containing content fails appropriately
- Output extraction from trace (with and without custom `output_key`)
- Full trace context in prompts (for multi-turn conversations)

Category coverage:
- Individual tests for all 9 PII categories (email, phone, SSN, credit card, IP address, name, address, medical, financial)
- Category filtering validation

Mode testing:
- Pattern mode functionality
- LLM mode functionality
- Hybrid mode functionality

Edge cases:
- Direct output overrides trace output
- None reason handling
- Default categories applied when not specified
- Serialization round-trip (model_dump + model_validate)

### 📦 Updated Exports

- `libs/giskard-checks/src/giskard/checks/judges/__init__.py` - Added PIIDetection export
- `libs/giskard-checks/src/giskard/checks/__init__.py` - Added PIIDetection to top-level exports

## Validation

✅ **All 22 new tests pass**
```
libs/giskard-checks/tests/builtin/test_pii_detection.py::test_clean_content_passes PASSED
libs/giskard-checks/tests/builtin/test_pii_detection.py::test_pii_content_fails PASSED
libs/giskard-checks/tests/builtin/test_pii_detection.py::test_output_extracted_from_trace PASSED
[... 19 more tests ...]
```

✅ **No regressions** - All 11 existing toxicity tests still pass

✅ **Serialization verified** - Round-trip serialization works correctly

✅ **Top-level import works** - `from giskard.checks import PIIDetection`

✅ **Template registered** - Prompt template correctly referenced and loaded

## Architecture Alignment

This implementation strictly follows the established patterns:
- Inherits from `BaseLLMCheck` (same as Toxicity)
- Uses `@Check.register()` decorator for registration
- Implements `get_prompt()` returning `TemplateReference`
- Implements `get_inputs()` building template context
- Uses Pydantic `Field()` with proper descriptions
- Follows naming conventions and code style
- Minimal, idiomatic Python

## Usage Example

```python
from giskard.checks import PIIDetection, Scenario
from giskard.agents.generators import Generator

# Check with default settings (all categories, LLM mode)
check = PIIDetection(
    generator=Generator(model="openai/gpt-4o"),
)

# Or with custom configuration
check = PIIDetection(
    output="Response to evaluate for PII",
    categories=["email", "phone", "ssn"],
    mode="hybrid",
    generator=Generator(model="openai/gpt-4o"),
)

# Use in a scenario
scenario = (
    Scenario(name="pii_safety")
    .interact(
        inputs="What's your contact info?",
        outputs="My email is john@example.com"
    )
    .check(PIIDetection())
)
```

## Files Changed

```
libs/giskard-checks/src/giskard/checks/judges/pii_detection.py (NEW)
├── PIICategory type alias (9 categories)
├── DEFAULT_PII_CATEGORIES tuple
├── PIIDetectionMode type alias (3 modes)
└── PIIDetection class (126 lines)

libs/giskard-checks/src/giskard/checks/prompts/judges/pii_detection.j2 (NEW)
└── Comprehensive Jinja2 template (110 lines)

libs/giskard-checks/tests/builtin/test_pii_detection.py (NEW)
└── 22 async test functions (400+ lines)

libs/giskard-checks/src/giskard/checks/judges/__init__.py (MODIFIED)
└── Added PIIDetection import and export

libs/giskard-checks/src/giskard/checks/__init__.py (MODIFIED)
└── Added PIIDetection to judges import and __all__ list
```

## Checklist

- [x] Follows existing Toxicity check architecture exactly
- [x] Type-safe implementation with Literal types
- [x] Complete docstring with examples
- [x] Comprehensive test coverage (22 tests)
- [x] All new tests pass
- [x] No regressions in existing tests
- [x] Serialization round-trip verified
- [x] Proper exports in __init__ files
- [x] Prompt template correctly registered
- [x] Code style matches repository conventions
- [x] Minimal, idiomatic implementation
- [x] Ready for merge

## Related Issue

Closes #2374
