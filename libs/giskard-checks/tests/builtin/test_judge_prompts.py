"""Static checks that built-in judge prompts request a clear reason."""

from pathlib import Path

import pytest

PROMPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "giskard"
    / "checks"
    / "prompts"
    / "judges"
)

JUDGE_PROMPTS = sorted(PROMPTS_DIR.glob("*.j2"))


@pytest.mark.parametrize("prompt_path", JUDGE_PROMPTS, ids=lambda p: p.name)
def test_built_in_judge_prompt_requests_clear_reason(prompt_path: Path) -> None:
    text = prompt_path.read_text(encoding="utf-8").lower()
    assert "reason" in text
    # Every built-in judge prompt must ask for a reason on the evaluation decision.
    asks_for_reason = (
        "provide a clear reason" in text
        or "provide a concise reason" in text
        or "with a brief reason" in text
        or "clear reason for your evaluation decision" in text
    )
    assert asks_for_reason, (
        f"{prompt_path.name} must instruct the model to provide a reason"
    )
