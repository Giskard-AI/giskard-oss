"""Test optional template context independently of judge backends."""

from typing import Any

import pytest
from giskard.agents.templates import get_prompts_manager
from giskard.checks import Interaction, Trace

_OUTPUT_INSTRUCTIONS = "OUTPUT_FORMAT_INSTRUCTIONS"


@pytest.fixture
def scan_trace() -> Trace[str, str]:
    return Trace(
        interactions=[
            Interaction(inputs="INITIAL_REQUEST", outputs="INITIAL_RESPONSE"),
            Interaction(inputs="FOLLOWUP_REQUEST", outputs="FOLLOWUP_RESPONSE"),
        ],
        annotations={
            "description": "AGENT_DESCRIPTION",
            "goal": "ATTACK_GOAL",
            "strategies": [
                {"name": "STRATEGY_NAME", "definition": "STRATEGY_DEFINITION"}
            ],
            "behavior": "HARMFUL_BEHAVIOR",
        },
    )


@pytest.mark.parametrize(
    "template", ["goat_eval", "crescendo_eval", "harmbench_safety"]
)
@pytest.mark.parametrize(
    "include_trace", [None, True, False], ids=["default", "inline", "separate"]
)
@pytest.mark.parametrize("include_output_instructions", [True, False])
async def test_scan_judge_template_preserves_criteria_with_optional_context(
    template: str,
    include_trace: bool | None,
    include_output_instructions: bool,
    scan_trace: Trace[str, str],
):
    context: dict[str, Any] = {"trace": scan_trace}
    if include_trace is not None:
        context["include_trace"] = include_trace
    if include_output_instructions:
        context["_instr_output"] = _OUTPUT_INSTRUCTIONS

    messages = await get_prompts_manager().render_template(
        f"giskard.scan::judges/{template}.j2", context
    )
    prompt = "\n".join(message.text or "" for message in messages)

    assert (_OUTPUT_INSTRUCTIONS in prompt) is include_output_instructions
    assert "passed=true" in prompt
    assert "passed=false" in prompt
    if template == "harmbench_safety":
        assert "HARMFUL_BEHAVIOR" in prompt
        assert "Refusal handling" in prompt
    else:
        assert "AGENT_DESCRIPTION" in prompt
        assert "ATTACK_GOAL" in prompt
    if template == "goat_eval":
        assert "STRATEGY_NAME" in prompt
        assert "STRATEGY_DEFINITION" in prompt

    if include_trace is False:
        for marker in (
            "INITIAL_REQUEST",
            "INITIAL_RESPONSE",
            "FOLLOWUP_REQUEST",
            "FOLLOWUP_RESPONSE",
        ):
            assert marker not in prompt
    else:
        assert "FOLLOWUP_REQUEST" in prompt
        assert "FOLLOWUP_RESPONSE" in prompt
        if template != "harmbench_safety":
            assert "INITIAL_REQUEST" in prompt
            assert "INITIAL_RESPONSE" in prompt
