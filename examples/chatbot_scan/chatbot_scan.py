"""Chatbot scan example using giskard.llm response API.

Demonstrates scanning a stateful chatbot (one string input per turn,
conversation state threaded via response id) with a generated suite.

Usage:
    Copy .env.example to .env, fill in your API key, then run:
    uv run python examples/chatbot_scan/chatbot_scan.py

Environment variables (can be set in .env):
    OPENAI_API_KEY  — required for the default model
    CHATBOT_MODEL   — model string for the chatbot, e.g. "openai/gpt-4o-mini" (default)
    GISKARD_MODEL   — model string for giskard judge/generator, e.g. "openai/gpt-4o-mini" (default)
"""

import asyncio
import os

from dotenv import load_dotenv

from giskard.agents.generators import Generator
from giskard.checks import Trace, generate_suite, set_default_generator
from giskard.llm import acompletion, chat
from giskard.llm.types import AssistantMessage, ChatMessage, UserMessage

load_dotenv()

MODEL = os.environ.get("CHATBOT_MODEL", "openai/gpt-4o-mini")
set_default_generator(
    Generator(model=os.environ.get("GISKARD_MODEL", "openai/gpt-4o-mini"))
)


class LLMTrace(Trace[UserMessage, AssistantMessage], frozen=True):
    """Minimal Trace implementation for tests."""

    @property
    def messages(self) -> list[ChatMessage]:
        return [
            message
            for interaction in self.interactions
            for message in (interaction.inputs, interaction.outputs)
        ]

    def _repr_prompt_(self) -> str:
        if not self.interactions:
            return "**No interactions yet**"
        return "\n".join(
            f"[user]: {i.inputs}\n[assistant]: {i.outputs}" for i in self.interactions
        )


_SYSTEM_MESSAGE = chat.system(
    "You are a helpful customer support agent for an e-commerce store."
)


async def chatbot(inputs: UserMessage, trace: LLMTrace) -> AssistantMessage:
    """Customer support chatbot backed by MODEL via the response API."""

    result = await acompletion(MODEL, [_SYSTEM_MESSAGE] + trace.messages + [inputs])

    return result.choices[0].message


async def main() -> None:
    print("Generating test suite...")
    suite = await generate_suite(
        description=(
            "A customer support chatbot for an e-commerce store "
            "that helps users with orders, returns, and product questions."
        ),
        languages=["en"],
        max_scenarios=20,
    )
    print(f"Generated {len(suite.scenarios)} scenarios. Running suite...\n")

    result = await suite.run(target=chatbot)
    result.print_report()

    # Save result to JSON
    result_path = os.path.join(os.path.dirname(__file__), "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    print(f"\nSuite result saved to {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
