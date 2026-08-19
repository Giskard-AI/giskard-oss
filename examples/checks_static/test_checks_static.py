"""Offline checks happy path — mirrors the canonical example contract."""

import asyncio

from giskard.checks import Equals, Scenario


def echo(inputs: str) -> str:
    return inputs


async def main() -> None:
    result = await (
        Scenario("echo")
        .interact(inputs="hello", outputs=echo)
        .check(Equals(target_key="trace.last.inputs", expected_value="hello"))
        .check(Equals(target_key="trace.last.outputs", expected_value="hello"))
        .run()
    )
    assert result.passed
    result.print_report()


async def test_checks_static_happy_path() -> None:
    await main()


if __name__ == "__main__":
    asyncio.run(main())
