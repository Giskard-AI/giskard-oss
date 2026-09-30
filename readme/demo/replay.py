"""Replay a pre-recorded vulnerability-scan suite result for README demos.

This is intentionally offline: it does not call LLMs. It loads a saved
``SuiteResult`` fixture (from a realistic scan shape), animates suite progress
with small delays, then prints the same Rich report ``vulnerability_scan`` would.

Usage::

    GISKARD_QUIET=1 uv run python readme/demo/replay.py

Optional speed control::

    GISKARD_DEMO_DELAY_SCALE=0.5 GISKARD_QUIET=1 uv run python readme/demo/replay.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from giskard.checks.core.interaction import Trace
from giskard.checks.core.result import (
    STATUS_MAPPING,
    ScenarioStatus,
    SuiteResult,
    format_status_count_text,
)
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "vulnerability_scan_suite.json"

# Snippet mirrors the README quickstart (fake agent kept offline).
DEMO_SNIPPET = """\
import asyncio
from giskard.scan import vulnerability_scan


async def shopbot(inputs: str) -> str:
    return await call_my_agent(inputs)  # your e-commerce support agent


async def main() -> None:
    await vulnerability_scan(
        target=shopbot,
        description="A customer support chatbot for an e-commerce platform.",
        languages=["en"],
        max_scenarios=12,
        group_by="threat-type",
    )


asyncio.run(main())
"""


def _delay_scale() -> float:
    raw = os.getenv("GISKARD_DEMO_DELAY_SCALE", "1")
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.0


def _sleep(seconds: float, scale: float) -> None:
    time.sleep(seconds * scale)


def _typewrite(console: Console, text: str, *, scale: float, cps: float = 55.0) -> None:
    """Print ``text`` with a light typing delay for tutorial feel."""
    delay = (1.0 / cps) if cps > 0 else 0.0
    for char in text:
        console.file.write(char)
        console.file.flush()
        if char not in (" ", "\n", "\t") and delay:
            _sleep(delay, scale)
    if not text.endswith("\n"):
        console.file.write("\n")
        console.file.flush()


def _load_suite_result() -> SuiteResult:
    if not FIXTURE_PATH.is_file():
        msg = (
            f"Missing fixture at {FIXTURE_PATH}. "
            "Run: GISKARD_QUIET=1 uv run python readme/demo/build_fixture.py"
        )
        raise FileNotFoundError(msg)
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    # SuiteResult is parameterized with Any for traces; rebuild Trace objects so
    # Rich uses Interaction.__rich_console__ instead of dumping a plain dict.
    for scenario in payload.get("results", []):
        final_trace = scenario.get("final_trace")
        if isinstance(final_trace, dict):
            scenario["final_trace"] = Trace.model_validate(final_trace)
    return SuiteResult.model_validate(payload)


def _scenario_delay_seconds(duration_ms: int) -> float:
    """Map recorded duration to a short on-screen delay (capped)."""
    # Real scans take seconds per scenario; compress to ~0.25–0.7s for the GIF.
    return min(0.7, max(0.22, duration_ms / 2500.0))


def _replay_progress(result: SuiteResult, *, scale: float) -> None:
    """Animate a suite-like progress bar from saved scenario timings/outcomes."""
    scenarios = result.results
    counts = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        overall = progress.add_task("Running scenarios", total=len(scenarios))
        for scenario in scenarios:
            name = scenario.scenario_name
            progress.update(overall, description=f"Running: {name}")
            row = progress.add_task(f"  ↳ {name}", total=None)
            _sleep(_scenario_delay_seconds(scenario.duration_ms), scale)
            progress.remove_task(row)
            counts[ScenarioStatus(scenario.status).value] += 1
            progress.advance(overall)
        _sleep(0.55, scale)
    summary = format_status_count_text(counts, prefix="  ")
    if summary is not None:
        Console().print(summary)


def _print_generation_phase(console: Console, *, scale: float, n: int) -> None:
    console.print(
        Text.assemble(
            ("Generating adversarial suite", "bold cyan"),
            (f" · {n} scenarios from description", "dim"),
        )
    )
    steps = [
        "Sampling Prompt Injection generators…",
        "Sampling Harmful Content + Jailbreak generators…",
        "Sampling Stereotypes, Misinformation, Prompt Leakage…",
        "Building Scenario objects and checks…",
    ]
    for step in steps:
        console.print(f"  • {step}")
        _sleep(0.35, scale)
    console.print()


async def _run() -> None:
    scale = _delay_scale()
    console = Console(force_terminal=True, color_system="truecolor", width=100)

    console.print()
    console.print(Text("$ python scan_shopbot.py", style="bold green"))
    console.print()
    _sleep(0.35, scale)

    # Show the script being “reviewed” before execution (tutorial framing).
    console.print(Text("# scan_shopbot.py", style="dim"))
    _typewrite(console, DEMO_SNIPPET, scale=scale, cps=90.0)
    console.print()
    _sleep(0.4, scale)

    result = _load_suite_result()
    _print_generation_phase(console, scale=scale, n=len(result.results))

    console.print(Text("Running suite", style="bold cyan"))
    _replay_progress(result, scale=scale)
    console.print()

    # Match vulnerability_scan default grouping.
    result.print_report(console=console, group_by="threat-type")
    console.print()

    symbols = "".join(STATUS_MAPPING[r.status]["symbol"] for r in result.results)
    counts = format_status_count_text(
        {
            "pass": result.passed_count,
            "fail": result.failed_count,
            "error": result.errored_count,
            "skip": result.skipped_count,
        }
    )
    if counts is None:
        console.print(Text.assemble(("Demo replay complete ", "dim"), (symbols, "")))
    else:
        console.print(
            Text.assemble(
                ("Demo replay complete ", "dim"),
                (symbols, ""),
                ("  ", ""),
                counts,
            )
        )


def main() -> None:
    # Ensure quiet welcome for clean recordings unless the user opted out.
    os.environ.setdefault("GISKARD_QUIET", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
