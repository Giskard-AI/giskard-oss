"""Render ``readme/vulnerability_scan.gif`` from Rich SVG frames (no GUI).

Requires optional tooling (not part of the package deps)::

    uv pip install cairosvg pillow
    GISKARD_QUIET=1 uv run python readme/demo/render_gif.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from giskard.checks.core.result import (
    STATUS_MAPPING,
    SuiteResult,
    format_status_count_parts,
)
from PIL import Image
from replay import FIXTURE_PATH, _load_suite_result
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[2]
OUT_GIF = ROOT / "readme" / "vulnerability_scan.gif"
WIDTH = 800


def _console() -> Console:
    return Console(
        record=True,
        width=88,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        file=io.StringIO(),
    )


def _to_image(console: Console) -> Image.Image:
    svg = console.export_svg(title="")
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=WIDTH)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _pad(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Pad/crop to a fixed canvas so GIF frames share one size."""
    canvas = Image.new("RGBA", size, (13, 17, 23, 255))  # #0d1117
    canvas.paste(img, (0, 0), img if img.mode == "RGBA" else None)
    return canvas


def _generation_frame(n: int, step: int) -> Image.Image:
    console = _console()
    console.print(Text("$ python scan_shopbot.py", style="bold green"))
    console.print()
    console.print(
        Text.assemble(
            ("Generating adversarial suite", "bold cyan"),
            (f" · {n} scenarios from description", "dim"),
        )
    )
    steps = [
        "Sampling Prompt Injection generators…",
        "Sampling Harmful Content + Jailbreak generators…",
        "Sampling Misinformation + Excessive Agency…",
        "Building Scenario objects and checks…",
    ]
    for i, line in enumerate(steps):
        if i <= step:
            console.print(f"  • {line}")
    return _to_image(console)


def _progress_frame(result: SuiteResult, completed: int) -> Image.Image:
    console = _console()
    console.print(Text("$ python scan_shopbot.py", style="bold green"))
    console.print()
    console.print(Text("Running suite", style="bold cyan"))
    total = len(result.results)
    done = min(completed, total)
    bar_width = 28
    filled = int(bar_width * done / total) if total else 0
    bar = "━" * filled + "─" * (bar_width - filled)
    if done < total:
        name = result.results[done].scenario_name
        console.print(f"  Running: {name}  {bar}  {done}/{total}")
        console.print(f"    ↳ {name}")
    else:
        console.print(f"  Running scenarios  {bar}  {done}/{total}")
    counts = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
    for scenario in result.results[:done]:
        counts[scenario.status.value] += 1
    parts = format_status_count_parts(counts)
    if parts:
        console.print(Text.from_markup("  " + ", ".join(parts)))
    return _to_image(console)


def _report_frame(result: SuiteResult) -> Image.Image:
    console = _console()
    console.print(Text("$ python scan_shopbot.py", style="bold green"))
    console.print()
    console.print(Rule("Suite Results", style="bold blue"))
    console.print(
        "".join(
            f"[{STATUS_MAPPING[r.status]['color']}]"
            f"{STATUS_MAPPING[r.status]['symbol']}"
            f"[/{STATUS_MAPPING[r.status]['color']}]"
            for r in result.results
        )
    )
    console.print()
    console.print(Rule(style="bold blue"))
    count_parts = [
        (
            f"[{STATUS_MAPPING['total']['color']} bold]{len(result.results)} total"
            f"[/{STATUS_MAPPING['total']['color']} bold]"
        )
    ]
    count_parts.extend(
        format_status_count_parts(
            {
                "error": result.errored_count,
                "fail": result.failed_count,
                "skip": result.skipped_count,
                "pass": result.passed_count,
            }
        )
    )
    pass_rate = f"{result.pass_rate:.1%}" if result.pass_rate is not None else "—"
    console.print(
        "Summary: "
        + ", ".join(count_parts)
        + f" | Pass Rate: [default bold]{pass_rate}[/default bold]"
        + f" | Total Duration: {result.duration_ms}ms"
    )
    grouped = result.group_by("threat-type")
    table = Table(title="Results by threat-type")
    table.add_column("threat-type", style="bold")
    table.add_column("Pass Rate", justify="right")
    for group_value, stats in grouped.groups.items():
        display_name = "(untagged)" if group_value is None else group_value or "true"
        rate = (
            f"{stats.passed} / {stats.non_skipped}"
            if stats.pass_rate is not None
            else "—"
        )
        table.add_row(display_name, rate)
    console.print(table)
    if result.recommendation and result.recommendation.strip():
        console.print(
            Panel(
                Markdown(result.recommendation),
                title="Recommendation",
                border_style="blue",
            )
        )
    return _to_image(console)


def main() -> None:
    if not FIXTURE_PATH.is_file():
        from build_fixture import main as write_fixture

        write_fixture()
    result = _load_suite_result()

    frames: list[Image.Image] = []
    durations: list[int] = []

    # Generation phase (reveal bullets)
    for step in range(4):
        frames.append(_generation_frame(len(result.results), step))
        durations.append(450 if step < 3 else 900)

    # Progress phase
    for completed in range(len(result.results) + 1):
        frames.append(_progress_frame(result, completed))
        durations.append(380 if completed < len(result.results) else 900)

    # Final report
    frames.append(_report_frame(result))
    durations.append(3500)

    max_h = max(im.height for im in frames)
    canvas_size = (WIDTH, max_h)
    padded = [_pad(im, canvas_size) for im in frames]

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    padded[0].save(
        OUT_GIF,
        save_all=True,
        append_images=padded[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Wrote {OUT_GIF} ({OUT_GIF.stat().st_size} bytes, {len(padded)} frames)")


if __name__ == "__main__":
    main()
