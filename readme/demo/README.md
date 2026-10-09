# README demo replay

Offline replay of a realistic `vulnerability_scan` / suite terminal session for
README GIFs and local previews.

## Regenerate the fixture

```bash
GISKARD_QUIET=1 uv run python readme/demo/build_fixture.py
```

## Play the demo (live terminal)

Compact phased replay (default — clears between generation / progress / report):

```bash
GISKARD_QUIET=1 uv run python readme/demo/replay.py
```

Full typed-snippet tutorial:

```bash
GISKARD_DEMO_COMPACT=0 GISKARD_QUIET=1 uv run python readme/demo/replay.py
```

Speed control via `GISKARD_DEMO_DELAY_SCALE` (default `1`).

## Render the README GIF

Needs optional tooling (`pillow`) and `fonts-dejavu-mono` — not part of package deps:

```bash
uv pip install pillow
GISKARD_QUIET=1 uv run python readme/demo/render_gif.py
```

Writes `readme/vulnerability_scan.gif` (800px wide). Frames come from Rich
`export_svg()` text layout, rasterized with DejaVu Sans Mono so box-drawing
glyphs render (instead of tofu/squares).
