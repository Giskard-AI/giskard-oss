# README demo replay

Offline replay of a realistic `vulnerability_scan` / suite terminal session for
README GIFs and local previews.

## Regenerate the fixture

```bash
GISKARD_QUIET=1 uv run python readme/demo/build_fixture.py
```

## Play the demo

```bash
GISKARD_QUIET=1 uv run python readme/demo/replay.py
```

Speed up or slow down with `GISKARD_DEMO_DELAY_SCALE` (default `1`):

```bash
GISKARD_DEMO_DELAY_SCALE=0.5 GISKARD_QUIET=1 uv run python readme/demo/replay.py
```
