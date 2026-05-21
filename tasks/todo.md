# Task: Suite Live Progress Output

## Approach

- Add an opt-out `verbose` parameter to `Suite.run()`.
- Emit live progress only when `verbose=True` and `sys.stdout.isatty()`.
- Reuse existing scenario status symbols from `STATUS_MAPPING`.
- Cover TTY output, `verbose=False`, non-TTY suppression, and incremental serial output in tests.

## Verification

- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run pytest -q libs/giskard-checks -m "not functional"`
- `uv tool run vermin --target=3.12- --no-tips --violations .`
- `uv run pip-audit --skip-editable`
- `uv tool run licensecheck --license MIT --show-only-failing --zero --skip-dependencies giskard-agents giskard-checks giskard-core`

## Review Notes

- `make` is not available on this Windows machine, so equivalent `uv` commands were used.
- Full and targeted `basedpyright` commands timed out locally on Windows; no output diagnostics were produced.
