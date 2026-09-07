---
name: check
description: Run the giskard-oss verification gate (format, check, unit tests) for a package
disable-model-invocation: true
---

# check

Run the giskard-oss verification gate that `AGENTS.md` mandates before any task is "done":

```
make format && make check && make test-unit PACKAGE=<affected-lib>
```

## Usage

`/check <package>` — run the full gate scoped to one lib's unit tests.

`/check` — run the gate; unit tests cover all libs (slower).

`<package>` is a directory under `libs/`: `giskard-agents`, `giskard-checks`, `giskard-core`, `giskard-llm`, `giskard-scan`.

## What to run

With a package:

```bash
make format && make check && make test-unit PACKAGE=<package>
```

Without a package:

```bash
make format && make check && make test-unit
```

`format` and `check` are repo-wide and take no `PACKAGE`. Only `test-unit` is scoped.

## Notes

- Run the three steps as one `&&` chain so a failure stops the gate early.
- Report failures with the command's output — do not claim the gate passed unless every step exited 0.
- `make format` mutates files (ruff format + check --fix); show the resulting diff if anything changed.
