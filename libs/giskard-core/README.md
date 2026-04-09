# Giskard Core

Core shared utilities and foundational components for the Giskard library ecosystem.

## Installation

```bash
uv add giskard-core
# or
pip install giskard-core
```

## Requirements

- Python >= 3.12
- pydantic >= 2.11.0, < 3

## Telemetry

`giskard-core` includes **optional usage analytics** to help maintainers understand how the libraries are used (versions, environment, coarse feature usage). Analytics are sent to **[PostHog](https://posthog.com)** hosted in the **EU** (`https://eu.i.posthog.com`).

### What we collect

Installed versions of `giskard-core`, `giskard-checks`, and `giskard-agents` (each the package version or `not_installed`), a coarse **environment** label (`ci`, `colab`, `kaggle`, or `local`), and when you run **giskard-checks** flows, aggregated non-content metrics such as step counts, counts of checks by `kind`, booleans (e.g. custom trace type or target present), durations, and pass/fail/skip-style outcomes. **Scenario names, prompts, model outputs, trace content, and exception messages are not sent.**

If an error propagates through the telemetry context, a single event may record **`exception_type`** (the Python class name only), not the exception string or traceback.

### Anonymous identifier

A random **persistent ID** may be stored under `~/.giskard/id` so repeated sessions can be counted without logging in. If the home directory is not writable, a one-off anonymous value is used for that process instead.

### How to disable telemetry

Set any of these environment variables to a truthy value **before** importing `giskard` packages (values are matched case-insensitively; common examples: `1`, `true`, `yes`, `on`):

- `DO_NOT_TRACK`
- `GISKARD_TELEMETRY_DISABLED`

You can also call `disable_telemetry()` from `giskard.core` at runtime to turn off further sends for that process (for example from test harnesses). That does not remove `~/.giskard/id` if it was already created; use env-based opt-out before import to avoid writing the file.

```python
from giskard.core import disable_telemetry

disable_telemetry()
```

### For library authors

The client and helpers are exported from `giskard.core`: `telemetry`, `telemetry_tag`, `telemetry_run_context`, `scoped_telemetry`, and `disable_telemetry`. New events should follow the same rules: **no user strings, secrets, file paths, or model I/O** in analytics payloads.

## Development

```bash
make setup    # Install dependencies
make test     # Run tests
make lint     # Run linting
make format   # Format code
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
