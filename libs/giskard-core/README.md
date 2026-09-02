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

Installed versions of Giskard libraries (each the package version or `not_installed`), a coarse **environment** label (`ci`, `colab`, `kaggle`, or `local`), and aggregated non-content metrics for Checks runs, Scan runs, and Hub-format exports. These metrics include counts, fixed feature modes, booleans, durations, and pass/fail/error/skip-style outcomes. **Descriptions, language values, scenario and check names, prompts, model outputs, recommendations, trace content, file paths, and exception messages are not sent.**

If an error propagates through the telemetry context, a single event may record **`exception_type`** (the Python class name only), not the exception string or traceback.

### Anonymous identifier

A random **persistent ID** may be stored under `~/.giskard/id` so repeated sessions can be counted without logging in. If the home directory is not writable, a one-off anonymous value is used for that process instead.

### How to disable telemetry

Set any of these environment variables to a truthy value (matched case-insensitively; wrapping quotes are ignored; common examples: `1`, `true`, `yes`, `on`):

- `DO_NOT_TRACK`
- `GISKARD_TELEMETRY_DISABLED`

Flags are read from the **process environment** and from a **`.env` file in the working directory** (process env wins). Set them **before importing** Giskard packages to skip creating `~/.giskard/id` and to never start the analytics sender. If you set them later, further events are dropped and the sender is stopped so **no requests are made** to PostHog (including in environments that block `eu.i.posthog.com`).

You can also call `disable_telemetry()` from `giskard.core` at runtime to turn off further sends for that process (for example from test harnesses). That does not remove `~/.giskard/id` if it was already created; use env-based opt-out before import to avoid writing the file.

```python
from giskard.core import disable_telemetry

disable_telemetry()
```

### GeoIP

When telemetry is **enabled**, PostHog may apply **GeoIP** enrichment to events (server-side location metadata used in dashboards). That enrichment is **off** whenever telemetry is fully disabled (see above) or when you call `disable_telemetry()`.

To **keep usage analytics** but **disable GeoIP only**, set this environment variable to a truthy value (same matching rules and `.env` behavior as in [How to disable telemetry](#how-to-disable-telemetry)):

- `GISKARD_TELEMETRY_DISABLE_GEOIP`

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
