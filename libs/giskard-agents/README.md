# giskard-agents

LLM workflow orchestration — generators, chat pipelines, Jinja templates, and tools. Used by `giskard-checks` for LLM-backed checks and user simulation.

```bash
pip install giskard-agents
```

Requires Python 3.12+.

**Telemetry:** depends on `giskard-core`. See [Telemetry](../giskard-core/README.md#telemetry).

## Development

From the repository root — see [CONTRIBUTING.md](../../CONTRIBUTING.md):

```bash
make test-unit PACKAGE=giskard-agents
```
