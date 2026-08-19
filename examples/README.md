# Examples

These examples show public Giskard APIs. They run offline and are checked in CI.

Run all examples from the repository root:

```bash
make test-examples
```

## Checks static

`checks_static/test_checks_static.py` shows a small `Scenario` with two
serializable `Equals` checks. It is a minimal happy-path example for the checks
API.

## Scan custom generators

`scan_custom_generators/test_scan_custom_generators.py` shows how to pass a
custom `ScenarioGenerator` to `generate_suite`. The generator returns a static
scenario, so the whole scan runs without monkeypatching, a model provider, or
network access.

`generate_suite` gives each generator the same agent context and a scenario
budget. This example returns one deterministic scenario, then runs the suite
against the target. Use this pattern when you want to test a known risk or add
domain-specific scenarios to a scan.

Repository maintenance tools do not belong in this directory. See `tools/`
for checks such as the README snippet linter.
