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

## Scan stub

`scan_stub/test_scan_stub.py` shows an offline `vulnerability_scan` flow. It
uses an empty generator list and a static suite stub, so it does not need a
model provider or network access.

Repository maintenance tools do not belong in this directory. See `tools/`
for checks such as the README snippet linter.
