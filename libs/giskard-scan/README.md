# giskard-scan

Agent vulnerability scanner — red teaming, prompt injection, adversarial scenario generation.

## Third-party scanners (experimental)

`third_party_scan` runs an external security scanner against a Giskard target and
returns a `SuiteResult`. Only [garak](https://github.com/NVIDIA/garak) is supported
today, and it ships as an optional extra:

```bash
pip install giskard-scan[garak]
```

```python
import asyncio

from giskard.scan import third_party_scan


def target(inputs: str) -> str:
    # Your agent / model call. Structured (BaseModel) inputs also work.
    return call_my_agent(inputs)


result = asyncio.run(
    third_party_scan(
        target,
        tool="garak",
        probes=["probes.goodside.ThreatenJSON"],  # omit to run all active probes
        target_mode="multiturn",  # "singleturn" skips garak's iterative probes
    )
)

print(result)
```

Probes run in parallel; the target is invoked concurrently, so it must be safe to
call from multiple threads (per-conversation state is tracked in the `Trace`, not on
the target).

### API keys and LLM-judge detectors

Some garak detectors need an LLM or a third-party API to score a probe:

- **LLM-judge detectors** (garak's `judge.*`, e.g. refusal detection) normally require
  their own OpenAI key. Instead, they are automatically backed by Giskard's default
  generator (`giskard.checks.get_default_generator()`), so they run with the same
  credentials as the rest of Giskard — no separate OpenAI key needed.
- **Detectors that need a third-party API key** you have not set (for example
  `perspective.*`, which needs `PERSPECTIVE_API_KEY`) are **skipped** rather than
  silently dropping the whole probe. Each skipped detector surfaces as a skip result
  (`CheckResult.skip`) in the returned `SuiteResult`, with the missing key named in the
  message, so the rest of the probe's detectors still run.
