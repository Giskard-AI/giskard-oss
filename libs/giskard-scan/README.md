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
