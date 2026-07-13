# giskard-scan

Agent vulnerability scanner — red teaming, prompt injection, adversarial scenario generation.

## Third-party scanners (experimental)

`third_party_scan` runs an external security scanner against a Giskard target and
returns a `SuiteResult`. Two scanners are supported, each shipping as an optional
extra: [garak](https://github.com/NVIDIA/garak) and
[deepteam](https://github.com/confident-ai/deepteam).

```bash
pip install giskard-scan[garak]
pip install giskard-scan[deepteam]
```

### garak

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
        description="A helpful assistant",  # required by the API; garak ignores it
        probes=["probes.goodside.ThreatenJSON"],  # omit to run all active probes
        target_mode="multiturn",  # "singleturn" skips garak's iterative probes
    )
)

print(result)
```

Probes run in parallel; the target is invoked concurrently, so it must be safe to
call from multiple threads (per-conversation state is tracked in the `Trace`, not on
the target).

Unknown probe names are logged and skipped rather than raising.

### deepteam

Deepteam generates adversarial attacks with an LLM and judges the responses with an
LLM, so it needs a working Giskard default generator (see
`giskard.checks.get_default_generator()`) — there is no keyless mode.

```python
result = asyncio.run(
    third_party_scan(
        target,
        tool="deepteam",
        description="A helpful assistant",  # becomes deepteam's target_purpose
        vulnerabilities=["Bias", "Toxicity"],  # omit for a curated default set
        attacks=["PromptInjection", "LinearJailbreaking"],  # omit for defaults
        target_mode="multiturn",
    )
)
```

`vulnerabilities` accepts `Bias`, `Toxicity`, `PIILeakage`, `PromptLeakage`, and
`Misinformation`.

`attacks` accepts the single-turn `PromptInjection`, `Roleplay`, `Leetspeak`, and
`ROT13`, plus the multi-turn `LinearJailbreaking`, `CrescendoJailbreaking`,
`TreeJailbreaking`, `SequentialJailbreak`, and `BadLikertJudge`. An unrecognized
name raises `ValueError` listing the valid ones.

`target_mode="singleturn"` **drops every multi-turn attack** from the run. If that
leaves nothing to run — for example `attacks=["CrescendoJailbreaking"]` with
`target_mode="singleturn"` — the scan returns an empty `SuiteResult`, which reads
as "nothing failed" rather than as an error.

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
