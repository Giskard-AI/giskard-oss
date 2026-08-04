# giskard-scan

Agent vulnerability scanner — red teaming, prompt injection, adversarial scenario generation.

## Scan entrypoints

`quality_scan` and `vulnerability_scan` share the same explicit execution options.
Pass shared settings by keyword on either entrypoint; each scan adds scan-specific
options (`knowledge_base` for quality, `commercial_use` for vulnerability).

```python
from giskard.scan import quality_scan, vulnerability_scan


async def echo(inputs: str) -> str:
    return inputs


vulnerability_result = await vulnerability_scan(
    target=echo,
    description="Customer support chatbot for an e-commerce store.",
    languages=["en"],
    max_scenarios=20,
    seed=42,
    group_by="threat-type",
    parallel=True,
    max_concurrency=8,
    return_exception=False,
    target_mode="multiturn",
    commercial_use=False,
)

quality_result = await quality_scan(
    target=echo,
    description="Customer support chatbot for an e-commerce store.",
    languages=["en"],
    knowledge_base=["Paris is the capital of France."],
    max_scenarios=20,
    seed=42,
    group_by="component",
    parallel=True,
    max_concurrency=8,
    return_exception=False,
    target_mode="multiturn",
)
```

## Shared defaults

`DEFAULT_TARGET_MODE` (`"multiturn"`) is shared by `generate_suite`,
`vulnerability_scan`, `quality_scan`, and `third_party_scan`. Pass
`target_mode="singleturn"` to skip multi-turn generators/attacks and cap turn
budgets to 1.

Discover selectable items for any tool with `list_scan_items`:

```python
from giskard.scan import list_scan_items

list_scan_items("giskard")  # scenario generator class names
list_scan_items("garak")  # active garak probe names
list_scan_items("deepteam")  # vulnerability + attack names
```

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

from giskard.scan import list_scan_items, third_party_scan


def target(inputs: str) -> str:
    # Your agent / model call. Structured (BaseModel) inputs also work.
    return call_my_agent(inputs)


result = asyncio.run(
    third_party_scan(
        target,
        tool="garak",
        description="A helpful assistant",  # required by the API; garak ignores it
        # probes=None -> curated default set; probes="all" -> every active probe
        probes=["probes.goodside.ThreatenJSON"],
        # target_mode defaults to "multiturn"; pass "singleturn" to drop iterative probes
    )
)

print(result)
print(list_scan_items("garak")[:5])
```

Probes run in parallel; the target is invoked concurrently, so it must be safe to
call from multiple threads (per-conversation state is tracked in the `Trace`, not on
the target).

Omitting `probes` runs a small curated default set aligned with DeepTeam themes
(Bias/Toxicity, PII, Misinformation, PromptLeakage, jailbreak/injection, data
exfil) — not the full garak catalog. Pass `probes="all"` for every active probe,
or an explicit name list.

Unknown, inactive, or unloadable probe names are logged and emitted as
``CheckResult.skip`` scenarios rather than raising.

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
        attacks_per_vulnerability_type=1,  # default; each vuln subtype × this many
        # target_mode defaults to "multiturn" (shared with native Giskard scans)
    )
)
```

`vulnerabilities` accepts `Bias`, `Toxicity`, `PIILeakage`, `PromptLeakage`, and
`Misinformation`. Instantiating those classes without subtypes runs **all** of
their types (for example Bias → race, gender, politics, religion), so cost scales
with types × `attacks_per_vulnerability_type` × attacks.

`attacks` accepts the single-turn `PromptInjection`, `Roleplay`, `Leetspeak`, and
`ROT13`, plus the multi-turn `LinearJailbreaking`, `CrescendoJailbreaking`,
`TreeJailbreaking`, `SequentialJailbreak`, and `BadLikertJudge`. Unrecognized
names are logged and emitted as skip scenarios (valid names in the same call still
run).

`target_mode` defaults to the shared `DEFAULT_TARGET_MODE` (`"multiturn"`).
Pass `"singleturn"` to drop multi-turn attacks (they surface as skip scenarios).
Unknown names also skip — the suite is never an empty "everything passed"
result when you asked for attacks that could not run.

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
