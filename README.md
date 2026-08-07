<p align="center">
  <img alt="giskardlogo" src="readme/logo_light.png#gh-light-mode-only">
  <img alt="giskardlogo" src="readme/logo_dark.png#gh-dark-mode-only">
</p>
<h1 align="center" weight='300' >Evals, Red Teaming and Test Generation for Agentic Systems</h1>
<h3 align="center" weight='300' >Modular, Lightweight, Dynamic and Async-first </h3>
<div align="center">

[![GitHub release](https://img.shields.io/github/v/release/Giskard-AI/giskard-oss)](https://github.com/Giskard-AI/giskard-oss/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Giskard-AI/giskard/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/giskard/month)](https://pepy.tech/project/giskard)
[![CI](https://github.com/Giskard-AI/giskard-oss/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Giskard-AI/giskard-oss/actions/workflows/ci.yml/badge.svg?branch=main)
[![Giskard on Discord](https://img.shields.io/discord/939190303397666868?label=Discord)](https://gisk.ar/discord)

<a rel="me" href="https://fosstodon.org/@Giskard"></a>

</div>
<h3 align="center">
   <a href="https://docs.giskard.ai/oss"><b>Docs</b></a> &bull;
  <a href="https://www.giskard.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readmeblog"><b>Website</b></a> &bull;
  <a href="https://gisk.ar/discord"><b>Community</b></a>
 </h3>
<br />

> [!IMPORTANT]
> **Giskard v3** is a fresh rewrite designed for dynamic, multi-turn testing of AI agents. This release drops heavy dependencies for better efficiency while introducing a more powerful AI vulnerability scanner and enhanced RAG evaluation — both now shipping natively in `giskard-scan` (beta), with no dependency on v2. Only the legacy scan for **tabular/ML models** remains v2-only.
> **Giskard v2 remains available but is no longer actively maintained.**
> Follow progress → [Read the v3 Announcement](https://github.com/orgs/Giskard-AI/discussions/2250) · [Roadmap](https://github.com/Giskard-AI/giskard-oss/issues/2252)

## Install

```sh
pip install giskard           # checks (+ agents, llm, core)
pip install "giskard[scan]"   # + vulnerability / quality scan
pip install "giskard[openai]" # provider SDK for LLM judges / generators
```

Requires Python 3.12+.

| Extra | Adds |
| --- | --- |
| *(none)* | `giskard-checks` and dependencies |
| `scan` | `giskard-scan` |
| `openai` / `anthropic` / … | provider SDKs (see `pyproject.toml` optional deps) |

**Telemetry:** optional aggregated analytics via `giskard-core`. No prompts or outputs are sent.
Opt out **before importing Giskard**: `export DO_NOT_TRACK=1` or `export GISKARD_TELEMETRY_DISABLED=1`.
Details: [`giskard-core` README](libs/giskard-core/README.md#telemetry).

---

Giskard is an open-source Python library for **testing and evaluating agentic systems**. The v3 architecture is a modular set of focused packages — each carrying only the dependencies it needs — built from scratch to wrap anything: an LLM, a black-box agent, or a multi-step pipeline.

| Status         | Package          | Description                                                                                                                                                              |
| -------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ✅ Beta        | `giskard-checks` | Testing & evaluation — scenario API, built-in checks, LLM-as-judge                                                                                                       |
| ✅ Beta        | `giskard-scan`   | Agent vulnerability scanner + RAG/quality evaluation — red teaming, prompt injection, jailbreaks & harmful content (`vulnerability_scan`, successor of [v2 Scan](https://legacy-docs.giskard.ai/en/stable/open_source/scan/index.html)), plus knowledge-base quality eval (`quality_scan`, successor of [v2 RAGET](https://legacy-docs.giskard.ai/en/stable/open_source/testset_generation/index.html)) |

These build on three foundational libraries — `giskard-core` (shared utilities & telemetry), `giskard-llm` (provider-agnostic LLM routing), and `giskard-agents` (agent & workflow orchestration) — which are pulled in automatically and rarely used directly.

## Giskard Checks — create and apply evals for testing agents

```sh
pip install giskard-checks
```

**[Giskard Checks](https://docs.giskard.ai/oss/checks)** is a lightweight library for creating evaluations (evals) that test LLM-based systems — from simple assertions to LLM-as-judge assessments. Unlike traditional unit tests, evals are designed for **non-deterministic outputs** where the same input can produce different valid responses.

Use Giskard Checks to:

- **Catch regressions** — verify your system still behaves correctly after changes
- **Validate RAG quality** — check if answers are grounded in retrieved context
- **Enforce safety rules** — ensure outputs conform to your content policies
- **Evaluate multi-turn agents** — test full conversations, not just single exchanges

Built-in evals include string matching, comparisons, regex, semantic similarity, and LLM-as-judge checks (`Groundedness`, `Conformity`, `LLMJudge`).

### Concepts

- **Target** — your system under test: any sync/async callable `(inputs) -> outputs` (optionally with `trace`)
- **Scenario** — one eval: interactions + checks
- **Check** — assertion or LLM judge over the trace
- **Suite** — many scenarios run together

`giskard.agents.Generator` is an LLM client for workflows/judges — not the same as
`giskard.checks` input generators (`LLMGenerator`) that synthesize user messages.

### Quickstart

```python
import asyncio
from giskard.checks import Scenario, Groundedness


def get_answer(inputs: str) -> str:
    return "Paris"  # replace with your model / agent


async def main() -> None:
    scenario = (
        Scenario("test_france_capital")
        .interact(inputs="What is the capital of France?", outputs=get_answer)
        .check(
            Groundedness(
                name="answer is grounded",
                context="France is in Western Europe. Its capital is Paris.",
            )
        )
    )
    result = await scenario.run()
    result.print_report()


asyncio.run(main())
```

`Groundedness` is an LLM judge — install a provider extra (e.g. `pip install "giskard[openai]"`) and set the matching API key. Default model: `openai/gpt-4o-mini`.

See the [full docs](https://docs.giskard.ai/oss/checks) for `Suites`, `LLMJudge`, multi-turn scenarios, and more.

---

## Giskard Scan — vulnerability scanner for AI agents

```sh
pip install "giskard[scan]"   # or: pip install giskard-scan
```

**Giskard Scan** is the red-teaming and vulnerability scanning layer for agentic systems. It generates adversarial test suites automatically from a plain-language description of your agent, covering prompt injection, harmful content, stereotypes, misinformation, and more.

Use Giskard Scan to:

- **Red-team your agent** — automatically generate adversarial inputs across OWASP LLM Top-10 threat categories
- **Run prompt-injection probes** — built-in dataset of injection payloads ready to use
- **Extend with custom generators** — pass your own `ScenarioGenerator` instances to `generate_suite`, or register them on `vulnerability_suite_generator_registry`

### Quickstart

```python
import asyncio
from giskard.scan import vulnerability_scan


async def my_agent(inputs: str) -> str:
    # Replace with your agent / model call
    return f"Echo: {inputs}"


async def main() -> None:
    await vulnerability_scan(
        target=my_agent,
        description="A customer support chatbot for an e-commerce platform.",
        languages=["en"],
    )


asyncio.run(main())
```

Scan generators also need an LLM provider extra and API key (same as Checks judges above).

## Looking for Giskard v2?

Giskard v2 included **Scan** (automatic vulnerability detection) and **RAGET** (RAG evaluation test set generation).

For **LLM agents**, both are superseded in v3 by `giskard-scan`: use [`vulnerability_scan`](#giskard-scan--vulnerability-scanner-for-ai-agents) in place of the v2 LLM scan, and `quality_scan` (with `KnowledgeBase`) in place of RAGET.

v3 works with ML models too — wrap one as a target and evaluate it with `giskard-checks` or `giskard-scan`. What the examples below cover is the **v2-only automatic tabular scan** — the detector suite that introspects a `giskard.Model` + `giskard.Dataset` to auto-detect performance, bias, and robustness issues — along with the `giskard.testing` ML test suite and the Giskard Hub. These are not planned for v3.

```sh
pip install "giskard[llm]>2,<3"
```

### [Scan](https://legacy-docs.giskard.ai/en/stable/open_source/scan/index.html) — automatically detect performance, bias & security issues

Wrap your model and run the scan:

```python
import giskard
import pandas as pd


# Replace my_llm_chain with your actual LLM chain or model inference logic
def model_predict(df: pd.DataFrame):
    """The function takes a DataFrame and must return a list of outputs (one per row)."""
    return [my_llm_chain.run({"query": question}) for question in df["question"]]


giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="My LLM Application",
    description="A question answering assistant",
    feature_names=["question"],
)

scan_results = giskard.scan(giskard_model)
display(scan_results)
```

<p align="center">
  <img src="readme/scan_updated.gif" alt="Scan Example" width="800">
</p>

### [RAGET](https://legacy-docs.giskard.ai/en/stable/open_source/testset_generation/index.html) — generate evaluation datasets for RAG applications

Automatically generate questions, reference answers, and context from your knowledge base:

```python
import pandas as pd
from giskard.rag import generate_testset, KnowledgeBase

# Load your knowledge base documents
df = pd.read_csv("path/to/your/knowledge_base.csv")
knowledge_base = KnowledgeBase.from_pandas(df, columns=["column_1", "column_2"])

testset = generate_testset(
    knowledge_base,
    num_questions=60,
    language="en",
    agent_description="A customer support chatbot for company X",
)
```

<p align="center">
  <img src="readme/RAGET_updated.gif" alt="RAGET Example" width="800">
</p>

[Full v2 docs](https://legacy-docs.giskard.ai)

<h1 id="community">👋 Community</h1>

We welcome contributions from the AI community! Read this [guide](./CONTRIBUTING.md) to get started, and join our thriving community on [Discord](https://gisk.ar/discord).

Follow the progress and share feedback:
[v3 Announcement](https://github.com/orgs/Giskard-AI/discussions/2250) · [Roadmap](https://github.com/Giskard-AI/giskard-oss/issues/2252)

🌟 [Leave us a star](https://github.com/Giskard-AI/giskard), it helps the project to get discovered by others and keeps us motivated to build awesome open-source tools! 🌟

❤️ If you find our work useful, please consider [sponsoring us](https://github.com/sponsors/Giskard-AI) on GitHub. With a monthly sponsoring, you can get a sponsor badge, display your company in this readme, and get your bug reports prioritized. We also offer one-time sponsoring if you want us to get involved in a consulting project, run a workshop, or give a talk at your company.
