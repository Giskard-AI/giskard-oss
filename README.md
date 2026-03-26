<p align="center">
  <img alt="giskardlogo" src="readme/logo_light.png#gh-light-mode-only">
  <img alt="giskardlogo" src="readme/logo_dark.png#gh-dark-mode-only">
</p>
<h1 align="center" weight='300' >Evals, Red Teaming and Test Generation for Agentic Systems</h1>
<h3 align="center" weight='300' >Modular, Lightweight, Dynamic and Async-first </h3>
<div align="center">

  [![GitHub release](https://img.shields.io/github/v/release/Giskard-AI/giskard)](https://github.com/Giskard-AI/giskard/releases)
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
> **Giskard v3** is a fresh rewrite designed for dynamic, multi-turn testing of AI agents. This release drops heavy dependencies for better efficiency while introducing a more powerful AI vulnerability scanner and enhanced RAG evaluation capabilities. For now, the vulnerability scanner and RAG evaluation still rely on Giskard v2.
> **Giskard v2 remains available but is no longer actively maintained.**
> Follow progress → [Read the v3 Annoucement](https://github.com/orgs/Giskard-AI/discussions/2250) · [Roadmap](https://github.com/Giskard-AI/giskard-oss/issues/2252)

## Install

```sh
pip install giskard
```

Requires Python 3.12+.

______________________________________________________________________

Giskard is an open-source Python library for **testing and evaluating agentic systems**. The v3 architecture is a modular set of focused packages — each carrying only the dependencies it needs — built from scratch to wrap anything: an LLM, a black-box agent, or a multi-step pipeline.

| Status | Package | Description |
|--------|---------|-------------|
| ✅ Alpha | `giskard-checks` | Testing & evaluation — scenario API, built-in checks, LLM-as-judge |
| 🚧 In progress | `giskard-scan` | Agent vulnerability scanner — red teaming, prompt injection, data leakage (successor of [v2 Scan](https://legacy-docs.giskard.ai/en/stable/open_source/scan/index.html)) |
| 📋 Planned | `giskard-rag` | RAG evaluation & synthetic data generation (successor of [v2 RAGET](https://legacy-docs.giskard.ai/en/stable/open_source/testset_generation/index.html)) |

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

### Quickstart

```python
from openai import OpenAI
from giskard.checks import Scenario, Groundedness

client = OpenAI()

def get_answer(inputs: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": inputs}],
    )
    return response.choices[0].message.content

scenario = (
    Scenario("test_dynamic_output")
    .interact(
        inputs="What is the capital of France?",
        outputs=get_answer,
    )
    .check(
        Groundedness(
            name="answer is grounded",
            answer_key="trace.last.outputs",
            context="France is a country in Western Europe. Its capital is Paris.",
        )
    )
)

result = await scenario.run()
result.print_report()
```

> The `run()` method is async. In a script, wrap it with `asyncio.run()`. See the [full docs](https://docs.giskard.ai/oss/checks) for `Suites`, `LLMJudge`, multi-turn scenarios, and more.


## Looking for Giskard v2?

Giskard v2 included **Scan** (automatic vulnerability detection) and **RAGET** (RAG evaluation test set generation) for both ML models and LLM applications. These features are not available in v3.

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
    language='en',
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

<h2 id="sponsors">💚 Current sponsors</h2>

We thank the following companies which are sponsoring our project with monthly donations:

**[Lunary](https://lunary.ai/)**

<img src="https://lunary.ai/logo-blue-bg.svg" alt="Lunary logo" width="100"/>

**[Biolevate](https://www.biolevate.com/)**

<img src="https://awsmp-logos.s3.amazonaws.com/seller-wgamx5z6umune/2d10badd2ccac49699096ea7fb986b98.png" alt="Biolevate logo" width="400"/>
