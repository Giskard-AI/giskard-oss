# Changelog

All notable user-facing changes are documented in this file.

Giskard v3 is a new, incompatible generation of the project. It is a
rewrite for testing, evaluating, and red-teaming AI agents. It is not an
in-place upgrade of the v2 API.

## [3.0.0] - Unreleased

### Breaking changes

- **Python 3.12 or newer is required.** Python 3.9, 3.10, and 3.11 are no
  longer supported.
- **The v2 monolith and its ML-model workflow are not part of v3.** This
  includes `giskard.Model`, `giskard.Dataset`, the automatic tabular/ML scan,
  the `giskard.testing` ML test suite, and Giskard Hub integration. Keep using
  `giskard[llm]>2,<3` for these v2-only features.
- **v2 LLM Scan and RAGET have new APIs.** Use
  `giskard.scan.vulnerability_scan` for agent vulnerability scanning, and
  `giskard.scan.quality_scan` with a knowledge base for RAG-quality
  evaluation. Existing v2 scan reports, test sets, and scripts need migration.
- **Imports now use focused packages.** The primary testing and scanning APIs
  live under `giskard.checks` and `giskard.scan`. `giskard.agents`,
  `giskard.llm`, and `giskard.core` remain focused public packages for agent
  workflows, LLM routing, and shared utilities.

### Added

- A modular Python workspace with five public libraries:
  `giskard-core`, `giskard-llm`, `giskard-agents`, `giskard-checks`, and
  `giskard-scan`.
- `giskard-checks`: an async-first API for building `Scenario` objects from
  interactions and checks, then running them in `Suite` objects. It supports
  synchronous and asynchronous targets, multi-turn traces, parallel scenario
  execution, JUnit export, and LLM-as-judge checks.
- Built-in checks for comparisons, text and regex matching, JSON validity,
  semantic similarity, readability, Rego policies, composition, and custom
  functions.
- `giskard-scan`: native red teaming and quality evaluation for AI agents.
  `vulnerability_scan` covers prompt injection, jailbreaks, harmful content,
  stereotypes, misinformation, and related threats. `quality_scan` evaluates
  knowledge-base and RAG behavior.
- Scan generators for adversarial prompts, indirect prompt injection,
  Crescendo, GOAT, GCG, HarmBench, refusal, sycophancy, and knowledge-base
  quality scenarios.
- Experimental third-party scanner support through `third_party_scan`, with
  optional `garak` and `deepteam` integrations.
- Provider-agnostic LLM routing plus optional OpenAI, Google, Anthropic,
  Azure, and LiteLLM integrations.
- Optional, aggregated telemetry. Prompts, outputs, and scenario text are not
  sent. Set `DO_NOT_TRACK=1` or `GISKARD_TELEMETRY_DISABLED=1` before import to
  opt out.

### Installation

```sh
pip install giskard
pip install "giskard[scan]"
pip install "giskard[openai]"
```

The base `giskard` package installs checks. Add `scan` for automatic
vulnerability and quality scans, and add the provider extra needed by your
LLM-backed checks or generators.

### Migration guide

1. Move agent tests to `giskard.checks`. Wrap the system under test as a
   synchronous or asynchronous callable, then model each evaluation as a
   `Scenario` made of interactions and checks.

2. Replace v2 LLM Scan calls with `vulnerability_scan`. Describe the target
   agent and choose the threat coverage, language, concurrency, and target
   mode needed by your application.

3. Replace RAGET test-set generation and evaluation with `quality_scan` and a
   `KnowledgeBase`. Review the new scenario-based result model before porting
   any thresholds or report processing.

4. If your project uses tabular/ML automatic scanning, Hub, or the legacy ML
   test suite, keep that part on v2. Those features are deliberately outside
   the v3 product scope.

5. Rebuild test suites and report processing around the new scenario and suite
   result models. Then re-run your test suite.

### Notes

- This changelog is intentionally migration-focused. It covers the release
  boundary from `v2.19.2` to the v3 codebase, rather than reproducing every
  commit from the rewrite.

[3.0.0]: https://github.com/Giskard-AI/giskard-oss/compare/v2.19.2...main
