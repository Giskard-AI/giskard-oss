<p align="center">
  <img alt="giskardlogo" src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/giskard_logo.png#gh-light-mode-only">
  <img alt="giskardlogo" src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/giskard_logo_green.png#gh-dark-mode-only">
</p>
<h1 align="center" weight='300' >The Evaluation & Testing framework for AI systems</h1>
<h3 align="center" weight='300' >Control risks of performance, bias and security issues in AI systems</h3>
<div align="center">

  [![GitHub release](https://img.shields.io/github/v/release/Giskard-AI/giskard)](https://github.com/Giskard-AI/giskard/releases)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Giskard-AI/giskard/blob/main/LICENSE)
  [![Downloads](https://static.pepy.tech/badge/giskard/month)](https://pepy.tech/project/giskard)
  [![CI](https://github.com/Giskard-AI/giskard/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Giskard-AI/giskard/actions/workflows/ci.yml?query=branch%3Amain)
  [![Giskard on Discord](https://img.shields.io/discord/939190303397666868?label=Discord)](https://gisk.ar/discord)

  <a rel="me" href="https://fosstodon.org/@Giskard"></a>

</div>
<h3 align="center">
   <a href="https://docs.giskard.ai/en/stable/getting_started/index.html"><b>Docs</b></a> &bull;
  <a href="https://www.giskard.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readmeblog"><b>Website</b></a> &bull;
  <a href="https://gisk.ar/discord"><b>Community</b></a>
 </h3>
<br />

## Status and breaking changes
- This branch is **Giskard v3 alpha**. The API is unstable.
- Giskard v3 is a **breaking change** from legacy Giskard (`<2`).
- The codebase was fully rewritten to simplify the architecture.

## Scope today
- LLM systems are the primary focus.
- Traditional ML models are no longer a product focus, even if it is technically possible to use them.
- A new scan is planned for v3 (LLM-only).
- RAGET will be rewritten for v3 in a future release.

## Install
Giskard v3 is published as a pre-release.
```sh
pip install --pre giskard
```
Python >= 3.12 is required.

## Repository structure
This repo is a Python workspace with three packages:
- `giskard-core`
- `giskard-checks`
- `giskard-agents`

## Development
Use the Makefile for common tasks.
```sh
make setup
make test
make ci
```

## Community
We welcome contributions from the AI community. Read the [contributing guide](./CONTRIBUTING.md) to get started, and join the community on [Discord](https://gisk.ar/discord).

🌟 [Leave us a star](https://github.com/Giskard-AI/giskard) to help the project get discovered and keep the momentum.

❤️ If you find our work useful, consider [sponsoring us](https://github.com/sponsors/Giskard-AI) on GitHub. We also offer one-time sponsoring for consulting, workshops, or talks.

<h2 id="sponsors">💚 Current sponsors</h2>

We thank the following companies which are sponsoring our project with monthly donations:

**[Lunary](https://lunary.ai/)**

<img src="https://lunary.ai/logo-blue-bg.svg" alt="Lunary logo" width="100"/>

**[Biolevate](https://www.biolevate.com/)**

<img src="https://awsmp-logos.s3.amazonaws.com/seller-wgamx5z6umune/2d10badd2ccac49699096ea7fb986b98.png" alt="Biolevate logo" width="400"/>
