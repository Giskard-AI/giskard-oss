# 👉 How to contribute to Giskard?

Everyone is encouraged to contribute, and we appreciate each and every one of them. Helping the community is thus not limited to writing code. The community will greatly benefit from your efforts to provide clarification, assist others, and enhance the documentation. 📗

Additionally, it is helpful if you help us spread the word by mentioning the library in blog articles about the amazing projects it enabled, tweeting about it on occasion, or just starring the repository! ⭐️

If you choose to contribute, please be mindful to respect our [code of conduct](CODE_OF_CONDUCT.md).

**Coding agents and similar automation:** read **[AGENTS.md](AGENTS.md)** at the repository root before you work in this repo or open pull requests. It is the authoritative guide for agent setup, commands, and PR requirements (including how agent PRs are handled).

## The different ways you can contribute to Giskard!

There are 5 ways you can contribute to Giskard:
* Submitting issues related to bugs or desired new features.
* Contributing to the examples or to the documentation;
* Fixing outstanding issues with the existing code;
* Implementing new checks or evaluation scenarios for agents and LLM-based systems;
* Implementing new features to Giskard.

### Did you find a bug?

First, we would really appreciate it if you could **make sure the bug was not
already reported** (search this repository's Issues tab on GitHub).

If you did not find it, please follow these steps to inform us:

* Include your **OS type and version**, the versions of **Python**, and different Python libraries you used;
* A short, self-contained, code snippet that allows us to reproduce the bug in less than 30s;
* Provide the *full* stack trace if an exception is raised.

### Do you want to implement a new check?

Custom and domain-based checks are welcome. If you have an idea, you can inform us by providing us a short description of the check and possibly a link to its documentation (paper, etc.).

Checks can be built using the `@Check.register("kind")` decorator and the fluent Scenario API. See the [checks documentation](https://docs.giskard.ai/oss/checks) for end-user usage.

For contributing built-in checks, prompts, and interaction specs in this repo, see **Creating Custom Checks and Interaction Specs** in the [`giskard-checks` README](libs/giskard-checks/README.md#creating-custom-checks-and-interaction-specs).

If you are willing to contribute the check yourself, let us know so we can best guide you.

### Do you want a new feature (that is not a check)?

An awesome feature request addresses the following points:

1. Motivation first: Is it related to a problem/frustration with the library? Is it related to something you would need for a project? Is it something you worked on and think could benefit the community?
2. Write a *full paragraph* describing the feature;
3. Attach any additional information (drawings, screenshots, etc.) you think may help.

## Style guide

The repository is a **uv** workspace and requires Python 3.12+. We use several tools to ensure code quality:

* **Ruff** for formatting and linting.
* **basedpyright** for type checking.
* **pre-commit** hooks (Ruff, basedpyright, and a few file checks) to catch issues before you push.

`make setup` runs `uv sync`, installs these CLI tools, and enables the hooks.

From the repository root:

```bash
make setup     # uv sync + dev tools + pre-commit install
make format    # Ruff format and apply safe lint fixes (`ruff check --fix`)
make lint      # Ruff check only (no writes)
make check     # Format check, lint, Python 3.12 compat (vermin), types, security, licenses
make test      # pytest for packages under libs/
```

Run `make help` for other targets (for example scoped tests with `PACKAGE=giskard-checks`).

**This guide was heavily inspired by the awesome [Hugging Face guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**
