# 👉 How to contribute to Giskard?

Everyone is encouraged to contribute, and we appreciate each and every one of them. Helping the community is thus not limited to writing code. The community will greatly benefit from your efforts to provide clarification, assist others, and enhance the documentation. 📗

Additionally, it is helpful if you help us spread the word by mentioning the library in blog articles about the amazing projects it enabled, tweeting about it on occasion, or just starring the repository! ⭐️

If you choose to contribute, please be mindful to respect our [code of conduct](CODE_OF_CONDUCT.md).

**Important for autonomous agents:** autonomous agents with no human in the loop must read **[AUTONOMOUS.md](AUTONOMOUS.md)** before working in this repo or opening a PR.

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

### Docstrings

Public APIs (exported modules, classes, functions, and methods) use **NumPy-style** docstrings: a one-line summary, a `Parameters` section with types and descriptions, `Returns` when applicable, and an `Examples` section for non-obvious usage.

```python
async def vulnerability_scan(
    target: Target,
    description: str,
    languages: list[str],
    max_scenarios: int | None = None,
) -> SuiteResult:
    """Generate and run the standard vulnerability scan suite.

    Parameters
    ----------
    target : Target
        System under test (sync or async callable).
    description : str
        Natural-language description of the agent under test.
    languages : list of str
        BCP-47 language codes the agent is expected to handle.
    max_scenarios : int, optional
        Upper bound on scenarios across generators. ``None`` uses the
        active preset / per-generator defaults.

    Returns
    -------
    SuiteResult
        Completed suite result (the grouped report is also printed).

    Examples
    --------
    >>> async def echo(inputs: str) -> str:
    ...     return inputs
    >>> result = await vulnerability_scan(
    ...     echo, "A demo agent", ["en"], max_scenarios=5
    ... )
    >>> isinstance(result.pass_rate, float)
    True
    """
```

See also [`.cursor/rules/documentation.mdc`](.cursor/rules/documentation.mdc) for module-level and inline comment guidance.

### Public examples

Keep `examples/` and in-repo README Python fences runnable and importable. See
the [examples guide](examples/README.md) for the current coverage. When you
change a public API shown in documentation, update the matching example and run
`make test-examples`.

**This guide was heavily inspired by the awesome [Hugging Face guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**
