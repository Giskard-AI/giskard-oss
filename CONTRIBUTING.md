# 👉 How to contribute to Giskard?

Everyone is encouraged to contribute, and we appreciate each and every one of them. Helping the community is thus not limited to writing code. The community will greatly benefit from your efforts to provide clarification, assist others, and enhance the documentation. 📗

Additionally, it is helpful if you help us spread the word by mentioning the library in blog articles about the amazing projects it enabled, tweeting about it on occasion, or just starring the repository! ⭐️

If you choose to contribute, please be mindful to respect our [code of conduct](https://github.com/Giskard-AI/giskard/blob/main/CODE_OF_CONDUCT.md).

## The different ways you can contribute to Giskard!

There are 5 ways you can contribute to Giskard:
* Submitting issues related to bugs or desired new features.
* Contributing to the examples or to the documentation;
* Fixing outstanding issues with the existing code;
* Implementing new checks or evaluation scenarios for agents and LLM-based systems;
* Implementing new features to Giskard

### Did you find a bug?

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

If you did not find it, please follow these steps to inform us:

* Include your **OS type and version**, the versions of **Python**, and different Python libraries you used;
* A short, self-contained, code snippet that allows us to reproduce the bug in less than 30s;
* Provide the *full* stack trace if an exception is raised.

### Do you want to implement a new check?

Custom and domain-based checks are welcome. If you have an idea, you can inform us by providing us a short description of the check and possibly a link to its documentation (paper, etc.).

Checks can be built using the `@Check.register("kind")` decorator and the fluent Scenario API. See the [checks documentation](https://docs.giskard.ai/oss/checks) for details.

If you are willing to contribute the check yourself, let us know so we can best guide you.

### Do you want a new feature (that is not a check)?

An awesome feature request addresses the following points:

1. Motivation first: Is it related to a problem/frustration with the library? Is it related to something you would need for a project? Is it something you worked on and think could benefit the community?
2. Write a *full paragraph* describing the feature;
3. Attach any additional information (drawings, screenshots, etc.) you think may help.

## Style guide

We use `ruff` for both formatting and linting. You can run the following commands to ensure your code conforms:

```bash
make setup     # Install dependencies + dev tools
make format    # Auto-format code
make lint      # Check for lint errors
make test      # Run the test suite
```

**This guide was heavily inspired by the awesome [HuggingFace guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**
