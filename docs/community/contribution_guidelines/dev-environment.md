# How to configure local development environment

After cloning the repository here are the useful commands to set up the local environment:

## Install dependencies

We work with [uv](https://docs.astral.sh/uv/) as our Python package and project manager. You can find the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

After installing uv, we can install the dependencies by running:

```shell
uv sync --all-extras --all-groups
```

## Pre-commit hooks installation
Pre-commit hooks ensure that the basic code quality checks are performed before the code is committed. Pre-commit has already been installed as a dev dependency using `uv`.

The current pre-commit hook configuration is defined in `.pre-commit-config.yaml`.

To install it run:

```shell
pre-commit install
```

Make sure to comment out `skip: true` in the `.pre-commit-config.yaml` file to enable the pre-commit hooks for `ggshield` to use GitGuardian.

## Run pre-commit hook manually to fix easy issues
In case the build is failing because of the pre-commit checks that don't pass it's possible to fix easy issues by running:

```shell
pre-commit run --all-files
```

The pre-commit hooks are also automatically run when committing.

## Troubleshooting

### `Fatal Python error: segmentation fault` when running `pytest` on MacOS

This is probably caused by a [known issue](https://github.com/microsoft/LightGBM/issues/4707) with `libomp>=12` and LightGBM on MacOS.
You can avoid this problem by downgrading `libomp` to version 11.1.0, which is compatible with LightGBM.

```shell
$ wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
$ brew install ./libomp.rb
```
