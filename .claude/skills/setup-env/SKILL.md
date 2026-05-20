---
description: Set up the full workspace dev environment. Use when the user wants to install dependencies, sync libs, or prepare the environment.
disable-model-invocation: true
allowed-tools: Bash(make *) Bash(uv *)
---

## Setup dev env

!`make setup 2>&1 | tail -20`

## Validate installation and notify user

Check the output above for errors. If any errors occurred, try to fix them and re-run the failed step. Report whether installation was successful.
