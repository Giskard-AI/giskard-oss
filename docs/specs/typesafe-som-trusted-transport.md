# TypeSafe SOM trusted transport

## Problem

Scenario JSON is an untrusted data boundary, but a serialized TypeSafe SOM judge can currently choose both an HTTP destination and the name of an environment variable used as its bearer credential. Running such a scenario can therefore send a local secret to an attacker-controlled service.

## Requirements

- R1: Scenario and check data must not control the TypeSafe API URL.
- R2: Scenario and check data must not select which environment variable supplies the TypeSafe credential.
- R3: Local operators must retain gateway support through environment configuration.
- R4: Safe TypeSafe judge identity and model selection must continue to round-trip in scenarios.
- R5: Payloads containing removed transport configuration must fail closed.

## Acceptance examples

- A TypeSafe model configured as `typesafe/jev` uses the locally configured TypeSafe URL and `TYPESAFE_API_KEY`.
- Scenario JSON containing `base_url` or `api_key_env` inside a TypeSafe judge is rejected during validation.
- A scenario containing only the TypeSafe judge kind and model serializes, deserializes, and runs with locally supplied transport configuration.

## Compatibility

Direct constructor calls and serialized payloads that set `base_url` or `api_key_env` will no longer validate. Gateway users must move the URL to `TYPESAFE_BASE_URL` (or `TYPESAFE_API_BASE`) and the gateway credential value to `TYPESAFE_API_KEY`.
