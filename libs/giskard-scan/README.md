# giskard-scan

Agent vulnerability scanner — red teaming, prompt injection, adversarial scenario generation.

## Scan entrypoints

`quality_scan` and `vulnerability_scan` share the same explicit execution options.
Pass shared settings by keyword on either entrypoint; each scan adds scan-specific
options (`knowledge_base` for quality, `commercial_use` for vulnerability).

```python
from giskard.scan import quality_scan, vulnerability_scan


async def echo(inputs: str) -> str:
    return inputs


vulnerability_result = await vulnerability_scan(
    target=echo,
    description="Customer support chatbot for an e-commerce store.",
    languages=["en"],
    max_scenarios=20,
    seed=42,
    group_by="threat-type",
    parallel=True,
    max_concurrency=8,
    return_exception=False,
    target_mode="multiturn",
    commercial_use=False,
)

quality_result = await quality_scan(
    target=echo,
    description="Customer support chatbot for an e-commerce store.",
    languages=["en"],
    knowledge_base=["Paris is the capital of France."],
    max_scenarios=20,
    seed=42,
    group_by="component",
    parallel=True,
    max_concurrency=8,
    return_exception=False,
    target_mode="multiturn",
)
```
