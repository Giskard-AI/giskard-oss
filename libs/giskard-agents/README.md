# Giskard Agents

Giskard Agents is a lightweight library that orchestrates LLM completions and agents in parallel workflows. It enables multiple AI workflows to run independently while maintaining coordination and synchronization between them.

## Requirements

- Python 3.10 or higher

## Installation

### Using uv (recommended)

Install the package:

```bash
uv add giskard-agents
```

For development, install with dev dependencies:

```bash
uv add giskard-agents --dev
```

# Docs

Three basic elements to keep in mind:

- `Generator` corresponds to a conversational text generator. In short, it represents a model with certain params, and can run completions.
- `ChatWorkflow` defines all what's needed to run a chat with the generator. It handles templates, parsing, and tools.
- `Chat` is the result of a pipeline run. It contains the generated messages and everything you would expect.

Also important to keep in mind: everything is async.

## Basic usage

### Running a chat

```python
from giskard import agents

generator = agents.Generator(model="openai/gpt-4o-mini")

# generator.chat automatically creates a workflow that can be run.
chat = await generator.chat("Hello, how are you?").run()

# print the content of the last message (in this case, the assistant's response)
print(chat.last.content)
```

You can run multiple chats in parallel:

```python
chats = await generator.chat("Hello, how are you?").run_many(n=3)
```

Or add multiple messages to the workflow:

```python
# The chat message role is "user" by default.
chat = await (
    generator
    .chat("You are a helpful assistant.", role="system")
    .chat("Hello, how are you?")
    .chat("I'm fine, thank you!", role="assistant")
    .chat("What's your name?")
    .run()
)
```

## Structured output

You can specify the output model for the workflow, and this will be passed to
each completion call:

```python
class SimpleOutput(BaseModel):
    mood: str
    greeting: str

chat = await (
    generator.chat("Hello!")
    .with_output(SimpleOutput)
    .run()
)

assert isinstance(chat.output, SimpleOutput)
assert chat.output.mood == "happy"
```

## Inputs and templates

### Inline templates

You can associate input variables to a workflow, and use them in the messages thanks to jinja2 templating. Here's an example:

```python

# This will run a chat with the message "Hello Test Bot, how are you?"
chat = await (
    generator.chat("Hello {{ name_of_the_bot }}, how are you?")
    .with_inputs(name_of_the_bot="Test Bot")
    .run()
)
```

### External templates

For more complicated prompts you can define your template in a separate file. First tell `giskard.agents` where to find the templates (you probably want to do this in your `__init__.py` file):

```python
agents.set_prompts_path("path/to/the/prompts")
```

Write your templates in jinja2:

```jinja
Hello {{ name_of_the_bot }}, how are you?
```

```python
chat = await (
    generator.template("hello_template.j2")
    .with_inputs(name_of_the_bot="Test Bot")
    .run()
)
```

### Multi-message templates

Sometimes you may want to use more complex, multi-message prompts. This is particularly useful when you need a few-shots chat that includes examples.
For this need, `giskard.agents` provides a special syntax to define multi-message prompts.

```jinja
{% message system %}
You are an impartial evaluator of scientific theories. Your only job is to rate them on a scale of 1-5, where:
1 = "This theory belongs in the same category as 'the Earth is flat'"
2 = "More holes than Swiss cheese, but at least it's creative"
3 = "Could be true, could be false, Schrödinger's theory"
4 = "Almost as solid as the theory of gravity"
5 = "This theory is so good, even the experimentalists are convinced!"

The user will provide you with a scientific theory to evaluate. Respond with ONLY a number from 1-5.
{% endmessage %}

{# Example #}
{% message user %}
The universe is actually a giant simulation running on a quantum computer in a higher dimension, and we're all just NPCs in someone's cosmic video game.
{% endmessage %}

{% message assistant %}
3
{% endmessage %}

{# Actual input #}
{% message user %}
{{ theory }}
{% endmessage %}
```

You can then load the template as usual:

```python

chat = await (
    generator.template("evaluators.scientific_theory")
    .with_inputs(theory="Normandy is actually the center of the universe because its perfect balance of rain, cheese, and cider creates a quantum field that bends space-time, making it the most harmonious place on Earth.")
    .run()
)

score = chat.last.parse(int)
assert score == 5
```

## Input batches

You can run multiple chats with different inputs by passing a list of inputs to the `run_batch` method.

```python
chats = await (
    generator.chat("What's the weather in {{ city }}?")
    .run_batch([{"city": "Paris"}, {"city": "London"}])
)
assert len(chats) == 2
```

## Tools

You can define tools using the `@agents.tool` decorator. Tools will be automatically called when the workflow is run.

When defining tools, you need to make sure that all tool arguments have type hints. These will be used to define the tool schema. You must also provide a docstring, which will be used to describe the tool to the LLM. If you include the parameters in the docstring, their descriptions will be automatically added to the tool schema.

This can be combined with all functionalities described earlier. Here's an example:

```python
from giskard import agents

@agents.tool
def get_weather(city: str) -> str:
    """Get the weather in a city.

    Parameters
    ----------
    city: str
        The city to get the weather for.
    """
    if city == "Paris":
        return f"It's raining in {city}."

    return f"It's sunny in {city}."

# Run parallel chats with tools
chats = await (generator.chat("Hello, what's the weather in {{ city }}?")
    .with_tools(get_weather)
    .run_batch([{"city": "Paris"}, {"city": "London"}])
)

assert "rain" in chats[0].last.content
assert "sun" in chats[1].last.content
```

### Run context

Tools can access a `RunContext` object that acts as a storage memory for the run. This can be useful to store information that is needed for the next tool calls.

The `RunContext` object will be automatically passed to the tool if you specify the type hint.

```python
@agents.tool
def get_weather(city: str, run_context: agents.RunContext) -> str:
    previously_asked_cities = run_context.get("previously_asked_cities", [])

    if city in previously_asked_cities:
        return f"I've already asked this!"

    run_context.set("previously_asked_cities", previously_asked_cities + [city])
    return f"It's raining in {city}."
```

The run context will be shared between all tool calls in the same run.

You can also retrieve it after the run is complete:

```python
chat = await (generator.chat("Hello, what's the weather in {{ city }}?")
    .with_tools(get_weather)
    .with_inputs(city="Paris")
    .run()
)

assert "Paris" in chat.context.get("previously_asked_cities")
```

To initialize the run context, you can pass it to the workflow with the `with_context` method:

```python
run_context = agents.RunContext()
run_context.set("previously_asked_cities", ["Paris"])

chat = await (generator.chat("Hello, what's the weather in {{ city }}?")
    .with_context(run_context)
    .with_tools(get_weather)
    .run()
)
```

## Error handling

### Errors during workflow execution

You can specify the error handling policy for the workflow. By default, the workflow will raise an error if an error occurs. You can change this behavior by passing the `on_error` method.

You can choose to:

- Raise an error (`ErrorPolicy.RAISE`)
- Return the chat with the error (`ErrorPolicy.RETURN`). The chat will have a `failed` attribute set to `True`, and an `error` attribute with a serializable error message.
- For multi-run methods (e.g. `run_many` or `run_batch`), you can discard the failed chats (`ErrorPolicy.SKIP`). You will then only get the successful chats (potentially an empty list).

Note: when running a single chat (`workflow.run(...)`), error policy is `SKIP` will behave as `RETURN`, returning `Chat` object with the error.

```python
# This may return less than 3 chats if some fail.
chats = await generator.chat("Hello!", role="user").on_error(ErrorPolicy.SKIP).run_many(n=3)

# This will return 3 chats, some may be in failed state.
chats = await generator.chat("Hello!", role="user").on_error(ErrorPolicy.RETURN).run_many(n=3)

for chat in chats:
    if chat.failed:
        print("CHAT FAILED:", chat.error.message)
```

### Errors during tool calls

By default, `giskard.agents` will catch errors during tool calls and return the error message as a tool result. This will let the agent decide what to do with the error (whether retrying or moving on).
You can change this behavior by passing the `catch=None` on the tool decorator. In this case, the error will be raised and passed to the workflow, which will then handle it according to the workflow error handling policy.

```python
# Default behavior, will catch errors
@agents.tool
def get_weather(city: str) -> str:
    raise ValueError("City not found")

result = await get_weather.run(arguments={"city": "Paris"})
print(result) # "ERROR: City not found"


# Opt out of the catch
@agents.tool(catch=None)
def get_weather(city: str) -> str:
    raise ValueError("City not found")

# This will raise an exception
result = await get_weather.run(arguments={"city": "Paris"})
```

## Development

### Quick Setup

For quick development setup, use the provided Makefile:

```bash
make setup  # Install deps + tools
make help   # See all available commands
```

### Manual Setup

Install the project dependencies:

```bash
uv sync
```

Install development tools:

```bash
uv tool install ruff
uv tool install vermin
uv tool install pre-commit --with pre-commit-uv
```

Note: `pytest` and `pip-audit` are included in dev dependencies since they need access to the project code.

### Common Tasks

```bash
make test          # Run tests
make lint          # Run linting
make format        # Format code
make check-format  # Check if code is formatted
make check         # Run all checks (lint + format + compatibility + security)
make ci            # Simulate CI locally
make clean         # Clean build artifacts
```

### Python Compatibility

This project maintains compatibility with Python 3.11+. We use [vermin](https://github.com/netromdk/vermin) to ensure code compatibility:

```bash
# Check Python 3.11 compatibility
make check-compat
# or manually:
uv tool run vermin --target=3.11- --no-tips --violations .
```

#### Setting up Pre-commit Hooks

To automatically check compatibility on every commit:

```bash
# Quick setup (if you ran `make setup` this is already done)
make pre-commit-install

# Or manually:
uv tool install pre-commit --with pre-commit-uv
pre-commit install

# Run on all files
make pre-commit-run
```

The hooks will now run automatically on `git commit` and prevent commits that don't meet Python 3.10 compatibility requirements.

### Security

We use [pip-audit](https://pypi.org/project/pip-audit/) to scan for known security vulnerabilities in dependencies:

```bash
# Check for security vulnerabilities
make security
# or manually:
uv run pip-audit .
```

Both compatibility and security checks are automatically run in CI for every pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
