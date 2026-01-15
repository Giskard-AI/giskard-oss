"""Core tool functionality for Giskard Agents."""

import inspect
from typing import Any, Callable, Literal, TypeVar

import logfire_api as logfire
from pydantic import BaseModel, Field, create_model

from ..context import RunContext
from ..errors.serializable import Error
from ._docstring_parser import parse_docstring

F = TypeVar("F", bound=Callable[..., Any])


class Function(BaseModel):
    """Represents a function call in a tool call."""

    arguments: str
    name: str | None


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""

    id: str
    type: Literal["function"] = "function"
    function: Function


class Tool(BaseModel):
    """A tool that can be used with LLM completions."""

    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    fn: Callable
    catch: Callable[[Exception], Any] | None = Field(default=None)

    run_context_param: str | None = Field(default=None)

    @classmethod
    def from_callable(
        cls, fn: Callable, catch: Callable[[Exception], Any] | None = None
    ) -> "Tool":
        """Create a Tool from a callable function.

        Parameters
        ----------
        fn : Callable
            The function to convert to a tool.
        catch : Callable[[Exception], Any] | None, optional
            Error handler. If ``None``, errors are not caught. If not provided,
            a default handler returning a serializable ``Error`` is used.

        Returns
        -------
        Tool
            A Tool instance.

        Raises
        ------
        ValueError
            If the function lacks proper annotations or docstring.
        """
        sig = inspect.signature(fn)
        description, parameter_descriptions = parse_docstring(fn, sig)

        fields = {}
        run_context_param = None

        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Tool `{fn.__name__}` parameter `{name}` must have a type annotation"
                )

            # Check if this parameter is a RunContext
            if param.annotation is RunContext:
                run_context_param = name
                continue  # Skip adding RunContext to the schema

            field = Field(
                default=(
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else ...
                ),
                description=parameter_descriptions.get(name, None),
            )

            fields[name] = (param.annotation, field)

        model = create_model(
            fn.__name__,
            **fields,
        )

        return cls(
            name=fn.__name__,
            description=description,
            parameters_schema=model.model_json_schema(),
            fn=fn,
            run_context_param=run_context_param,
            catch=catch,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function without modification.

        Notes
        -----
        This preserves the original sync/async behavior and exception
        propagation of the wrapped function. Error handling and serialization
        are applied only in ``run()``.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the function.
        **kwargs : Any
            Keyword arguments to pass to the function.

        Returns
        -------
        Any
            The direct return value (or awaitable) of the wrapped function.
        """
        return self.fn(*args, **kwargs)

    @logfire.instrument("tool.run")
    async def run(
        self, arguments: dict[str, Any], ctx: RunContext | None = None
    ) -> Any:
        """Run the tool's function asynchronously.

        This method handles both sync and async functions by awaiting async
        functions. Errors are handled based on ``self.catch``.

        Parameters
        ----------
        arguments : dict[str, Any]
            Arguments to pass to the function.
        ctx : RunContext | None, optional
            The run context to inject if the tool expects it.

        Returns
        -------
        Any
            The result of calling the function.
        """

        # Inject the context if the tool expects it
        if ctx and self.run_context_param:
            arguments = arguments.copy()
            arguments[self.run_context_param] = ctx

        try:
            res = self.fn(**arguments)
            if inspect.isawaitable(res):
                res = await res
        except Exception as error:
            if self.catch is not None:
                res = self.catch(error)
            else:
                raise

        if isinstance(res, Error):
            logfire.error("tool.run.error", error=res)
            return str(res)

        if isinstance(res, BaseModel):
            res = res.model_dump()

        return res

    def to_litellm_function(self) -> dict[str, Any]:
        """Convert the tool to a LiteLLM function format.

        Returns
        -------
        dict[str, Any]
            A dictionary in the LiteLLM function format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolMethod:
    """Descriptor to handle tool methods on classes."""

    def __init__(self, func: Callable, catch: Callable[[Exception], Any] | None = None):
        self.func = func
        self._catch = catch

    def __get__(self, instance, owner):
        if instance is None:
            # Accessing from class, return unbound tool
            return Tool.from_callable(self.func, catch=self._catch)

        # Accessing from instance, create tool from bound method
        bound_method = self.func.__get__(instance, owner)
        return Tool.from_callable(bound_method, catch=self._catch)


def _default_catch(exception: Exception) -> Any:
    """Default error handler for tools.

    Parameters
    ----------
    exception : Exception
        The exception that was raised.

    Returns
    -------
    Any
        A serializable ``Error`` instance containing the message.
    """
    return Error(message=str(exception))


def tool(
    _func: F | None = None, *, catch: Callable[[Exception], Any] | None = _default_catch
) -> Tool:
    """Decorator to create a tool from a function.

    The function should have type annotations and a docstring in numpy or Google
    format. The docstring should describe the function and its parameters.

    Usage
    -----
    - ``@tool``: uses default error catching (returns ``Error`` objects).
    - ``@tool(catch=None)``: disables catching and propagates exceptions.
    - ``@tool(catch=handler)``: uses a custom handler ``handler(Exception) -> Any``.

    Parameters
    ----------
    _func : F | None, optional
        The function to convert to a tool (when used as ``@tool``).
    catch : Callable[[Exception], Any] | None, optional
        Error handler. ``None`` disables catching; if omitted, a default handler
        returning ``Error`` is used.

    Returns
    -------
    Tool
        A Tool instance that can be called like the original function, or a
        ``ToolMethod`` if applied to a method.
    """

    def decorator(func: F) -> Tool:
        # Check if this is a class method by looking for 'self' as first parameter
        sig = inspect.signature(func)
        first_param = next(iter(sig.parameters.keys()), None)
        if first_param == "self":
            return ToolMethod(func, catch=catch)
        return Tool.from_callable(func, catch=catch)

    if _func is not None:
        return decorator(_func)
    return decorator
