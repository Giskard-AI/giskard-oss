import functools
import inspect
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel
from typing_extensions import TypeVar


class ParameterInjectionRequirement(BaseModel, frozen=True):
    class_info: type[Any] | Any
    optional: bool = False


class ParameterInjection[T](BaseModel, frozen=True):
    position: int | None
    name: str | None
    class_info: type[T]

    def resolve(self, *args, **kwargs) -> T:
        if self.position is not None:
            return args[self.position]
        if self.name is not None:
            return kwargs[self.name]

        raise ValueError(f"ParameterInjection {self} has no position or name")


class CallableInjectionMapping(BaseModel, frozen=True):
    args: list[ParameterInjection[Any]]
    kwargs: dict[str, ParameterInjection[Any]]

    def inject_parameters[**P, R](
        self, value: Callable[..., R], *args: P.args, **kwargs: P.kwargs
    ) -> Callable[[], R]:
        return functools.partial(
            value,
            *[arg.resolve(*args, **kwargs) for arg in self.args],
            **{
                name: kwarg.resolve(*args, **kwargs)
                for name, kwarg in self.kwargs.items()
            },
        )

    @classmethod
    def from_callable(
        cls,
        callable: Callable[..., Any],
        *args_reqs: ParameterInjectionRequirement,
        **kwargs_reqs: ParameterInjectionRequirement,
    ) -> "CallableInjectionMapping":
        signature = inspect.signature(callable)

        reqs_by_type: dict[type[Any] | Any, ParameterInjectionRequirement] = {}

        for req in args_reqs:
            if req.class_info in reqs_by_type:
                pass
            reqs_by_type[req.class_info] = req

        for req in kwargs_reqs.values():
            reqs_by_type[req.class_info] = req

        resolved_args: list[ParameterInjection[Any]] = []
        resolved_kwargs: dict[str, ParameterInjection[Any]] = {}

        satisfied_reqs: set[type[Any] | Any] = set()

        for idx, parameter in enumerate(signature.parameters.values()):
            param_name = parameter.name
            param_annotation = parameter.annotation

            if isinstance(param_annotation, TypeVar):
                param_annotation = param_annotation.__bound__ or Any

            param_default = parameter.default

            if param_default is not inspect.Parameter.empty:
                continue

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            matching_req: ParameterInjectionRequirement | None = None

            # First pass: try to find exact matches or type-based matches
            for req_class_info, req in reqs_by_type.items():
                if (
                    param_annotation is inspect.Parameter.empty
                    or param_annotation is Any
                ):
                    # For untyped parameters, prefer Any if available, but continue searching
                    if req_class_info is Any:
                        matching_req = req
                        break
                elif req_class_info is Any:
                    matching_req = req
                    break
                elif param_annotation is req_class_info:
                    matching_req = req
                    break
                elif (
                    isinstance(param_annotation, type)
                    and isinstance(req_class_info, type)
                    and issubclass(cast(type, param_annotation), req_class_info)
                ):
                    matching_req = req
                    break

            # Second pass: if no match found and parameter is untyped, allow it to match any requirement
            if matching_req is None and (
                param_annotation is inspect.Parameter.empty or param_annotation is Any
            ):
                # Match the first available requirement for untyped parameters
                for req_class_info, req in reqs_by_type.items():
                    matching_req = req
                    break

            if matching_req is None:
                raise TypeError(
                    f"Parameter '{param_name}' of type '{param_annotation}' is required but no matching `ParameterInjectionRequirement` was provided."
                )

            injection = ParameterInjection(
                position=(
                    idx
                    if parameter.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    else None
                ),
                name=param_name,
                class_info=(
                    param_annotation
                    if param_annotation is not inspect.Parameter.empty
                    else matching_req.class_info
                ),
            )

            if injection.position is not None:
                resolved_args.append(injection)
            else:
                resolved_kwargs[param_name] = injection

            satisfied_reqs.add(matching_req.class_info)

        for req_class_info, req in reqs_by_type.items():
            if not req.optional and req_class_info not in satisfied_reqs:
                pass

        return cls(args=resolved_args, kwargs=resolved_kwargs)
