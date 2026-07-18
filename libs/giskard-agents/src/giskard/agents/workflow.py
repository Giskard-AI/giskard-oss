import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import (
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
    cast,
)

import logfire_api as logfire
import tenacity as t
from giskard.llm import chat
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    ChatMessageParam,
    ToolMessage,
)
from giskard.llm.utils import deserialize_arguments
from pydantic import BaseModel, Field, ValidationError

from .chat import Chat
from .context import RunContext
from .errors.serializable import Error
from .errors.workflow_errors import ModelRefusalError, WorkflowError
from .generators import BaseGenerator, GenerationParams
from .templates import MessageTemplate, PromptsManager, get_prompts_manager
from .tools.tool import Tool


class TemplateReference(BaseModel):
    """A reference to a template file that will be loaded at runtime."""

    template_name: str


class ErrorPolicy(StrEnum):
    """The policy for handling errors."""

    RAISE = "raise"
    RETURN = "return"
    SKIP = "skip"


class StepType(StrEnum):
    """Discriminator for workflow step types."""

    TOOL_RESULT = "tool_result"
    COMPLETION = "completion"


class WorkflowStep(BaseModel):
    """A step in a workflow."""

    index: int = Field(default=0)
    step_type: StepType
    workflow: "ChatWorkflow[Any]"
    chat: Chat[Any]
    message: ChatMessage
    previous: "WorkflowStep | None" = Field(default=None)


StepGenerator = AsyncGenerator[WorkflowStep, None]


OutputType = TypeVar("OutputType", bound=BaseModel)
NewOutputType = TypeVar("NewOutputType", bound=BaseModel)


class _StepRunner:
    """Encapsulates per-run state and step execution.

    Parameters
    ----------
    workflow : ChatWorkflow[Any]
        The workflow instance.
    params : GenerationParams
        Generator parameters (including tools, response format).
    init_chat : Chat[Any]
        Initial chat (rendered messages + context).
    """

    def __init__(
        self,
        workflow: "ChatWorkflow[Any]",
        params: GenerationParams,
        init_chat: Chat[Any],
    ):
        self._workflow = workflow
        self._params = params
        self._init_chat = init_chat

    async def execute(
        self, max_steps: int | None
    ) -> AsyncGenerator[WorkflowStep, None]:
        """Drive the workflow step-by-step.

        Parameters
        ----------
        max_steps : int or None
            Maximum number of steps to run.

        Yields
        ------
        WorkflowStep
            Steps produced by the workflow as it advances. One step correspond to a message added to the chat.
        """
        if max_steps is not None and max_steps <= 0:
            return

        chat = self._init_chat  # will be cloned for each step

        step = None
        step_index = 0
        while max_steps is None or step_index < max_steps:
            # First, consume any pending tool calls on the current chat
            async for tool_message in self._run_tools(chat):
                chat = chat.clone().add(tool_message)
                step = WorkflowStep(
                    step_type=StepType.TOOL_RESULT,
                    workflow=self._workflow,
                    chat=chat,
                    message=tool_message,
                    previous=step,
                    index=step_index,
                )
                logfire.info(
                    "step.completed",
                    step_index=step.index,
                    message=step.message,
                )
                yield step

                step_index += 1
                if max_steps is not None and step_index >= max_steps:
                    return

            # Now we run the generator to create a completion
            message = await self._run_completion(chat)
            chat = chat.clone().add(message)
            step = WorkflowStep(
                step_type=StepType.COMPLETION,
                workflow=self._workflow,
                chat=chat,
                message=message,
                previous=step,
                index=step_index,
            )
            logfire.info(
                "step.completed",
                step_index=step.index,
                message=step.message,
            )
            yield step
            step_index += 1
            if max_steps is not None and step_index >= max_steps:
                break

            # If the last message has no tool calls, we're done.
            if not message.tool_calls:
                break

    async def _run_tools(self, chat: Chat[Any]) -> AsyncGenerator[ToolMessage, None]:
        if (
            not chat.messages
            or not isinstance(chat.last, AssistantMessage)
            or not chat.last.tool_calls
        ):
            return

        for tool_call in chat.last.tool_calls:
            tool_name = tool_call.function.name or "<missing>"
            if tool_name not in self._workflow.tools:
                registered_tools = ", ".join(sorted(self._workflow.tools)) or "<none>"
                raise ValueError(
                    f"Unknown tool call '{tool_name}' "
                    f"(tool_call_id='{tool_call.id}'). "
                    f"Registered tools: {registered_tools}."
                )

            tool = self._workflow.tools[tool_name]
            tool_content = await tool.run(
                deserialize_arguments(tool_call.function.arguments),
                ctx=chat.context,
            )
            yield ToolMessage(
                tool_call_id=tool_call.id,
                content=tool_content,
            )

    async def _run_completion(self, chat: Chat[Any]) -> AssistantMessage:
        # Determine if strict output parsing is enabled
        strict_parsing = chat.output_model and self._workflow.output_model_strict
        if strict_parsing:
            max_attempts = 1 + int(self._workflow.output_model_num_retries or 0)
            retrier = t.AsyncRetrying(
                stop=t.stop_after_attempt(max_attempts),
                retry=t.retry_if_exception_type(ValidationError),
                reraise=True,
            )
            return await retrier(
                self._run_completion_with_output_validation,
                chat,
                output_model=chat.output_model,
            )

        # Simple completion without output validation
        response = await self._workflow.generator.complete(chat.messages, self._params)

        if not response.choices:
            raise RuntimeError("Provider returned an empty choices list.")

        return response.choices[0].message

    async def _run_completion_with_output_validation(
        self, chat: Chat[OutputType], output_model: type[OutputType]
    ) -> AssistantMessage:
        response = await self._workflow.generator.complete(chat.messages, self._params)

        if not response.choices:
            raise RuntimeError("Provider returned an empty choices list.")

        # If the assistant produced tool calls, defer parsing to after tools are run
        if response.choices[0].message.tool_calls:
            return response.choices[0].message

        # If the model refused to generate content, raise a ModelRefusalError
        choice = response.choices[0]
        if choice.finish_reason == "refusal" or choice.message.is_refusal:
            raise ModelRefusalError(refusal=choice.message.text)

        # Attempt the parsing to raise ValidationError if output is not compatible
        output_model.model_validate_json(choice.message.text or "")

        return choice.message


class ChatWorkflow(BaseModel, Generic[OutputType]):
    """A workflow for handling chat completions.

    Attributes
    ----------
    messages : list[Message | MessageTemplate | TemplateReference]
        Conversation messages or templates to render.
    tools : dict[str, Tool]
        Tools available to the workflow by name.
    generator : BaseGenerator
        The generator instance to use for completions.
    prompt_manager : PromptsManager
        The prompt manager to use for rendering templates.
    output_model : type[OutputType] or None
        Optional Pydantic model used for response formatting.
    output_model_strict : bool, default True
        Whether to raise an error if the generator output is not compatible with the output schema.
    output_model_num_retries : int or None, default 2
        The number of times to retry the generation if the output does not match the expected schema. Requires `output_model_strict` to be `True`.
    context : RunContext
        Execution context copied per run.
    error_mode : Literal["raise", "pass"]
        Error handling behavior.
    """

    generator: "BaseGenerator"

    messages: list[ChatMessage | MessageTemplate | TemplateReference] = Field(
        default_factory=list
    )
    tools: dict[str, Tool] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_model: type[OutputType] | None = Field(default=None)
    output_model_strict: bool = Field(default=True)
    output_model_num_retries: int | None = Field(default=2, ge=0)
    prompt_manager: PromptsManager = Field(default_factory=get_prompts_manager)
    context: RunContext = Field(default_factory=RunContext)
    error_policy: ErrorPolicy = Field(default=ErrorPolicy.RAISE)

    def chat(
        self,
        message: str | ChatMessage | ChatMessageParam | MessageTemplate,
        role: Literal["user", "assistant", "system", "developer"] = "user",
        *,
        as_template: bool = False,
    ) -> Self:
        """Add a chat message to the workflow.

        Parameters
        ----------
        message : str | Message | MessageTemplate
            The message to add. Plain strings are appended as literal text by default.
        role : Role, default "user"
            The role for the message, used when ``message`` is a string.
        as_template : bool, default False
            When True, parse a string ``message`` as a Jinja2 template.
            For other message types, this parameter is ignored.
            Alternatively, you can pass a :class:`~.templates.MessageTemplate` instance.

            .. warning::

                Treating a string as a template evaluates Jinja2 syntax at render time.
                If any part of ``message`` can be influenced by untrusted input, this
                can lead to template injection and unintended disclosure or execution
                of logic exposed by the template environment. Only enable this for
                trusted, developer-authored template strings.
        """
        if isinstance(message, str):
            if as_template:
                message = MessageTemplate(role=role, content_template=message)
            else:
                message = chat.message(message, role)

        return self.model_copy(update={"messages": [*self.messages, message]})

    def template(self, template_name: str) -> Self:
        """Load messages from a template file.

        Parameters
        ----------
        template_name : str
            The template name in dot notation (e.g., "crescendo.master_prompt")

        Returns
        -------
        ChatWorkflow
            The workflow instance for method chaining.
        """
        template_message = TemplateReference(template_name=template_name)
        return self.model_copy(update={"messages": [*self.messages, template_message]})

    def with_tools(self, *tools: Tool) -> Self:
        """Add tools to the workflow.

        Parameters
        ----------
        *tools : Tool
            Tools to add to the workflow.

        Returns
        -------
        ChatWorkflow
            The workflow instance for method chaining.
        """
        new_tools = self.tools.copy()
        new_tools.update({tool.name: tool for tool in tools})
        return self.model_copy(update={"tools": new_tools})

    def with_output(
        self: "ChatWorkflow[Any]",
        output_model: type[NewOutputType],
        strict: bool = True,
        num_retries: int | None = 2,
    ) -> "ChatWorkflow[NewOutputType]":
        """Set the output model for the workflow.

        Parameters
        ----------
        output_model : Type[OutputType]
            The output model to use for the workflow.
        strict : bool, default True
            Whether to raise an error if the generator output is not compatible with the output schema.
        num_retries : int or None, default 2
            The number of times to retry the generation if the output does not match the expected schema. Requires `strict` to be `True`.

        Returns
        -------
        ChatWorkflow
            The workflow instance for method chaining.
        """
        return cast(
            "ChatWorkflow[NewOutputType]",
            self.model_copy(
                update={
                    "output_model": output_model,
                    "output_model_strict": strict,
                    "output_model_num_retries": num_retries,
                }
            ),
        )

    def with_inputs(self, **kwargs: Any) -> Self:
        """Set the input for the workflow.

        Parameters
        ----------
        **kwargs : Any
            The input for the workflow.

        Returns
        -------
        ChatWorkflow
            The workflow instance for method chaining.
        """
        return self.model_copy(update={"inputs": {**self.inputs, **kwargs}})

    def with_context(self, context: RunContext) -> Self:
        """Set the context for the workflow."""
        return self.model_copy(update={"context": context})

    def on_error(self, error_policy: ErrorPolicy) -> Self:
        """Set the error handling behavior for the workflow."""
        return self.model_copy(update={"error_policy": error_policy})

    @asynccontextmanager
    async def steps(self, max_steps: int | None = None) -> AsyncIterator[StepGenerator]:
        """Create an async context for iterating workflow steps.

        Parameters
        ----------
        max_steps : int or None
            Maximum number of steps to run. If None, runs until completion.

        Yields
        ------
        StepGenerator
            An async generator producing `WorkflowStep` instances.
        """
        params = GenerationParams(
            tools=list(self.tools.values()),
            response_format=self.output_model,
        )
        init_chat = await self._init_chat()

        runner = _StepRunner(self, params, init_chat)
        agen = runner.execute(max_steps)
        try:
            yield agen
        finally:
            await agen.aclose()

    async def _init_chat(self) -> Chat[OutputType]:
        context = self.context.model_copy(deep=True)
        context.inputs = self.inputs.copy()
        return Chat(
            messages=await self._render_messages(),
            output_model=self.output_model,
            context=context,
        )

    @logfire.instrument("chat_workflow.run")
    async def run(self, max_steps: int | None = None) -> Chat[OutputType]:
        """Runs the workflow.

        Parameters
        ----------
        max_steps : int, optional
            The number of steps to run. If not provided, the workflow will run until the chat is complete.

        Returns
        -------
        Chat[OutputType]
            A Chat object containing the conversation messages.

        Raises
        ------
        WorkflowError
            If the workflow fails and the error policy is RAISE (default).
        """
        last_step: WorkflowStep | None = None

        try:
            # Run the steps, and store the last step.
            async with self.steps(max_steps=max_steps) as steps:
                async for step in steps:
                    last_step = step

            if last_step is not None:
                return last_step.chat

            raise WorkflowError("ChatWorkflow failed: no steps were executed.")

        except Exception as err:
            return await self._handle_error(err, last_step)

    async def _handle_error(
        self, err: Exception, last_step: WorkflowStep | None = None
    ) -> Chat[OutputType]:
        # Raise an error if the error mode is RAISE.
        if self.error_policy == ErrorPolicy.RAISE:
            raise WorkflowError(
                "Step processing failed",
                last_step=last_step,
                exception=err,
            ) from err

        # Otherwise return partial chat with error.
        if last_step and last_step.chat is not None:
            chat = last_step.chat.clone()
        else:
            chat = await self._init_chat()

        # Set the error on the chat, this will make it "failed".
        chat.error = Error(message=str(err))

        return chat

    @logfire.instrument("chat_workflow.run_many")
    async def run_many(
        self, n: int, max_steps: int | None = None
    ) -> list[Chat[OutputType]]:
        """Run multiple completions in parallel.

        Parameters
        ----------
        n : int
            Number of parallel completions to run.
        max_steps : int, optional
            The maximum number of steps to run for each completion.

        Returns
        -------
        List[Chat]
            List of Chat objects containing the conversation messages.

        Raises
        ------
        WorkflowError
            If the workflow fails and the error policy is RAISE (default).
        """
        results = await asyncio.gather(
            *[self.run(max_steps=max_steps) for _ in range(n)],
            return_exceptions=False,
        )

        # If the error mode is SKIP, we return only the successful chats.
        if self.error_policy == ErrorPolicy.SKIP:
            results = [chat for chat in results if not chat.failed]

        return results

    @logfire.instrument("chat_workflow.run_batch")
    async def run_batch(
        self, inputs: list[dict[str, Any]], max_steps: int | None = None
    ) -> list[Chat[OutputType]]:
        """Run a batch of completions with different parameters.

        Parameters
        ----------
        params_list : list[dict]
            List of parameter dictionaries for each completion.
        max_steps : int, optional
            The maximum number of steps to run for each completion.

        Returns
        -------
        List[Chat]
            List of completion results.
        """
        workflows = [
            self.model_copy(update={"inputs": {**self.inputs, **params}})
            for params in inputs
        ]

        chats = await asyncio.gather(
            *[workflow.run(max_steps=max_steps) for workflow in workflows],
            return_exceptions=False,
        )

        if self.error_policy == ErrorPolicy.SKIP:
            chats = [chat for chat in chats if not chat.failed]

        return chats

    async def stream_many(
        self, n: int, max_steps: int | None = None
    ) -> AsyncIterator[Chat[OutputType]]:
        """Stream multiple completions as they complete.

        Parameters
        ----------
        n : int
            Number of parallel completions to run.
        max_steps : int, optional
            The maximum number of steps to run for each completion.

        Yields
        ------
        Chat
            Chat objects as they complete.
        """
        tasks = [self.run(max_steps=max_steps) for _ in range(n)]

        for coro in asyncio.as_completed(tasks):
            result = await coro

            # Skip failed chats if the error policy is SKIP
            if result.failed and self.error_policy == ErrorPolicy.SKIP:
                continue

            yield result

    async def stream_batch(
        self, inputs: list[dict[str, Any]], max_steps: int | None = None
    ) -> AsyncIterator[Chat[OutputType]]:
        """Stream a batch of completions as they complete.

        Parameters
        ----------
        inputs : list[dict]
            List of parameter dictionaries for each completion.
        max_steps : int, optional
            The maximum number of steps to run for each completion.

        Yields
        ------
        Chat
            Chat objects as they complete.
        """
        workflows = [
            self.model_copy(update={"inputs": {**self.inputs, **params}})
            for params in inputs
        ]
        tasks = [workflow.run(max_steps=max_steps) for workflow in workflows]

        for coro in asyncio.as_completed(tasks):
            result = await coro

            # Skip failed chats if the error policy is SKIP
            if result.failed and self.error_policy == ErrorPolicy.SKIP:
                continue

            yield result

    async def _render_messages(self) -> list[ChatMessage]:
        rendered_messages = []
        context_vars = {}
        if self.output_model is not None:
            context_vars["_instr_output"] = _output_instructions(self.output_model)
        context_vars.update(self.inputs)
        for message in self.messages:
            if isinstance(message, MessageTemplate):
                rendered_messages.append(message.render(**context_vars))
            elif isinstance(message, TemplateReference):
                template_messages = await self.prompt_manager.render_template(
                    message.template_name, context_vars
                )
                rendered_messages.extend(template_messages)
            else:
                rendered_messages.append(message)
        return rendered_messages


def _output_instructions(output_model: type[BaseModel]) -> str:
    return f"Provide your answer in JSON format, respecting this schema:\n{output_model.model_json_schema()}"
