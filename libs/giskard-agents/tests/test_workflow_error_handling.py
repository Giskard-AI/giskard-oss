import pytest
from giskard import agents
from giskard.agents.errors import WorkflowError
from giskard.agents.workflow import ErrorPolicy
from pydantic import Field, PrivateAttr


class FailingGenerator(agents.generators.BaseGenerator):
    fail_after: int = Field(default=0)
    _num_calls: int = PrivateAttr(default=0)

    async def _complete(
        self,
        messages: list[agents.chat.Message],
        params: agents.generators.GenerationParams | None = None,
    ) -> agents.generators.Response:
        if self._num_calls >= self.fail_after:
            raise ValueError("Test error")
        self._num_calls += 1
        return agents.generators.Response(
            message=agents.chat.Message(
                role="assistant", content=f"Test response {self._num_calls}"
            ),
            finish_reason="stop",
        )


async def test_run_raises_error(generator):
    """Test that errors are handled correctly."""
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=0))

    # By default, will raise an error
    with pytest.raises(WorkflowError):
        await workflow.chat("Hello!", role="user").run()


async def test_run_returns_chat_with_error(generator):
    # We can define policy to return the chat with error
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=0))
    chat = await workflow.chat("Hello!", role="user").on_error(ErrorPolicy.RETURN).run()
    assert chat.last.content == "Hello!"
    assert chat.last.role == "user"
    assert chat.failed


async def test_run_skips_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=0))

    # If we choose to skip, we will get the same
    chat = await workflow.chat("Hello!", role="user").on_error(ErrorPolicy.SKIP).run()
    assert chat.last.content == "Hello!"
    assert chat.last.role == "user"
    assert chat.failed


async def test_run_many_raises_error(generator):
    """Test that errors are handled correctly."""

    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    # By default, will raise an error
    with pytest.raises(WorkflowError):
        await workflow.chat("Hello!", role="user").run_many(n=3)


async def test_run_many_returns_chat_with_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    # If we choose to return, we will get the same
    chats = (
        await workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.RETURN)
        .run_many(n=3)
    )

    assert len(chats) == 3

    # First is successful, others are failed
    assert not chats[0].failed
    assert chats[1].failed
    assert chats[2].failed

    # Check that the successful chat is ok
    assert not chats[0].failed
    assert len(chats[0].messages) == 2
    assert chats[0].last.role == "assistant"
    assert chats[0].last.content == "Test response 1"

    # Others should be failed
    assert chats[1].failed
    assert chats[2].failed

    assert len(chats[1].messages) == 1
    assert chats[1].last.role == "user"
    assert chats[1].last.content == "Hello!"

    assert len(chats[2].messages) == 1
    assert chats[2].last.role == "user"
    assert chats[2].last.content == "Hello!"


async def test_run_many_skips_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    # We can skip errors
    chats = (
        await workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.SKIP)
        .run_many(n=3)
    )

    assert len(chats) == 1
    assert not chats[0].failed
    assert len(chats[0].messages) == 2
    assert chats[0].last.content == "Test response 1"


async def test_run_batch_raises_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=0))

    with pytest.raises(WorkflowError):
        await workflow.chat("Hello!", role="user").run_batch(inputs=[{}, {}, {}])


async def test_run_batch_returns_chat_with_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    chats = (
        await workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.RETURN)
        .run_batch(inputs=[{}, {}, {}])
    )

    assert len(chats) == 3

    successes = [c for c in chats if not c.failed]
    failures = [c for c in chats if c.failed]

    assert len(successes) == 1
    assert len(failures) == 2

    # Successful chat
    assert len(successes[0].messages) == 2
    assert successes[0].last.role == "assistant"
    assert successes[0].last.content == "Test response 1"

    # Failed chats
    for chat in failures:
        assert len(chat.messages) == 1
        assert chat.last.role == "user"
        assert chat.last.content == "Hello!"


async def test_run_batch_skips_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    chats = (
        await workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.SKIP)
        .run_batch(inputs=[{}, {}, {}])
    )

    assert len(chats) == 1
    assert not chats[0].failed
    assert len(chats[0].messages) == 2
    assert chats[0].last.role == "assistant"
    assert chats[0].last.content == "Test response 1"


async def test_stream_many_raises_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    with pytest.raises(WorkflowError):
        async for _ in workflow.chat("Hello!", role="user").stream_many(n=3):
            pass


async def test_stream_many_returns_chat_with_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    results = []
    async for chat in (
        workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.RETURN)
        .stream_many(n=3)
    ):
        results.append(chat)

    assert len(results) == 3

    successes = [c for c in results if not c.failed]
    failures = [c for c in results if c.failed]

    assert len(successes) == 1
    assert len(failures) == 2

    # Successful chat
    assert len(successes[0].messages) == 2
    assert successes[0].last.role == "assistant"
    assert successes[0].last.content == "Test response 1"

    # Failed chats
    for chat in failures:
        assert len(chat.messages) == 1
        assert chat.last.role == "user"
        assert chat.last.content == "Hello!"


async def test_stream_many_skips_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    results = []
    async for chat in (
        workflow.chat("Hello!", role="user").on_error(ErrorPolicy.SKIP).stream_many(n=3)
    ):
        results.append(chat)

    assert len(results) == 1
    assert not results[0].failed
    assert len(results[0].messages) == 2
    assert results[0].last.role == "assistant"
    assert results[0].last.content == "Test response 1"


async def test_stream_batch_raises_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    with pytest.raises(WorkflowError):
        async for _ in workflow.chat("Hello!", role="user").stream_batch(
            inputs=[{}, {}, {}]
        ):
            pass


async def test_stream_batch_returns_chat_with_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    results = []
    async for chat in (
        workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.RETURN)
        .stream_batch(inputs=[{}, {}, {}])
    ):
        results.append(chat)

    assert len(results) == 3

    successes = [c for c in results if not c.failed]
    failures = [c for c in results if c.failed]

    assert len(successes) == 1
    assert len(failures) == 2

    # Successful chat
    assert len(successes[0].messages) == 2
    assert successes[0].last.role == "assistant"
    assert successes[0].last.content == "Test response 1"

    # Failed chats
    for chat in failures:
        assert len(chat.messages) == 1
        assert chat.last.role == "user"
        assert chat.last.content == "Hello!"


async def test_stream_batch_skips_error(generator):
    workflow = agents.ChatWorkflow(generator=FailingGenerator(fail_after=1))

    results = []
    async for chat in (
        workflow.chat("Hello!", role="user")
        .on_error(ErrorPolicy.SKIP)
        .stream_batch(inputs=[{}, {}, {}])
    ):
        results.append(chat)

    assert len(results) == 1
    assert not results[0].failed
    assert len(results[0].messages) == 2
    assert results[0].last.role == "assistant"
    assert results[0].last.content == "Test response 1"
