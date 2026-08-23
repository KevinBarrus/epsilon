import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from core.agent_loop import AgentLoop, AgentLoopCancelled, ToolExecutionEvent
from types import SimpleNamespace
import core.agent_loop as agent_loop
from core.context import ContextBuildResult
from core.errors import AgentError
from core.model import (
    Message,
    ModelEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from core.tools import ToolManager, create_read_file_tool


class FakeModelClient:
    """按请求次数返回工具调用和最终文本的模型客户端。"""

    def __init__(self) -> None:
        self.requests: list[list[Message]] = []
        self.tools: list[list[dict[str, object]]] = []

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]] = (),
        thinking_level: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append(list(messages))
        self.tools.append(list(tools))
        if len(self.requests) == 1:
            yield ToolCallEvent(
                ToolCall(
                    call_id="call-1",
                    name="read_file",
                    arguments={"path": "README.md"},
                )
            )
            return
        yield TextDelta("文件已经读取")


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_and_continues_model_request(
    tmp_path,
) -> None:
    """测试 Agent Loop 执行工具后继续请求模型。"""

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    client = FakeModelClient()
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    events: list[object] = []

    async def collect_event(event: object) -> None:
        events.append(event)

    result = await AgentLoop(tool_manager=manager, client=client).run(
        [Message(role="user", content="读取说明")],
        on_event=collect_event,
    )

    assert result.final_content == "文件已经读取"
    assert result.messages[-1] == Message(role="assistant", content="文件已经读取")
    assert result.messages[-2] == Message(
        role="tool",
        content="项目说明",
        tool_call_id="call-1",
    )
    assert result.new_messages == result.messages[1:]
    assert any(isinstance(event, ToolExecutionEvent) for event in events)
    assert len(client.requests) == 2
    assert client.tools[0][0]["function"]["name"] == "read_file"  # type: ignore[index]


@pytest.mark.asyncio
async def test_agent_loop_yields_after_text_delta_for_ui_refresh() -> None:
    """测试连续文本分片之间会让出事件循环给界面刷新任务。"""

    refresh_ran = False

    class BufferedClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield TextDelta("第一段")
            assert refresh_ran is True
            yield TextDelta("第二段")

    async def handle_event(event: ModelEvent) -> None:
        nonlocal refresh_ran
        if isinstance(event, TextDelta) and not refresh_ran:

            async def mark_refreshed() -> None:
                nonlocal refresh_ran
                refresh_ran = True

            asyncio.create_task(mark_refreshed())

    result = await AgentLoop(BufferedClient(), ToolManager()).run(
        [Message(role="user", content="继续")],
        on_event=handle_event,
    )

    assert result.final_content == "第一段第二段"


@pytest.mark.asyncio
async def test_agent_loop_rebuilds_context_after_tool_result(tmp_path) -> None:
    """测试每轮工具结果都会在下一次模型请求前重建上下文。"""

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    client = FakeModelClient()
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    build_inputs: list[list[Message]] = []

    async def build_context(messages, force_compaction: bool) -> ContextBuildResult:
        assert force_compaction is False
        build_inputs.append(list(messages))
        if len(build_inputs) == 2:
            return ContextBuildResult(
                [
                    Message(role="system", content="已压缩的历史"),
                    *messages[-2:],
                ]
            )
        return ContextBuildResult(list(messages))

    result = await AgentLoop(client, manager).run(
        [Message(role="user", content="读取说明")],
        build_context=build_context,
    )

    assert [len(messages) for messages in build_inputs] == [1, 3]
    assert client.requests[1][0] == Message(role="system", content="已压缩的历史")
    assert result.messages[-3:] == (
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "README.md"}),),
        ),
        Message(role="tool", content="项目说明", tool_call_id="call-1"),
        Message(role="assistant", content="文件已经读取"),
    )


@pytest.mark.asyncio
async def test_agent_loop_returns_unknown_tool_result_to_model() -> None:
    """测试未知工具不会被执行，而是将错误返回模型。"""

    class UnknownToolClient(FakeModelClient):
        async def stream_response(
            self,
            messages: Sequence[Message],
            tools: Sequence[dict[str, object]] = (),
            thinking_level: str | None = None,
        ) -> AsyncIterator[ModelEvent]:
            self.requests.append(list(messages))
            self.tools.append(list(tools))
            if len(self.requests) == 1:
                yield ToolCallEvent(
                    ToolCall("call-1", "missing", {})
                )
            else:
                yield TextDelta("工具不存在")

    client = UnknownToolClient()
    result = await AgentLoop(client, ToolManager()).run(
        [Message(role="user", content="调用未知工具")]
    )

    assert result.final_content == "工具不存在"
    assert "tool not found: missing" in client.requests[1][-1].content


@pytest.mark.asyncio
async def test_agent_loop_retries_model_network_error_once() -> None:
    """测试模型网络错误会按统一策略重试一次。"""

    class RetryClient:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_response(
            self,
            messages: Sequence[Message],
            tools: Sequence[dict[str, object]] = (),
            thinking_level: str | None = None,
        ) -> AsyncIterator[ModelEvent]:
            self.calls += 1
            if self.calls == 1:
                raise AgentError(
                    category="network",
                    operation="model_request",
                    user_message="模型网络请求失败",
                    retryable=True,
                )
            yield TextDelta("重试成功")

    client = RetryClient()
    result = await AgentLoop(client, ToolManager()).run(
        [Message(role="user", content="继续")]
    )

    assert result.final_content == "重试成功"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_agent_loop_uses_exponential_backoff_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试重试使用指数退避，并优先遵守服务端等待时间。"""

    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    class RetryClient:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.calls += 1
            if self.calls == 1:
                raise AgentError("network", "model_request", "网络失败")
            if self.calls == 2:
                cause = type(
                    "RateLimitCause",
                    (),
                    {"response": type("Response", (), {"headers": {"Retry-After": "3"}})()},
                )()
                raise AgentError("rate_limit", "model_request", "限流", cause=cause)
            yield TextDelta("重试成功")

    monkeypatch.setattr("core.agent_loop.asyncio.sleep", sleep)
    monkeypatch.setattr("core.agent_loop.random.uniform", lambda start, end: 0)

    result = await AgentLoop(RetryClient(), ToolManager()).run(
        [Message(role="user", content="继续")]
    )

    assert result.final_content == "重试成功"
    assert delays == [0.5, 3]


@pytest.mark.asyncio
async def test_agent_loop_does_not_retry_after_partial_model_output() -> None:
    """测试模型已经输出内容后失败不会重复请求。"""

    class PartialFailureClient:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_response(
            self,
            messages: Sequence[Message],
            tools: Sequence[dict[str, object]] = (),
            thinking_level: str | None = None,
        ) -> AsyncIterator[ModelEvent]:
            self.calls += 1
            yield TextDelta("部分内容")
            raise AgentError(
                category="network",
                operation="model_request",
                user_message="模型网络请求失败",
                retryable=True,
            )

    client = PartialFailureClient()
    with pytest.raises(AgentError):
        await AgentLoop(client, ToolManager()).run(
            [Message(role="user", content="继续")]
        )

    assert client.calls == 1


@pytest.mark.asyncio
async def test_agent_loop_keeps_completed_tool_chain_when_cancelled(tmp_path) -> None:
    """测试取消后仍保留已完成的工具调用和结果。"""

    class BlockingClient(FakeModelClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def stream_response(
            self,
            messages: Sequence[Message],
            tools: Sequence[dict[str, object]] = (),
            thinking_level: str | None = None,
        ) -> AsyncIterator[ModelEvent]:
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                yield ToolCallEvent(
                    ToolCall("call-1", "read_file", {"path": "README.md"})
                )
                return
            self.started.set()
            await asyncio.Event().wait()
            yield TextDelta("不会返回")

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    client = BlockingClient()
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    task = asyncio.create_task(
        AgentLoop(client, manager).run([Message(role="user", content="读取说明")])
    )
    await client.started.wait()
    task.cancel()

    with pytest.raises(AgentLoopCancelled) as error:
        await task

    assert error.value.new_messages == (
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "README.md"}),),
            status="cancelled",
        ),
        Message(role="tool", content="项目说明", tool_call_id="call-1"),
    )


@pytest.mark.asyncio
async def test_agent_loop_emits_retry_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试重试前通过 on_event 发出 RetryEvent 供界面展示。"""

    async def sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(agent_loop, "asyncio", SimpleNamespace(sleep=sleep))

    class RetryClient:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.calls += 1
            if self.calls == 1:
                raise AgentError("network", "model_request", "网络失败")
            yield TextDelta("成功")

    events: list[object] = []

    async def on_event(event) -> None:
        events.append(event)

    await AgentLoop(RetryClient(), ToolManager()).run(
        [Message(role="user", content="继续")],
        on_event=on_event,
    )

    retries = [e for e in events if isinstance(e, agent_loop.RetryEvent)]
    assert len(retries) == 1
    assert retries[0].attempt == 1
    assert retries[0].max_attempts == 2
