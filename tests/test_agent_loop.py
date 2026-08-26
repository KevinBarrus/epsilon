import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from core.agent_loop import AgentLoop, AgentLoopCancelled, ToolBatchEvent, ToolExecutionEvent
from core.end_policy import (
    FAILED_VERIFICATION_REMINDER,
    VERIFICATION_REMINDER,
    WriteVerificationPolicy,
    is_verification_command,
)
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
    ToolResult,
)
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionManager,
    ToolDefinition,
    ToolManager,
    create_read_file_tool,
)


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
    batches = [event for event in events if isinstance(event, ToolBatchEvent)]
    assert len(batches) == 1
    assert batches[0].execution_mode == "parallel"
    assert batches[0].duration_ms >= 0
    assert len(client.requests) == 2
    assert client.tools[0][0]["function"]["name"] == "read_file"  # type: ignore[index]


@pytest.mark.parametrize(
    "command",
    (
        "uv run pytest tests/test_agent_loop.py",
        "pytest && python -m py_compile module.py",
        "python -m unittest",
        "python manage.py test sessions",
        "python -m py_compile module.py",
        "ruff check src",
        "make test",
        "cargo test",
        "go test ./...",
    ),
)
def test_is_verification_command_accepts_common_checks(command: str) -> None:
    """测试常见测试与静态检查命令会被识别。"""

    assert is_verification_command(ToolCall("check", "run_command", {"command": command}))


@pytest.mark.parametrize(
    "command",
    ("pwd", "ls -la", "ls test.txt", "echo test", "pip list", "git status", "which python"),
)
def test_is_verification_command_rejects_information_commands(command: str) -> None:
    """测试环境查询命令不能作为代码验证证据。"""

    assert not is_verification_command(ToolCall("info", "run_command", {"command": command}))


def test_is_verification_command_rejects_missing_command() -> None:
    """测试缺少字符串命令参数时保持保守判断。"""

    assert not is_verification_command(ToolCall("missing", "run_command", {}))


def test_write_verification_policy_separates_check_commands_from_information_commands() -> None:
    """测试写后轨迹保留全部命令，同时单独记录有效检查命令。"""

    policy = WriteVerificationPolicy()
    information = ToolCall("info", "run_command", {"command": "pip list"})
    check = ToolCall("check", "run_command", {"command": "pytest tests"})
    information_result = ToolResult("info", "packages")
    check_result = ToolResult("check", "passed")

    policy.observe_tool_results(
        (ToolCall("write", "write_file", {}), information, check),
        (ToolResult("write", "written"), information_result, check_result),
    )

    assert policy.summary.post_write_command_results == (information_result, check_result)
    assert policy.summary.verification_command_results == (check_result,)


@pytest.mark.asyncio
async def test_write_verification_policy_does_not_change_read_only_task() -> None:
    """测试启用策略后只读任务仍在模型自然结束时直接完成。"""

    class TextOnlyClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            yield TextDelta("直接回答")

    client = TextOnlyClient()
    manager = ToolManager()
    result = await AgentLoop(
        client,
        manager,
        end_policy=WriteVerificationPolicy(),
    ).run([Message("user", "只回答问题")])

    assert result.verification_reminder_injected is False
    assert result.write_count == 0
    assert client.requests == 1


@pytest.mark.asyncio
async def test_write_verification_policy_reminds_once_after_unverified_write() -> None:
    """测试成功写入后未执行命令时只追加一次验证提醒。"""

    class WriteThenStopClient:
        def __init__(self) -> None:
            self.requests: list[list[Message]] = []

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                yield ToolCallEvent(ToolCall("write-1", "write_file", {}))
            elif len(self.requests) == 2:
                yield TextDelta("已经完成")
            else:
                assert self.requests[-1][-1] == Message("system", VERIFICATION_REMINDER)
                yield TextDelta("无法运行验证")

    manager = ToolManager()
    manager.register_local(
        ToolDefinition("write_file", "write", {"type": "object"}, "local", "read", False),
        lambda call: _tool_result(call, "written"),
    )
    client = WriteThenStopClient()
    result = await AgentLoop(
        client,
        manager,
        end_policy=WriteVerificationPolicy(),
    ).run([Message("user", "修改文件")])

    assert result.final_content == "无法运行验证"
    assert result.verification_reminder_injected is True
    assert result.write_count == 1
    assert result.post_write_command_results == ()
    assert len(client.requests) == 3


@pytest.mark.asyncio
async def test_write_verification_policy_reminds_once_after_failed_command() -> None:
    """测试写后命令失败时追加一次受限的收尾提醒。"""

    class WriteThenCommandClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests == 1:
                yield ToolCallEvent(ToolCall("write-1", "write_file", {}))
            elif self.requests == 2:
                yield ToolCallEvent(ToolCall("check-1", "run_command", {"command": "pytest"}))
            elif self.requests == 3:
                yield TextDelta("验证失败，已说明原因")
            else:
                assert messages[-1] == Message("system", FAILED_VERIFICATION_REMINDER)
                yield TextDelta("无法修复验证环境")

    manager = ToolManager()
    manager.register_local(
        ToolDefinition("write_file", "write", {"type": "object"}, "local", "read", False),
        lambda call: _tool_result(call, "written"),
    )
    manager.register_local(
        ToolDefinition(
            "run_command",
            "check",
            {"type": "object", "properties": {"command": {"type": "string"}}},
            "local",
            "read",
            False,
        ),
        lambda call: _tool_result(call, "failed", is_error=True),
    )
    client = WriteThenCommandClient()
    result = await AgentLoop(
        client,
        manager,
        end_policy=WriteVerificationPolicy(),
    ).run([Message("user", "修改并验证")])

    assert result.verification_reminder_injected is True
    assert result.write_count == 1
    assert result.post_write_command_results == (
        ToolResult("check-1", "failed", is_error=True),
    )
    assert client.requests == 4


@pytest.mark.asyncio
async def test_write_verification_policy_accepts_success_after_failed_command() -> None:
    """测试模型自行重跑成功后不会再注入收尾提醒。"""

    class RetryCommandClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests == 1:
                yield ToolCallEvent(ToolCall("write-1", "write_file", {}))
            elif self.requests in {2, 3}:
                yield ToolCallEvent(
                    ToolCall(f"check-{self.requests}", "run_command", {"command": "pytest"})
                )
            else:
                yield TextDelta("验证成功")

    attempts = 0

    async def command_result(call: ToolCall) -> ToolResult:
        nonlocal attempts
        attempts += 1
        return await _tool_result(
            call,
            "failed" if attempts == 1 else "passed",
            attempts == 1,
        )

    manager = ToolManager()
    manager.register_local(
        ToolDefinition("write_file", "write", {"type": "object"}, "local", "read", False),
        lambda call: _tool_result(call, "written"),
    )
    manager.register_local(
        ToolDefinition(
            "run_command",
            "check",
            {"type": "object", "properties": {"command": {"type": "string"}}},
            "local",
            "read",
            False,
        ),
        command_result,
    )

    result = await AgentLoop(
        RetryCommandClient(),
        manager,
        end_policy=WriteVerificationPolicy(),
    ).run([Message("user", "修改并验证")])

    assert result.final_content == "验证成功"
    assert result.verification_reminder_injected is False
    assert len(result.post_write_command_results) == 2


@pytest.mark.asyncio
async def test_write_verification_policy_does_not_run_after_tool_limit() -> None:
    """测试工具上限结束时不会额外注入验证提醒。"""

    class WriteOnlyClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield ToolCallEvent(ToolCall("write-1", "write_file", {}))

    manager = ToolManager()
    manager.register_local(
        ToolDefinition("write_file", "write", {"type": "object"}, "local", "read", False),
        lambda call: _tool_result(call, "written"),
    )
    result = await AgentLoop(
        WriteOnlyClient(),
        manager,
        max_tool_rounds=1,
        end_policy=WriteVerificationPolicy(),
    ).run([Message("user", "修改")])

    assert result.stop_reason == "tool_limit"
    assert result.verification_reminder_injected is False


@pytest.mark.asyncio
async def test_write_verification_policy_does_not_remind_after_cancellation() -> None:
    """测试取消模型请求时不会额外发起验证提醒。"""

    class WriteThenCancelClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests == 1:
                yield ToolCallEvent(ToolCall("write-1", "write_file", {}))
                return
            raise asyncio.CancelledError

    manager = ToolManager()
    manager.register_local(
        ToolDefinition("write_file", "write", {"type": "object"}, "local", "read", False),
        lambda call: _tool_result(call, "written"),
    )
    client = WriteThenCancelClient()

    with pytest.raises(AgentLoopCancelled):
        await AgentLoop(
            client,
            manager,
            end_policy=WriteVerificationPolicy(),
        ).run([Message("user", "修改")])

    assert client.requests == 2


async def _tool_result(
    tool_call: ToolCall,
    content: str,
    is_error: bool = False,
) -> ToolResult:
    """为收尾策略测试生成固定工具结果。"""

    return ToolResult(tool_call.call_id, content, is_error=is_error)


@pytest.mark.asyncio
async def test_agent_loop_runs_parallel_tools_concurrently_and_keeps_message_order() -> None:
    """测试同批只读工具并行运行，结果消息仍按模型调用顺序保存。"""

    class BatchClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests == 1:
                yield ToolCallEvent(ToolCall("call-1", "first", {}))
                yield ToolCallEvent(ToolCall("call-2", "second", {}))
                return
            yield TextDelta("完成")

    running = 0
    peak_running = 0

    async def handler(tool_call: ToolCall) -> ToolResult:
        nonlocal running, peak_running
        running += 1
        peak_running = max(peak_running, running)
        await asyncio.sleep(0.02 if tool_call.name == "first" else 0)
        running -= 1
        return ToolResult(tool_call.call_id, tool_call.name)

    manager = ToolManager()
    for name in ("first", "second"):
        manager.register_local(
            ToolDefinition(
                name=name,
                description=name,
                parameters={"type": "object"},
                source="local",
                permission="read",
                idempotent=True,
                execution_mode="parallel",
            ),
            handler,
        )
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    result = await AgentLoop(BatchClient(), manager).run(
        [Message("user", "读取")],
        on_event=collect,
    )

    assert peak_running == 2
    assert [message.content for message in result.new_messages if message.role == "tool"] == [
        "first",
        "second",
    ]
    assert [
        event.tool_call.name
        for event in events
        if isinstance(event, ToolExecutionEvent)
    ] == ["second", "first"]
    assert next(event for event in events if isinstance(event, ToolBatchEvent)).execution_mode == "parallel"


@pytest.mark.asyncio
async def test_agent_loop_serializes_batch_with_a_sequential_tool() -> None:
    """测试同批出现写工具时，所有工具保持串行。"""

    class BatchClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests == 1:
                yield ToolCallEvent(ToolCall("call-1", "read", {}))
                yield ToolCallEvent(ToolCall("call-2", "write", {}))
                return
            yield TextDelta("完成")

    running = 0
    peak_running = 0

    async def handler(tool_call: ToolCall) -> ToolResult:
        nonlocal running, peak_running
        running += 1
        peak_running = max(peak_running, running)
        await asyncio.sleep(0)
        running -= 1
        return ToolResult(tool_call.call_id, tool_call.name)

    async def approve(definition, tool_call, allow_session):
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    manager = ToolManager(permission_manager=PermissionManager(approve))
    manager.register_local(
        ToolDefinition(
            "read", "read", {"type": "object"}, "local", "read", True, "parallel"
        ),
        handler,
    )
    manager.register_local(
        ToolDefinition("write", "write", {"type": "object"}, "local", "write", True),
        handler,
    )
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    await AgentLoop(BatchClient(), manager).run([Message("user", "修改")], on_event=collect)

    assert peak_running == 1
    assert next(event for event in events if isinstance(event, ToolBatchEvent)).execution_mode == "sequential"


@pytest.mark.asyncio
async def test_agent_loop_cancels_parallel_tool_tasks() -> None:
    """测试取消 Agent 时会停止同批仍在运行的工具。"""

    class BatchClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield ToolCallEvent(ToolCall("call-1", "first", {}))
            yield ToolCallEvent(ToolCall("call-2", "second", {}))

    started = 0
    both_started = asyncio.Event()
    cancelled = 0

    async def handler(tool_call: ToolCall) -> ToolResult:
        nonlocal started, cancelled
        started += 1
        if started == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    manager = ToolManager()
    for name in ("first", "second"):
        manager.register_local(
            ToolDefinition(
                name,
                name,
                {"type": "object"},
                "local",
                "read",
                True,
                "parallel",
            ),
            handler,
        )
    task = asyncio.create_task(
        AgentLoop(BatchClient(), manager).run([Message("user", "读取")])
    )
    await asyncio.wait_for(both_started.wait(), timeout=1)

    task.cancel()

    with pytest.raises(AgentLoopCancelled):
        await task
    assert cancelled == 2


@pytest.mark.asyncio
async def test_agent_loop_returns_completed_tool_chain_at_round_limit(tmp_path) -> None:
    """测试工具轮次耗尽时保留已完成工具链，不再发起下一次请求"""

    class ToolOnlyClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            yield ToolCallEvent(
                ToolCall("call-1", "read_file", {"path": "README.md"})
            )

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    client = ToolOnlyClient()

    result = await AgentLoop(client, manager, max_tool_rounds=1).run(
        [Message(role="user", content="读取说明")]
    )

    assert result.stop_reason == "tool_limit"
    assert result.tool_rounds == 1
    assert result.new_messages == (
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "README.md"}),),
        ),
        Message(role="tool", content="项目说明", tool_call_id="call-1"),
    )
    assert result.messages == (
        Message(role="user", content="读取说明"),
        *result.new_messages,
    )
    assert client.requests == 1


@pytest.mark.asyncio
async def test_agent_loop_without_limit_can_finish_after_ten_tool_rounds(tmp_path) -> None:
    """测试默认 Agent Loop 不会在第十个工具回合强制停止。"""

    class ManyToolsClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests <= 11:
                yield ToolCallEvent(
                    ToolCall(
                        f"call-{self.requests}",
                        "read_file",
                        {"path": "README.md"},
                    )
                )
                return
            yield TextDelta("完成")

    (tmp_path / "README.md").write_text("项目说明", encoding="utf-8")
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    client = ManyToolsClient()

    result = await AgentLoop(client, manager).run(
        [Message(role="user", content="读取说明")]
    )

    assert result.stop_reason == "completed"
    assert result.tool_rounds == 11
    assert result.final_content == "完成"


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
