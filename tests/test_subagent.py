import asyncio
from collections.abc import Sequence

import pytest

import core.subagent as subagent
from core.context import ContextBudget
from core.model import Message, TextDelta, ToolCall, ToolCallEvent, UsageEvent
from core.subagent import SCOUT_SUMMARY_MAX_CHARS, create_spawn_agent_tool


class ScoutClient:
    """按任务内容返回 Scout 工具调用或最终摘要。"""

    def __init__(self) -> None:
        self.requests: list[list[Message]] = []
        self.tools: list[list[dict[str, object]]] = []

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools=(),
        thinking_level=None,
    ):
        self.requests.append(list(messages))
        self.tools.append(list(tools))
        if len(self.requests) == 1:
            yield ToolCallEvent(ToolCall("read-1", "read_file", {"path": "target.txt"}))
            return
        yield TextDelta("结论：目标内容已定位")
        yield UsageEvent(10, 2, 12)


@pytest.mark.asyncio
async def test_spawn_agent_uses_independent_read_only_loop(tmp_path) -> None:
    """测试 Scout 只获得三个只读工具并返回独立会话摘要。"""

    (tmp_path / "target.txt").write_text("证据", encoding="utf-8")
    client = ScoutClient()
    metrics = []
    definition, handler = create_spawn_agent_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        metrics.append,
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "定位目标"}))

    assert result.content == "结论：目标内容已定位"
    assert definition.execution_mode == "parallel"
    assert [tool["function"]["name"] for tool in client.tools[0]] == [  # type: ignore[index]
        "read_file",
        "list_files",
        "search_files",
    ]
    assert all(tool["function"]["name"] != "spawn_agent" for tool in client.tools[0])  # type: ignore[index]
    assert client.requests[0][-1] == Message("user", "定位目标")
    assert metrics[0].total_tokens == 12
    assert metrics[0].outcome == "completed"


@pytest.mark.asyncio
async def test_spawn_agent_limits_summary(tmp_path) -> None:
    """测试 Scout 摘要超过固定字符上限时会被截断。"""

    class LongSummaryClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield TextDelta("x" * (SCOUT_SUMMARY_MAX_CHARS + 100))

    _, handler = create_spawn_agent_tool(
        tmp_path,
        LongSummaryClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))

    assert len(result.content) == SCOUT_SUMMARY_MAX_CHARS
    assert result.content.endswith("[Scout summary truncated]")


@pytest.mark.asyncio
async def test_spawn_agent_times_out_with_structured_error(tmp_path, monkeypatch) -> None:
    """测试 Scout 超时后返回可恢复的结构化错误。"""

    class SlowClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            await asyncio.sleep(1)
            yield TextDelta("不会返回")

    monkeypatch.setattr(subagent, "SCOUT_TIMEOUT_SECONDS", 0.01)
    _, handler = create_spawn_agent_tool(
        tmp_path,
        SlowClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))

    assert result.is_error is True
    assert result.error_category == "timeout"


@pytest.mark.asyncio
async def test_spawn_agent_reports_tool_round_limit(tmp_path, monkeypatch) -> None:
    """测试 Scout 达到工具轮次上限后返回结构化错误。"""

    class LoopingClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield ToolCallEvent(ToolCall("list-1", "list_files", {"path": "."}))

    monkeypatch.setattr(subagent, "SCOUT_MAX_TOOL_ROUNDS", 1)
    _, handler = create_spawn_agent_tool(
        tmp_path,
        LoopingClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))

    assert result.is_error is True
    assert result.error_category == "tool_execution"
    assert "tool round limit" in result.content


@pytest.mark.asyncio
async def test_spawn_agent_cancels_running_scout_with_parent(tmp_path) -> None:
    """测试父调用取消时同步取消仍在运行的 Scout。"""

    cancelled = asyncio.Event()

    class SlowClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
            yield TextDelta("不会返回")

    _, handler = create_spawn_agent_tool(
        tmp_path,
        SlowClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )
    task = asyncio.create_task(
        handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))
    )
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_spawn_agent_caps_parallel_runs_at_three(tmp_path) -> None:
    """测试同一工具实例最多同时运行三个 Scout。"""

    active = 0
    peak = 0

    class BlockingClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            yield TextDelta("完成")

    _, handler = create_spawn_agent_tool(
        tmp_path,
        BlockingClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    await asyncio.gather(
        *(
            handler(ToolCall(f"spawn-{index}", "spawn_agent", {"task": "读取"}))
            for index in range(4)
        )
    )

    assert peak == 3
