"""子 Agent 上下文边界（fork / fresh）与结构化回传的测试。"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from core.agent_loop import AgentLoop, ToolBatchEvent, ToolExecutionEvent, parent_context_snapshot
from core.context import ContextBudget
from core.loop_guard import config_from_settings
from core.model import Message, TextDelta, ToolCall, ToolCallEvent, ToolResult, UsageEvent
from core.subagent import (
    DEFAULT_FORK_TURNS,
    FORK_CONTEXT_NOTICE,
    SPAWN_MODES,
    SubagentRunMetrics,
    _limit_summary,
    changed_paths,
    create_spawn_agent_tool,
    fork_history,
    resolve_mode,
    resolve_turns,
    unverified_changed_paths,
)
from core.tools import ToolManager, create_read_file_tool

NEEDLE = "紫色独角兽"
NOTES = "notes.txt"


def _message(role: str, content: str, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
    """构造一条测试消息。"""

    return Message(role=role, content=content, tool_calls=tool_calls)


# --- fork_history 截断 -------------------------------------------------------


def test_fork_history_drops_trailing_tool_call_message() -> None:
    """尾部尚未产生结果的 assistant 工具调用消息要被去掉（避免半个回合）。"""

    snapshot = (
        _message("user", "任务"),
        _message("assistant", "", (ToolCall("c1", "read_file", {"path": NOTES}),)),
        _message("tool", "内容"),
        _message("assistant", "", (ToolCall("c2", "spawn_agent", {"task": "x"}),)),
    )

    history = fork_history(snapshot)

    assert [message.content for message in history] == ["任务", "", "内容"]


def test_fork_history_keeps_full_history_when_turns_is_zero() -> None:
    """turns=0 表示不做回合截断。"""

    snapshot = (_message("user", "一"), _message("assistant", "a"), _message("user", "二"))

    assert fork_history(snapshot, 0) == snapshot


def test_fork_history_truncates_at_user_turn_boundary() -> None:
    """fork_last_n 只保留最后 n 个 user 回合，且起点落在 user 消息上。"""

    snapshot = (
        _message("user", "第一轮"),
        _message("assistant", "读到 X"),
        _message("user", "第二轮"),
        _message("assistant", "读到 Y"),
        _message("user", "第三轮"),
        _message("assistant", "读到 Z"),
    )

    history = fork_history(snapshot, 2)

    assert history[0] == _message("user", "第二轮")
    assert [message.content for message in history] == ["第二轮", "读到 Y", "第三轮", "读到 Z"]


def test_fork_history_degrades_when_turns_exceed_available() -> None:
    """回合数超出实际数量时退化为全量。"""

    snapshot = (_message("user", "一"), _message("assistant", "a"))

    assert fork_history(snapshot, 99) == snapshot


# --- 模式解析 ---------------------------------------------------------------


def test_resolve_mode_prefers_forced_mode() -> None:
    """强制配置优先于模型给出的 mode。"""

    assert resolve_mode("fresh", "fresh", "fork") == "fork"
    assert resolve_mode(None, "fork", None) == "fork"
    assert resolve_mode("fork_last_n", "fresh", None) == "fork_last_n"


def test_resolve_mode_ignores_unknown_value() -> None:
    """非法 mode 回落到默认值。"""

    assert resolve_mode("banana", "fresh", None) == "fresh"


def test_resolve_turns_validates() -> None:
    """回合数非法时回落默认值。"""

    assert resolve_turns(6) == 6
    assert resolve_turns(0) == DEFAULT_FORK_TURNS
    assert resolve_turns("4") == DEFAULT_FORK_TURNS
    assert resolve_turns(True) == DEFAULT_FORK_TURNS


def test_spawn_modes_match_config_whitelist() -> None:
    """子 Agent 模式白名单与 config 校验保持一致。"""

    from core import config as config_module

    assert tuple(config_module._SPAWN_MODES) == SPAWN_MODES


# --- 结构化回传 -------------------------------------------------------------


def test_worker_summary_always_has_three_sections() -> None:
    """Worker 摘要总是补齐 CHANGED / EVIDENCE / BLOCKED。"""

    result = _limit_summary("## CHANGED\n- a.py", "worker")

    assert "## CHANGED" in result and "- a.py" in result
    assert "## EVIDENCE\n（未提供）" in result
    assert "## BLOCKED\n（未提供）" in result


def test_worker_summary_without_sections_is_not_prose() -> None:
    """完全没有小节的散文会被规范成三个小节。"""

    result = _limit_summary("我做完了，应该没问题。", "worker")

    assert "## CHANGED" in result
    assert "## EVIDENCE" in result
    assert "## BLOCKED" in result
    # 三个小节都在，且原始说明被保留而不是当成唯一内容
    assert "我做完了" in result


def test_reviewer_summary_keeps_its_sections() -> None:
    """Reviewer 保持 Passed / Findings / Evidence 但同样补齐。"""

    result = _limit_summary("## Passed\n- 通过", "reviewer")

    assert "## Passed" in result
    assert "## Findings\n（未提供）" in result
    assert "## Evidence\n（未提供）" in result


def test_changed_paths_and_verification(tmp_path: Path) -> None:
    """CHANGED 里的路径可被父侧核对，编造的路径会被点名。"""

    (tmp_path / "real.py").write_text("x", encoding="utf-8")
    summary = "## CHANGED\n- real.py\n- ghost.py\n## EVIDENCE\n- pytest\n## BLOCKED\n- 无"

    assert changed_paths(summary) == ("real.py", "ghost.py")
    assert unverified_changed_paths(summary, tmp_path) == ("ghost.py",)


# --- 缓存键透传 -------------------------------------------------------------


def test_prompt_cache_key_sent_only_for_non_deepseek() -> None:
    """prompt_cache_key 只对支持的 provider 透传。"""

    from core.openai_client import _apply_prompt_cache_key

    request: dict[str, object] = {}
    _apply_prompt_cache_key(request, "key-1", is_deepseek=True)
    assert "prompt_cache_key" not in request

    _apply_prompt_cache_key(request, "key-1", is_deepseek=False)
    assert request["prompt_cache_key"] == "key-1"

    empty: dict[str, object] = {}
    _apply_prompt_cache_key(empty, None, is_deepseek=False)
    assert empty == {}


# --- 关键集成测试：父先读 X，子 Agent 再用 ------------------------------------


class ContentAwareClient:
    """只有上下文里已包含目标内容时才能直接回答，否则先读一遍。

    `inherits_context=True` 用来模拟“继承了父前缀”的子 Agent：它继承到的
    整段前缀都可以命中提示缓存；fresh 子 Agent 没有可命中的前缀。
    """

    def __init__(self, needle: str, path: str, inherits_context: bool = False) -> None:
        self._needle = needle
        self._path = path
        self._inherits_context = inherits_context
        self.requests: list[list[Message]] = []
        self.reads = 0

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]] = (),
        thinking_level: str | None = None,
    ):
        self.requests.append(list(messages))
        text = "\n".join(message.content for message in messages)
        prompt_tokens = max(1, sum(len(message.content) for message in messages))
        yield UsageEvent(
            prompt_tokens,
            5,
            prompt_tokens + 5,
            cached_tokens=prompt_tokens if self._inherits_context else 0,
        )
        if self._needle in text:
            yield TextDelta(
                "## Files Read\n- notes.txt\n## Key Evidence\n- notes.txt:1 "
                f"{self._needle}\n## Conclusion\n- 内容已确认"
            )
            return
        self.reads += 1
        yield ToolCallEvent(
            ToolCall(f"read-{self.reads}", "read_file", {"path": self._path})
        )


class ParentClient:
    """父 Agent：先读文件，再把问题委派出去，最后收尾。"""

    def __init__(self) -> None:
        self._step = 0
        self.spawn_calls: list[ToolCall] = []

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]] = (),
        thinking_level: str | None = None,
    ):
        self._step += 1
        if self._step == 1:
            yield ToolCallEvent(ToolCall("read-1", "read_file", {"path": NOTES}))
            return
        if self._step == 2:
            call = ToolCall("spawn-1", "spawn_agent", {"task": f"{NOTES} 里是什么？"})
            self.spawn_calls.append(call)
            yield ToolCallEvent(call)
            return
        yield TextDelta("完成")


async def _run_parent(force_mode: str, tmp_path: Path) -> tuple[list[object], list[SubagentRunMetrics]]:
    """跑一遍"父读文件 → 委派子 Agent"，返回子事件与子 Agent 指标。"""

    (tmp_path / NOTES).write_text(f"备注：{NEEDLE}", encoding="utf-8")
    child = ContentAwareClient(NEEDLE, NOTES, inherits_context=force_mode != "fresh")
    metrics: list[SubagentRunMetrics] = []
    child_events: list[object] = []

    async def collect_child(event: object) -> None:
        child_events.append(event)

    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    manager.register_local(
        *create_spawn_agent_tool(
            tmp_path,
            lambda: child,
            lambda: "high",
            ContextBudget(10_000, 1_000, 2_000),
            on_metrics=metrics.append,
            on_event=collect_child,
            force_mode=force_mode,
        )
    )

    await AgentLoop(ParentClient(), manager).run([Message(role="user", content="总结 notes.txt")])
    return child_events, metrics


@pytest.mark.asyncio
async def test_fresh_child_must_read_the_file_itself(tmp_path: Path) -> None:
    """fresh 子 Agent 拿不到父已读内容，必须自己再读一遍。"""

    child_events, metrics = await _run_parent("fresh", tmp_path)
    reads = [
        event
        for event in child_events
        if isinstance(event, ToolExecutionEvent) and event.tool_call.name == "read_file"
    ]

    assert reads, "fresh 子 Agent 应该自己调用 read_file"
    assert metrics[0].mode == "fresh"
    assert metrics[0].cache_hit_tokens == 0


@pytest.mark.asyncio
async def test_fork_child_answers_without_reading(tmp_path: Path) -> None:
    """fork 子 Agent 继承了父已读内容，无需再读，且缓存命中更高。"""

    child_events, metrics = await _run_parent("fork", tmp_path)
    reads = [
        event
        for event in child_events
        if isinstance(event, ToolExecutionEvent) and event.tool_call.name == "read_file"
    ]

    assert reads == [], "fork 子 Agent 不应该再调用 read_file"
    assert metrics[0].mode == "fork"
    assert metrics[0].cache_hit_tokens > 0
    assert (metrics[0].cache_hit_rate or 0) > 0


@pytest.mark.asyncio
async def test_fork_child_inherits_parent_messages_verbatim(tmp_path: Path) -> None:
    """fork 子 Agent 的首批消息逐字节等于父快照（缓存前缀不能被改写）。"""

    (tmp_path / NOTES).write_text(f"备注：{NEEDLE}", encoding="utf-8")
    child = ContentAwareClient(NEEDLE, NOTES, inherits_context=True)
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    manager.register_local(
        *create_spawn_agent_tool(
            tmp_path,
            lambda: child,
            lambda: "high",
            ContextBudget(10_000, 1_000, 2_000),
            force_mode="fork",
        )
    )

    await AgentLoop(ParentClient(), manager).run([Message(role="user", content="总结 notes.txt")])

    inherited = child.requests[0][:-1]
    roles = [message.role for message in child.requests[0]]
    assert roles[0] == "system"
    assert any(
        message.role == "user" and message.content == "总结 notes.txt"
        for message in inherited
    )
    assert any(message.role == "tool" and NEEDLE in message.content for message in inherited)
    # 子 Agent 自己的任务消息是最后一条，继承历史在其前面
    assert child.requests[0][-1].content == "notes.txt 里是什么？"


def test_fork_fresh_smoke_builds_fixture(tmp_path: Path) -> None:
    """微基准的测试项目能建出来，且只包含预设文件。"""

    from evaluation.fork_fresh_smoke import ANSWER_FACT, build_workspace

    workspace = build_workspace(tmp_path)
    architecture = workspace / "docs/architecture.md"
    assert architecture.is_file()
    assert ANSWER_FACT in architecture.read_text(encoding="utf-8")
    assert not (workspace / "node_modules").exists()


@pytest.mark.asyncio
async def test_parent_context_snapshot_is_cleared_after_tools(tmp_path: Path) -> None:
    """工具执行结束后父上下文快照被清理，子 Agent 不会污染父 Agent。"""

    (tmp_path / NOTES).write_text("x", encoding="utf-8")
    seen: list[tuple[Message, ...] | None] = []

    async def probe(tool_call: ToolCall):
        seen.append(parent_context_snapshot())
        return ToolResult(tool_call.call_id, "ok")

    class OneShotClient:
        def __init__(self) -> None:
            self._step = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self._step += 1
            if self._step == 1:
                yield ToolCallEvent(ToolCall("p1", "probe", {}))
                return
            yield TextDelta("完成")

    manager = ToolManager()
    manager.register_local(
        __import__("core.tools", fromlist=["ToolDefinition"]).ToolDefinition(
            name="probe",
            description="probe",
            parameters={"type": "object", "properties": {}},
            source="local",
            permission="read",
            idempotent=True,
        ),
        probe,
    )

    await AgentLoop(OneShotClient(), manager).run([Message(role="user", content="t")])

    assert seen and seen[0] is not None
    assert parent_context_snapshot() is None


@pytest.mark.asyncio
async def test_fork_child_is_told_it_may_cite_inherited_reads(tmp_path: Path) -> None:
    """fork 子 Agent 收到“可引用继承内容、但不得引用未出现内容”的边界说明。"""

    (tmp_path / NOTES).write_text(f"备注：{NEEDLE}", encoding="utf-8")
    child = ContentAwareClient(NEEDLE, NOTES, inherits_context=True)
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    manager.register_local(
        *create_spawn_agent_tool(
            tmp_path,
            lambda: child,
            lambda: "high",
            ContextBudget(10_000, 1_000, 2_000),
            force_mode="fork",
        )
    )

    await AgentLoop(ParentClient(), manager).run([Message(role="user", content="总结 notes.txt")])

    notices = [
        message
        for message in child.requests[0]
        if message.role == "system" and FORK_CONTEXT_NOTICE in message.content
    ]
    assert notices, "fork 子 Agent 必须收到可引用继承内容的说明"


@pytest.mark.asyncio
async def test_fresh_child_is_not_told_to_reuse_context(tmp_path: Path) -> None:
    """fresh 子 Agent 不该收到 fork 说明。"""

    (tmp_path / NOTES).write_text(f"备注：{NEEDLE}", encoding="utf-8")
    child = ContentAwareClient(NEEDLE, NOTES)
    manager = ToolManager()
    manager.register_local(*create_read_file_tool(tmp_path))
    manager.register_local(
        *create_spawn_agent_tool(
            tmp_path,
            lambda: child,
            lambda: "high",
            ContextBudget(10_000, 1_000, 2_000),
            force_mode="fresh",
        )
    )

    await AgentLoop(ParentClient(), manager).run([Message(role="user", content="总结 notes.txt")])

    assert not any(FORK_CONTEXT_NOTICE in message.content for message in child.requests[0])


def test_tool_call_message_always_carries_reasoning_field() -> None:
    """DeepSeek 思考模式下，历史里的工具调用消息必须带 reasoning_content（可为空串）。

    服务端对"自己签发的 id"的豁免不可靠（id 状态会过期），真机复现过 400。
    """

    from core.model import ToolCall
    from core.openai_client import _serialize_message

    with_reasoning = Message(
        role="assistant",
        content="",
        reasoning="想了一下",
        tool_calls=(ToolCall("c1", "read_file", {"path": "a"}),),
    )
    without = Message(
        role="assistant",
        content="",
        reasoning="",
        tool_calls=(ToolCall("c2", "read_file", {"path": "a"}),),
    )
    plain = Message(role="assistant", content="普通回复", reasoning="")

    assert _serialize_message(with_reasoning, include_reasoning=True)["reasoning_content"] == "想了一下"
    assert _serialize_message(without, include_reasoning=True)["reasoning_content"] == ""
    # 非工具调用、无推理的普通回复不加该字段
    assert "reasoning_content" not in _serialize_message(plain, include_reasoning=True)
    # 非 DeepSeek 端点仍然完全不加
    assert "reasoning_content" not in _serialize_message(with_reasoning, include_reasoning=False)
