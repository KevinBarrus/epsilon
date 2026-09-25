import asyncio
import subprocess
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import pytest

import core.subagent as subagent
from core.agent_loop import AgentLoop, ToolBatchEvent
from core.context import ContextBudget
from core.loop_guard import LoopGuardConfig
from core.model import Message, TextDelta, ToolCall, ToolCallEvent, UsageEvent
from core.session import Session
from core.subagent import (
    SCOUT_SUMMARY_MAX_CHARS,
    SCOUT_SYSTEM_PROMPT,
    SUBAGENT_PARENT_PROMPT,
    create_spawn_agent_tool,
    create_spawn_reviewer_tool,
    create_spawn_worker_tool,
    _limit_summary,
)
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionManager,
    ToolManager,
)
from core.tools.command_executor import CommandExecution


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


class RoleClient:
    """执行一次可选工具调用，再返回预设的角色摘要。"""

    def __init__(
        self,
        call_batches: tuple[tuple[ToolCall, ...], ...],
        summary: str,
    ) -> None:
        self.call_batches = call_batches
        self.summary = summary
        self.requests: list[list[Message]] = []
        self.tools: list[list[dict[str, object]]] = []

    async def stream_response(self, messages, tools=(), thinking_level=None):
        self.requests.append(list(messages))
        self.tools.append(list(tools))
        request_index = len(self.requests) - 1
        if request_index < len(self.call_batches):
            for call in self.call_batches[request_index]:
                yield ToolCallEvent(call)
        else:
            yield TextDelta(self.summary)


class FakeCommandExecutor:
    """记录命令并返回成功结果，避免测试执行真实 shell 命令。"""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def execute(self, command, cwd, timeout_seconds):
        self.commands.append(command)
        return CommandExecution(b"checks passed", b"", 0)


def test_parent_prompt_requires_worker_to_reviewer_handoff() -> None:
    """父 Agent 必须把 Worker 修改范围和验证命令交给 Reviewer。"""

    assert "spawn_reviewer" in SUBAGENT_PARENT_PROMPT
    assert "context" in SUBAGENT_PARENT_PROMPT
    assert "Worker 的实际改动内容" in SUBAGENT_PARENT_PROMPT
    assert "从什么改成什么" in SUBAGENT_PARENT_PROMPT
    assert "验证命令" in SUBAGENT_PARENT_PROMPT


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
    assert metrics[0].context_chars == 0


@pytest.mark.asyncio
async def test_spawn_agent_detects_its_own_loop_and_reports_injections(tmp_path) -> None:
    """子 Agent 自己抳到空转，并把纠偏次数写进指标。"""

    (tmp_path / "target.txt").write_text("证据", encoding="utf-8")
    repeats = tuple(
        (ToolCall(f"read-{index}", "read_file", {"path": "target.txt"}),)
        for index in range(3)
    )
    client = RoleClient(repeats, "结论：仍是同一份证据")
    metrics = []
    _, handler = create_spawn_agent_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        metrics.append,
    )

    result = await handler(ToolCall("spawn-loop", "spawn_agent", {"task": "重复读取"}))

    assert result.is_error is False
    assert metrics[0].loop_guard_injections == 1


@pytest.mark.asyncio
async def test_spawn_agent_honors_disabled_loop_guard(tmp_path) -> None:
    """显式关闭时空转不再注入纠偏。"""

    (tmp_path / "target.txt").write_text("证据", encoding="utf-8")
    repeats = tuple(
        (ToolCall(f"read-{index}", "read_file", {"path": "target.txt"}),)
        for index in range(4)
    )
    client = RoleClient(repeats, "结论：仍是同一份证据")
    metrics = []
    _, handler = create_spawn_agent_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        metrics.append,
        loop_guard_config=LoopGuardConfig(enabled=False),
    )

    result = await handler(ToolCall("spawn-off", "spawn_agent", {"task": "重复读取"}))

    assert result.is_error is False
    assert metrics[0].loop_guard_injections == 0


@pytest.mark.asyncio
async def test_spawn_agent_passes_optional_context_to_scout(tmp_path) -> None:
    """测试可选 context 会进入 Scout 首条用户消息。"""

    client = ScoutClient()
    metrics = []
    definition, handler = create_spawn_agent_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        metrics.append,
    )

    assert "context" in definition.parameters["properties"]
    assert "context" not in definition.parameters["required"]
    assert "已知的项目结构" in definition.description

    await handler(
        ToolCall(
            "spawn-context",
            "spawn_agent",
            {"task": "检查配置", "context": "配置入口在 src/core/config.py"},
        )
    )

    assert client.requests[0][-1] == Message(
        "user",
        "任务：检查配置\n\n已知背景：\n配置入口在 src/core/config.py",
    )
    assert client.requests[0][-1].content.count("src/core/config.py") == 1
    assert metrics[0].context_chars == len("配置入口在 src/core/config.py")


@pytest.mark.asyncio
async def test_spawn_agent_rejects_non_string_context(tmp_path) -> None:
    """测试可选 context 存在时必须是字符串。"""

    _, handler = create_spawn_agent_tool(
        tmp_path,
        ScoutClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    with pytest.raises(ValueError, match="context must be a string"):
        await handler(
            ToolCall("spawn-invalid", "spawn_agent", {"task": "检查", "context": 3})
        )


def test_scout_prompt_requires_structured_sections() -> None:
    """测试 Scout 提示词要求固定的结构化分节。"""

    assert "## Files Read" in SCOUT_SYSTEM_PROMPT
    assert "## Key Evidence" in SCOUT_SYSTEM_PROMPT
    assert "## Conclusion" in SCOUT_SYSTEM_PROMPT


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
async def test_spawn_agent_truncates_files_before_evidence_and_conclusion(
    tmp_path,
) -> None:
    """测试超长结构化摘要先裁 Files Read 并保留证据与结论。"""

    summary = (
        "## Files Read\n" + "- path.py\n" * 900
        + "## Key Evidence\n- key-evidence-needle\n"
        + "## Conclusion\n- conclusion-needle"
    )

    class LongStructuredSummaryClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield TextDelta(summary)

    _, handler = create_spawn_agent_tool(
        tmp_path,
        LongStructuredSummaryClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))

    assert len(result.content) <= SCOUT_SUMMARY_MAX_CHARS
    assert "## Key Evidence" in result.content
    assert "key-evidence-needle" in result.content
    assert "## Conclusion" in result.content
    assert "conclusion-needle" in result.content
    assert "[truncated]" in result.content


@pytest.mark.asyncio
async def test_spawn_agent_times_out_with_structured_error(tmp_path, monkeypatch) -> None:
    """测试 Scout 超时后返回可恢复的结构化错误。"""

    class SlowClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            await asyncio.sleep(1)
            yield TextDelta("不会返回")

    _, handler = create_spawn_agent_tool(
        tmp_path,
        SlowClient,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        timeout_seconds=0.01,
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))

    assert result.is_error is True
    assert result.error_category == "timeout"


@pytest.mark.asyncio
async def test_spawn_agent_can_exceed_eight_tool_rounds(tmp_path) -> None:
    """测试 Scout 不会在固定的第八轮工具调用处被截断。"""

    class LoopingClient:
        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.requests += 1
            if self.requests <= 9:
                yield ToolCallEvent(
                    ToolCall(f"list-{self.requests}", "list_files", {"path": "."})
                )
            else:
                yield TextDelta("Scout completed after nine tool rounds")

    client = LoopingClient()
    _, handler = create_spawn_agent_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
    )

    result = await handler(ToolCall("spawn-1", "spawn_agent", {"task": "读取"}))

    assert result.is_error is False
    assert "nine tool rounds" in result.content
    assert client.requests == 10


@pytest.mark.asyncio
async def test_spawn_worker_parallel_reads_sequential_write_with_approval(tmp_path) -> None:
    """测试 Worker 只读批次并行，写入批次串行且复用父级审批。"""

    approvals = []

    async def approve(definition, tool_call, allow_session):
        approvals.append((definition.name, allow_session))
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    client = RoleClient(
        (
            (
                ToolCall("list-1", "list_files", {"path": "."}),
                ToolCall("search-1", "search_files", {"pattern": "result"}),
            ),
            (
                ToolCall(
                    "write-1",
                    "write_file",
                    {"path": "result.txt", "content": "done"},
                ),
            ),
        ),
        "## Changes\n- result.txt\n## Verification\n- not run\n## Result\n- done",
    )
    events = []

    async def collect(event):
        events.append(event)

    definition, handler = create_spawn_worker_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        permission_manager=PermissionManager(approve),
        isolation_enabled=False,
        on_event=collect,
    )

    result = await handler(ToolCall("worker-1", "spawn_worker", {"task": "修改文件"}))

    names = [tool["function"]["name"] for tool in client.tools[0]]  # type: ignore[index]
    assert definition.execution_mode == "sequential"
    assert names == [
        "read_file", "list_files", "search_files", "write_file", "edit_file", "run_command"
    ]
    assert all(not name.startswith("spawn_") for name in names)
    assert approvals == [("write_file", True)]
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "done"
    child_batches = [event for event in events if isinstance(event, ToolBatchEvent)]
    assert [batch.execution_mode for batch in child_batches] == ["parallel", "sequential"]
    assert len(child_batches[0].tool_calls) == 2
    assert [call.name for call in child_batches[1].tool_calls] == ["write_file"]
    assert "result.txt" in result.content and "Verification" in result.content
    assert any("修改文件" in message.content for message in client.requests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_isolated_workers_run_concurrently_and_merge(tmp_path: Path, conflict: bool, monkeypatch) -> None:
    """并发 Worker 的独立写入可合并，冲突时保留失败分支。"""
    repo = tmp_path / "main"
    repo.mkdir()
    for args in (
        ("init",),
        ("config", "user.name", "Worker Test"),
        ("config", "user.email", "worker@example.invalid"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    (repo / "base.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "baseline"], check=True, capture_output=True)

    started = 0
    simultaneous = asyncio.Event()
    roots: list[str] = []
    command_cwds: list[Path] = []

    class ParallelWorkerClient:
        """两次首请求都到达后才允许写入，以验证并发而非串行。"""

        def __init__(self) -> None:
            self.requests = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            nonlocal started
            self.requests += 1
            if self.requests == 1:
                roots.append(next(message.content for message in messages if "Current workspace root:" in message.content))
                started += 1
                if started == 2:
                    simultaneous.set()
                await asyncio.wait_for(simultaneous.wait(), timeout=3)
                task = messages[-1].content
                filename = "shared.txt" if conflict else f"{task}.txt"
                yield ToolCallEvent(ToolCall(task, "write_file", {"path": filename, "content": task}))
            elif self.requests == 2:
                yield ToolCallEvent(ToolCall("verify", "run_command", {"command": "verify"}))
            else:
                yield TextDelta("## Changes\n- file written\n## Verification\n- checked\n## Result\n- done")

    class PathRecordingExecutor:
        async def execute(self, command, cwd, timeout_seconds):
            command_cwds.append(cwd)
            return CommandExecution(b"ok", b"", 0)

    async def approve(definition, tool_call, allow_session):
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    merge_started: dict[str, float] = {}
    original_merge = subagent.merge_branch

    def observed_merge(root: Path, branch: str):
        merge_started[branch] = perf_counter()
        return original_merge(root, branch)

    monkeypatch.setattr(subagent, "merge_branch", observed_merge)
    metrics = []
    definition, handler = create_spawn_worker_tool(
        repo, ParallelWorkerClient, lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        permission_manager=PermissionManager(approve),
        isolation_enabled=True,
        command_executor_factory=lambda path: PathRecordingExecutor(),
        on_metrics=metrics.append,
    )
    results = await asyncio.gather(
        handler(ToolCall("one", "spawn_worker", {"task": "one"})),
        handler(ToolCall("two", "spawn_worker", {"task": "two"})),
    )

    assert definition.execution_mode == "parallel"
    assert started == 2
    assert len(set(roots)) == 2
    assert len(set(command_cwds)) == 2
    assert all(path != repo for path in command_cwds)
    assert max(metric.started_at for metric in metrics) < min(metric.finished_at for metric in metrics)
    assert all(metric.finished_at <= merge_started[metric.branch] for metric in metrics)
    if conflict:
        failed = [result for result in results if result.is_error]
        assert len(failed) == 1
        assert failed[0].error_category == "merge_conflict"
        assert "shared.txt" in failed[0].content
        rejected_branch = next(metric.branch for metric in metrics if metric.merge_status == "conflict")
        assert rejected_branch is not None
        assert subprocess.run(["git", "-C", str(repo), "show", f"{rejected_branch}:shared.txt"],
                              check=True, capture_output=True).stdout.strip() in {b"one", b"two"}
    else:
        assert all(not result.is_error for result in results)
        assert (repo / "one.txt").read_text(encoding="utf-8") == "one"
        assert (repo / "two.txt").read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_isolated_worker_reports_non_git_workspace(tmp_path: Path) -> None:
    """开启隔离但没有 Git 检出时明确失败，不回退共享写入。"""
    _, handler = create_spawn_worker_tool(
        tmp_path, lambda: RoleClient((), "done"), lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        permission_manager=PermissionManager(), isolation_enabled=True,
    )
    result = await handler(ToolCall("worker", "spawn_worker", {"task": "change"}))
    assert result.is_error
    assert result.error_category == "worktree"
    assert "Git checkout" in result.content


@pytest.mark.asyncio
async def test_spawn_reviewer_is_sequential_and_approves_verification_command(
    tmp_path,
) -> None:
    """测试 Reviewer 只有只读工具与命令，运行命令仍经父级审批。"""

    approvals = []

    async def approve(definition, tool_call, allow_session):
        approvals.append((definition.name, allow_session))
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    (tmp_path / "target.txt").write_text("检查内容", encoding="utf-8")
    executor = FakeCommandExecutor()
    client = RoleClient(
        (
            (
                ToolCall("read-1", "read_file", {"path": "target.txt"}),
                ToolCall("list-1", "list_files", {"path": "."}),
            ),
            (ToolCall("command-1", "run_command", {"command": "pytest tests/test_x.py"}),),
        ),
        "## Passed\n- tests passed\n## Findings\n- none\n## Evidence\n- exit code 0",
    )
    events = []

    async def collect(event):
        events.append(event)

    definition, handler = create_spawn_reviewer_tool(
        tmp_path,
        lambda: client,
        lambda: "high",
        ContextBudget(10_000, 1_000, 2_000),
        permission_manager=PermissionManager(approve),
        command_executor=executor,
        on_event=collect,
    )

    result = await handler(
        ToolCall("reviewer-1", "spawn_reviewer", {"task": "验证修改"})
    )

    names = [tool["function"]["name"] for tool in client.tools[0]]  # type: ignore[index]
    assert definition.execution_mode == "sequential"
    assert names == ["read_file", "list_files", "search_files", "run_command"]
    assert all(not name.startswith("spawn_") for name in names)
    assert approvals == [("run_command", False)]
    assert executor.commands == ["pytest tests/test_x.py"]
    child_batches = [event for event in events if isinstance(event, ToolBatchEvent)]
    assert [batch.execution_mode for batch in child_batches] == ["parallel", "sequential"]
    assert "Passed" in result.content and "Findings" in result.content
    assert any("只运行验证相关命令" in message.content for message in client.requests[0])


@pytest.mark.asyncio
async def test_scout_worker_and_reviewer_have_separate_initial_contexts(
    tmp_path,
) -> None:
    """测试三个角色每次都从各自任务开始，不共享此前消息。"""

    client = RoleClient((), "完成")
    budget = ContextBudget(10_000, 1_000, 2_000)
    manager = PermissionManager()
    factories = (
        ("Scout 专属任务", create_spawn_agent_tool),
        ("Worker 专属任务", create_spawn_worker_tool),
        ("Reviewer 专属任务", create_spawn_reviewer_tool),
    )
    for task, factory in factories:
        kwargs = {"permission_manager": manager} if factory is not create_spawn_agent_tool else {}
        _, handler = factory(
            tmp_path,
            lambda: client,
            lambda: "high",
            budget,
            **kwargs,
        )
        await handler(ToolCall(f"call-{task}", factory.__name__, {"task": task}))

    for request, (task, _) in zip(client.requests, factories):
        user_messages = [message.content for message in request if message.role == "user"]
        system_messages = [message.content for message in request if message.role == "system"]
        assert user_messages == [task]
        assert any(f"Current workspace root: `{tmp_path}`" in content for content in system_messages)
        assert all(other_task not in "\n".join(user_messages) for other_task, _ in factories if other_task != task)


@pytest.mark.parametrize(
    ("role", "sections"),
    [
        ("worker", ("Changes", "Verification", "Result")),
        ("reviewer", ("Passed", "Findings", "Evidence")),
    ],
)
def test_role_summary_truncation_keeps_required_sections(role, sections) -> None:
    """测试 Worker 与 Reviewer 的超长摘要仍保留各自必需分节。"""

    first, second, third = sections
    content = (
        f"## {first}\n- required-{first.lower()}\n- " + "x" * 8_000
        + f"\n## {second}\n- required-{second.lower()}\n"
        + f"## {third}\n- required-{third.lower()}"
    )

    result = _limit_summary(content, role)

    assert len(result) <= SCOUT_SUMMARY_MAX_CHARS
    for section in sections:
        assert f"## {section}" in result
        assert f"required-{section.lower()}" in result
    assert "[truncated]" in result


@pytest.mark.asyncio
async def test_spawn_agent_cancels_running_scout_with_parent(tmp_path) -> None:
    """测试父调用取消时同步取消仍在运行的 Scout。"""

    cancelled = asyncio.Event()
    started = asyncio.Event()

    class SlowClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            started.set()
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
    await asyncio.wait_for(started.wait(), timeout=1)

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


@pytest.mark.asyncio
async def test_parent_session_persists_scout_task_and_summary(tmp_path) -> None:
    """测试父会话只落盘 Scout 任务与最终摘要，恢复后调用链完整。"""

    class ParentAndScoutClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            if any(
                message.role == "system" and "你是 Scout" in message.content
                for message in messages
            ):
                yield TextDelta("结论：入口位于 src/core/ui.py")
                return
            if not any(message.role == "tool" for message in messages):
                yield ToolCallEvent(
                    ToolCall("spawn-1", "spawn_agent", {"task": "定位程序入口"})
                )
                return
            yield TextDelta("已根据 Scout 摘要完成定位")

    client = ParentAndScoutClient()
    manager = ToolManager()
    manager.register_local(
        *create_spawn_agent_tool(
            tmp_path,
            lambda: client,
            lambda: "high",
            ContextBudget(10_000, 1_000, 2_000),
        )
    )
    result = await AgentLoop(client, manager).run(
        [Message("user", "入口在哪里？")]
    )
    session = Session(tmp_path)
    session.add_user_message("入口在哪里？")
    for message in result.new_messages:
        session.add_message(message)
    assert session.flush_persistence()
    session_id = session.session_id
    session.close()

    restored = Session.restore(tmp_path, session_id)

    assert restored.get_messages()[1].tool_calls == (
        ToolCall("spawn-1", "spawn_agent", {"task": "定位程序入口"}),
    )
    assert restored.get_messages()[2] == Message(
        "tool",
        "结论：入口位于 src/core/ui.py",
        tool_call_id="spawn-1",
    )
    assert all("read_file" not in message.content for message in restored.get_messages())
    restored.close()
