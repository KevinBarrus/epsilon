"""实现 Scout、Worker、Reviewer 三种职责隔离的子 Agent。"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

from .agent_loop import (
    AgentLoop,
    AgentLoopFailed,
    AgentRunResult,
    parent_context_snapshot,
)
from .context import ContextBudget, ContextBuildResult
from .context import ContextManager
from .loop_guard import LoopGuardConfig
from .model import Message, ModelClient, ToolCall, ToolResult, UsageEvent
from .project_instructions import load_project_instructions
from .prompts import load_prompt
from .session_store import CompactionRecord, EvictionRecord
from .tools import (
    PermissionManager,
    ToolDefinition,
    ToolManager,
    create_edit_file_tool,
    create_list_files_tool,
    create_read_file_tool,
    create_run_command_tool,
    create_search_files_tool,
    create_write_file_tool,
)
from .tools.args import string_argument
from .tools.command_executor import CommandExecutor
from .tools.types import ToolExecutionMode, ToolHandler
from .worktree import WorktreeError, commit_worktree, create_worktree, merge_branch, remove_worktree


# fork 子 Agent 的上下文说明：允许引用继承内容，但不得引用继承里从未出现过的内容
FORK_CONTEXT_NOTICE = (
    "你继承了父 Agent 的上下文：其中已经读过的文件内容可以直接引用，不必重新读取；"
    "但不得引用继承上下文里从未出现过的内容。"
)
SCOUT_MAX_CONCURRENCY = 3
SCOUT_SUMMARY_MAX_CHARS = 6_000
# 子 Agent 的上下文模式：fresh 空上下文；fork 继承父快照；fork_last_n 只继承最后 n 个 user 回合
SpawnMode = Literal["fresh", "fork", "fork_last_n"]
DEFAULT_FORK_TURNS = 4
ROLE_DEFAULT_MODE: dict[str, SpawnMode] = {
    "scout": "fresh",
    "worker": "fork",
    "reviewer": "fresh",
}
SPAWN_MODES: tuple[SpawnMode, ...] = ("fresh", "fork", "fork_last_n")
SCOUT_SYSTEM_PROMPT = load_prompt("scout")
SCOUT_PARENT_PROMPT = (
    "Scout delegation is enabled. Use spawn_agent for independent read-only "
    "codebase exploration that would otherwise add noisy search and file output "
    "to the main context. Give each Scout a precise task and use its returned "
    "summary as evidence, not as unverified fact. "
    "委派时主动在 context 里写下你已定位的关键路径，避免 Scout 从零探索。"
)
SUBAGENT_PARENT_PROMPT = (
    SCOUT_PARENT_PROMPT
    + " 定位使用 spawn_agent；重构或功能修改使用 spawn_worker；改完后使用"
    " spawn_reviewer 独立检查并运行验证。Worker 负责按明确任务修改代码，"
    "Reviewer 只负责检查改动和运行验证命令；写操作和命令由用户逐次审批。"
    "优先按“探索 → 修改 → 独立验证”的顺序委派。"
    "委派 spawn_reviewer 时，必须在 context 里转述 Worker 的实际改动内容"
    "（改了哪个函数、从什么改成什么）和验证命令，"
    "让 Reviewer 聚焦检查这些改动；不要让它盲目扫描整个工作区。"
    "需要复用你已读到的项目结构或已冻结的接口时，把 spawn 的 mode 设为 fork"
    "（继承你的上下文）；只是独立探索或独立验证时用 fresh（默认按角色）。"
)
WORKER_SYSTEM_PROMPT = load_prompt("worker")
REVIEWER_SYSTEM_PROMPT = load_prompt("reviewer")


@dataclass(frozen=True)
class SubagentRunMetrics:
    """保存一次子 Agent 调用的运行指标。"""

    call_id: str
    outcome: Literal["completed", "timeout", "error", "cancelled"]
    total_tokens: int
    duration_ms: float
    summary_chars: int
    context_chars: int
    role: Literal["scout", "worker", "reviewer"] = "scout"
    merge_status: str | None = None
    branch: str | None = None
    started_at: float = 0.0  # 子 Agent 运行开始，不含 worktree 准备
    finished_at: float = 0.0  # 子 Agent 运行结束，不含提交和合并
    loop_guard_injections: int = 0
    mode: SpawnMode = "fresh"  # 本次子 Agent 的上下文模式
    cache_hit_tokens: int = 0
    prompt_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float | None:
        """返回本次子 Agent 的提示缓存命中率；没有 token 时为 None。"""

        if self.prompt_tokens <= 0:
            return None
        return round(self.cache_hit_tokens / self.prompt_tokens, 4)


ScoutRunMetrics = SubagentRunMetrics


def changed_paths(summary: str) -> tuple[str, ...]:
    """从子 Agent 摘要的 CHANGED 小节提取路径清单（去重保序）。"""

    section = ROLE_SUMMARY_SECTIONS["worker"][0]
    paths: dict[str, None] = {}
    inside = False
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped in ROLE_SUMMARY_SECTIONS["worker"]:
            inside = stripped == section
            continue
        if not inside:
            continue
        candidate = stripped.lstrip("-* ").strip().strip("`")
        if candidate and candidate != MISSING_SECTION_PLACEHOLDER:
            paths.setdefault(candidate, None)
    return tuple(paths)


def unverified_changed_paths(summary: str, workspace: Path) -> tuple[str, ...]:
    """返回 CHANGED 里在工作区中不存在的路径，供父侧校验、防幻觉。"""

    missing = []
    for path in changed_paths(summary):
        candidate = (workspace / path).resolve()
        if not candidate.exists():
            missing.append(path)
    return tuple(missing)


def scout_parent_message() -> Message:
    """返回只在 Scout 开启时注入父 Agent 的运行时说明。"""

    return Message(role="system", content=SCOUT_PARENT_PROMPT)


def subagent_parent_message() -> Message:
    """返回 Scout、Worker、Reviewer 统一启用时的主 Agent 使用说明。"""

    return Message(role="system", content=SUBAGENT_PARENT_PROMPT)


def fork_history(snapshot: Sequence[Message], turns: int = 0) -> tuple[Message, ...]:
    """从父上下文快照切出 fork 子 Agent 的初始历史。

    - 去掉尾部尚未产生结果的 assistant 工具调用消息，避免把“半个回合”继承下去；
    - `turns > 0` 时只保留最后 `turns` 个 user 回合，不足则退化为全量。
    """

    end = len(snapshot)
    while end > 0:
        last = snapshot[end - 1]
        if last.role == "assistant" and last.tool_calls:
            end -= 1
            continue
        break
    history = list(snapshot[:end])
    if turns > 0:
        user_indices = [
            index for index, message in enumerate(history) if message.role == "user"
        ]
        if len(user_indices) > turns:
            history = history[user_indices[-turns] :]
    return tuple(history)


def resolve_mode(
    mode_argument: object,
    default_mode: SpawnMode,
    force_mode: SpawnMode | None,
) -> SpawnMode:
    """确定本次 spawn 的上下文模式，配置优先于模型选择。"""

    if force_mode is not None:
        return force_mode
    if isinstance(mode_argument, str) and mode_argument in SPAWN_MODES:
        return mode_argument
    return default_mode


def resolve_turns(turns_argument: object) -> int:
    """解析 fork_last_n 的回合数，非法值回落到默认值。"""

    if isinstance(turns_argument, int) and not isinstance(turns_argument, bool) and turns_argument > 0:
        return turns_argument
    return DEFAULT_FORK_TURNS


def create_spawn_agent_tool(
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    on_metrics: Callable[[SubagentRunMetrics], None] | None = None,
    timeout_seconds: float | None = None,
    on_event: Callable[[object], Awaitable[None]] | None = None,
    loop_guard_config: LoopGuardConfig | None = None,
    default_mode: SpawnMode = "fresh",
    force_mode: SpawnMode | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """创建最多并行三个、只返回有界摘要的 Scout 工具。"""

    tools = [
        create_read_file_tool(workspace),
        create_list_files_tool(workspace),
        create_search_files_tool(workspace),
    ]
    return _create_spawn_role_tool(
        "scout",
        "spawn_agent",
        "Delegate a read-only codebase exploration task to an independent Scout. "
        "在 context 参数里提供你已知的项目结构、关键文件路径和约束，减少 Scout 重复探索。",
        "parallel",
        SCOUT_SYSTEM_PROMPT,
        tools,
        workspace,
        client_provider,
        thinking_level_provider,
        context_budget,
        on_metrics=on_metrics,
        timeout_seconds=timeout_seconds,
        on_event=on_event,
        loop_guard_config=loop_guard_config,
        default_mode=default_mode,
        force_mode=force_mode,
    )


def create_spawn_worker_tool(
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    *,
    permission_manager: PermissionManager,
    command_executor: CommandExecutor | None = None,
    command_executor_factory: Callable[[Path], CommandExecutor] | None = None,
    isolation_enabled: bool = False,
    max_concurrency: int = 4,
    on_metrics: Callable[[SubagentRunMetrics], None] | None = None,
    on_event: Callable[[object], Awaitable[None]] | None = None,
    loop_guard_config: LoopGuardConfig | None = None,
    default_mode: SpawnMode = "fork",
    force_mode: SpawnMode | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """创建可修改代码、写操作仍需用户审批的串行 Worker 工具。"""

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")

    def worker_tools(path: Path) -> list[tuple[ToolDefinition, ToolHandler]]:
        """为每个隔离工作区绑定自己的文件与命令工具。"""
        executor = command_executor_factory(path) if command_executor_factory else command_executor
        return [
            create_read_file_tool(path),
            create_list_files_tool(path),
            create_search_files_tool(path),
            create_write_file_tool(path),
            create_edit_file_tool(path),
            create_run_command_tool(path, executor=executor),
        ]

    tools = worker_tools(workspace)
    return _create_spawn_role_tool(
        "worker",
        "spawn_worker",
        "Delegate a code change to an isolated parallel Worker." if isolation_enabled
        else "Delegate a code change to a Worker with workspace file and command tools.",
        "parallel" if isolation_enabled else "sequential",
        WORKER_SYSTEM_PROMPT,
        tools,
        workspace,
        client_provider,
        thinking_level_provider,
        context_budget,
        permission_manager=permission_manager,
        on_metrics=on_metrics,
        on_event=on_event,
        isolation_enabled=isolation_enabled,
        worker_tools_factory=worker_tools,
        max_concurrency=max_concurrency,
        loop_guard_config=loop_guard_config,
        default_mode=default_mode,
        force_mode=force_mode,
    )


def create_spawn_reviewer_tool(
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    *,
    permission_manager: PermissionManager,
    command_executor: CommandExecutor | None = None,
    on_metrics: Callable[[SubagentRunMetrics], None] | None = None,
    on_event: Callable[[object], Awaitable[None]] | None = None,
    loop_guard_config: LoopGuardConfig | None = None,
    default_mode: SpawnMode = "fresh",
    force_mode: SpawnMode | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """创建只读审查工具；Reviewer 执行命令仍要经过用户审批。"""

    tools = [
        create_read_file_tool(workspace),
        create_list_files_tool(workspace),
        create_search_files_tool(workspace),
        create_run_command_tool(workspace, executor=command_executor),
    ]
    return _create_spawn_role_tool(
        "reviewer",
        "spawn_reviewer",
        "Delegate a read-only code review and verification task to a Reviewer.",
        "sequential",
        REVIEWER_SYSTEM_PROMPT,
        tools,
        workspace,
        client_provider,
        thinking_level_provider,
        context_budget,
        permission_manager=permission_manager,
        on_metrics=on_metrics,
        on_event=on_event,
        loop_guard_config=loop_guard_config,
        default_mode=default_mode,
        force_mode=force_mode,
    )


def _create_spawn_role_tool(
    role: Literal["scout", "worker", "reviewer"],
    name: str,
    description: str,
    execution_mode: ToolExecutionMode,
    system_prompt: str,
    tools: Sequence[tuple[ToolDefinition, ToolHandler]],
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    *,
    permission_manager: PermissionManager | None = None,
    on_metrics: Callable[[SubagentRunMetrics], None] | None = None,
    timeout_seconds: float | None = None,
    on_event: Callable[[object], Awaitable[None]] | None = None,
    isolation_enabled: bool = False,
    worker_tools_factory: Callable[[Path], Sequence[tuple[ToolDefinition, ToolHandler]]] | None = None,
    max_concurrency: int = 4,
    loop_guard_config: LoopGuardConfig | None = None,
    default_mode: SpawnMode = "fresh",
    force_mode: SpawnMode | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """按角色组装独立 Agent；子工具集不包含任何 spawn 工具。"""

    semaphore = asyncio.Semaphore(
        SCOUT_MAX_CONCURRENCY if role == "scout"
        else max_concurrency if role == "worker" and isolation_enabled else 1
    )
    project_instructions = load_project_instructions(workspace).content

    async def spawn_role(tool_call: ToolCall) -> ToolResult:
        task = string_argument(tool_call, "task")
        context = tool_call.arguments.get("context", "")
        if not isinstance(context, str):
            raise ValueError("context must be a string")
        mode = resolve_mode(tool_call.arguments.get("mode"), default_mode, force_mode)
        turns = resolve_turns(tool_call.arguments.get("turns"))
        async with semaphore:
            started_at = perf_counter()
            usages: list[UsageEvent] = []
            outcome: Literal["completed", "error", "cancelled"] = "error"
            content = ""
            branch: str | None = None
            merge_status: str | None = None
            loop_guard_injections = 0
            history: tuple[Message, ...] = ()
            worktree_path: Path | None = None
            scout_task: asyncio.Task[AgentRunResult] | None = None
            run_started_at = 0.0
            run_finished_at = 0.0
            try:
                if mode != "fresh":
                    # fork：继承父上下文快照；fork_last_n 再按 user 回合截断
                    snapshot = parent_context_snapshot() or ()
                    history = fork_history(snapshot, turns if mode == "fork_last_n" else 0)
                run_workspace = workspace
                run_tools = tools
                if role == "worker" and isolation_enabled:
                    worktree_path = await asyncio.to_thread(create_worktree, workspace, tool_call.call_id)
                    run_workspace = worktree_path
                    assert worker_tools_factory is not None
                    run_tools = worker_tools_factory(run_workspace)
                scout_task = asyncio.create_task(
                    _run_subagent(
                        role,
                        task,
                        run_workspace,
                        client_provider(),
                        thinking_level_provider(),
                        context_budget,
                        project_instructions,
                        context,
                        usages,
                        run_tools,
                        permission_manager,
                        system_prompt,
                        on_event,
                        loop_guard_config,
                        tool_call.call_id,
                        history,
                    )
                )
                run_started_at = perf_counter()
                try:
                    if timeout_seconds is None:
                        result = await scout_task
                    else:
                        done, _ = await asyncio.wait(
                            (scout_task,), timeout=timeout_seconds
                        )
                        if not done:
                            outcome = "timeout"
                            scout_task.cancel()
                            await asyncio.gather(scout_task, return_exceptions=True)
                            content = "Scout timed out before producing a summary"
                            return ToolResult(
                                tool_call.call_id,
                                content,
                                is_error=True,
                                error_category="timeout",
                            )
                        result = scout_task.result()
                finally:
                    run_finished_at = perf_counter()
                loop_guard_injections = result.loop_guard_injections
                outcome = "completed"
                content = _limit_summary(result.final_content, role)
                if worktree_path is not None:
                    branch = await asyncio.to_thread(commit_worktree, worktree_path)
                    merge = await asyncio.to_thread(merge_branch, workspace, branch)
                    merge_status = merge.status
                    await asyncio.to_thread(remove_worktree, workspace, worktree_path)
                    if merge.status == "conflict":
                        outcome = "error"
                        return ToolResult(
                            tool_call.call_id,
                            f"merge conflict in {', '.join(merge.conflicted_paths)}; "
                            f"Worker changes preserved on branch {branch}",
                            is_error=True,
                            error_category="merge_conflict",
                        )
                return ToolResult(tool_call.call_id, content)
            except asyncio.CancelledError:
                outcome = "cancelled"
                if scout_task is not None:
                    scout_task.cancel()
                    await asyncio.gather(scout_task, return_exceptions=True)
                if worktree_path is not None:
                    try:
                        branch = await asyncio.to_thread(commit_worktree, worktree_path)
                    except WorktreeError:
                        pass  # 提交失败时保留整个工作区，不能丢失部分改动。
                raise
            except AgentLoopFailed as exc:
                content = exc.model_message or exc.user_message
                if worktree_path is not None:
                    try:
                        branch = await asyncio.to_thread(commit_worktree, worktree_path)
                        content += f"\nWorker partial changes preserved on branch {branch}"
                    except WorktreeError as save_error:
                        content += f"\nWorker worktree preserved at {worktree_path}: {save_error}"
                return ToolResult(
                    tool_call.call_id,
                    content,
                    is_error=True,
                    error_category=exc.category,
                )
            except WorktreeError as exc:
                content = str(exc)
                if worktree_path is not None:
                    content += f"; worktree preserved at {worktree_path}"
                return ToolResult(tool_call.call_id, content, is_error=True, error_category="worktree")
            except Exception as exc:
                if worktree_path is None:
                    raise
                # 预算熔断等子循环异常不能丢弃已写文件；留分支供父级恢复。
                try:
                    branch = await asyncio.to_thread(commit_worktree, worktree_path)
                    content = f"Worker stopped: {type(exc).__name__}; partial changes on branch {branch}"
                except WorktreeError as save_error:
                    content = f"Worker stopped: {type(exc).__name__}; worktree at {worktree_path}: {save_error}"
                return ToolResult(tool_call.call_id, content, is_error=True, error_category="worker")
            finally:
                if on_metrics is not None:
                    finished_at = perf_counter()
                    on_metrics(
                        SubagentRunMetrics(
                            tool_call.call_id,
                            outcome,
                            sum(usage.total_tokens for usage in usages),
                            (finished_at - started_at) * 1000,
                            len(content),
                            len(context),
                            role,
                            merge_status,
                            branch,
                            run_started_at,
                            run_finished_at,
                            loop_guard_injections,
                            mode,
                            sum((usage.cached_tokens or 0) for usage in usages),
                            sum(usage.prompt_tokens for usage in usages),
                        )
                    )

    return (
        ToolDefinition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "context": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": list(SPAWN_MODES),
                        "description": (
                            "上下文模式。fresh = 空上下文（默认，适合独立探索或独立验证）；"
                            "fork = 继承你的完整上下文（适合需要复用你已读到的项目结构、"
                            "已冻结的接口的任务）；fork_last_n = 只继承最后若干回合。"
                        ),
                    },
                    "turns": {
                        "type": "integer",
                        "description": "fork_last_n 继承的最后 user 回合数，默认 4。",
                    },
                },
                "required": ["task"],
            },
            source="local",
            permission="read",
            idempotent=True,
            execution_mode=execution_mode,
            capability=f"agent.{role}",
        ),
        spawn_role,
    )


async def _run_subagent(
    role: Literal["scout", "worker", "reviewer"],
    task: str,
    workspace: Path,
    client: ModelClient,
    thinking_level: str,
    context_budget: ContextBudget,
    project_instructions: str,
    context: str,
    usages: list[UsageEvent],
    tools: Sequence[tuple[ToolDefinition, ToolHandler]],
    permission_manager: PermissionManager | None,
    system_prompt: str,
    on_event: Callable[[object], Awaitable[None]] | None,
    loop_guard_config: LoopGuardConfig | None = None,
    run_id: str = "",
    history: Sequence[Message] = (),
) -> AgentRunResult:
    """用独立上下文和当前角色限定的工具运行一次子 Agent。"""

    tool_manager = ToolManager(permission_manager=permission_manager)
    for definition, handler in tools:
        # 只读工具沿用并行定义；写入和命令保持串行，避免同批修改相互冲突。
        tool_manager.register_local(
            replace(
                definition,
                execution_mode=(
                    definition.execution_mode
                    if definition.permission == "read"
                    else "sequential"
                ),
            ),
            handler,
        )
    context_manager = ContextManager(
        context_budget,
        {
            definition.name: definition.capability
            for definition in tool_manager.list_definitions()
            if definition.capability is not None
        },
        model_tools=tool_manager.model_tools(),
        system_prompt=system_prompt,
    )
    if history:
        # fork 子 Agent：把“可引用继承内容”的边界写进系统消息
        context_manager.set_extra_system_messages(
            [Message(role="system", content=FORK_CONTEXT_NOTICE)]
        )
    context_manager.set_workspace_path(str(workspace))
    context_manager.set_project_instructions(project_instructions)
    compactions: list[CompactionRecord] = []
    evictions: list[EvictionRecord] = []

    async def build_context(
        messages: Sequence[Message],
        force_compaction: bool,
    ) -> ContextBuildResult:
        """为本次子 Agent 维护独立的压缩记录。"""

        result = await context_manager.build_for_model_result(
            client,
            messages,
            compactions,
            force_compaction,
            evictions,
        )
        if result.compaction is not None:
            compactions.append(result.compaction)
        if result.eviction is not None:
            evictions.append(result.eviction)
        return result

    async def collect_usage(event: object) -> None:
        """收集 token 并按需把子 Agent 事件交给调用方。"""

        if isinstance(event, UsageEvent):
            usages.append(event)
        if on_event is not None:
            await on_event(event)

    return await AgentLoop(
        client,
        tool_manager,
        max_tool_rounds=None,
        thinking_level=thinking_level,
        firewall_enabled=False,
        agent_role=role,
        loop_guard_config=loop_guard_config,
        run_id=run_id,
    ).run(
        [
            *history,
            Message(
                role="user",
                content=(
                    f"任务：{task}\n\n已知背景：\n{context}"
                    if context
                    else task
                ),
            ),
        ],
        on_event=collect_usage,
        build_context=build_context,
    )


# 子 Agent 摘要的固定小节：父 Agent 只接受可核对的证据，而不是散文
ROLE_SUMMARY_SECTIONS: dict[str, tuple[str, ...]] = {
    "worker": ("## CHANGED", "## EVIDENCE", "## BLOCKED"),
    "reviewer": ("## Passed", "## Findings", "## Evidence"),
}
MISSING_SECTION_PLACEHOLDER = "（未提供）"


def _structured_summary(content: str, headings: tuple[str, ...]) -> str:
    """把子 Agent 摘要规范成固定小节；缺失的小节补占位，父 Agent 不会丢证据。"""

    sections: dict[str, list[str]] = {heading: [] for heading in headings}
    preamble: list[str] = []
    current: str | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in headings:
            current = stripped
            continue
        (preamble if current is None else sections[current]).append(line)

    lines = [line for line in preamble if line.strip()]
    for heading in headings:
        lines.append(heading)
        body = [line for line in sections[heading] if line.strip()]
        lines.extend(body or [MISSING_SECTION_PLACEHOLDER])
    return "\n".join(lines)


def _limit_summary(
    content: str,
    role: Literal["scout", "worker", "reviewer"] = "scout",
) -> str:
    """把子 Agent 摘要规范成固定小节，并在超限时优先保留必需分节。"""

    if role == "scout":
        return _limit_scout_summary(content)
    headings = ROLE_SUMMARY_SECTIONS[role]
    normalized = _structured_summary(content, headings)
    if len(normalized) <= SCOUT_SUMMARY_MAX_CHARS:
        return normalized
    return _limit_required_sections(normalized, headings)


def _limit_scout_summary(content: str) -> str:
    """限制 Scout 摘要长度，超限时优先保留关键证据和结论分节。"""

    if len(content) <= SCOUT_SUMMARY_MAX_CHARS:
        return content
    headings = ("## Files Read", "## Key Evidence", "## Conclusion")
    sections: dict[str, str] = {}
    current: str | None = None
    for line in content.splitlines():
        if line in headings:
            current = line
            sections[current] = line
        elif current is not None:
            sections[current] += f"\n{line}"
    if not all(heading in sections for heading in headings):
        notice = "\n\n[Scout summary truncated]"
        return f"{content[: SCOUT_SUMMARY_MAX_CHARS - len(notice)]}{notice}"

    notice = "\n[truncated]"
    evidence = sections["## Key Evidence"].rstrip()
    conclusion = sections["## Conclusion"].rstrip()
    files = sections["## Files Read"].rstrip()
    separator = "\n\n"
    capacity = SCOUT_SUMMARY_MAX_CHARS - len(evidence) - len(conclusion) - 2 * len(separator)
    if len(files) > capacity:
        if capacity >= len("## Files Read") + len(notice):
            files = files[: capacity - len(notice)].rsplit("\n", 1)[0].rstrip() + notice
        else:
            # 文件清单让位于证据与结论；若二者本身超限，再优先保留结论。
            priority_capacity = SCOUT_SUMMARY_MAX_CHARS - len("## Files Read\n[truncated]") - 2 * len(separator)
            conclusion_capacity = min(len(conclusion), priority_capacity)
            evidence_capacity = max(0, priority_capacity - conclusion_capacity)
            if len(evidence) > evidence_capacity:
                evidence = evidence[: max(0, evidence_capacity - len(notice))].rstrip() + notice
            if len(conclusion) > conclusion_capacity:
                conclusion = conclusion[: max(0, conclusion_capacity - len(notice))].rstrip() + notice
            files = "## Files Read\n[truncated]"
    result = separator.join((files, evidence, conclusion))
    return result[:SCOUT_SUMMARY_MAX_CHARS]


def _limit_required_sections(content: str, headings: tuple[str, ...]) -> str:
    """裁剪 Worker/Reviewer 摘要时保留全部必需分节标题。"""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        if line in headings:
            current = line
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    if len(sections) != len(headings):
        notice = "\n\n[Subagent summary truncated]"
        return f"{content[: SCOUT_SUMMARY_MAX_CHARS - len(notice)]}{notice}"

    separator = "\n\n"
    notice = "[truncated]"
    remaining = (
        SCOUT_SUMMARY_MAX_CHARS
        - sum(len(heading) for heading in headings)
        - (len(headings) - 1) * len(separator)
        - len(headings) * len(notice)
    )
    rendered = []
    for index, heading in enumerate(headings):
        body = "\n".join(sections[heading]).rstrip()
        budget = remaining // (len(headings) - index)
        if len(body) > budget:
            body = f"{body[: max(0, budget - len(notice))].rstrip()}{notice}"
        remaining -= len(body)
        rendered.append(f"{heading}\n{body}" if body else f"{heading}\n{notice}")
    return separator.join(rendered)[:SCOUT_SUMMARY_MAX_CHARS]
