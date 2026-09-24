"""实现 Scout、Worker、Reviewer 三种职责隔离的子 Agent。"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

from .agent_loop import AgentLoop, AgentLoopFailed, AgentRunResult
from .context import ContextBudget, ContextBuildResult
from .context import ContextManager
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


SCOUT_MAX_CONCURRENCY = 3
SCOUT_SUMMARY_MAX_CHARS = 6_000
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


ScoutRunMetrics = SubagentRunMetrics


def scout_parent_message() -> Message:
    """返回只在 Scout 开启时注入父 Agent 的运行时说明。"""

    return Message(role="system", content=SCOUT_PARENT_PROMPT)


def subagent_parent_message() -> Message:
    """返回 Scout、Worker、Reviewer 统一启用时的主 Agent 使用说明。"""

    return Message(role="system", content=SUBAGENT_PARENT_PROMPT)


def create_spawn_agent_tool(
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    on_metrics: Callable[[SubagentRunMetrics], None] | None = None,
    timeout_seconds: float | None = None,
    on_event: Callable[[object], Awaitable[None]] | None = None,
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
    )


def create_spawn_worker_tool(
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    *,
    permission_manager: PermissionManager,
    command_executor: CommandExecutor | None = None,
    on_metrics: Callable[[SubagentRunMetrics], None] | None = None,
    on_event: Callable[[object], Awaitable[None]] | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """创建可修改代码、写操作仍需用户审批的串行 Worker 工具。"""

    tools = [
        create_read_file_tool(workspace),
        create_list_files_tool(workspace),
        create_search_files_tool(workspace),
        create_write_file_tool(workspace),
        create_edit_file_tool(workspace),
        create_run_command_tool(workspace, executor=command_executor),
    ]
    return _create_spawn_role_tool(
        "worker",
        "spawn_worker",
        "Delegate a code change to a Worker with workspace file and command tools.",
        "sequential",
        WORKER_SYSTEM_PROMPT,
        tools,
        workspace,
        client_provider,
        thinking_level_provider,
        context_budget,
        permission_manager=permission_manager,
        on_metrics=on_metrics,
        on_event=on_event,
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
) -> tuple[ToolDefinition, ToolHandler]:
    """按角色组装独立 Agent；子工具集不包含任何 spawn 工具。"""

    semaphore = asyncio.Semaphore(SCOUT_MAX_CONCURRENCY if role == "scout" else 1)
    project_instructions = load_project_instructions(workspace).content

    async def spawn_role(tool_call: ToolCall) -> ToolResult:
        task = string_argument(tool_call, "task")
        context = tool_call.arguments.get("context", "")
        if not isinstance(context, str):
            raise ValueError("context must be a string")
        async with semaphore:
            started_at = perf_counter()
            usages: list[UsageEvent] = []
            outcome: Literal["completed", "error", "cancelled"] = "error"
            content = ""
            try:
                scout_task = asyncio.create_task(
                    _run_subagent(
                        role,
                        task,
                        workspace,
                        client_provider(),
                        thinking_level_provider(),
                        context_budget,
                        project_instructions,
                        context,
                        usages,
                        tools,
                        permission_manager,
                        system_prompt,
                        on_event,
                    )
                )
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
                outcome = "completed"
                content = _limit_summary(result.final_content, role)
                return ToolResult(tool_call.call_id, content)
            except asyncio.CancelledError:
                outcome = "cancelled"
                scout_task.cancel()
                await asyncio.gather(scout_task, return_exceptions=True)
                raise
            except AgentLoopFailed as exc:
                content = exc.model_message or exc.user_message
                return ToolResult(
                    tool_call.call_id,
                    content,
                    is_error=True,
                    error_category=exc.category,
                )
            finally:
                if on_metrics is not None:
                    on_metrics(
                        SubagentRunMetrics(
                            tool_call.call_id,
                            outcome,
                            sum(usage.total_tokens for usage in usages),
                            (perf_counter() - started_at) * 1000,
                            len(content),
                            len(context),
                            role,
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
    ).run(
        [
            Message(
                role="user",
                content=(
                    f"任务：{task}\n\n已知背景：\n{context}"
                    if context
                    else task
                ),
            )
        ],
        on_event=collect_usage,
        build_context=build_context,
    )


def _limit_summary(
    content: str,
    role: Literal["scout", "worker", "reviewer"] = "scout",
) -> str:
    """限制摘要长度，超限时优先保留关键证据和结论分节。"""

    if len(content) <= SCOUT_SUMMARY_MAX_CHARS:
        return content
    if role != "scout":
        headings = {
            "worker": ("## Changes", "## Verification", "## Result"),
            "reviewer": ("## Passed", "## Findings", "## Evidence"),
        }[role]
        return _limit_required_sections(content, headings)

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
