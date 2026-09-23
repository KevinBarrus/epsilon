"""实现第一版只读 Scout 子 Agent。"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    ToolDefinition,
    ToolManager,
    create_list_files_tool,
    create_read_file_tool,
    create_search_files_tool,
)
from .tools.args import string_argument
from .tools.types import ToolHandler


SCOUT_MAX_CONCURRENCY = 3
SCOUT_MAX_TOOL_ROUNDS = 8
SCOUT_TIMEOUT_SECONDS = 120.0
SCOUT_SUMMARY_MAX_CHARS = 6_000
SCOUT_SYSTEM_PROMPT = load_prompt("scout")


@dataclass(frozen=True)
class ScoutRunMetrics:
    """保存一次 Scout 调用的易失运行指标。"""

    call_id: str
    outcome: Literal["completed", "timeout", "tool_limit", "error"]
    total_tokens: int
    duration_ms: float
    summary_chars: int


def create_spawn_agent_tool(
    workspace: Path,
    client_provider: Callable[[], ModelClient],
    thinking_level_provider: Callable[[], str],
    context_budget: ContextBudget,
    on_metrics: Callable[[ScoutRunMetrics], None] | None = None,
) -> tuple[ToolDefinition, ToolHandler]:
    """创建最多并行三个、只返回有界摘要的 Scout 工具。"""

    semaphore = asyncio.Semaphore(SCOUT_MAX_CONCURRENCY)
    project_instructions = load_project_instructions(workspace).content

    async def spawn_agent(tool_call: ToolCall) -> ToolResult:
        task = string_argument(tool_call, "task")
        async with semaphore:
            started_at = perf_counter()
            usages: list[UsageEvent] = []
            outcome: Literal["completed", "timeout", "tool_limit", "error"] = "error"
            content = ""
            try:
                scout_task = asyncio.create_task(
                    _run_scout(
                        task,
                        workspace,
                        client_provider(),
                        thinking_level_provider(),
                        context_budget,
                        project_instructions,
                        usages,
                    )
                )
                done, _ = await asyncio.wait(
                    (scout_task,),
                    timeout=SCOUT_TIMEOUT_SECONDS,
                )
                if not done:
                    scout_task.cancel()
                    await asyncio.gather(scout_task, return_exceptions=True)
                    outcome = "timeout"
                    content = "Scout timed out before producing a summary"
                    return ToolResult(
                        tool_call.call_id,
                        content,
                        is_error=True,
                        error_category="timeout",
                    )
                result = scout_task.result()
                if result.stop_reason == "tool_limit":
                    outcome = "tool_limit"
                    content = "Scout reached the tool round limit before producing a summary"
                    return ToolResult(
                        tool_call.call_id,
                        content,
                        is_error=True,
                        error_category="tool_execution",
                    )
                outcome = "completed"
                content = _limit_summary(result.final_content)
                return ToolResult(tool_call.call_id, content)
            except asyncio.CancelledError:
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
                        ScoutRunMetrics(
                            tool_call.call_id,
                            outcome,
                            sum(usage.total_tokens for usage in usages),
                            (perf_counter() - started_at) * 1000,
                            len(content),
                        )
                    )

    return (
        ToolDefinition(
            name="spawn_agent",
            description=(
                "Delegate a read-only codebase exploration task to an independent "
                "Scout. The Scout can read, list, and search files, then returns a "
                "bounded summary. Use multiple calls together for independent searches."
            ),
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            source="local",
            permission="read",
            idempotent=True,
            execution_mode="parallel",
            capability="agent.scout",
        ),
        spawn_agent,
    )


async def _run_scout(
    task: str,
    workspace: Path,
    client: ModelClient,
    thinking_level: str,
    context_budget: ContextBudget,
    project_instructions: str,
    usages: list[UsageEvent],
) -> AgentRunResult:
    """用独立上下文和只读工具运行一次 Scout。"""

    tool_manager = ToolManager()
    for create_tool in (
        create_read_file_tool,
        create_list_files_tool,
        create_search_files_tool,
    ):
        tool_manager.register_local(*create_tool(workspace))
    context_manager = ContextManager(
        context_budget,
        {
            definition.name: definition.capability
            for definition in tool_manager.list_definitions()
            if definition.capability is not None
        },
        model_tools=tool_manager.model_tools(),
        system_prompt=SCOUT_SYSTEM_PROMPT,
    )
    context_manager.set_project_instructions(project_instructions)
    compactions: list[CompactionRecord] = []
    evictions: list[EvictionRecord] = []

    async def build_context(
        messages: Sequence[Message],
        force_compaction: bool,
    ) -> ContextBuildResult:
        """为 Scout 维护仅存在于本次调用内的压缩记录。"""

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
        """只收集评测需要的实际 token，不保存 Scout 内部轨迹。"""

        if isinstance(event, UsageEvent):
            usages.append(event)

    return await AgentLoop(
        client,
        tool_manager,
        max_tool_rounds=SCOUT_MAX_TOOL_ROUNDS,
        thinking_level=thinking_level,
        firewall_enabled=False,
    ).run(
        [Message(role="user", content=task)],
        on_event=collect_usage,
        build_context=build_context,
    )


def _limit_summary(content: str) -> str:
    """限制写入父上下文的 Scout 摘要长度。"""

    if len(content) <= SCOUT_SUMMARY_MAX_CHARS:
        return content
    notice = "\n\n[Scout summary truncated]"
    return f"{content[: SCOUT_SUMMARY_MAX_CHARS - len(notice)]}{notice}"
