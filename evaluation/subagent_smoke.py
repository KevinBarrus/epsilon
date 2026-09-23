"""运行 Scout 开关前后的单次只读长任务冒烟。"""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from core.agent_loop import AgentLoop, ToolExecutionEvent
from core.config import Settings, load_settings
from core.context import ContextBudget, ContextBuildResult, ContextManager
from core.model import Message, ModelClient
from core.openai_client import OpenAICompatibleClient
from core.project_instructions import load_project_instructions
from core.prompts import load_prompt
from core.session_store import CompactionRecord, EvictionRecord
from core.subagent import (
    ScoutRunMetrics,
    create_spawn_agent_tool,
    scout_parent_message,
)
from core.tools import (
    ToolManager,
    create_list_files_tool,
    create_read_file_tool,
    create_search_files_tool,
)

from .online import TimedModelClient


SUBAGENT_SMOKE_PROMPT = """Read this repository and explain its runtime architecture.

Answer all four items with concrete file paths and implementation evidence:
1. where the main model-tool loop runs;
2. how model context is built and bounded;
3. how tools become visible and are executed;
4. how session messages are persisted and restored.

This is a read-only investigation. Do not modify files or run commands. If spawn_agent is
available, delegate independent parts to Scouts. Keep the final answer concise but complete.
"""
REQUIRED_EVIDENCE = (
    "agent_loop.py",
    "context.py",
    "tools/manager.py",
    "session_store.py",
)


@dataclass(frozen=True)
class SubagentSmokeResult:
    """保存一个开关档位的描述性结果。"""

    arm: str
    subagent_enabled: bool
    task_completed: bool
    parent_actual_tokens: int | None
    scout_actual_tokens: int | None
    all_agent_actual_tokens: int | None
    duration_ms: float
    scout_calls: int
    parent_scout_result_chars: int
    parent_model_requests: int
    final_content: str


async def run_subagent_smoke_arm(
    workspace: Path,
    client: ModelClient,
    settings: Settings,
    *,
    enabled: bool,
    thinking_level: str = "high",
    max_tool_rounds: int = 30,
) -> SubagentSmokeResult:
    """运行一个 on/off 档位，不写工作区且不保存 Scout 内部轨迹。"""

    started_at = perf_counter()
    parent_client = TimedModelClient(client)
    scout_client = TimedModelClient(client)
    scout_metrics: list[ScoutRunMetrics] = []
    manager = ToolManager()
    for create_tool in (
        create_read_file_tool,
        create_list_files_tool,
        create_search_files_tool,
    ):
        manager.register_local(*create_tool(workspace))
    budget = ContextBudget(
        settings.context_window or 100_000,
        settings.reserve_tokens,
        settings.keep_recent_tokens,
    )
    if enabled:
        manager.register_local(
            *create_spawn_agent_tool(
                workspace,
                lambda: scout_client,
                lambda: thinking_level,
                budget,
                scout_metrics.append,
            )
        )

    context_manager = ContextManager(
        budget,
        {
            definition.name: definition.capability
            for definition in manager.list_definitions()
            if definition.capability is not None
        },
        model_tools=manager.model_tools(),
        system_prompt=load_prompt("agent"),
    )
    context_manager.set_model_name(settings.model_name)
    context_manager.set_project_instructions(
        load_project_instructions(workspace).content
    )
    if enabled:
        context_manager.set_extra_system_messages([scout_parent_message()])
    compactions: list[CompactionRecord] = []
    evictions: list[EvictionRecord] = []

    async def build_context(
        messages: Sequence[Message],
        force_compaction: bool,
    ) -> ContextBuildResult:
        """维护仅属于本次评测档位的上下文记录。"""

        result = await context_manager.build_for_model_result(
            parent_client,
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

    parent_scout_result_chars = 0

    async def collect_event(event: object) -> None:
        """只累计注入父上下文的 Scout 结果字符数。"""

        nonlocal parent_scout_result_chars
        if (
            isinstance(event, ToolExecutionEvent)
            and event.tool_call.name == "spawn_agent"
        ):
            parent_scout_result_chars += len(event.result.content)

    result = await AgentLoop(
        parent_client,
        manager,
        max_tool_rounds=max_tool_rounds,
        thinking_level=thinking_level,
        firewall_enabled=False,
    ).run(
        [Message(role="user", content=SUBAGENT_SMOKE_PROMPT)],
        on_event=collect_event,
        build_context=build_context,
    )
    parent_tokens = parent_client.total_actual_tokens
    scout_tokens = (
        scout_client.total_actual_tokens
        if scout_client.requests
        else 0
    )
    total_tokens = (
        parent_tokens + scout_tokens
        if parent_tokens is not None and scout_tokens is not None
        else None
    )
    final_lower = result.final_content.lower()
    return SubagentSmokeResult(
        arm="on" if enabled else "off",
        subagent_enabled=enabled,
        task_completed=(
            result.stop_reason == "completed"
            and all(evidence in final_lower for evidence in REQUIRED_EVIDENCE)
        ),
        parent_actual_tokens=parent_tokens,
        scout_actual_tokens=scout_tokens,
        all_agent_actual_tokens=total_tokens,
        duration_ms=(perf_counter() - started_at) * 1000,
        scout_calls=len(scout_metrics),
        parent_scout_result_chars=parent_scout_result_chars,
        parent_model_requests=len(parent_client.requests),
        final_content=result.final_content,
    )


async def run_subagent_smoke(
    workspace: Path,
    settings: Settings,
    arms: Sequence[str],
    thinking_level: str,
    max_tool_rounds: int,
) -> list[SubagentSmokeResult]:
    """依次运行指定档位，每档使用独立模型客户端。"""

    results = []
    for arm in arms:
        client = OpenAICompatibleClient(settings)
        try:
            results.append(
                await run_subagent_smoke_arm(
                    workspace,
                    client,
                    settings,
                    enabled=arm == "on",
                    thinking_level=thinking_level,
                    max_tool_rounds=max_tool_rounds,
                )
            )
        finally:
            await client.close()
    return results


def _write_results(path: Path, results: Sequence[SubagentSmokeResult]) -> None:
    """覆盖写入本次冒烟结果，避免把不同配置误当重复样本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(asdict(result), ensure_ascii=False) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )


def main() -> int:
    """处理 Scout 冒烟命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Scout 开/关只读长任务冒烟")
    parser.add_argument("--confirm", action="store_true", help="确认发起真实模型请求")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--arm", choices=("off", "on", "both"), default="both")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--max-tool-rounds", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-results/subagent-smoke.jsonl"),
    )
    args = parser.parse_args()
    if not args.confirm:
        print("Scout 冒烟会发起真实模型请求，请添加 --confirm 后运行")
        return 2
    if args.max_tool_rounds <= 0:
        parser.error("--max-tool-rounds 必须大于 0")
    arms = ("off", "on") if args.arm == "both" else (args.arm,)
    results = asyncio.run(
        run_subagent_smoke(
            args.workspace.resolve(),
            load_settings(project_dir=args.workspace),
            arms,
            args.thinking,
            args.max_tool_rounds,
        )
    )
    _write_results(args.output, results)
    for result in results:
        print(
            f"{result.arm}: completed={result.task_completed} "
            f"parent_tokens={result.parent_actual_tokens} "
            f"all_tokens={result.all_agent_actual_tokens} "
            f"duration_ms={result.duration_ms:.0f} scouts={result.scout_calls} "
            f"summary_chars={result.parent_scout_result_chars}"
        )
    print(f"results: {args.output}")
    return 0 if all(result.task_completed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
