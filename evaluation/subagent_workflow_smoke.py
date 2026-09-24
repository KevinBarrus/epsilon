"""运行 Scout → Worker → Reviewer 的小任务真机冒烟。"""

import argparse
import asyncio
import json
import shlex
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from core.agent_loop import AgentLoop, AgentLoopFailed, ToolExecutionEvent
from core.config import Settings, load_settings
from core.context import ContextBudget, ContextBuildResult, ContextManager
from core.model import Message, ModelClient, ToolCall
from core.openai_client import OpenAICompatibleClient
from core.project_instructions import load_project_instructions
from core.prompts import load_prompt
from core.session_store import CompactionRecord, EvictionRecord
from core.subagent import (
    SubagentRunMetrics,
    create_spawn_agent_tool,
    create_spawn_reviewer_tool,
    create_spawn_worker_tool,
    subagent_parent_message,
)
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionManager,
    ToolManager,
)
from core.tools.command_executor import CommandExecutor, HostCommandExecutor

from .online import TimedModelClient


ROLES = ("scout", "worker", "reviewer")
DELEGATION_NAMES = {
    "spawn_agent": "scout",
    "spawn_worker": "worker",
    "spawn_reviewer": "reviewer",
}


@dataclass(frozen=True)
class SubagentWorkflowResult:
    """保存委派顺序、各角色用量、审批和验证结果。"""

    model_name: str
    task_completed: bool
    parent_actual_tokens: int | None
    role_actual_tokens: dict[str, int]
    role_requests: dict[str, int]
    role_missing_usage: dict[str, int]
    total_actual_tokens: int | None
    duration_ms: float
    delegation_order: tuple[str, ...]
    delegation_counts: dict[str, int]
    role_outcomes: dict[str, dict[str, int]]
    approvals: tuple[dict[str, object], ...]
    verification_results: tuple[dict[str, object], ...]
    reviewer_context: str
    reviewer_read_paths: tuple[str, ...]
    final_content: str
    error: str | None


async def run_subagent_workflow_smoke(
    workspace: Path,
    client: ModelClient,
    settings: Settings,
    *,
    verification_command: str = "python -m unittest -v",
    command_executor: CommandExecutor | None = None,
) -> SubagentWorkflowResult:
    """在隔离工作区运行固定的小修复任务，并记录各角色实际用量。"""

    started_at = asyncio.get_running_loop().time()
    clients = {
        "parent": TimedModelClient(client),
        **{role: TimedModelClient(client) for role in ROLES},
    }
    metrics: list[SubagentRunMetrics] = []
    delegation_order: list[str] = []
    approvals: list[dict[str, object]] = []
    verification_results: list[dict[str, object]] = []
    reviewer_context = ""
    reviewer_read_paths: list[str] = []

    async def collect_event(event: object) -> None:
        """记录父级委派顺序和 Reviewer 实际验证结果。"""

        nonlocal reviewer_context
        if not isinstance(event, ToolExecutionEvent):
            return
        name = event.tool_call.name
        if event.agent_role == "parent" and name in DELEGATION_NAMES:
            delegation_order.append(name)
            if name == "spawn_reviewer":
                reviewer_context = str(event.tool_call.arguments.get("context", ""))
        if event.agent_role == "reviewer" and name == "read_file":
            reviewer_read_paths.append(str(event.tool_call.arguments.get("path", "")))
        if event.agent_role == "reviewer" and name == "run_command":
            output = event.result.content
            verification_results.append(
                {
                    "command": event.tool_call.arguments.get("command"),
                    "passed": (
                        not event.result.is_error
                        and "Ran 1 test" in output
                        and "OK" in output
                    ),
                    "output": output,
                }
            )

    async def approve_tool(definition, tool_call, allow_session):
        """只自动批准冒烟任务所需的单文件写入和固定测试命令。"""

        arguments = tool_call.arguments
        if definition.name in {"write_file", "edit_file"}:
            target = Path(str(arguments.get("path", "")))
            if not target.is_absolute():
                target = workspace / target
            allowed = target.resolve() == (workspace / "calculator.py").resolve()
        elif definition.name == "run_command":
            allowed = arguments.get("command") == verification_command
        else:
            allowed = False
        approvals.append(
            {
                "tool": definition.name,
                "approved": allowed,
                "allow_session": allow_session,
            }
        )
        return ApprovalResult(
            ApprovalDecision.ALLOW_ONCE if allowed else ApprovalDecision.DENY,
            "冒烟测试只批准 calculator.py 和固定单元测试命令",
        )

    permission_manager = PermissionManager(approve_tool)
    executor = command_executor or HostCommandExecutor()
    manager = ToolManager()
    budget = ContextBudget(
        settings.context_window or 100_000,
        settings.reserve_tokens,
        settings.keep_recent_tokens,
    )
    manager.register_local(
        *create_spawn_agent_tool(
            workspace,
            lambda: clients["scout"],
            lambda: "high",
            budget,
            metrics.append,
            on_event=collect_event,
        )
    )
    for role, create_tool in (
        ("worker", create_spawn_worker_tool),
        ("reviewer", create_spawn_reviewer_tool),
    ):
        manager.register_local(
            *create_tool(
                workspace,
                lambda role=role: clients[role],
                lambda: "high",
                budget,
                permission_manager=permission_manager,
                command_executor=executor,
                on_metrics=metrics.append,
                on_event=collect_event,
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
    context_manager.set_workspace_path(str(workspace))
    context_manager.set_model_name(settings.model_name)
    context_manager.set_project_instructions(
        load_project_instructions(workspace).content
    )
    context_manager.set_extra_system_messages([subagent_parent_message()])
    compactions: list[CompactionRecord] = []
    evictions: list[EvictionRecord] = []

    async def build_context(
        messages: Sequence[Message], force_compaction: bool
    ) -> ContextBuildResult:
        """把本次临时工作区路径和父 Agent 提示词带入每次模型请求。"""

        result = await context_manager.build_for_model_result(
            clients["parent"], messages, compactions, force_compaction, evictions
        )
        if result.compaction is not None:
            compactions.append(result.compaction)
        if result.eviction is not None:
            evictions.append(result.eviction)
        return result

    final_content = ""
    error: str | None = None

    verification_task = (
        "修改 calculator.py 中 multiply 的错误实现，让它返回两个参数的乘积。"
        "必须先用 Scout 找到实现和测试证据，再让 Worker 修改，最后让 Reviewer 独立检查并运行："
        f"{verification_command}。不要由主 Agent 直接修改文件。"
    )
    try:
        result = await AgentLoop(
            clients["parent"],
            manager,
            thinking_level="high",
            firewall_enabled=False,
        ).run(
            [Message(role="user", content=verification_task)],
            on_event=collect_event,
            build_context=build_context,
        )
        final_content = result.final_content
        if result.stop_reason != "completed":
            error = f"parent stopped: {result.stop_reason}"
    except AgentLoopFailed as exc:
        final_content = exc.model_message or exc.user_message
        error = exc.user_message

    role_actual_tokens = {
        role: sum(metric.total_tokens for metric in metrics if metric.role == role)
        for role in ROLES
    }
    role_requests = {
        role: len(clients[role].requests)
        for role in ROLES
    }
    role_missing_usage = {
        role: sum(usage is None for usage in clients[role].usages)
        for role in ROLES
    }
    role_outcomes = {
        role: dict(
            Counter(metric.outcome for metric in metrics if metric.role == role)
        )
        for role in ROLES
    }
    delegation_counts = {
        role: sum(DELEGATION_NAMES[name] == role for name in delegation_order)
        for role in ROLES
    }
    parent_tokens = clients["parent"].total_actual_tokens
    child_tokens_complete = all(
        role_missing_usage[role] == 0 for role in ROLES
    )
    total_tokens = (
        parent_tokens + sum(role_actual_tokens.values())
        if parent_tokens is not None and child_tokens_complete
        else None
    )
    verification_passed = any(
        result["passed"]
        and result["command"] == verification_command
        for result in verification_results
    )
    changed_correctly = (
        (workspace / "calculator.py").is_file()
        and "return a * b" in (workspace / "calculator.py").read_text(
            encoding="utf-8"
        )
    )
    task_completed = (
        error is None
        and changed_correctly
        and tuple(delegation_order)
        == ("spawn_agent", "spawn_worker", "spawn_reviewer")
        and all(
            role_outcomes[role].get("completed", 0) >= 1 for role in ROLES
        )
        and verification_passed
    )
    return SubagentWorkflowResult(
        model_name=settings.model_name,
        task_completed=task_completed,
        parent_actual_tokens=parent_tokens,
        role_actual_tokens=role_actual_tokens,
        role_requests=role_requests,
        role_missing_usage=role_missing_usage,
        total_actual_tokens=total_tokens,
        duration_ms=(asyncio.get_running_loop().time() - started_at) * 1000,
        delegation_order=tuple(delegation_order),
        delegation_counts=delegation_counts,
        role_outcomes=role_outcomes,
        approvals=tuple(approvals),
        verification_results=tuple(verification_results),
        reviewer_context=reviewer_context,
        reviewer_read_paths=tuple(reviewer_read_paths),
        final_content=final_content,
        error=error,
    )


def _write_result(path: Path, result: SubagentWorkflowResult) -> None:
    """将一次工作流冒烟结果写为单条 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_workspace(workspace: Path) -> None:
    """准备一个单文件乘法错误和对应单元测试。"""

    (workspace / "calculator.py").write_text(
        "def multiply(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (workspace / "test_calculator.py").write_text(
        "import unittest\nfrom calculator import multiply\n\n"
        "class CalculatorTest(unittest.TestCase):\n"
        "    def test_multiply(self):\n"
        "        self.assertEqual(multiply(3, 4), 12)\n",
        encoding="utf-8",
    )


def main() -> int:
    """确认后运行真实模型请求并输出工作流统计。"""

    parser = argparse.ArgumentParser(description="Scout → Worker → Reviewer 冒烟评测")
    parser.add_argument("--confirm", action="store_true", help="确认发起真实模型请求")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-results/subagent-workflow-smoke.jsonl"),
    )
    if not (args := parser.parse_args()).confirm:
        print("子 Agent 工作流冒烟会发起真实模型请求，请添加 --confirm 后运行")
        return 2

    settings = load_settings()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="epsilon-subagent-workflow-"
    ) as workspace_name:
        workspace = Path(workspace_name)
        _prepare_workspace(workspace)

        async def run() -> SubagentWorkflowResult:
            client = OpenAICompatibleClient(settings)
            try:
                command = f"{shlex.quote(sys.executable)} -m unittest -v"
                return await run_subagent_workflow_smoke(
                    workspace,
                    client,
                    settings,
                    verification_command=command,
                )
            finally:
                await client.close()

        result = asyncio.run(run())
    _write_result(args.output, result)
    print(
        f"completed={result.task_completed} total_tokens={result.total_actual_tokens} "
        f"parent_tokens={result.parent_actual_tokens} "
        f"roles={result.role_actual_tokens} calls={result.delegation_counts} "
        f"duration_ms={result.duration_ms:.0f}"
    )
    print(f"results: {args.output}")
    return 0 if result.task_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
