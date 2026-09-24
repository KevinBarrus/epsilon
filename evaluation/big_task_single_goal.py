"""在隔离副本运行带持续目标的单 Agent Python→TypeScript 大任务。"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from core.agent_loop import AgentLoop, AgentLoopFailed, ToolBatchEvent, ToolExecutionEvent
from core.config import load_settings
from core.context import ContextBudget, ContextBuildResult, ContextManager
from core.errors import AgentError
from core.goal import Goal, GoalPolicy, create_goal_tool
from core.model import Message, ToolCallEvent
from core.openai_client import OpenAICompatibleClient
from core.project_instructions import load_project_instructions
from core.prompts import load_prompt
from core.session_store import CompactionRecord, EvictionRecord
from core.tools import (
    ApprovalDecision, ApprovalResult, PermissionManager, ToolManager,
    create_edit_file_tool, create_list_files_tool, create_read_file_tool,
    create_run_command_tool, create_search_files_tool, create_write_file_tool,
)
from core.tools.command_executor import CommandExecution, terminate_process_group

from .online import TimedModelClient


SOURCE = Path(__file__).resolve().parents[1]
EXCLUDED = (".venv", ".git", "evaluation-results", ".epsilon", "__pycache__")
IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-11001:latest"
OBJECTIVE = "把副本 src/core 的全部 Python 模块重构成等价的 TypeScript，直到全部模块都有对应 TS 实现且类型检查通过"
TASK = (
    "你的工作区是 Epsilon 项目的副本。把 src/core/ 下的全部 Python 源代码重构成 TypeScript，"
    "保持功能等价，输出到 ts/ 目录并保持对应的模块结构。系统规划、逐步完成；"
    "可以用进度文件记录已完成和待完成模块。完成前检查全部模块映射和 TypeScript 类型检查。"
)


def source_hash() -> str:
    """对原仓库 src/、tests/ 求稳定哈希，确认评测没有改原代码。"""
    digest = hashlib.sha256()
    for root in (SOURCE / "src", SOURCE / "tests"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                digest.update(str(path.relative_to(SOURCE)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_copy() -> Path:
    """创建不含仓库元数据和运行时状态的独立副本。"""
    root = Path(tempfile.mkdtemp(prefix="epsilon-single-goal-"))
    workspace = root / "workspace"
    shutil.copytree(SOURCE, workspace, ignore=shutil.ignore_patterns(*EXCLUDED))
    if not (workspace / "src/core/agent_loop.py").is_file():
        raise RuntimeError("副本缺少 src/core/agent_loop.py")
    modules = sorted(str(path.relative_to(workspace / "src/core")) for path in (workspace / "src/core").rglob("*.py"))
    (root / "baseline.json").write_text(
        json.dumps({"source_hash": source_hash(), "source_modules": modules}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return workspace


def coverage(workspace: Path, modules: list[str]) -> dict[str, object]:
    """统计真正生成的 TS 文件及与 Python 模块的一一对应关系。"""
    ts_root = workspace / "ts"
    files: list[str] = []
    if ts_root.exists():
        for directory, child_dirs, names in os.walk(ts_root):
            child_dirs[:] = [name for name in child_dirs if name != "node_modules"]
            for name in names:
                path = Path(directory) / name
                if path.suffix in {".ts", ".tsx"} and not name.endswith(".d.ts"):
                    files.append(path.relative_to(ts_root).as_posix())
    files.sort()
    names = set(files)
    mapped: dict[str, str] = {}
    for module in modules:
        source = Path(module)
        target = source.with_name("index.ts") if source.name == "__init__.py" else source.with_suffix(".ts")
        for candidate in (f"src/{target.as_posix()}", target.as_posix()):
            if candidate in names:
                mapped[module] = candidate
                break
    return {
        "ts_files": files,
        "ts_file_count": len(files),
        "module_mapping_matched": mapped,
        "py_modules_converted": len(mapped),
        "modules_without_matching_ts": [module for module in modules if module not in mapped],
    }


def typecheck(workspace: Path) -> dict[str, object]:
    """只在副本运行无输出类型检查，不依赖模型自己的声明。"""
    ts_root = workspace / "ts"
    if not (ts_root / "tsconfig.json").is_file():
        return {"tsc_pass": None, "tsc_note": "tsconfig.json 尚未生成"}
    executable = shutil.which("tsc")
    if executable is None:
        return {"tsc_pass": None, "tsc_note": "评测环境缺少 tsc"}
    try:
        result = subprocess.run(
            [executable, "--noEmit"], cwd=ts_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"tsc_pass": False, "tsc_note": "类型检查超过 90 秒"}
    return {"tsc_pass": result.returncode == 0, "tsc_note": result.stdout[-1000:]}


def completion_verdict(status: str, mapped: int, total: int, tsc_pass: bool | None) -> str:
    """完成声明与独立的模块映射、类型检查结果分开判定。"""
    verified = mapped == total and tsc_pass is True
    if status == "complete":
        return "structurally_complete" if verified else "false_completion_claim"
    if status == "budget_limited":
        return "budget_limited_complete_structure" if verified else "budget_limited_incomplete"
    return "stopped_without_terminal_goal"


class CopyCommandExecutor:
    """将模型命令限制在只挂载副本的容器内。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("评测环境缺少 Node.js")
        self.node_root = Path(node).resolve().parent.parent

    async def execute(self, command: str, cwd: Path, timeout_seconds: float) -> CommandExecution:
        """容器只获得副本的读写挂载和 Node 工具链的只读挂载。"""
        if cwd.resolve() != self.workspace:
            raise ValueError("命令工作目录必须是实验副本")
        name = f"epsilon-goal-{uuid.uuid4().hex[:12]}"
        process = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "--name", name,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/workspace",
            "-e", "PATH=/opt/node/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin",
            "-v", f"{self.workspace}:/workspace",
            "-v", f"{self.node_root}:/opt/node:ro",
            "-w", "/workspace", "--entrypoint", "/bin/sh", IMAGE, "-lc", command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
        except (TimeoutError, asyncio.CancelledError):
            await terminate_process_group(process)
            cleanup = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await cleanup.wait()
            raise
        return CommandExecution(stdout, stderr, process.returncode)


class TokenBudgetReached(Exception):
    """总用量达到 Goal 的同一 token 预算时停止新请求。"""


async def run(workspace: Path) -> dict[str, object]:
    """运行完整任务，并在中断或失败时仍保存真实进度。"""
    baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
    settings = load_settings()
    if settings.model_name != "deepseek-flash":
        raise ValueError(f"本实验要求 deepseek-flash，当前为 {settings.model_name}")
    model = OpenAICompatibleClient(settings)
    client = TimedModelClient(model)
    goal = Goal(OBJECTIVE, max_rounds=300, token_budget=15_000_000, time_budget_seconds=3600)
    goal_events = workspace.parent / "goal.jsonl"

    def save_goal(current: Goal) -> None:
        """即时落盘预算与状态，避免长任务异常后丢失轨迹。"""
        with goal_events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(vars(current), ensure_ascii=False) + "\n")

    policy = GoalPolicy(goal, on_change=save_goal)
    save_goal(goal)

    async def approve(definition, tool_call, allow_session):
        """副本内的写入和容器命令可自动批准。"""
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    tools = ToolManager(permission_manager=PermissionManager(approve))
    for factory in (
        create_read_file_tool, create_list_files_tool, create_search_files_tool,
        create_write_file_tool, create_edit_file_tool,
    ):
        tools.register_local(*factory(workspace))
    tools.register_local(*create_run_command_tool(workspace, executor=CopyCommandExecutor(workspace)))
    tools.register_local(*create_goal_tool(lambda: policy))
    assert {item.name for item in tools.list_definitions()} == {
        "read_file", "list_files", "search_files", "write_file", "edit_file", "run_command", "goal"
    }

    budget = ContextBudget(settings.context_window or 100_000, settings.reserve_tokens, settings.keep_recent_tokens)
    context = ContextManager(
        budget, {item.name: item.capability for item in tools.list_definitions() if item.capability is not None},
        model_tools=tools.model_tools(), system_prompt=load_prompt("agent"),
    )
    context.set_model_name(settings.model_name)
    context.set_workspace_path(str(workspace))
    context.set_project_instructions(load_project_instructions(workspace).content)
    context.set_extra_system_messages([policy.instruction_message()])
    compactions: list[CompactionRecord] = []
    evictions: list[EvictionRecord] = []
    modules = baseline["source_modules"]
    started = perf_counter()
    tool_rounds = 0
    tool_calls = 0
    progress: list[dict[str, object]] = []
    sample_lock = asyncio.Lock()
    events_path = workspace.parent / "events.jsonl"
    progress_path = workspace.parent / "progress.jsonl"
    last_step: dict[str, object] | None = None

    def received_tokens() -> int:
        """所有请求已收到的 usage 下界，包含可能的上下文压缩请求。"""
        return sum(usage.total_tokens for usage in client.usages if usage is not None)

    async def sample(trigger: str) -> None:
        """按工具轮次、墙钟和最终状态采样，并立即持久化。"""
        async with sample_lock:
            state = coverage(workspace, modules)
            check = await asyncio.to_thread(typecheck, workspace)
            point = {
                "trigger": trigger,
                "tool_rounds": tool_rounds,
                "goal_rounds_started": goal.rounds_started,
                "elapsed_seconds": round(perf_counter() - started, 3),
                "ts_file_count": state["ts_file_count"],
                "py_modules_converted": state["py_modules_converted"],
                "tsc_pass": check["tsc_pass"],
                "tsc_note": check["tsc_note"],
                "actual_tokens_received": received_tokens(),
            }
            progress.append(point)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(point, ensure_ascii=False) + "\n")
            print(
                f"progress tool_round={tool_rounds} goal_round={goal.rounds_started} "
                f"mapped={state['py_modules_converted']}/{len(modules)} "
                f"ts={state['ts_file_count']} tsc={check['tsc_pass']} "
                f"tokens={received_tokens()}", flush=True,
            )

    async def periodic_sample() -> None:
        """工具事件稀疏时也每两分钟保留进度点。"""
        next_sample = asyncio.get_running_loop().time() + 120
        while True:
            await asyncio.sleep(max(0, next_sample - asyncio.get_running_loop().time()))
            await sample("120_seconds")
            next_sample += 120

    async def build_context(messages: Sequence[Message], force_compaction: bool) -> ContextBuildResult:
        """在模型请求边界执行与 Goal 相同的总 token 预算安全网。"""
        if received_tokens() >= goal.token_budget:
            raise TokenBudgetReached
        result = await context.build_for_model_result(client, messages, compactions, force_compaction, evictions)
        if result.compaction is not None:
            compactions.append(result.compaction)
        if result.eviction is not None:
            evictions.append(result.eviction)
        if received_tokens() >= goal.token_budget:
            raise TokenBudgetReached
        return result

    async def collect_event(event: object) -> None:
        """保留工具轨迹并在每 25 个工具批次采样。"""
        nonlocal tool_rounds, tool_calls, last_step
        record: dict[str, object] | None = None
        if isinstance(event, ToolCallEvent):
            tool_calls += 1
            record = {"type": "call", "tool": event.tool_call.name, "arguments": event.tool_call.arguments}
        elif isinstance(event, ToolExecutionEvent):
            record = {
                "type": "result", "tool": event.tool_call.name,
                "is_error": event.result.is_error, "result_excerpt": event.result.content[:500],
            }
        elif isinstance(event, ToolBatchEvent):
            tool_rounds += 1
            record = {
                "type": "batch", "tool_round": tool_rounds,
                "tools": [call.name for call in event.tool_calls],
                "duration_ms": event.duration_ms,
            }
        if record is not None:
            record["elapsed_seconds"] = round(perf_counter() - started, 3)
            last_step = record
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["type"] == "batch" and tool_rounds % 25 == 0:
                await sample("25_tool_rounds")

    stop_reason = "error"
    guard: str | None = None
    error: str | None = None
    error_category: str | None = None
    final_content = ""
    sampler = asyncio.create_task(periodic_sample())
    try:
        await sample("initial")
        async with asyncio.timeout(goal.time_budget_seconds):
            outcome = await AgentLoop(
                client, tools, max_tool_rounds=None, thinking_level="high", end_policy=policy,
            ).run([Message(role="user", content=TASK)], on_event=collect_event, build_context=build_context)
        stop_reason = outcome.stop_reason
        tool_rounds = outcome.tool_rounds
        final_content = outcome.final_content
    except TokenBudgetReached:
        guard = "token_budget_request_boundary"
        goal.status = "budget_limited"
        save_goal(goal)
        stop_reason = "budget_limited"
    except TimeoutError:
        guard = "time_budget_wall_clock"
        goal.status = "budget_limited"
        save_goal(goal)
        stop_reason = "budget_limited"
    except AgentError as exc:
        error_category = exc.category
        error = exc.user_message
        stop_reason = "error"
        if isinstance(exc, AgentLoopFailed):
            final_content = exc.model_message or ""
    except Exception as exc:
        error_category = type(exc).__name__
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError:
            pass
        await client.close()

    await sample("final")
    state = coverage(workspace, modules)
    check = typecheck(workspace)
    result = {
        "workspace": str(workspace), "model_name": settings.model_name, "thinking": "high",
        "delegation_tools_registered": False, "max_tool_rounds": None,
        "goal": vars(goal).copy(), "goal_final_status": goal.status,
        "continuation_count": goal.rounds_started,
        "stop_reason": stop_reason, "budget_guard": guard,
        "error_category": error_category, "error": error,
        "tool_rounds": tool_rounds, "tool_calls": tool_calls,
        "model_requests": len(client.requests), "actual_tokens_received": received_tokens(),
        "requests_missing_usage": sum(usage is None for usage in client.usages),
        "duration_seconds": round(perf_counter() - started, 3),
        "compaction_count": len(compactions), "eviction_count": len(evictions),
        "source_modules": modules, **state, **check, "progress": progress,
        "completion_verdict": completion_verdict(goal.status, state["py_modules_converted"], len(modules), check["tsc_pass"]),
        "semantic_equivalence_verified": False,
        "final_content": final_content, "last_step": last_step,
        "events_path": str(events_path), "progress_path": str(progress_path),
        "original_code_unchanged": baseline["source_hash"] == source_hash(),
    }
    return result


def main() -> int:
    """准备副本不花钱；真实模型实验必须明确 --confirm。"""
    parser = argparse.ArgumentParser(description="带 Goal 的单 Agent TS 迁移基线")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        workspace = prepare_copy()
        print(json.dumps({"workspace": str(workspace)}, ensure_ascii=False))
        return 0
    if not args.confirm or args.workspace is None:
        parser.error("真机运行需要 --workspace 和 --confirm，并须先获得费用确认")
    workspace = args.workspace.resolve()
    if not workspace.is_dir() or not workspace.parent.name.startswith("epsilon-single-goal-"):
        parser.error("workspace 必须是本脚本创建的独立副本")
    result = asyncio.run(run(workspace))
    output = workspace.parent / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "result": str(output), "goal_final_status": result["goal_final_status"],
        "completion_verdict": result["completion_verdict"],
        "actual_tokens_received": result["actual_tokens_received"],
    }, ensure_ascii=False), flush=True)
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
