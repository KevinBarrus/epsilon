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
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from core.agent_loop import AgentLoop, AgentLoopCancelled, AgentLoopFailed, RetryEvent, ToolBatchEvent, ToolExecutionEvent
from core.config import load_settings
from core.loop_guard import config_from_settings
from core.context import ContextBudget, ContextBuildResult, ContextManager
from core.errors import AgentError
from core.goal import Goal, GoalPolicy, create_goal_tool
from core.model import Message, ModelClient, TextDelta, ToolCallEvent, UsageLedger, UsageTrackingClient
from core.openai_client import OpenAICompatibleClient
from core.project_instructions import load_project_instructions
from core.prompts import load_prompt
from core.session_store import CompactionRecord, EvictionRecord
from core.subagent import (
    SubagentRunMetrics, create_spawn_agent_tool, create_spawn_worker_tool,
    create_spawn_reviewer_tool, subagent_parent_message,
)
from core.worktree import commit_worktree
from core.tools import (
    ApprovalDecision, ApprovalResult, PermissionManager, ToolManager,
    create_edit_file_tool, create_list_files_tool, create_read_file_tool,
    create_run_command_tool, create_search_files_tool, create_write_file_tool,
)
from core.tools.command_executor import CommandExecution, terminate_process_group

from .events import child_event_record
from .online import TimedModelClient


SOURCE = Path(__file__).resolve().parents[1]
EXCLUDED = (".venv", ".git", "evaluation-results", ".epsilon", "__pycache__")
IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-11001:latest"
TOKEN_FUSE = 120_000_000
TIME_FUSE_SECONDS = 10_800
OBJECTIVE = "把副本 src/core 的全部 Python 模块重构成等价的 TypeScript，直到全部模块都有对应 TS 实现且类型检查通过"
TASK = (
    "你的工作区是 Epsilon 项目的副本。把 src/core/ 下的全部 Python 源代码重构成 TypeScript，"
    "保持功能等价，输出到 ts/ 目录并保持对应的模块结构。系统规划、逐步完成；"
    "可以用进度文件记录已完成和待完成模块。完成前检查全部模块映射和 TypeScript 类型检查。"
)
DELEGATED_TASK = (
    TASK + " 请按独立模块分组，使用 spawn_worker 完成各组迁移，"
    "使用 spawn_reviewer 独立检查修改并验证；定位时可使用 spawn_agent。"
    "不要把全部迁移工作留给主 Agent。"
)


def usage_breakdown(parent: TimedModelClient, children: dict[str, list[TimedModelClient]]) -> dict[str, object]:
    """按客户端请求统计父与每次委派用量，包括压缩请求。"""
    def total(client: TimedModelClient) -> int:
        return sum(usage.total_tokens for usage in client.usages if usage is not None)
    return {"parent": total(parent), **{role: [total(client) for client in clients] for role, clients in children.items()}}


def peak_worker_concurrency(metrics: Sequence[SubagentRunMetrics]) -> int:
    """用实际子 Agent 运行区间计算并发峰值，不把排队调用算作并行。"""
    points = sorted(
        point
        for metric in metrics if metric.role == "worker" and metric.finished_at > metric.started_at
        for point in ((metric.started_at, 1), (metric.finished_at, -1))
    )
    active = peak = 0
    for _, change in points:
        active += change
        peak = max(peak, active)
    return peak


def source_hash() -> str:
    """对原仓库 src/、tests/ 求稳定哈希，确认评测没有改原代码。"""
    digest = hashlib.sha256()
    for root in (SOURCE / "src", SOURCE / "tests"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                digest.update(str(path.relative_to(SOURCE)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_copy(*, isolation_enabled: bool = False, source_core: Path | None = None) -> Path:
    """创建不含仓库元数据和运行时状态的独立副本。"""
    if isolation_enabled and source_core is None:
        raise ValueError("worktree comparison requires archived source_core")
    root = Path(tempfile.mkdtemp(prefix="epsilon-single-goal-"))
    workspace = root / "workspace"
    shutil.copytree(SOURCE, workspace, ignore=shutil.ignore_patterns(*EXCLUDED))
    if source_core is not None:
        # 对照实验从已归档的 75 模块源码恢复任务面，避免新加的 worktree.py 改变题目。
        if not (source_core / "agent_loop.py").is_file():
            raise ValueError("source_core must contain agent_loop.py")
        shutil.rmtree(workspace / "src/core")
        shutil.copytree(source_core, workspace / "src/core", ignore=shutil.ignore_patterns("__pycache__"))
    if not (workspace / "src/core/agent_loop.py").is_file():
        raise RuntimeError("副本缺少 src/core/agent_loop.py")
    modules = sorted(str(path.relative_to(workspace / "src/core")) for path in (workspace / "src/core").rglob("*.py"))
    (root / "baseline.json").write_text(
        json.dumps({"source_hash": source_hash(), "source_modules": modules,
                    "isolation_enabled": isolation_enabled,
                    "source_core": str(source_core) if source_core else None}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if isolation_enabled:
        _initialize_copy_git(workspace)
    return workspace


def _initialize_copy_git(workspace: Path) -> None:
    """仅在临时副本建 Git 基线，让后续新增 AGENTS 也能随 Worker 合并。"""
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(workspace), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

    ignore_file = workspace / ".gitignore"
    if ignore_file.is_file():
        # 原项目忽略 AGENTS；临时 Git 对照要追踪模型后续新写的模块说明。
        lines = ignore_file.read_text(encoding="utf-8").splitlines()
        ignore_file.write_text(
            "\n".join(line for line in lines if line.strip() != "AGENTS.md") + "\n",
            encoding="utf-8",
        )
    git("init")
    git("config", "user.name", "Epsilon Evaluation")
    git("config", "user.email", "evaluation@example.invalid")
    # node_modules 由模型在副本安装；不纳入 Worker 分支或基线提交。
    exclude = workspace / ".git/info/exclude"
    with exclude.open("a", encoding="utf-8") as stream:
        # npm 把缓存写到副本 .npm；不能让并行 Worker 为缓存文件互相冲突。
        stream.write("\nnode_modules/\n.npm/\n")
    git("add", "-A")
    git("commit", "-m", "baseline")


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
        for candidate in (
            f"core/{target.as_posix()}", f"src/core/{target.as_posix()}",
            f"src/{target.as_posix()}", target.as_posix(),
        ):
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
    config = next((path for path in (ts_root / "tsconfig.json", workspace / "tsconfig.json") if path.is_file()), None)
    if config is None:
        return {"tsc_pass": None, "tsc_note": "tsconfig.json 尚未生成"}
    executable = shutil.which("tsc")
    if executable is None:
        return {"tsc_pass": None, "tsc_note": "评测环境缺少 tsc"}
    try:
        result = subprocess.run(
            [executable, "--noEmit"], cwd=config.parent, text=True,
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


class BudgetedClient:
    """所有角色在发起新请求前共享同一 token 安全熔断。"""

    def __init__(self, client: ModelClient, ledger: UsageLedger, limit: int | None) -> None:
        self.client = client
        self.ledger = ledger
        self.limit = limit

    async def stream_response(self, messages, tools=(), thinking_level=None):
        """子 Agent 也受总账约束，不再等父 Agent 下一轮才检查。"""
        if self.limit is not None and self.ledger.total_tokens >= self.limit:
            raise TokenBudgetReached
        async for event in self.client.stream_response(messages, tools, thinking_level):
            yield event

    async def stream_chat(self, messages):
        """摘要压缩请求同样先检查总账。"""
        async for event in self.stream_response(messages):
            if isinstance(event, TextDelta):
                yield event.content

    async def close(self) -> None:
        """关闭底层真实网络客户端。"""
        await self.client.close()


async def run_with_wall_clock(awaitable, seconds: float):
    """把 AgentLoop 包装过的超时取消还原成可记录的墙钟熔断。"""
    async with asyncio.timeout(seconds) as deadline:
        try:
            return await awaitable
        except AgentLoopCancelled as exc:
            if deadline.expired():
                raise TimeoutError from exc
            raise


async def run(workspace: Path, *, delegate: bool = False, isolate_workers: bool = False,
              allow_token_overrun: bool = False) -> dict[str, object]:
    """运行完整任务，并在中断或失败时仍保存真实进度。"""
    baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
    settings = load_settings()
    loop_guard_config = config_from_settings(settings)
    if isolate_workers and not delegate:
        raise ValueError("worktree isolation requires delegation")
    if isolate_workers and not (workspace / ".git").exists():
        raise ValueError("worktree isolation requires a Git checkout")
    if settings.model_name != "deepseek-flash":
        raise ValueError(f"本实验要求 deepseek-flash，当前为 {settings.model_name}")
    model = OpenAICompatibleClient(settings)
    timed_client = TimedModelClient(model)
    ledger = UsageLedger()
    token_limit = None if allow_token_overrun else TOKEN_FUSE
    client = BudgetedClient(UsageTrackingClient(timed_client, ledger), ledger, token_limit)
    goal = Goal(OBJECTIVE, token_budget=token_limit, time_budget_seconds=TIME_FUSE_SECONDS)
    goal_events = workspace.parent / "goal.jsonl"

    def save_goal(current: Goal) -> None:
        """即时落盘预算与状态，避免长任务异常后丢失轨迹。"""
        with goal_events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(vars(current), ensure_ascii=False) + "\n")

    policy = GoalPolicy(goal, on_change=save_goal, usage_ledger=ledger)
    save_goal(goal)

    async def approve(definition, tool_call, allow_session):
        """副本内的写入和容器命令可自动批准。"""
        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    permissions = PermissionManager(approve)
    tools = ToolManager(permission_manager=permissions)
    for factory in (
        create_read_file_tool, create_list_files_tool, create_search_files_tool,
        create_write_file_tool, create_edit_file_tool,
    ):
        tools.register_local(*factory(workspace))
    tools.register_local(*create_run_command_tool(workspace, executor=CopyCommandExecutor(workspace)))
    tools.register_local(*create_goal_tool(lambda: policy))
    budget = ContextBudget(settings.context_window or 100_000, settings.reserve_tokens, settings.keep_recent_tokens)
    child_clients: dict[str, list[TimedModelClient]] = {role: [] for role in ("scout", "worker", "reviewer")}
    child_metrics: list[SubagentRunMetrics] = []
    child_tool_errors = 0
    child_retry_events = 0
    child_events_path = workspace.parent / "child_events.jsonl"

    child_rounds: dict[str, int] = {}

    async def collect_child_event(event: object) -> None:
        """子 Agent 轨迹单独落盘，保留可离线回放的轮次、签名与 Worker 标识。"""
        nonlocal child_tool_errors, child_retry_events
        if isinstance(event, ToolExecutionEvent):
            child_tool_errors += int(event.result.is_error)
        elif isinstance(event, RetryEvent):
            child_retry_events += 1
        run_id = getattr(event, "agent_run_id", "")
        if isinstance(event, ToolBatchEvent):
            child_rounds[run_id] = child_rounds.get(run_id, 0) + 1
        record = child_event_record(event, child_rounds.get(run_id, 0))
        if record is None:
            return
        with child_events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def child_provider(role: str):
        """每次委派独立计量，并汇入 Goal 的共享用量总账。"""
        def provide() -> BudgetedClient:
            timed = TimedModelClient(model)
            child_clients[role].append(timed)
            return BudgetedClient(UsageTrackingClient(timed, ledger), ledger, token_limit)
        return provide

    if delegate:
        command_executor = CopyCommandExecutor(workspace)
        for factory, role in (
            (create_spawn_agent_tool, "scout"),
            (create_spawn_worker_tool, "worker"),
            (create_spawn_reviewer_tool, "reviewer"),
        ):
            kwargs = {
                "on_metrics": child_metrics.append,
                "on_event": collect_child_event,
                "loop_guard_config": loop_guard_config,
            }
            if role != "scout":
                kwargs.update(permission_manager=permissions, command_executor=command_executor)
            if role == "worker" and isolate_workers:
                kwargs.update(
                    isolation_enabled=True,
                    max_concurrency=settings.worker_max_concurrency,
                    command_executor_factory=CopyCommandExecutor,
                )
            definition, handler = factory(workspace, child_provider(role), lambda: "high", budget, **kwargs)
            if role == "worker" and isolate_workers:
                async def snapshot_then_delegate(call, run_worker=handler):
                    """仅在临时副本提交父侧准备文件，确保新 worktree 看见它们。"""
                    await asyncio.to_thread(commit_worktree, workspace, "chore(eval): 固化委派前主副本状态")
                    return await run_worker(call)
                handler = snapshot_then_delegate
            tools.register_local(definition, handler)
    expected_tools = {"read_file", "list_files", "search_files", "write_file", "edit_file", "run_command", "goal"}
    if delegate:
        expected_tools.update({"spawn_agent", "spawn_worker", "spawn_reviewer"})
    assert {item.name for item in tools.list_definitions()} == expected_tools

    context = ContextManager(
        budget, {item.name: item.capability for item in tools.list_definitions() if item.capability is not None},
        model_tools=tools.model_tools(), system_prompt=load_prompt("agent"),
    )
    context.set_model_name(settings.model_name)
    context.set_workspace_path(str(workspace))
    context.set_project_instructions(load_project_instructions(workspace).content)
    context.set_extra_system_messages(
        [policy.instruction_message(), subagent_parent_message()] if delegate else [policy.instruction_message()]
    )
    compactions: list[CompactionRecord] = []
    evictions: list[EvictionRecord] = []
    modules = baseline["source_modules"]
    started = perf_counter()
    tool_rounds = 0
    tool_calls = 0
    tool_errors = 0
    retry_events = 0
    tsc_failures = 0
    progress: list[dict[str, object]] = []
    sample_lock = asyncio.Lock()
    events_path = workspace.parent / "events.jsonl"
    progress_path = workspace.parent / "progress.jsonl"
    last_step: dict[str, object] | None = None
    delegation_order: list[str] = []

    def received_tokens() -> int:
        """读取客户端边界共享总账，包含上下文压缩请求。"""
        return ledger.total_tokens

    async def sample(trigger: str) -> None:
        """按工具轮次、墙钟和最终状态采样，并立即持久化。"""
        nonlocal tsc_failures
        async with sample_lock:
            state = coverage(workspace, modules)
            check = await asyncio.to_thread(typecheck, workspace)
            if check["tsc_pass"] is False:
                tsc_failures += 1
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
                "tool_errors": tool_errors + child_tool_errors,
                "retry_events": retry_events + child_retry_events,
                "tsc_failures": tsc_failures,
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
        if token_limit is not None and received_tokens() >= token_limit:
            raise TokenBudgetReached
        result = await context.build_for_model_result(client, messages, compactions, force_compaction, evictions)
        if result.compaction is not None:
            compactions.append(result.compaction)
        if result.eviction is not None:
            evictions.append(result.eviction)
        if token_limit is not None and received_tokens() >= token_limit:
            raise TokenBudgetReached
        return result

    async def collect_event(event: object) -> None:
        """保留工具轨迹并在每 25 个工具批次采样。"""
        nonlocal tool_rounds, tool_calls, tool_errors, retry_events, last_step
        record: dict[str, object] | None = None
        if isinstance(event, ToolCallEvent):
            tool_calls += 1
            if event.tool_call.name in {"spawn_agent", "spawn_worker", "spawn_reviewer"}:
                delegation_order.append(event.tool_call.name)
            record = {"type": "call", "tool": event.tool_call.name, "arguments": event.tool_call.arguments}
        elif isinstance(event, ToolExecutionEvent):
            if event.result.is_error:
                tool_errors += 1
            record = {
                "type": "result", "tool": event.tool_call.name,
                "is_error": event.result.is_error, "result_excerpt": event.result.content[:500],
            }
        elif isinstance(event, ToolBatchEvent):
            tool_rounds += 1
            record = {
                "type": "batch", "tool_round": tool_rounds,
                "tools": [call.name for call in event.tool_calls],
                "execution_mode": event.execution_mode,
                "duration_ms": event.duration_ms,
            }
        elif isinstance(event, RetryEvent):
            retry_events += 1
            record = {
                "type": "retry", "attempt": event.attempt,
                "max_attempts": event.max_attempts,
                "delay_seconds": event.delay_seconds,
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
    parent_loop_guard_injections = 0
    sampler = asyncio.create_task(periodic_sample())
    try:
        await sample("initial")
        outcome = await run_with_wall_clock(
            AgentLoop(
                client, tools, max_tool_rounds=None, thinking_level="high", end_policy=policy,
                loop_guard_config=loop_guard_config,
            ).run([Message(role="user", content=DELEGATED_TASK if delegate else TASK)], on_event=collect_event, build_context=build_context),
            TIME_FUSE_SECONDS,
        )
        stop_reason = outcome.stop_reason
        tool_rounds = outcome.tool_rounds
        final_content = outcome.final_content
        parent_loop_guard_injections = outcome.loop_guard_injections
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
    split = usage_breakdown(timed_client, child_clients)
    client_usage_total = split["parent"] + sum(sum(split[role]) for role in child_clients)
    result = {
        "workspace": str(workspace), "model_name": settings.model_name, "thinking": "high",
        "delegation_tools_registered": delegate, "isolation_enabled": isolate_workers,
        "token_fuse": token_limit,
        "worker_max_concurrency": settings.worker_max_concurrency if isolate_workers else 1,
        "max_tool_rounds": None,
        "goal": vars(goal).copy(), "goal_final_status": goal.status,
        "continuation_count": goal.rounds_started,
        "stop_reason": stop_reason, "budget_guard": guard,
        "error_category": error_category, "error": error,
        "tool_rounds": tool_rounds, "tool_calls": tool_calls,
        "model_requests": len(timed_client.requests) + sum(
            len(child.requests) for clients in child_clients.values() for child in clients
        ),
        "parent_model_requests": len(timed_client.requests),
        "role_model_requests": {role: [len(child.requests) for child in clients]
                                for role, clients in child_clients.items()},
        "role_requests_missing_usage": {
            "parent": sum(usage is None for usage in timed_client.usages),
            **{role: [sum(usage is None for usage in child.usages) for child in clients]
               for role, clients in child_clients.items()},
        },
        "actual_tokens_received": received_tokens(),
        "client_usage_total": client_usage_total,
        "requests_missing_usage": ledger.requests_missing_usage,
        "accounting_matches": goal.tokens_used == ledger.total_tokens == client_usage_total,
        "accounting_complete": ledger.requests_missing_usage == 0,
        "tool_errors": tool_errors + child_tool_errors, "retry_events": retry_events + child_retry_events,
        "parent_tool_errors": tool_errors, "child_tool_errors": child_tool_errors,
        "role_tokens": split, "delegation_counts": dict(Counter(delegation_order)),
        "loop_guard_injections": parent_loop_guard_injections,
        "role_loop_guard_injections": {
            role: [metric.loop_guard_injections for metric in child_metrics if metric.role == role]
            for role in child_clients
        },
        "delegation_order": delegation_order,
        "subagent_runs": [vars(metric) for metric in child_metrics],
        "peak_worker_concurrency": peak_worker_concurrency(child_metrics),
        "worker_intervals": [
            {"call_id": metric.call_id,
             "start_seconds": round(metric.started_at - started, 3),
             "end_seconds": round(metric.finished_at - started, 3)}
            for metric in child_metrics if metric.role == "worker"
        ],
        "merge_conflicts": sum(metric.merge_status == "conflict" for metric in child_metrics),
        "merged_workers": sum(metric.merge_status == "merged" for metric in child_metrics),
        "child_events_path": str(child_events_path),
        "tsc_failures_at_samples": tsc_failures,
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
    parser.add_argument("--delegate", action="store_true", help="开启 Scout/Worker/Reviewer 委派对照")
    parser.add_argument("--isolate-workers", action="store_true", help="在临时 Git worktree 并行执行 Worker")
    parser.add_argument("--no-token-fuse", action="store_true", help="已获授权时只保留三小时墙钟熔断")
    parser.add_argument("--source-core", type=Path, help="使用归档的 src/core 作为对照任务源码")
    args = parser.parse_args()
    if args.prepare:
        workspace = prepare_copy(isolation_enabled=args.isolate_workers, source_core=args.source_core)
        print(json.dumps({"workspace": str(workspace)}, ensure_ascii=False))
        return 0
    if not args.confirm or args.workspace is None:
        parser.error("真机运行需要 --workspace 和 --confirm，并须先获得费用确认")
    workspace = args.workspace.resolve()
    if not workspace.is_dir() or not workspace.parent.name.startswith("epsilon-single-goal-"):
        parser.error("workspace 必须是本脚本创建的独立副本")
    baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
    if bool(baseline.get("isolation_enabled")) != args.isolate_workers:
        parser.error("运行档位必须与副本准备时的隔离档位一致")
    result = asyncio.run(run(workspace, delegate=args.delegate, isolate_workers=args.isolate_workers,
                             allow_token_overrun=args.no_token_fuse))
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
