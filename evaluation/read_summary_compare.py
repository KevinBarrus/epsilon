"""只读汇总任务的多 Agent 对比实验（oncall）。

三档，只有"是否注册委派工具"和"是否在任务里要求拆分"不同：

- `single`：不注册委派工具，中性任务描述；
- `multi_delegated`：注册只读 Scout（`spawn_agent`），任务里明确要求按主题拆给只读子 Agent；
- `multi_neutral`：注册只读 Scout，但任务描述不提委派（测模型会不会自发并行）。

实验只读原仓库、只在临时副本里写 `report.md`，原仓库零改动。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

from core.agent_loop import (
    AgentLoop,
    RetryEvent,
    ToolBatchEvent,
    ToolExecutionEvent,
)
from core.config import load_settings
from core.context import ContextBudget, ContextBuildResult, ContextManager
from core.errors import AgentError
from core.goal import CompletionGateEvent, Goal, GoalPolicy, create_goal_tool, verifier_task
from core.loop_guard import action_digest, config_from_settings
from core.model import Message, ModelClient, UsageLedger, UsageTrackingClient
from core.openai_client import OpenAICompatibleClient
from core.prompts import load_prompt
from core.subagent import (
    SubagentRunMetrics,
    create_spawn_agent_tool,
    run_readonly_audit,
    scout_parent_message,
)
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionManager,
    ToolManager,
    create_edit_file_tool,
    create_list_files_tool,
    create_read_file_tool,
    create_search_files_tool,
    create_write_file_tool,
)

from .big_task_single_goal import (
    BudgetedClient,
    TokenBudgetReached,
    run_with_wall_clock,
    usage_breakdown,
)
from .events import child_event_record, event_to_record
from .online import TimedModelClient
from .read_summary_score import score

SOURCE = Path("/home/kevinbarrus/projects/oncall")
_TEST_FILE_PATTERN = re.compile(r"tests?\.rs$")


def is_test_code(path: Path) -> bool:
    """判断路径是否测试代码：目录名为 test/tests，或文件名以 tests.rs / _test.rs 结尾。"""

    if any(part in {"test", "tests"} for part in path.parts):
        return True
    return bool(_TEST_FILE_PATTERN.search(path.name))
EXCLUDED_DIRS = {".git", "node_modules", ".venv", "__pycache__"}
MAX_FILE_BYTES = 1_000_000
TOKEN_FUSE = 40_000_000
TIME_FUSE_SECONDS = 3_600
REPORT_NAME = "report.md"
# 三档：A 单 Agent（串行读）/ B fresh 扇出（并行读）/ C 先测绘再 fork（延续式探索）
ARMS = ("single", "fanout_fresh", "map_then_fork")
DEFAULT_SCOUT_MODE: dict[str, str | None] = {
    "single": None,
    "fanout_fresh": "fresh",
    "map_then_fork": "fork",
}
SCOUT_MODES = ("fresh", "fork", "fork_last_n")

OBJECTIVE = "通读工作区里的 oncall 项目，产出覆盖关键设计要点的 report.md 总结报告"
# 除项目级总结外的源码级问题；三档完全相同，答案只在源码里
SOURCE_QUESTIONS = (
    "\n\n除项目级总结外，报告还必须回答下面的源码级问题（答案不在文档里，必须读源码）；"
    "无法确定时明确写“未能确定”，不许编造。每项都要写出对应的文件路径与关键标识符：\n"
    "1. BM25 那一路的中文分词是怎么实现的？\n"
    "2. 信念压缩（SOP 信念更新）的阈值规则具体是什么？\n"
    "3. 会话记忆有哪几档？写出它们的字面枚举值，并说明压缩发生在哪一层。\n"
    "4. Planner / Executor / Replanner / Report 分别实现在哪里？是一个文件还是多个文件？"
    "关键状态字段有哪些？\n"
    "5. 工具证据压缩相关的表是做什么用的？有哪些关键字段？\n"
    "6. 索引任务状态机有哪些字面状态值？失败后如何重试或重建？\n"
)
TASK = (
    "你的工作区是 oncall 项目的副本。请通读它的文档、规格与后端/前端源码，把项目讲清楚，"
    f"写一份中文总结报告到工作区根目录的 {REPORT_NAME}。"
    "报告要让不了解项目的人看懂：它解决什么问题、整体架构与主要模块、各条关键链路怎么走、"
    "有哪些设计取舍与已知局限。提到模块或文件时请写出真实路径。"
    + SOURCE_QUESTIONS
)
DELEGATION_SENTENCE = (
    " 请把阅读工作按主题/目录拆给若干只读子 Agent 并行完成，再由你汇总成报告。"
)
# C 档必须先把项目图建起来，否则 fork 出来的子 Agent 继承不到任何东西
MAP_FIRST_SENTENCE = (
    " 先自己快速测绘项目结构（主要目录、关键模块及它们的关系），"
    "再把测绘结果与你正在阅读的分区交给若干只读子 Agent，"
    "让它们在你已有上下文的基础上深读各自负责的部分。"
)
# 每档在共同任务之外追加的唯一一句话；A 档不加
_ARM_SENTENCE: dict[str, str] = {
    "fanout_fresh": DELEGATION_SENTENCE,
    "map_then_fork": MAP_FIRST_SENTENCE,
}


def repository_hash(
    root: Path,
    max_file_bytes: int = MAX_FILE_BYTES,
    excluded_dirs: set[str] | None = None,
    skip_tests: bool = False,
    include: Callable[[Path], bool] | None = None,
) -> str:
    """对仓库（按排除规则）求稳定哈希，用于确认原仓库零改动。"""

    excluded = EXCLUDED_DIRS if excluded_dirs is None else excluded_dirs
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        relative = path.relative_to(root)
        if skip_tests and is_test_code(relative):
            continue
        if include is not None and not include(relative):
            continue
        if path.stat().st_size > max_file_bytes:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def copy_repository(
    source: Path,
    target: Path,
    max_file_bytes: int = MAX_FILE_BYTES,
    excluded_dirs: set[str] | None = None,
    skip_tests: bool = False,
    include: Callable[[Path], bool] | None = None,
) -> dict[str, int]:
    """按排除规则复制仓库：跳过元数据目录、测试代码与超过阈值的大文件。"""

    excluded = EXCLUDED_DIRS if excluded_dirs is None else excluded_dirs
    files = 0
    total_bytes = 0
    skipped_large = 0
    for root, dirs, names in os.walk(source):
        dirs[:] = sorted(name for name in dirs if name not in excluded)
        if skip_tests:
            dirs[:] = [name for name in dirs if name not in {"test", "tests"}]
        relative_root = Path(root).relative_to(source)
        (target / relative_root).mkdir(parents=True, exist_ok=True)
        for name in sorted(names):
            origin = Path(root) / name
            try:
                size = origin.stat().st_size
            except OSError:
                continue
            if size > max_file_bytes:
                skipped_large += 1
                continue
            relative = relative_root / name
            if skip_tests and is_test_code(relative):
                continue
            if include is not None and not include(relative):
                continue
            shutil.copy2(origin, target / relative_root / name)
            files += 1
            total_bytes += size
    return {"files": files, "bytes": total_bytes, "skipped_large": skipped_large}


def prepare(arm: str, source: Path) -> Path:
    """创建一份内容一致的只读副本，并记录基线哈希。"""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    root = Path(tempfile.mkdtemp(prefix=f"epsilon-read-{arm}-"))
    workspace = root / "workspace"
    stats = copy_repository(source, workspace)
    (root / "baseline.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "source": str(source),
                "source_hash": repository_hash(source),
                "copy_hash": repository_hash(workspace),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return workspace


def peak_concurrency(metrics: Sequence[SubagentRunMetrics]) -> int:
    """用实际运行区间的重叠计算子 Agent 并发峰值。"""

    points = sorted(
        point
        for metric in metrics
        if metric.finished_at > metric.started_at
        for point in ((metric.started_at, 1), (metric.finished_at, -1))
    )
    active = peak = 0
    for _, change in points:
        active += change
        peak = max(peak, active)
    return peak


def read_duplication(
    parent_events: list[dict[str, object]],
    child_events: list[dict[str, object]],
) -> dict[str, object]:
    """统计"重复读率"：多个 Agent 是否读了同一份内容。"""

    per_agent: dict[str, list[str]] = {}
    for record in parent_events:
        if record.get("type") == "tool_started" and record.get("name") == "read_file":
            arguments = record.get("arguments")
            if isinstance(arguments, dict):
                per_agent.setdefault("parent", []).append(action_digest("read_file", arguments))
    for record in child_events:
        if record.get("type") != "batch":
            continue
        run_id = str(record.get("agent_run_id", ""))
        for call in record.get("calls", []):  # type: ignore[union-attr]
            if call.get("tool") == "read_file":
                per_agent.setdefault(run_id, []).append(str(call.get("args_digest", "")))
    total = sum(len(digests) for digests in per_agent.values())
    unique = len({digest for digests in per_agent.values() for digest in digests})
    return {
        "total_reads": total,
        "unique_reads": unique,
        "duplication_rate": round(1 - unique / total, 4) if total else None,
        "agents_with_reads": len(per_agent),
    }


def scout_summary(
    metrics: Sequence[SubagentRunMetrics],
    child_events: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """按上下文模式汇总子 Agent 的 token、缓存命中与读调用次数。"""

    reads: dict[str, int] = {}
    for record in child_events:
        if record.get("type") != "batch":
            continue
        run_id = str(record.get("agent_run_id", ""))
        reads[run_id] = reads.get(run_id, 0) + sum(
            1 for call in record.get("calls", []) if call.get("tool") == "read_file"  # type: ignore[union-attr]
        )

    summary: dict[str, dict[str, object]] = {}
    for metric in metrics:
        bucket = summary.setdefault(
            metric.mode,
            {
                "runs": 0,
                "tokens": 0,
                "prompt_tokens": 0,
                "cache_hit_tokens": 0,
                "read_calls": 0,
            },
        )
        bucket["runs"] = int(bucket["runs"]) + 1
        bucket["tokens"] = int(bucket["tokens"]) + metric.total_tokens
        bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + metric.prompt_tokens
        bucket["cache_hit_tokens"] = int(bucket["cache_hit_tokens"]) + metric.cache_hit_tokens
        bucket["read_calls"] = int(bucket["read_calls"]) + reads.get(metric.call_id, 0)
    for bucket in summary.values():
        prompt_tokens = int(bucket["prompt_tokens"])
        bucket["cache_hit_rate"] = (
            round(int(bucket["cache_hit_tokens"]) / prompt_tokens, 4)
            if prompt_tokens
            else None
        )
    return summary


async def run(
    workspace: Path,
    arm: str,
    scout_mode: str | None = None,
    *,
    source: Path = SOURCE,
    objective: str = OBJECTIVE,
    task_text: str = TASK,
    arm_sentence: dict[str, str] | None = None,
    score_fn: Callable[[str, Path], dict[str, object]] = score,
    token_fuse: int = TOKEN_FUSE,
    time_fuse_seconds: float = TIME_FUSE_SECONDS,
    report_name: str = REPORT_NAME,
    source_hash_fn: Callable[[Path], str] | None = None,
    criteria: str = "",
) -> dict[str, object]:
    """运行一档任务，并在中断或失败时仍保存真实进度。

    任务/评分/熔断都由调用方通过 profile 参数传入，便于复用同一套 harness。
    """

    sentences = arm_sentence if arm_sentence is not None else _ARM_SENTENCE
    # 校验"原仓库零改动"必须用与 baseline 完全相同的范围口径，否则比的是两个哈希
    if source_hash_fn is not None:
        hash_source = source_hash_fn
    else:
        hash_source = lambda root: repository_hash(root, MAX_FILE_BYTES, EXCLUDED_DIRS)

    baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
    settings = load_settings()
    loop_guard_config = config_from_settings(settings)
    model = OpenAICompatibleClient(settings)
    timed_client = TimedModelClient(model)
    ledger = UsageLedger()
    client = BudgetedClient(UsageTrackingClient(timed_client, ledger), ledger, token_fuse)
    # 验收标准由调用方提供（**必须是泛化的，绝不点名任何隐藏清单条目**）
    goal = Goal(
        objective,
        acceptance_criteria=criteria,
        token_budget=token_fuse,
        time_budget_seconds=time_fuse_seconds,
    )
    goal_events = workspace.parent / "goal.jsonl"

    def save_goal(current: Goal) -> None:
        """即时落盘预算与状态，便于中断后复盘。"""

        with goal_events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(vars(current), ensure_ascii=False) + "\n")

    async def approve(definition, tool_call, allow_session):
        """副本内的写入自动放行，避免评测被审批卡住。"""

        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    permissions = PermissionManager(approve)
    tools = ToolManager(permission_manager=permissions)
    for create_tool in (
        create_read_file_tool,
        create_list_files_tool,
        create_search_files_tool,
        create_write_file_tool,
        create_edit_file_tool,
    ):
        tools.register_local(*create_tool(workspace))
    tools.register_local(*create_goal_tool(lambda: policy))

    budget = ContextBudget(
        settings.context_window or 100_000,
        settings.reserve_tokens,
        settings.keep_recent_tokens,
    )
    child_clients: dict[str, list[TimedModelClient]] = {"scout": []}
    child_metrics: list[SubagentRunMetrics] = []
    child_tool_errors = 0
    parent_events: list[dict[str, object]] = []
    child_events: list[dict[str, object]] = []
    gate_events: list[CompletionGateEvent] = []
    child_rounds: dict[str, int] = {}
    events_path = workspace.parent / "events.jsonl"
    child_events_path = workspace.parent / "child_events.jsonl"

    async def collect_event(event: object) -> None:
        """父 Agent 事件落盘。"""

        if isinstance(event, CompletionGateEvent):
            gate_events.append(event)
        record = event_to_record(event)
        parent_events.append(record)
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def collect_child_event(event: object) -> None:
        """Scout 事件单独落盘，保留可回放的轮次与签名。"""

        nonlocal child_tool_errors
        if isinstance(event, ToolExecutionEvent):
            child_tool_errors += int(event.result.is_error)
        elif isinstance(event, RetryEvent):
            pass
        run_id = getattr(event, "agent_run_id", "")
        if isinstance(event, ToolBatchEvent):
            child_rounds[run_id] = child_rounds.get(run_id, 0) + 1
        record = child_event_record(event, child_rounds.get(run_id, 0))
        if record is None:
            return
        child_events.append(record)
        with child_events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def verify_completion(current: Goal) -> str:
        """独立、只读的完成审计；预算耗尽或不可用一律 inconclusive。"""

        verifier_client = BudgetedClient(
            UsageTrackingClient(TimedModelClient(model), ledger),
            ledger,
            settings.completion_gate_verifier_token_budget,
        )
        try:
            return await run_readonly_audit(
                verifier_task(current),
                workspace,
                verifier_client,
                settings.completion_gate_verifier_thinking,
                budget,
                timeout_seconds=settings.completion_gate_verifier_timeout_seconds,
                on_event=collect_child_event,
                loop_guard_config=loop_guard_config,
            )
        except TokenBudgetReached:
            return "VERDICT: inconclusive\nUNMET: 验证器 token 预算耗尽"

    policy = GoalPolicy(
        goal,
        on_change=save_goal,
        usage_ledger=ledger,
        verifier=verify_completion if settings.completion_gate_enabled else None,
        max_rejections=settings.completion_gate_max_rejections,
        verifier_timeout_seconds=settings.completion_gate_verifier_timeout_seconds,
        no_tool_nudge_rounds=settings.completion_gate_no_tool_nudge_rounds,
        on_event=collect_event,
    )
    save_goal(goal)


    def child_provider() -> ModelClient:
        """每次委派独立计量，并汇入共享用量总账。"""

        timed = TimedModelClient(model)
        child_clients["scout"].append(timed)
        return BudgetedClient(UsageTrackingClient(timed, ledger), ledger, token_fuse)

    delegation_enabled = arm != "single"
    resolved_mode = scout_mode or DEFAULT_SCOUT_MODE[arm]
    if delegation_enabled:
        assert resolved_mode in SCOUT_MODES
        tools.register_local(
            *create_spawn_agent_tool(
                workspace,
                child_provider,
                lambda: "high",
                budget,
                on_metrics=child_metrics.append,
                on_event=collect_child_event,
                loop_guard_config=loop_guard_config,
                force_mode=resolved_mode,  # type: ignore[arg-type]
            )
        )

    context = ContextManager(
        budget,
        {
            definition.name: definition.capability
            for definition in tools.list_definitions()
            if definition.capability is not None
        },
        model_tools=tools.model_tools(),
        system_prompt=load_prompt("agent"),
    )
    context.set_model_name(settings.model_name)
    context.set_workspace_path(str(workspace))
    extra_system = [policy.instruction_message()]
    if delegation_enabled:
        extra_system.append(scout_parent_message())
    context.set_extra_system_messages(extra_system)
    compactions: list[object] = []
    evictions: list[object] = []

    def received_tokens() -> int:
        """返回本轮已收到的 token 总量。"""

        return ledger.total_tokens

    async def build_context(messages, force_compaction) -> ContextBuildResult:
        """在请求边界执行与 Goal 相同的总 token 预算安全网。"""

        if received_tokens() >= token_fuse:
            raise TokenBudgetReached
        result = await context.build_for_model_result(
            client, messages, compactions, force_compaction, evictions
        )
        if result.compaction is not None:
            compactions.append(result.compaction)
        if result.eviction is not None:
            evictions.append(result.eviction)
        return result

    task = task_text + sentences.get(arm, "")
    agent = AgentLoop(
        client,
        tools,
        max_tool_rounds=None,
        thinking_level="high",
        end_policy=policy,
        loop_guard_config=loop_guard_config,
    )

    started = perf_counter()
    stop_reason = "error"
    guard: str | None = None
    error: str | None = None
    error_category: str | None = None
    final_content = ""
    outcome = None
    try:
        outcome = await run_with_wall_clock(
            agent.run(
                [Message(role="user", content=task)],
                on_event=collect_event,
                build_context=build_context,
            ),
            time_fuse_seconds,
        )
        stop_reason = outcome.stop_reason
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
    except Exception as exc:  # noqa: BLE001 - 评测必须把任何异常落盘
        error_category = type(exc).__name__
        error = f"{type(exc).__name__}: {exc}"
    duration = perf_counter() - started
    await client.close()

    report_path = workspace / report_name
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    split = usage_breakdown(timed_client, child_clients)
    client_usage_total = split["parent"] + sum(sum(split[role]) for role in child_clients)
    return {
        "arm": arm,
        "workspace": str(workspace),
        "model_name": settings.model_name,
        "thinking": "high",
        "delegation_tools_registered": delegation_enabled,
        "scout_mode": resolved_mode,
        "task_sentence": sentences.get(arm, ""),
        "token_fuse": token_fuse,
        "time_fuse_seconds": time_fuse_seconds,
        "goal": vars(goal).copy(),
        "goal_final_status": goal.status,
        "continuation_count": goal.rounds_started,
        "stop_reason": stop_reason,
        "budget_guard": guard,
        "error": error,
        "error_category": error_category,
        "tool_rounds": outcome.tool_rounds if outcome is not None else 0,
        "model_requests": len(timed_client.requests)
        + sum(len(child.requests) for clients in child_clients.values() for child in clients),
        "parent_model_requests": len(timed_client.requests),
        "role_tokens": split,
        "actual_tokens_received": ledger.total_tokens,
        "client_usage_total": client_usage_total,
        "accounting_matches": goal.tokens_used == ledger.total_tokens == client_usage_total,
        "tool_errors": sum(1 for record in parent_events if record.get("is_error"))
        + child_tool_errors,
        "child_tool_errors": child_tool_errors,
        "delegation_counts": {"spawn_agent": len(child_metrics)},
        "subagent_runs": [vars(metric) for metric in child_metrics],
        "peak_scout_concurrency": peak_concurrency(child_metrics),
        "loop_guard_injections": outcome.loop_guard_injections if outcome is not None else 0,
        "role_loop_guard_injections": {
            "scout": [metric.loop_guard_injections for metric in child_metrics]
        },
        "scout_summary": scout_summary(child_metrics, child_events),
        "completion_gate": {
            "enabled": settings.completion_gate_enabled,
            "rejections": goal.completion_rejections,
            "verified": goal.verified,
            "rejected_not_met": sum(1 for e in gate_events if e.reason == "not_met"),
            "rejected_inconclusive": sum(
                1 for e in gate_events if e.reason == "inconclusive"
            ),
            "rejected_verifier_error": sum(
                1 for e in gate_events if e.reason == "verifier_error"
            ),
            "attempts": [
                {
                    "attempt": e.attempt,
                    "verdict": e.verdict,
                    "reason": e.reason,
                    "accepted": e.accepted,
                    "verified": e.verified,
                    "unmet": list(e.unmet),
                }
                for e in gate_events
            ],
        },
        "duration_seconds": round(duration, 3),
        "compaction_count": len(compactions),
        "eviction_count": len(evictions),
        "report_path": str(report_path),
        "report_chars": len(report_text),
        "quality": score_fn(report_text, workspace) if report_text else None,
        "read_duplication": read_duplication(parent_events, child_events),
        "original_code_unchanged": baseline["source_hash"] == hash_source(source),
        "events_path": str(events_path),
        "final_content": final_content,
    }


def main() -> int:
    """准备副本不花钱；真机运行必须显式 --confirm。"""

    parser = argparse.ArgumentParser(description="只读汇总任务的多 Agent 对比实验")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--scout-mode",
        choices=SCOUT_MODES,
        help="覆盖该档默认的子 Agent 上下文模式（默认：fanout_fresh=fresh / map_then_fork=fork）",
    )
    args = parser.parse_args()

    if args.prepare:
        if not args.arm:
            parser.error("--prepare 需要 --arm")
        workspace = prepare(args.arm, args.source)
        print(json.dumps({"workspace": str(workspace)}, ensure_ascii=False))
        return 0
    if not args.confirm or args.workspace is None or not args.arm:
        parser.error("真机运行需要 --arm、--workspace 与 --confirm")
    workspace = args.workspace.resolve()
    if not workspace.is_dir() or not workspace.parent.name.startswith("epsilon-read-"):
        parser.error("workspace 必须是本脚本创建的独立副本")
    baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
    if baseline.get("arm") != args.arm:
        parser.error("运行档位必须与副本准备时的档位一致")
    result = asyncio.run(run(workspace, args.arm, args.scout_mode))
    output = workspace.parent / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(output),
                "stop_reason": result["stop_reason"],
                "actual_tokens_received": result["actual_tokens_received"],
                "duration_seconds": result["duration_seconds"],
                "quality": result["quality"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
