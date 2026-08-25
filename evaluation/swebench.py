"""运行隔离的 SWE-bench 真实代码修复任务。"""

import argparse
import asyncio
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from core.agent_loop import AgentLoop
from core.config import load_settings
from core.context import ContextBudget, ContextManager, DEFAULT_CONTEXT_BUDGET, estimate_context_tokens
from core.model import Message, ModelClientError
from core.openai_client import OpenAICompatibleClient
from core.prompts import load_prompt
from core.session import Session
from core.tools import (
    ApprovalDecision,
    ApprovalResult,
    PermissionManager,
    ToolManager,
    create_edit_file_tool,
    create_list_files_tool,
    create_read_file_tool,
    create_run_command_tool,
    create_search_files_tool,
    create_write_file_tool,
)

from .events import event_to_record, message_to_record
from .models import EvaluationAssertion, EvaluationResult
from .online import TimedModelClient
from .report import generate_report
from .storage import append_result
from .swebench_workspace import EvaluationWorkspaceError, prepare_evaluation_workspace


DATASETS = {
    "swebench-lite": "SWE-bench/SWE-bench_Lite",
    "swebench-full": "SWE-bench/SWE-bench",
}
EVALUATION_COMMAND_TIMEOUT_SECONDS = 300.0
DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS = 40


@dataclass(frozen=True)
class SwebenchTask:
    """保存运行真实任务所需的公开元数据，不保存参考补丁。"""

    instance_id: str
    repository: str
    base_commit: str
    issue: str
    source: str


@dataclass(frozen=True)
class HarnessResult:
    """保存官方 Harness 对单条补丁的验证结论。"""

    passed: bool
    environment_error: str | None = None


def load_task(instance_id: str, source: str) -> SwebenchTask:
    """从公开数据集读取任务描述，不向 Agent 暴露参考补丁或测试补丁。"""

    dataset_name = _dataset_name(source)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("缺少 datasets，请使用 SWE-bench 评测环境运行") from exc
    dataset = load_dataset(dataset_name, split="test")
    for record in dataset:
        if record["instance_id"] == instance_id:
            return SwebenchTask(
                instance_id=instance_id,
                repository=str(record["repo"]),
                base_commit=str(record["base_commit"]),
                issue=str(record["problem_statement"]),
                source=source,
            )
    raise ValueError(f"数据集不存在任务：{instance_id}")


def load_reference_patch(instance_id: str, source: str) -> str:
    """仅为预检读取参考补丁，正常 Agent 运行绝不调用此函数。"""

    dataset_name = _dataset_name(source)
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="test")
    for record in dataset:
        if record["instance_id"] == instance_id:
            return str(record["patch"])
    raise ValueError(f"数据集不存在任务：{instance_id}")


def prepare_repository(task: SwebenchTask, cache_root: Path) -> Path:
    """获取基线提交所在仓库，仓库历史只保存在 Agent 工作区外。"""

    repository = cache_root / task.repository.replace("/", "__")
    if not repository.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--no-checkout", f"https://github.com/{task.repository}.git", str(repository)])
    if _has_commit(repository, task.base_commit):
        return repository
    _run_git(["-C", str(repository), "fetch", "--depth", "1", "origin", task.base_commit])
    _run_git(["-C", str(repository), "cat-file", "-e", f"{task.base_commit}^{{commit}}"])
    return repository


async def run_task(
    task: SwebenchTask,
    result_root: Path,
    harness_python: str,
    compact: bool = False,
    max_tool_rounds: int | None = None,
) -> EvaluationResult:
    """在无 Git 历史工作区中执行 Agent，并使用官方 Harness 验证补丁。"""

    started_at = perf_counter()
    events: list[dict[str, object]] = []
    client: TimedModelClient | None = None
    stage = "repository"
    changed_files: tuple[str, ...] = ()
    try:
        repository = await asyncio.to_thread(
            prepare_repository, task, result_root / "repositories"
        )
        stage = "workspace"
        baseline = prepare_evaluation_workspace(repository, task.base_commit, result_root)
        prepared = prepare_evaluation_workspace(repository, task.base_commit, result_root)
        stage = "agent-loop"
        settings = load_settings()
        client = TimedModelClient(OpenAICompatibleClient(settings))
        manager = _tool_manager(prepared.workspace)
        session = Session(prepared.session_root)
        effective_tool_rounds = (
            max_tool_rounds
            if max_tool_rounds is not None
            else DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS
        )
        prompt = _agent_prompt(task)
        session.add_user_message(prompt)
        events.append(message_to_record(Message(role="user", content=prompt)))
        events.append({"type": "configuration", "max_tool_rounds": effective_tool_rounds})

        async def collect_event(event: object) -> None:
            """保存完整模型和工具轨迹，供结果报告复核。"""

            events.append(event_to_record(event))

        agent = AgentLoop(
            client,
            manager,
            max_tool_rounds=effective_tool_rounds,
        )
        context_builder = _context_builder(
            session,
            client,
            compact,
            events,
            settings.model_name,
            manager,
        )
        result = await agent.run(
            session.get_messages(),
            on_event=collect_event,
            build_context=context_builder,
        )
        events.append(
            {
                "type": "agent_end",
                "stop_reason": result.stop_reason,
                "tool_rounds": result.tool_rounds,
            }
        )
        for message in result.new_messages:
            session.add_message(message)
        events.append(message_to_record(Message(role="assistant", content=result.final_content)))
        persistence_ok = session.flush_persistence() and session.close()
        changed_files, patch = await asyncio.to_thread(
            create_patch, baseline.workspace, prepared.workspace
        )
        stage = "official-verification"
        verification = await asyncio.to_thread(
            verify_patch,
            task,
            patch,
            result_root / "harness",
            harness_python,
        )
        assertions = (
            EvaluationAssertion("official-harness", verification.passed, verification.environment_error or "官方 Harness 未通过"),
            EvaluationAssertion("patch-created", bool(patch), "Agent 没有生成代码补丁"),
            EvaluationAssertion("persistence", persistence_ok, "评测 Session 持久化失败"),
            EvaluationAssertion("workspace-isolated", not (prepared.workspace / ".git").exists(), "Agent 工作区包含 Git 历史"),
            EvaluationAssertion(
                "compaction-triggered",
                not compact or any(event.get("type") == "compaction" for event in events),
                "压缩专项没有实际触发上下文压缩",
            ),
        )
        return _result(
            task,
            started_at,
            client,
            events,
            assertions,
            changed_files,
            compact,
            not persistence_ok,
            verification.environment_error,
            tool_rounds=result.tool_rounds,
            stop_reason=result.stop_reason,
        )
    except Exception as exc:
        if isinstance(exc, ModelClientError):
            events.append(_model_error_record(exc))
        error_category = (
            "model"
            if isinstance(exc, ModelClientError)
            else "environment"
            if isinstance(exc, EvaluationWorkspaceError)
            else "evaluation"
        )
        return _result(
            task,
            started_at,
            client,
            events,
            (EvaluationAssertion("evaluation-error", False, f"{type(exc).__name__}: {exc}"),),
            changed_files,
            compact,
            False,
            f"{stage}: {type(exc).__name__}: {exc}",
            error_category,
        )
    finally:
        # 显式关闭 HTTP 客户端，避免事件循环关闭后 httpx 后台任务报错
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                await close()


async def precheck_task(
    task: SwebenchTask,
    result_root: Path,
    harness_python: str,
) -> EvaluationResult:
    """用参考补丁验证任务环境；该补丁不会进入 Agent 工作区或上下文。"""

    started_at = perf_counter()
    try:
        repository = await asyncio.to_thread(
            prepare_repository, task, result_root / "repositories"
        )
        prepared = prepare_evaluation_workspace(repository, task.base_commit, result_root)
        reference_patch = load_reference_patch(task.instance_id, task.source)
        verification = await asyncio.to_thread(
            verify_patch,
            task,
            reference_patch,
            result_root / "precheck-harness",
            harness_python,
        )
        if not verification.passed and verification.environment_error is None:
            verification = HarnessResult(False, "参考补丁未通过官方 Harness")
        return EvaluationResult(
            scenario=f"precheck:{task.instance_id}",
            task_id=task.instance_id,
            source=task.source,
            evaluation_group="precheck",
            base_commit=task.base_commit,
            duration_ms=(perf_counter() - started_at) * 1000,
            evaluation_type="real-task",
            error_category="environment" if verification.environment_error else None,
            error_message=verification.environment_error,
            assertions=(
                EvaluationAssertion("workspace-isolated", not (prepared.workspace / ".git").exists(), "预检工作区包含 Git 历史"),
                EvaluationAssertion("gold-patch", verification.passed, verification.environment_error or "参考补丁未通过官方 Harness"),
            ),
        )
    except Exception as exc:
        return EvaluationResult(
            scenario=f"precheck:{task.instance_id}",
            task_id=task.instance_id,
            source=task.source,
            evaluation_group="precheck",
            base_commit=task.base_commit,
            duration_ms=(perf_counter() - started_at) * 1000,
            evaluation_type="real-task",
            error_category="environment",
            error_message=f"{type(exc).__name__}: {exc}",
            assertions=(EvaluationAssertion("precheck", False, f"{type(exc).__name__}: {exc}"),),
        )


def create_patch(baseline: Path, workspace: Path) -> tuple[tuple[str, ...], str]:
    """比较两个无 Git 源码快照，生成可由官方 Harness 应用的补丁。"""

    baseline = baseline.resolve()
    workspace = workspace.resolve()
    _remove_runtime_artifacts(workspace)
    result = subprocess.run(
        ["git", "diff", "--no-index", "--src-prefix=a/", "--dst-prefix=b/", str(baseline), str(workspace)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise EvaluationWorkspaceError(result.stderr.strip() or "无法生成评测补丁")
    patch = _normalise_patch_paths(result.stdout, baseline, workspace)
    return _changed_files(patch), patch


def _remove_runtime_artifacts(workspace: Path) -> None:
    """移除 Python 执行产生的缓存，避免它们被误判为 Agent 补丁。"""

    for path in workspace.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)


def verify_patch(
    task: SwebenchTask,
    patch: str,
    result_root: Path,
    harness_python: str,
) -> HarnessResult:
    """调用官方 Harness 验证补丁，并区分任务失败与评测环境失败。"""

    result_root = result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    prediction_path = result_root / f"{task.instance_id}.jsonl"
    prediction_path.write_text(
        json.dumps(
            {"instance_id": task.instance_id, "model_name_or_path": "epsilon", "model_patch": patch}
        ) + "\n",
        encoding="utf-8",
    )
    run_id = task.instance_id.replace("__", "-")
    command = [
        harness_python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        _dataset_name(task.source),
        "--predictions_path",
        str(prediction_path),
        "--run_id",
        run_id,
    ]
    completed = subprocess.run(command, cwd=result_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (result_root / f"{task.instance_id}.harness.log").write_text(completed.stdout, encoding="utf-8")
    report_path = result_root / f"epsilon.{run_id}.json"
    if completed.returncode != 0 and not report_path.is_file():
        return HarnessResult(False, f"官方 Harness 执行失败，退出码 {completed.returncode}")
    if not report_path.is_file():
        return HarnessResult(False, f"官方 Harness 未生成结果，退出码 {completed.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("infra_failure_instances") or report.get("error_instances"):
        return HarnessResult(False, "官方 Harness 环境失败")
    return HarnessResult(task.instance_id in report.get("resolved_ids", []))


def _tool_manager(workspace: Path) -> ToolManager:
    """创建真实评测使用的全量本地工具集。"""

    async def approve_all(definition, tool_call, allow_session):
        """评测环境隔离后允许 Agent 执行工作区内操作。"""

        return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

    manager = ToolManager(permission_manager=PermissionManager(approve_all))
    for factory in (
        create_read_file_tool,
        create_list_files_tool,
        create_search_files_tool,
        create_write_file_tool,
        create_edit_file_tool,
    ):
        manager.register_local(*factory(workspace))
    manager.register_local(
        *create_run_command_tool(workspace, EVALUATION_COMMAND_TIMEOUT_SECONDS)
    )
    return manager


def _context_builder(
    session: Session,
    client: TimedModelClient,
    compact: bool,
    events: list[dict[str, object]],
    model_name: str,
    tool_manager: ToolManager,
):
    """复用生产上下文构建逻辑，并在专项中同步压缩记录。"""

    budget = ContextBudget(16_000, 4_000, 8_000) if compact else DEFAULT_CONTEXT_BUDGET
    manager = ContextManager(
        budget,
        {
            definition.name: definition.capability
            for definition in tool_manager.list_definitions()
            if definition.capability is not None
        },
        model_tools=tool_manager.model_tools(),
        system_prompt=load_prompt("agent"),
    )
    manager.set_model_name(model_name)

    async def build_context(messages: Sequence[Message], force_compaction: bool):
        """构建低预算模型上下文，并将新摘要写入评测 Session。"""

        result = await manager.build_for_model_result(
            client, messages, session.get_compactions(), force_compaction
        )
        if result.compaction is not None:
            session.add_compaction(result.compaction)
            events.append({"type": "compaction"})
        return result

    return build_context


def _agent_prompt(task: SwebenchTask) -> str:
    """组合真实 Issue 与最小工作约束。"""

    return (
        "Resolve the following repository issue. Inspect the source, make the smallest correct "
        "code change, and run relevant tests when possible. Do not modify tests merely to make "
        "them pass.\n\n"
        f"Issue:\n{task.issue}"
    )


def _result(
    task,
    started_at,
    client,
    events,
    assertions,
    changed_files,
    compact,
    persistence_degraded,
    error_message,
    error_category: str | None = None,
    tool_rounds: int = 0,
    stop_reason: str | None = None,
):
    """将运行统计汇总成统一评测结果。"""

    tool_events = [event for event in events if event.get("type") == "tool_result"]
    return EvaluationResult(
        scenario=task.instance_id,
        task_id=task.instance_id,
        source=task.source,
        evaluation_group="compaction" if compact else "normal",
        base_commit=task.base_commit,
        changed_files=changed_files,
        duration_ms=(perf_counter() - started_at) * 1000,
        evaluation_type="real-task",
        model_requests=len(client.requests) if client else 0,
        tool_rounds=tool_rounds,
        tool_calls=len(tool_events),
        tool_failures=sum(bool(event.get("is_error")) for event in tool_events),
        compactions=sum(event.get("type") == "compaction" for event in events),
        estimated_tokens=sum(estimate_context_tokens(request) for request in client.requests) if client else 0,
        actual_tokens=client.total_actual_tokens if client else None,
        persistence_degraded=persistence_degraded,
        error_category=(
            error_category
            or "environment"
            if error_message
            else "agent"
            if not all(assertion.passed for assertion in assertions)
            else None
        ),
        error_message=(
            error_message
            or next((assertion.message for assertion in assertions if not assertion.passed), None)
        ),
        stop_reason=stop_reason,
        model_request_durations_ms=tuple(client.durations_ms) if client else (),
        events=tuple(events),
        assertions=assertions,
    )


def _model_error_record(error: ModelClientError) -> dict[str, object]:
    """保存评测所需的安全模型失败诊断，不记录请求内容。"""

    return {
        "type": "model_error",
        "category": error.category,
        "retryable": error.retryable,
        "cause_type": type(error.cause).__name__ if error.cause is not None else None,
    }


def _normalise_patch_paths(patch: str, baseline: Path, workspace: Path) -> str:
    """移除临时目录前缀，使补丁可在基线仓库根目录应用。"""

    baseline_path = baseline.resolve().as_posix().lstrip("/")
    workspace_path = workspace.resolve().as_posix().lstrip("/")
    return patch.replace(f"a/{baseline_path}/", "a/").replace(f"b/{workspace_path}/", "b/")


def _changed_files(patch: str) -> tuple[str, ...]:
    """从标准 diff 头部提取被修改的相对路径。"""

    return tuple(
        line.removeprefix("+++ b/")
        for line in patch.splitlines()
        if line.startswith("+++ b/") and line != "+++ /dev/null"
    )


def _run_git(arguments: list[str]) -> None:
    """运行 Git 命令并将失败转换为清晰的评测环境错误。"""

    try:
        subprocess.run(["git", *arguments], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise EvaluationWorkspaceError(exc.stderr.strip() or "无法准备评测仓库") from exc


def _has_commit(repository: Path, commit: str) -> bool:
    """判断本地缓存是否已有目标基线提交，避免无意义的网络拉取。"""

    return subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _dataset_name(source: str) -> str:
    """校验任务来源并返回官方数据集名称。"""

    try:
        return DATASETS[source]
    except KeyError as exc:
        raise ValueError(f"不支持的 SWE-bench 来源：{source}") from exc


def main() -> int:
    """处理真实 SWE-bench 评测的命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Epsilon SWE-bench 真实任务")
    parser.add_argument("--confirm", action="store_true", help="确认发起真实模型请求")
    parser.add_argument("--instance-id", action="append", required=True, help="SWE-bench 任务 ID")
    parser.add_argument("--source", choices=tuple(DATASETS), default="swebench-lite")
    parser.add_argument("--result-root", type=Path, default=Path("evaluation-results/swebench"))
    parser.add_argument("--harness-python", default="python", help="安装 swebench 的 Python 可执行文件")
    parser.add_argument("--precheck", action="store_true", help="只用参考补丁验证环境，不调用模型")
    parser.add_argument("--compact", action="store_true", help="使用低上下文预算运行")
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS,
        help="本次评测的工具轮次上限，默认 40",
    )
    args = parser.parse_args()
    if not args.confirm:
        print("真实评测会发起模型请求，请添加 --confirm 后运行")
        return 2
    if args.max_tool_rounds is not None and args.max_tool_rounds <= 0:
        parser.error("--max-tool-rounds 必须大于 0")
    tasks = [load_task(instance_id, args.source) for instance_id in args.instance_id]
    output = args.result_root / "results.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    results = []
    for task in tasks:
        runner = (
            precheck_task(task, args.result_root, args.harness_python)
            if args.precheck
            else run_task(
                task,
                args.result_root,
                args.harness_python,
                args.compact,
                args.max_tool_rounds,
            )
        )
        result = asyncio.run(runner)
        results.append(result)
        append_result(output, result)
    generate_report(args.result_root / "report.html", results)
    print(f"swebench evaluation: {sum(result.passed for result in results)}/{len(results)} passed")
    print(f"results: {output}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
