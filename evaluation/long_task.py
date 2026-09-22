"""同一 Session 内连续下发多任务，量化驱逐在长任务中的净收益。"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from core.agent_loop import AgentLoop, AgentRunResult
from core.artifacts import ArtifactStore
from core.config import load_settings
from core.end_policy import WriteVerificationPolicy
from core.model import Message
from core.openai_client import OpenAICompatibleClient
from core.session import Session
from core.tools.command_executor import CommandExecutor

from .events import event_to_record, message_to_record
from .long_task_validation import (
    ORDERING_TESTS_COMMAND,
    StageValidation,
    run_harness_check,
    run_ordering_tests,
)
from .online import TimedModelClient
from .swebench import (
    DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS,
    SwebenchTask,
    _context_builder,
    _tool_manager,
    create_patch,
    load_task,
    prepare_repository,
)
from .swebench_container import SwebenchContainerExecutor, SwebenchTaskContainer
from .swebench_workspace import EvaluationWorkspace, prepare_evaluation_workspace


T1_INSTRUCTIONS = (
    "Resolve the following repository issue in the workspace. Inspect the source, make the "
    "smallest correct code change, and run relevant tests when possible. Do not modify tests "
    "merely to make them pass.\n\n"
    "Execution environment:\n"
    "- The repository workspace is already selected for every tool. Use relative paths.\n"
    "- run_command already runs from the repository workspace root.\n"
    "- File tool paths are relative to the workspace; search_files.path must be an existing "
    "directory; use '.' for the repository root.\n"
    "- This workspace is a source snapshot without Git history. Do not rely on Git commands "
    "for investigation.\n"
    "- Inspect the repository's existing test configuration before running tests.\n\n"
    "Issue:\n"
)
T2_INSTRUCTIONS = (
    "Continue in the same workspace. T1 fixed the ORDER BY deduplication bug for multi-line "
    "RawSQL expressions. Add a regression test for that behavior:\n"
    "- put the new test in tests/ordering/tests.py;\n"
    "- cover the case where multi-line RawSQL clauses share the same final line but differ "
    "overall, so they must not be treated as duplicates;\n"
    "- do not modify production code.\n"
    f"When done, run `{ORDERING_TESTS_COMMAND}` and make sure it passes."
)
T3_INSTRUCTIONS = (
    "Continue in the same workspace. Refactor T1's fix: extract the 'strip ORDER BY direction "
    "and compute the deduplication key' logic used by get_order_by and get_extra_select into "
    "a private SQLCompiler method, and reuse it in both places. Behavior must not change.\n"
    "- do not modify test files.\n"
    f"When done, run `{ORDERING_TESTS_COMMAND}` and make sure it passes."
)


@dataclass(frozen=True)
class LongTaskStage:
    """描述长任务中的一个阶段及其验证方式。"""

    name: str
    prompt: str
    validation: str
    allowed_changes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LongTaskSpec:
    """描述一个长任务实验：同一 Session 内依次执行的阶段序列。"""

    instance_id: str
    source: str
    stages: tuple[LongTaskStage, ...]


@dataclass(frozen=True)
class LongTaskStageResult:
    """保存单个阶段的验证结论与用量差分。"""

    stage: str
    passed: bool
    validation: StageValidation
    harness: StageValidation | None
    allowed_changes_ok: bool
    changed_files: tuple[str, ...]
    tool_rounds: int
    model_requests: int
    actual_tokens: int
    cached_tokens: int
    cache_hit_rate: float | None
    eviction_events: int
    compactions: int
    artifact_read_calls: int
    duration_ms: float


@dataclass(frozen=True)
class LongTaskResult:
    """保存一次长任务实验的 arm 级汇总。"""

    arm: str
    eviction_enabled: bool
    eviction_threshold_tokens: int | None
    stages: tuple[LongTaskStageResult, ...]

    @property
    def total_actual_tokens(self) -> int:
        """累计服务端实际 token。"""

        return sum(stage.actual_tokens for stage in self.stages)

    @property
    def total_cached_tokens(self) -> int:
        """累计缓存命中 token。"""

        return sum(stage.cached_tokens for stage in self.stages)

    @property
    def total_eviction_events(self) -> int:
        """累计驱逐触发次数。"""

        return sum(stage.eviction_events for stage in self.stages)

    @property
    def total_compactions(self) -> int:
        """累计压缩次数。"""

        return sum(stage.compactions for stage in self.stages)

    @property
    def total_tool_rounds(self) -> int:
        """累计工具回合数。"""

        return sum(stage.tool_rounds for stage in self.stages)

    @property
    def total_artifact_read_calls(self) -> int:
        """累计通过 artifact:// 取回工具输出的次数。"""

        return sum(stage.artifact_read_calls for stage in self.stages)


def default_long_task_spec(task: SwebenchTask) -> LongTaskSpec:
    """构造 T1/T2/T3 三阶段默认规格。"""

    return LongTaskSpec(
        instance_id=task.instance_id,
        source=task.source,
        stages=(
            LongTaskStage(
                "T1",
                T1_INSTRUCTIONS + task.issue,
                "harness",
                (),
            ),
            LongTaskStage("T2", T2_INSTRUCTIONS, "ordering-tests", ("tests/ordering/",)),
            LongTaskStage(
                "T3",
                T3_INSTRUCTIONS,
                "ordering-tests",
                ("django/db/models/sql/compiler.py", "tests/ordering/"),
            ),
        ),
    )


@dataclass
class _StageInputs:
    """承载单阶段运行所需的共享对象。"""

    task: SwebenchTask
    baseline: Path
    prepared: EvaluationWorkspace
    result_root: Path
    harness_python: str
    executor: CommandExecutor
    agent: AgentLoop
    session: Session
    client: TimedModelClient
    events: list[dict[str, object]]
    context_builder: object


async def run_long_task(
    spec: LongTaskSpec,
    result_root: Path,
    harness_python: str,
    *,
    eviction_enabled: bool,
    eviction_threshold_tokens: int | None,
    thinking: str = "high",
    firewall_enabled: bool = True,
    max_tool_rounds_per_stage: int = DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS,
) -> LongTaskResult:
    """在单个 Session 中依次执行阶段序列，并落盘阶段与 arm 记录。"""

    task = load_task(spec.instance_id, spec.source)
    repository = await asyncio.to_thread(
        prepare_repository, task, result_root / "repositories"
    )
    baseline = prepare_evaluation_workspace(repository, task.base_commit, result_root)
    prepared = prepare_evaluation_workspace(repository, task.base_commit, result_root)
    container = SwebenchTaskContainer(task.instance_image, prepared.workspace)
    run_id = str(uuid4())
    client: TimedModelClient | None = None
    try:
        async with container.running():
            settings = load_settings()
            client = TimedModelClient(OpenAICompatibleClient(settings))
            artifact_store = ArtifactStore.for_workspace(prepared.session_root)
            executor = SwebenchContainerExecutor(container)
            manager = _tool_manager(prepared.workspace, executor, artifact_store)
            session = Session(prepared.session_root)
            artifact_store.set_session_id(session.session_id)
            events: list[dict[str, object]] = []
            agent = AgentLoop(
                client,
                manager,
                max_tool_rounds=max_tool_rounds_per_stage,
                thinking_level=thinking,
                artifact_store=artifact_store,
                session_id=session.session_id,
                firewall_enabled=firewall_enabled,
                end_policy=WriteVerificationPolicy(),
            )
            context_builder = _context_builder(
                session,
                client,
                False,
                events,
                settings.model_name,
                manager,
                artifact_store=artifact_store,
                eviction_enabled=eviction_enabled,
                eviction_threshold_tokens=eviction_threshold_tokens,
            )
            inputs = _StageInputs(
                task=task,
                baseline=baseline,
                prepared=prepared,
                result_root=result_root,
                harness_python=harness_python,
                executor=executor,
                agent=agent,
                session=session,
                client=client,
                events=events,
                context_builder=context_builder,
            )
            stage_results = [
                await _run_stage(stage, inputs) for stage in spec.stages
            ]
            session.flush_persistence()
            session.close()
    finally:
        if client is not None:
            await client.close()

    arm = "eviction_on" if eviction_enabled else "eviction_off"
    records_path = result_root / "long_task.jsonl"
    for stage_result in stage_results:
        _append_record(
            records_path,
            _stage_record(run_id, arm, eviction_threshold_tokens, stage_result),
        )
    _append_record(
        records_path,
        _arm_record(arm, eviction_threshold_tokens, stage_results),
    )
    return LongTaskResult(
        arm=arm,
        eviction_enabled=eviction_enabled,
        eviction_threshold_tokens=eviction_threshold_tokens,
        stages=tuple(stage_results),
    )


async def _run_stage(
    stage: LongTaskStage,
    inputs: _StageInputs,
) -> LongTaskStageResult:
    """执行一个阶段：下发指令、跑 Agent、验证并计算用量差分。"""

    started_at = perf_counter()
    requests_before = len(inputs.client.requests)
    actual_before = inputs.client.total_actual_tokens
    cached_before = inputs.client.total_cached_tokens
    events_before = len(inputs.events)

    inputs.session.add_user_message(stage.prompt)
    inputs.events.append(
        message_to_record(Message(role="user", content=stage.prompt))
    )

    async def collect_event(event: object) -> None:
        """保存本阶段的完整模型与工具轨迹。"""

        inputs.events.append(event_to_record(event))

    agent_result = await inputs.agent.run(
        inputs.session.get_messages(),
        on_event=collect_event,
        build_context=inputs.context_builder,
    )
    for message in agent_result.new_messages:
        inputs.session.add_message(message)

    validation, harness = await _validate_stage(stage, inputs)
    changelog, _ = await asyncio.to_thread(
        create_patch, inputs.baseline, inputs.prepared.workspace
    )
    allowed_ok = _allowed_changes_ok(changelog, stage.allowed_changes)
    stage_events = inputs.events[events_before:]

    requests_after = len(inputs.client.requests)
    actual_after = inputs.client.total_actual_tokens
    cached_after = inputs.client.total_cached_tokens
    actual_tokens = _delta(actual_before, actual_after)
    cached_tokens = _delta(cached_before, cached_after)

    return LongTaskStageResult(
        stage=stage.name,
        passed=validation.passed and allowed_ok and (harness is None or harness.passed),
        validation=validation,
        harness=harness,
        allowed_changes_ok=allowed_ok,
        changed_files=changelog,
        tool_rounds=agent_result.tool_rounds,
        model_requests=requests_after - requests_before,
        actual_tokens=actual_tokens,
        cached_tokens=cached_tokens,
        cache_hit_rate=(
            cached_tokens / actual_tokens if actual_tokens > 0 else None
        ),
        eviction_events=sum(
            event.get("type") == "eviction" for event in stage_events
        ),
        compactions=sum(
            event.get("type") == "compaction" for event in stage_events
        ),
        artifact_read_calls=sum(_is_artifact_read(event) for event in stage_events),
        duration_ms=(perf_counter() - started_at) * 1000,
    )


async def _validate_stage(
    stage: LongTaskStage,
    inputs: _StageInputs,
) -> tuple[StageValidation, StageValidation | None]:
    """按阶段类型执行验证，返回主验证与可选的 Harness 回归验证。"""

    if stage.validation == "harness":
        harness = await run_harness_check(
            inputs.task,
            inputs.baseline,
            inputs.prepared.workspace,
            inputs.result_root,
            inputs.harness_python,
        )
        return harness, None
    if stage.validation == "ordering-tests":
        tests = await run_ordering_tests(inputs.executor, inputs.prepared.workspace)
        harness = await run_harness_check(
            inputs.task,
            inputs.baseline,
            inputs.prepared.workspace,
            inputs.result_root,
            inputs.harness_python,
        )
        return tests, harness
    raise ValueError(f"未知的阶段验证方式：{stage.validation}")


def _allowed_changes_ok(
    changed_files: tuple[str, ...],
    allowed_changes: tuple[str, ...],
) -> bool:
    """判断改动是否都落在允许的路径前缀内。"""

    if not allowed_changes:
        return True
    return all(
        any(path == allowed or path.startswith(allowed) for allowed in allowed_changes)
        for path in changed_files
    )


def _is_artifact_read(event: dict[str, object]) -> bool:
    """判断一条事件是否为通过 artifact:// 取回工具输出。"""

    if event.get("type") != "tool_call" or event.get("name") != "read_file":
        return False
    arguments = event.get("arguments")
    if not isinstance(arguments, dict):
        return False
    return "artifact://" in str(arguments.get("path", ""))


def _delta(before: int | None, after: int | None) -> int:
    """计算前后快照差值，任一侧缺失时按 0 处理。"""

    if before is None or after is None:
        return 0
    return max(0, after - before)


def _stage_record(
    run_id: str,
    arm: str,
    eviction_threshold_tokens: int | None,
    result: LongTaskStageResult,
) -> dict[str, object]:
    """把阶段结果转换为可落盘记录。"""

    return {
        "type": "long_task_stage",
        "run_id": run_id,
        "arm": arm,
        "eviction_threshold_tokens": eviction_threshold_tokens,
        "stage": result.stage,
        "passed": result.passed,
        "validation_kind": result.validation.kind,
        "command": result.validation.command,
        "exit_code": result.validation.exit_code,
        "detail": result.validation.detail,
        "harness_passed": None if result.harness is None else result.harness.passed,
        "harness_detail": None if result.harness is None else result.harness.detail,
        "allowed_changes_ok": result.allowed_changes_ok,
        "changed_files": list(result.changed_files),
        "tool_rounds": result.tool_rounds,
        "model_requests": result.model_requests,
        "actual_tokens": result.actual_tokens,
        "cached_tokens": result.cached_tokens,
        "cache_hit_rate": result.cache_hit_rate,
        "eviction_events": result.eviction_events,
        "compactions": result.compactions,
        "artifact_read_calls": result.artifact_read_calls,
        "duration_ms": result.duration_ms,
    }


def _arm_record(
    arm: str,
    eviction_threshold_tokens: int | None,
    stages: list[LongTaskStageResult],
) -> dict[str, object]:
    """把整个 arm 的阶段结果汇总为一条记录。"""

    return {
        "type": "long_task_arm",
        "arm": arm,
        "eviction_threshold_tokens": eviction_threshold_tokens,
        "stages": [stage.stage for stage in stages],
        "stage_passed": [stage.passed for stage in stages],
        "total_actual_tokens": sum(stage.actual_tokens for stage in stages),
        "total_cached_tokens": sum(stage.cached_tokens for stage in stages),
        "total_eviction_events": sum(stage.eviction_events for stage in stages),
        "total_compactions": sum(stage.compactions for stage in stages),
        "total_tool_rounds": sum(stage.tool_rounds for stage in stages),
        "total_artifact_read_calls": sum(
            stage.artifact_read_calls for stage in stages
        ),
    }


def _append_record(path: Path, record: dict[str, object]) -> None:
    """向 JSONL 追加一条记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")


def main() -> int:
    """处理长任务评测的命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Epsilon 长任务评测")
    parser.add_argument("--confirm", action="store_true", help="确认发起真实模型请求")
    parser.add_argument("--instance-id", required=True, help="SWE-bench 任务 ID")
    parser.add_argument("--source", default="swebench-lite")
    parser.add_argument(
        "--result-root", type=Path, default=Path("evaluation-results/long-task")
    )
    parser.add_argument("--harness-python", default="python")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--no-firewall", action="store_true")
    parser.add_argument("--eviction", action="store_true")
    parser.add_argument(
        "--eviction-threshold", type=int, default=None, help="驱逐阈值 token 数"
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS,
    )
    args = parser.parse_args()
    if not args.confirm:
        print("长任务评测会发起真实模型请求，请添加 --confirm 后运行")
        return 2
    if args.max_tool_rounds <= 0:
        parser.error("--max-tool-rounds 必须大于 0")
    if args.eviction_threshold is not None and args.eviction_threshold <= 0:
        parser.error("--eviction-threshold 必须大于 0")
    harness_python = Path(args.harness_python)
    if harness_python.exists():
        args.harness_python = str(harness_python.absolute())

    task = load_task(args.instance_id, args.source)
    spec = default_long_task_spec(task)
    result = asyncio.run(
        run_long_task(
            spec,
            args.result_root,
            args.harness_python,
            eviction_enabled=args.eviction,
            eviction_threshold_tokens=args.eviction_threshold,
            thinking=args.thinking,
            firewall_enabled=not args.no_firewall,
            max_tool_rounds_per_stage=args.max_tool_rounds,
        )
    )
    print(f"long task arm: {result.arm}")
    for stage in result.stages:
        print(
            f"  {stage.stage}: passed={stage.passed} "
            f"tokens={stage.actual_tokens} eviction={stage.eviction_events}"
        )
    print(f"total actual tokens: {result.total_actual_tokens}")
    print(f"results: {args.result_root / 'long_task.jsonl'}")
    return 0 if all(stage.passed for stage in result.stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
