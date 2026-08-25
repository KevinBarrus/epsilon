"""提供需要显式确认后才执行的真实模型冒烟评测"""

import argparse
import asyncio
import sys
import tempfile
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

from core.agent_loop import AgentLoop, ToolExecutionEvent
from core.config import ConfigError, load_settings
from core.context import estimate_context_tokens
from core.context import ContextBudget, ContextManager
from core.errors import AgentError
from core.model import Message, ModelClient, ModelEvent, TextDelta, ToolCallEvent, UsageEvent
from core.openai_client import OpenAICompatibleClient
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
    create_write_file_tool,
)

from .models import EvaluationAssertion, EvaluationResult
from .events import event_to_record, message_to_record
from .baseline import (
    compare_baseline,
    create_baseline,
    load_baseline,
    needs_performance_baseline_refresh,
    write_baseline,
)
from .report import generate_report
from .storage import append_result

ONLINE_SCENARIO_VERSION = "3"
DEFAULT_ONLINE_REPETITIONS = 7


@dataclass(frozen=True)
class _OnlineFileTask:
    """描述一条使用真实模型执行的确定性文件任务。"""

    name: str
    prompt: str
    initial_files: tuple[tuple[str, str], ...]
    expected_files: tuple[tuple[str, str], ...]
    unchanged_files: tuple[tuple[str, str], ...]
    required_reads: tuple[str, ...]
    required_edits: tuple[str, ...]
    reply_keywords: tuple[str, ...]
    failed_read_path: str | None = None


ONLINE_FILE_TASKS = (
    _OnlineFileTask(
        name="online_single_file_edit",
        prompt=(
            "请先读取 note.txt，再将其中的 before 改为 after。"
            "不要修改 keep.txt。完成后说明 note.txt 已完成修改"
        ),
        initial_files=(("note.txt", "before\n"), ("keep.txt", "keep\n")),
        expected_files=(("note.txt", "after\n"),),
        unchanged_files=(("keep.txt", "keep\n"),),
        required_reads=("note.txt",),
        required_edits=("note.txt",),
        reply_keywords=("note.txt", "完成"),
    ),
    _OnlineFileTask(
        name="online_multi_file_edit",
        prompt=(
            "请读取 config.txt 和 note.txt，并将 config.txt 中的 "
            "old-config 改为 new-config、note.txt 中的 old-note 改为 new-note。"
            "不要修改 keep.txt。完成后说明 config.txt 和 note.txt 已完成修改"
        ),
        initial_files=(
            ("config.txt", "old-config\n"),
            ("note.txt", "old-note\n"),
            ("keep.txt", "keep\n"),
        ),
        expected_files=(
            ("config.txt", "new-config\n"),
            ("note.txt", "new-note\n"),
        ),
        unchanged_files=(("keep.txt", "keep\n"),),
        required_reads=("config.txt", "note.txt"),
        required_edits=("config.txt", "note.txt"),
        reply_keywords=("config.txt", "note.txt", "完成"),
    ),
    _OnlineFileTask(
        name="online_tool_failure_recovery",
        prompt=(
            "请先尝试读取不存在的 missing.txt。收到工具错误后，读取 note.txt，"
            "再将其中的 before 改为 after。不要修改 keep.txt。完成后说明 "
            "note.txt 已完成修改"
        ),
        initial_files=(("note.txt", "before\n"), ("keep.txt", "keep\n")),
        expected_files=(("note.txt", "after\n"),),
        unchanged_files=(("keep.txt", "keep\n"),),
        required_reads=("note.txt",),
        required_edits=("note.txt",),
        reply_keywords=("note.txt", "完成"),
        failed_read_path="missing.txt",
    ),
)


@dataclass(frozen=True)
class _OnlineCodeTask:
    """描述一条以独立 pytest 结果验证代码正确性的真实任务。"""

    name: str
    prompt: str
    initial_files: tuple[tuple[str, str], ...]
    test_file: str
    required_reads: tuple[str, ...]
    required_edits: tuple[str, ...]
    reply_keywords: tuple[str, ...]
    unrelated_files: tuple[tuple[str, str], ...] = ()


ONLINE_CODE_TASKS = (
    _OnlineCodeTask(
        name="online_code_fix_bug",
        prompt=(
            "请先读取 math_utils.py 和 test_math_utils.py，找到 add 函数的错误并修复，"
            "使 test_math_utils.py 的测试通过。可以运行 pytest 验证结果。"
            "不要修改 test_math_utils.py 和 notes.txt。"
        ),
        initial_files=(
            ("math_utils.py", "def add(left, right):\n    return left - right\n"),
            (
                "test_math_utils.py",
                "from math_utils import add\n\n\ndef test_add_returns_sum():\n    assert add(2, 3) == 5\n",
            ),
            ("notes.txt", "不要修改\n"),
        ),
        test_file="test_math_utils.py",
        required_reads=("math_utils.py",),
        required_edits=("math_utils.py",),
        reply_keywords=("math_utils.py", "完成"),
        unrelated_files=(("notes.txt", "不要修改\n"),),
    ),
    _OnlineCodeTask(
        name="online_code_complete_skeleton",
        prompt=(
            "请先读取 string_utils.py 和 test_string_utils.py，补全 reverse_words 函数"
            "的实现，使 test_string_utils.py 的测试通过。可以运行 pytest 验证结果。"
            "不要修改 test_string_utils.py 和 notes.txt。"
        ),
        initial_files=(
            ("string_utils.py", "def reverse_words(text):\n    raise NotImplementedError\n"),
            (
                "test_string_utils.py",
                "from string_utils import reverse_words\n\n\n"
                "def test_reverse_words_reverses_order():\n"
                "    assert reverse_words(\"hello world\") == \"world hello\"\n",
            ),
            ("notes.txt", "不要修改\n"),
        ),
        test_file="test_string_utils.py",
        required_reads=("string_utils.py",),
        required_edits=("string_utils.py",),
        reply_keywords=("string_utils.py", "完成"),
        unrelated_files=(("notes.txt", "不要修改\n"),),
    ),
)


class TimedModelClient:
    """记录真实模型请求耗时的客户端包装器"""

    def __init__(self, client: ModelClient) -> None:
        """保存真实客户端和请求统计"""

        self._client = client
        self.requests: list[list[Message]] = []
        self.durations_ms: list[float] = []
        self.usages: list[int | None] = []

    async def close(self) -> None:
        """关闭被包装客户端持有的网络连接"""

        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    @property
    def total_actual_tokens(self) -> int | None:
        """汇总所有请求的 total Token，任一请求缺少 usage 时返回 None"""

        if not self.usages or any(usage is None for usage in self.usages):
            return None
        return sum(usage for usage in self.usages if usage is not None)

    async def stream_response(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, object]] = (),
        thinking_level: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        """记录一次真实流式请求并转发模型事件"""

        self.requests.append(list(messages))
        self.usages.append(None)
        usage_index = len(self.usages) - 1
        started_at = perf_counter()
        try:
            async for event in self._client.stream_response(messages, tools):
                if isinstance(event, UsageEvent):
                    self.usages[usage_index] = event.total_tokens
                yield event
        finally:
            self.durations_ms.append((perf_counter() - started_at) * 1000)

    async def stream_chat(self, messages: Sequence[Message]):
        """记录一次真实摘要请求并转发文本片段"""

        self.requests.append(list(messages))
        self.usages.append(None)
        usage_index = len(self.usages) - 1
        started_at = perf_counter()
        try:
            async for event in self._client.stream_response(messages):
                if isinstance(event, UsageEvent):
                    self.usages[usage_index] = event.total_tokens
                elif isinstance(event, TextDelta):
                    yield event.content
        finally:
            self.durations_ms.append((perf_counter() - started_at) * 1000)


@dataclass
class _OnlineRunState:
    """收集在线场景失败前的最小诊断信息。"""

    scenario: str
    evaluation_type: str
    started_at: float = field(default_factory=perf_counter)
    stage: str = "initialization"
    client: TimedModelClient | None = None
    events: list[dict[str, object]] = field(default_factory=list)

    def failure(self, error: Exception) -> EvaluationResult:
        """将场景异常转换为保留诊断信息的评测结果。"""

        tool_events = [event for event in self.events if event.get("type") == "tool_result"]
        message = f"{type(error).__name__}: {error}"
        return EvaluationResult(
            scenario=self.scenario,
            duration_ms=(perf_counter() - self.started_at) * 1000,
            evaluation_type=self.evaluation_type,  # type: ignore[arg-type]
            model_requests=len(self.client.requests) if self.client else 0,
            tool_calls=len(tool_events),
            tool_failures=sum(bool(event.get("is_error")) for event in tool_events),
            estimated_tokens=(
                sum(estimate_context_tokens(request) for request in self.client.requests)
                if self.client
                else 0
            ),
            model_request_durations_ms=(
                tuple(self.client.durations_ms) if self.client else ()
            ),
            error_category=_error_category(error),
            error_stage=self.stage,
            error_message=message,
            events=tuple(
                [
                    *self.events,
                    {
                        "type": "evaluation_error",
                        "category": _error_category(error),
                        "stage": self.stage,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    },
                ]
            ),
            assertions=(
                EvaluationAssertion(
                    "online-evaluation-error",
                    False,
                    f"在线评测在 {self.stage} 失败：{message}",
                ),
            ),
        )


async def _run_with_diagnostics(
    state: _OnlineRunState,
    runner: Awaitable[EvaluationResult],
) -> EvaluationResult:
    """执行在线场景并将未处理异常转换为评测结果。"""

    try:
        return await runner
    except Exception as error:
        return state.failure(error)


def _error_category(error: Exception) -> str:
    """按异常来源标记评测失败类别。"""

    if isinstance(error, AgentError):
        return error.category
    if isinstance(error, ConfigError):
        return "configuration"
    if isinstance(error, OSError):
        return "environment"
    return "runner"


async def run_online_smoke() -> EvaluationResult:
    """兼容原入口，执行单文件真实任务。"""

    return await run_online_file_task(ONLINE_FILE_TASKS[0])


async def run_online_file_task(
    task: _OnlineFileTask,
) -> EvaluationResult:
    """在独立工作区执行一条真实文件任务并保留诊断结果。"""

    state = _OnlineRunState(task.name, "real-task")
    return await _run_with_diagnostics(state, _run_online_file_task(state, task))


async def _run_online_file_task(
    state: _OnlineRunState,
    task: _OnlineFileTask,
) -> EvaluationResult:
    """执行真实文件任务并持续更新诊断状态。"""

    state.stage = "load-settings"
    settings = load_settings()
    with tempfile.TemporaryDirectory(prefix="epsilon-online-") as directory:
        workspace = Path(directory)
        for path, content in task.initial_files:
            (workspace / path).write_text(content, encoding="utf-8")
        client = TimedModelClient(OpenAICompatibleClient(settings))
        state.client = client

        async def approve_write(definition, tool_call, allow_session):
            """允许评测任务执行单次文件编辑。"""

            return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

        manager = ToolManager(
            permission_manager=PermissionManager(approve_write),
        )
        manager.register_local(*create_read_file_tool(workspace))
        manager.register_local(*create_edit_file_tool(workspace))
        session = Session(workspace)
        events = state.events
        events.append(
            message_to_record(
                Message(
                    role="user",
                    content=task.prompt,
                )
            )
        )
        session.add_user_message(task.prompt)

        async def collect_event(event: object) -> None:
            events.append(event_to_record(event))

        started_at = perf_counter()
        state.stage = "agent-loop"
        result = await AgentLoop(client, manager).run(
            session.get_messages(),
            on_event=collect_event,
        )
        for message in result.new_messages:
            session.add_message(message)
        state.stage = "session-persistence"
        persistence_ok = session.flush_persistence() and session.close()
        restored = Session.restore(workspace, session.session_id)
        restored_messages = restored.get_messages()
        restored.close()

        tool_events = [event for event in events if event["type"] == "tool_result"]
        events.append(message_to_record(Message(role="assistant", content=result.final_content)))
        assertions = [
            EvaluationAssertion(
                "file-content",
                _files_match(workspace, task.expected_files),
                "真实模型没有完成预期文件修改",
            ),
            EvaluationAssertion(
                "read-before-edit",
                _has_expected_file_read_before_edit(events, task),
                "真实模型在读取目标文件前尝试编辑",
            ),
            EvaluationAssertion(
                "unrelated-files",
                _files_match(workspace, task.unchanged_files),
                "真实模型修改了无关文件",
            ),
            EvaluationAssertion(
                "final-response",
                _contains_keywords(result.final_content, task.reply_keywords),
                "最终回复缺少任务完成关键信息",
            ),
            EvaluationAssertion(
                "session-restore",
                restored_messages == session.get_messages(),
                "真实 Session 恢复后的消息不一致",
            ),
            EvaluationAssertion(
                "persistence",
                persistence_ok,
                "真实 Session Flush 失败",
            ),
        ]
        if task.failed_read_path is not None:
            assertions.append(
                EvaluationAssertion(
                    "tool-failure-recovery",
                    _has_recovered_failed_read(events, task),
                    "工具失败后没有按要求恢复并继续执行",
                )
            )
        return EvaluationResult(
            scenario=task.name,
            duration_ms=(perf_counter() - started_at) * 1000,
            evaluation_type="real-task",
            model_requests=len(client.requests),
            tool_calls=len(tool_events),
            tool_failures=sum(bool(event["is_error"]) for event in tool_events),
            estimated_tokens=sum(
                estimate_context_tokens(request) for request in client.requests
            ),
            actual_tokens=client.total_actual_tokens,
            persistence_degraded=not persistence_ok,
            model_request_durations_ms=tuple(client.durations_ms),
            events=tuple(events),
            assertions=tuple(assertions),
        )


async def run_online_code_task(
    task: _OnlineCodeTask,
) -> EvaluationResult:
    """在独立工作区执行一条代码正确性任务并保留诊断结果。"""

    state = _OnlineRunState(task.name, "code-correctness")
    return await _run_with_diagnostics(state, _run_online_code_task(state, task))


async def _run_online_code_task(
    state: _OnlineRunState,
    task: _OnlineCodeTask,
) -> EvaluationResult:
    """执行代码正确性任务并用独立 pytest 结果判断成功。"""

    state.stage = "load-settings"
    settings = load_settings()
    with tempfile.TemporaryDirectory(prefix="epsilon-code-") as directory:
        workspace = Path(directory)
        for path, content in task.initial_files:
            (workspace / path).write_text(content, encoding="utf-8")
        client = TimedModelClient(OpenAICompatibleClient(settings))
        state.client = client

        async def approve_all(definition, tool_call, allow_session):
            """允许评测任务执行单次写入和命令。"""

            return ApprovalResult(ApprovalDecision.ALLOW_ONCE)

        manager = ToolManager(permission_manager=PermissionManager(approve_all))
        manager.register_local(*create_read_file_tool(workspace))
        manager.register_local(*create_list_files_tool(workspace))
        manager.register_local(*create_write_file_tool(workspace))
        manager.register_local(*create_edit_file_tool(workspace))
        manager.register_local(*create_run_command_tool(workspace))
        session = Session(workspace)
        events = state.events
        events.append(message_to_record(Message(role="user", content=task.prompt)))
        session.add_user_message(task.prompt)

        async def collect_event(event: object) -> None:
            events.append(event_to_record(event))

        started_at = perf_counter()
        state.stage = "agent-loop"
        result = await AgentLoop(client, manager).run(
            session.get_messages(),
            on_event=collect_event,
        )
        for message in result.new_messages:
            session.add_message(message)
        state.stage = "session-persistence"
        persistence_ok = session.flush_persistence() and session.close()
        restored = Session.restore(workspace, session.session_id)
        restored_messages = restored.get_messages()
        restored.close()

        state.stage = "pytest-verification"
        pytest_passed = await _run_pytest(workspace)

        tool_events = [event for event in events if event["type"] == "tool_result"]
        events.append(
            message_to_record(Message(role="assistant", content=result.final_content))
        )
        events.append(
            {
                "type": "reply_keywords",
                "matched": _contains_keywords(
                    result.final_content,
                    task.reply_keywords,
                ),
            }
        )
        test_file_content = _initial_file_content(task.initial_files, task.test_file)
        assertions = [
            EvaluationAssertion(
                "pytest-passed",
                pytest_passed,
                "代码任务没有通过独立 pytest 验证",
            ),
            EvaluationAssertion(
                "test-file-unchanged",
                _files_match(workspace, ((task.test_file, test_file_content),)),
                "模型修改了测试文件",
            ),
            EvaluationAssertion(
                "unrelated-files-unchanged",
                _files_match(workspace, task.unrelated_files),
                "模型修改了与任务无关的文件",
            ),
            EvaluationAssertion(
                "read-before-write",
                _has_expected_code_read_before_write(events, task),
                "模型在修改目标文件前没有先读取",
            ),
            EvaluationAssertion(
                "session-restore",
                restored_messages == session.get_messages(),
                "真实 Session 恢复后的消息不一致",
            ),
            EvaluationAssertion(
                "persistence",
                persistence_ok,
                "代码任务 Session Flush 失败",
            ),
        ]
        return EvaluationResult(
            scenario=task.name,
            duration_ms=(perf_counter() - started_at) * 1000,
            evaluation_type="code-correctness",
            model_requests=len(client.requests),
            tool_calls=len(tool_events),
            tool_failures=sum(bool(event["is_error"]) for event in tool_events),
            estimated_tokens=sum(
                estimate_context_tokens(request) for request in client.requests
            ),
            actual_tokens=client.total_actual_tokens,
            persistence_degraded=not persistence_ok,
            model_request_durations_ms=tuple(client.durations_ms),
            events=tuple(events),
            assertions=tuple(assertions),
        )


async def _run_pytest(workspace: Path) -> bool:
    """在独立工作区运行 pytest 并返回是否全部通过。"""

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()
    return process.returncode == 0


def _initial_file_content(
    initial_files: tuple[tuple[str, str], ...],
    path: str,
) -> str:
    """从初始文件列表查找指定路径的原始内容。"""

    for file_path, content in initial_files:
        if file_path == path:
            return content
    raise ValueError(f"初始文件未定义：{path}")


def _has_expected_code_read_before_write(
    events: Sequence[dict[str, object]],
    task: _OnlineCodeTask,
) -> bool:
    """确认每个目标源文件在被修改前都被读取。"""

    calls = [event for event in events if event.get("type") == "tool_call"]
    read_positions = _tool_positions(calls, "read_file", task.required_reads)
    write_positions = [
        position
        for path in task.required_edits
        if (position := _first_write_position(calls, path)) is not None
    ]
    return (
        len(read_positions) == len(task.required_reads)
        and len(write_positions) == len(task.required_edits)
        and all(
            (read_position := _tool_position(calls, "read_file", path)) is not None
            and read_position < write_position
            for path, write_position in zip(task.required_edits, write_positions)
        )
    )


def _first_write_position(
    calls: Sequence[dict[str, object]],
    path: str,
) -> int | None:
    """返回 write_file 或 edit_file 首次操作指定路径的位置。"""

    for index, call in enumerate(calls):
        arguments = call.get("arguments")
        if (
            call.get("name") in {"write_file", "edit_file"}
            and isinstance(arguments, dict)
            and arguments.get("path") == path
        ):
            return index
    return None


def _files_match(workspace: Path, expected_files: tuple[tuple[str, str], ...]) -> bool:
    """检查一组文件是否完全符合任务预期内容。"""

    return all(
        (workspace / path).is_file()
        and (workspace / path).read_text(encoding="utf-8") == content
        for path, content in expected_files
    )


def _has_expected_file_read_before_edit(
    events: Sequence[dict[str, object]],
    task: _OnlineFileTask,
) -> bool:
    """确认每个目标文件都在编辑前被读取。"""

    calls = [event for event in events if event.get("type") == "tool_call"]
    read_positions = _tool_positions(calls, "read_file", task.required_reads)
    edit_positions = _tool_positions(calls, "edit_file", task.required_edits)
    return (
        len(read_positions) == len(task.required_reads)
        and len(edit_positions) == len(task.required_edits)
        and all(
            (read_position := _tool_position(calls, "read_file", path)) is not None
            and read_position < edit_position
            for path, edit_position in zip(task.required_edits, edit_positions)
        )
    )


def _has_recovered_failed_read(
    events: Sequence[dict[str, object]],
    task: _OnlineFileTask,
) -> bool:
    """确认指定读取失败后才继续执行正确的读取和编辑。"""

    assert task.failed_read_path is not None
    calls = [event for event in events if event.get("type") == "tool_call"]
    failed_positions = _tool_positions(calls, "read_file", (task.failed_read_path,))
    if not failed_positions:
        return False
    failed_call = calls[failed_positions[0]]
    failed_call_id = failed_call.get("call_id")
    failed_result = any(
        event.get("type") == "tool_result"
        and event.get("call_id") == failed_call_id
        and event.get("is_error") is True
        for event in events
    )
    recovered_reads = _tool_positions(calls, "read_file", task.required_reads)
    return failed_result and recovered_reads and failed_positions[0] < min(recovered_reads)


def _tool_positions(
    calls: Sequence[dict[str, object]],
    tool_name: str,
    paths: tuple[str, ...],
) -> list[int]:
    """返回每个指定路径首次由目标工具调用的位置。"""

    return [
        position
        for path in paths
        if (position := _tool_position(calls, tool_name, path)) is not None
    ]


def _tool_position(
    calls: Sequence[dict[str, object]],
    tool_name: str,
    path: str,
) -> int | None:
    """返回目标工具首次操作指定路径的位置。"""

    for index, call in enumerate(calls):
        arguments = call.get("arguments")
        if (
            call.get("name") == tool_name
            and isinstance(arguments, dict)
            and arguments.get("path") == path
        ):
            return index
    return None


def _contains_keywords(content: str, keywords: tuple[str, ...]) -> bool:
    """检查最终回复是否包含任务要求的全部关键词。"""

    return all(keyword.casefold() in content.casefold() for keyword in keywords)


async def run_online_suite(
    repetitions: int = DEFAULT_ONLINE_REPETITIONS,
    scenario: str = "main",
) -> list[EvaluationResult]:
    """重复执行在线主链路并保留单次失败结果"""

    if repetitions <= 0:
        raise ValueError("在线评测重复次数必须大于 0")
    results: list[EvaluationResult] = []
    runners = {
        "context-compaction": run_online_compaction_smoke,
        "network-error": run_online_network_error_smoke,
    }
    if scenario != "main" and scenario not in runners:
        raise ValueError(f"不支持的在线评测场景：{scenario}")
    for repetition in range(1, repetitions + 1):
        if scenario == "main":
            for task in ONLINE_FILE_TASKS:
                result = await _run_suite_task(
                    task.name,
                    "real-task",
                    run_online_file_task(task),
                )
                results.append(replace(result, repetition=repetition))
            for task in ONLINE_CODE_TASKS:
                result = await _run_suite_task(
                    task.name,
                    "code-correctness",
                    run_online_code_task(task),
                )
                results.append(replace(result, repetition=repetition))
        else:
            result = await _run_suite_task(
                f"online_{scenario}",
                "online-special",
                runners[scenario](),
            )
            results.append(replace(result, repetition=repetition))
    return results


async def _run_suite_task(
    scenario_name: str,
    evaluation_type: str,
    runner: Awaitable[EvaluationResult],
) -> EvaluationResult:
    """执行单个评测任务并把异常转换为保留诊断的失败结果。"""

    try:
        return await runner
    except Exception as error:
        return EvaluationResult(
            scenario=scenario_name,
            duration_ms=0,
            evaluation_type=evaluation_type,  # type: ignore[arg-type]
            error_category=_error_category(error),
            error_stage="suite",
            error_message=f"{type(error).__name__}: {error}",
            events=(
                {
                    "type": "evaluation_error",
                    "category": _error_category(error),
                    "stage": "suite",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            ),
            assertions=(
                EvaluationAssertion(
                    "online-evaluation-error",
                    False,
                    f"在线评测在 suite 失败：{type(error).__name__}: {error}",
                ),
            ),
        )


async def run_online_compaction_smoke() -> EvaluationResult:
    """使用真实模型执行一次上下文压缩和 Session 恢复冒烟评测"""

    state = _OnlineRunState("online_context_compaction", "online-special")
    return await _run_with_diagnostics(state, _run_online_compaction_smoke(state))


async def _run_online_compaction_smoke(
    state: _OnlineRunState,
) -> EvaluationResult:
    """执行上下文压缩专项并持续更新诊断状态。"""

    state.stage = "load-settings"
    settings = load_settings()
    with tempfile.TemporaryDirectory(prefix="epsilon-compaction-") as directory:
        workspace = Path(directory)
        client = TimedModelClient(OpenAICompatibleClient(settings))
        state.client = client
        session = Session(workspace)
        for index in range(4):
            session.add_user_message(f"历史任务 {index}: " + "x" * 180)
            session.add_assistant_message(f"历史回复 {index}: " + "x" * 180)
        manager = ContextManager(ContextBudget(500, 100, 120))
        started_at = perf_counter()
        state.stage = "context-compaction"
        result = await manager.build_for_model_result(
            client,
            session.get_messages(),
            session.get_compactions(),
        )
        if result.compaction is not None:
            session.add_compaction(result.compaction)
        state.stage = "session-persistence"
        persistence_ok = session.flush_persistence() and session.close()
        restored = Session.restore(workspace, session.session_id)
        restored_compactions = restored.get_compactions()
        restored.close()
        assertions = (
            EvaluationAssertion(
                "compaction-created",
                result.compaction is not None and not result.fallback_used,
                "真实模型没有生成有效上下文摘要",
            ),
            EvaluationAssertion(
                "session-restore",
                bool(restored_compactions) == (result.compaction is not None),
                "压缩记录无法从 Session 恢复",
            ),
            EvaluationAssertion(
                "persistence",
                persistence_ok,
                "上下文压缩 Session 持久化失败",
            ),
        )
        return EvaluationResult(
            scenario="online_context_compaction",
            duration_ms=(perf_counter() - started_at) * 1000,
            evaluation_type="online-special",
            model_requests=len(client.requests),
            compactions=int(result.compaction is not None),
            estimated_tokens=sum(
                estimate_context_tokens(request) for request in client.requests
            ),
            actual_tokens=client.total_actual_tokens,
            persistence_degraded=not persistence_ok,
            model_request_durations_ms=tuple(client.durations_ms),
            assertions=assertions,
        )


async def run_online_network_error_smoke() -> EvaluationResult:
    """通过本机不可用端口验证真实网络异常处理"""

    state = _OnlineRunState("online_network_error", "online-special")
    return await _run_with_diagnostics(state, _run_online_network_error_smoke(state))


async def _run_online_network_error_smoke(
    state: _OnlineRunState,
) -> EvaluationResult:
    """执行网络异常专项并持续更新诊断状态。"""

    state.stage = "load-settings"
    settings = load_settings()
    unavailable_settings = replace(settings, base_url="http://127.0.0.1:1")
    client = TimedModelClient(OpenAICompatibleClient(unavailable_settings))
    state.client = client
    started_at = perf_counter()
    error: AgentError | None = None
    try:
        state.stage = "network-request"
        await AgentLoop(client, ToolManager()).run(
            [Message(role="user", content="测试网络异常处理")]
        )
    except AgentError as exc:
        error = exc
    assertions = (
        EvaluationAssertion(
            "network-category",
            error is not None and error.category == "network",
            "真实连接失败没有转换为 network 错误",
        ),
        EvaluationAssertion(
            "retry-once",
            len(client.requests) == 2,
            "网络错误没有按策略重试一次",
        ),
        EvaluationAssertion(
            "safe-error",
            error is not None and error.cause is not None,
            "网络错误没有保留内部诊断原因",
        ),
    )
    return EvaluationResult(
        scenario="online_network_error",
        duration_ms=(perf_counter() - started_at) * 1000,
        evaluation_type="online-special",
        model_requests=len(client.requests),
        retries=max(0, len(client.requests) - 1),
        estimated_tokens=sum(
            estimate_context_tokens(request) for request in client.requests
        ),
        actual_tokens=client.total_actual_tokens,
        model_request_durations_ms=tuple(client.durations_ms),
        events=(
            message_to_record(Message(role="user", content="测试网络异常处理")),
            {"type": "model_error", "category": error.category if error else None},
        ),
        assertions=assertions,
    )


def main() -> int:
    """处理真实在线评测命令行参数"""

    parser = argparse.ArgumentParser(description="运行 epsilon 在线冒烟评测")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="确认发起真实模型请求并可能产生费用",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_ONLINE_REPETITIONS,
        help="在线主链路重复次数，默认 7 次",
    )
    parser.add_argument(
        "--scenario",
        choices=("main", "context-compaction", "network-error"),
        default="main",
        help="在线专项场景",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-results/online.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation-results/online-report.html"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("evaluation-results/online-baseline.json"),
    )
    args = parser.parse_args()
    if not args.confirm:
        print("在线评测会发起真实模型请求，请添加 --confirm 后运行")
        return 2

    settings = load_settings()
    metadata = {
        "model_name": settings.model_name,
        "base_url": settings.base_url,
        "context_window": str(settings.context_window),
        "reserve_tokens": str(settings.reserve_tokens),
        "keep_recent_tokens": str(settings.keep_recent_tokens),
        "scenario": args.scenario,
        "scenario_version": ONLINE_SCENARIO_VERSION,
    }
    results = asyncio.run(
        run_online_suite(
            args.repetitions,
            scenario=args.scenario,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    for result in results:
        append_result(args.output, result)
    regression = None
    if args.baseline.exists():
        baseline = load_baseline(args.baseline)
        regression = compare_baseline(
            results,
            baseline,
            metadata,
        )
        if (
            regression.passed
            and all(result.passed for result in results)
            and needs_performance_baseline_refresh(results, baseline)
        ):
            write_baseline(args.baseline, create_baseline(results, metadata))
    elif all(result.passed for result in results):
        write_baseline(args.baseline, create_baseline(results, metadata))
    generate_report(args.report, results, regression)
    passed = sum(result.passed for result in results)
    print(f"online evaluation: {passed}/{len(results)} repetitions passed")
    if regression is not None:
        print(f"baseline regression: {'passed' if regression.passed else 'failed'}")
    print(f"results: {args.output}")
    print(f"report: {args.report}")
    return 0 if passed == len(results) and (regression is None or regression.passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
