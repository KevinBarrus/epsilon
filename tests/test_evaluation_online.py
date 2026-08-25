import pytest

from core.config import ConfigError, Settings
from core.model import Message, TextDelta, ToolCall, ToolCallEvent, UsageEvent
from evaluation.fakes import FakeModelClient
from evaluation.models import EvaluationAssertion, EvaluationResult
from evaluation.online import (
    DEFAULT_ONLINE_REPETITIONS,
    ONLINE_CODE_TASKS,
    ONLINE_FILE_TASKS,
    TimedModelClient,
    _has_expected_file_read_before_edit,
    run_online_code_task,
    run_online_file_task,
    run_online_suite,
)


@pytest.mark.asyncio
async def test_timed_model_client_records_request_duration() -> None:
    """测试真实客户端包装器记录请求和耗时"""

    client = TimedModelClient(FakeModelClient([[TextDelta("完成")]]))

    events = [event async for event in client.stream_response([])]

    assert events == [TextDelta("完成")]
    assert len(client.requests) == 1
    assert len(client.durations_ms) == 1
    assert client.durations_ms[0] >= 0


@pytest.mark.asyncio
async def test_timed_model_client_closes_wrapped_client() -> None:
    """测试评测包装器会关闭底层网络客户端。"""

    class ClosableClient(FakeModelClient):
        def __init__(self) -> None:
            super().__init__([])
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    wrapped = ClosableClient()
    client = TimedModelClient(wrapped)

    await client.close()

    assert wrapped.closed is True


@pytest.mark.asyncio
async def test_timed_model_client_records_summary_request_duration() -> None:
    """测试客户端包装器记录摘要请求耗时"""

    client = TimedModelClient(FakeModelClient([[TextDelta("摘要")]]))

    chunks = [chunk async for chunk in client.stream_chat([Message("user", "历史")])]

    assert chunks == ["摘要"]
    assert len(client.requests) == 1
    assert len(client.durations_ms) == 1


@pytest.mark.asyncio
async def test_timed_model_client_collects_usage() -> None:
    """测试客户端包装器汇总各请求的实际 total Token。"""

    client = TimedModelClient(
        FakeModelClient([[TextDelta("完成"), UsageEvent(10, 2, 12)]])
    )

    events = [event async for event in client.stream_response([])]

    assert events == [TextDelta("完成"), UsageEvent(10, 2, 12)]
    assert client.total_actual_tokens == 12


@pytest.mark.asyncio
async def test_timed_model_client_returns_none_when_usage_missing() -> None:
    """测试任一请求缺少 usage 时汇总结果保持 None。"""

    client = TimedModelClient(FakeModelClient([[TextDelta("完成")]]))

    _ = [event async for event in client.stream_response([])]

    assert client.total_actual_tokens is None


@pytest.mark.asyncio
async def test_timed_model_client_collects_usage_from_summary_request() -> None:
    """测试摘要请求的 usage 也会被收集并只向调用方返回文本。"""

    client = TimedModelClient(
        FakeModelClient([[TextDelta("摘要"), UsageEvent(5, 1, 6)]])
    )

    chunks = [chunk async for chunk in client.stream_chat([Message("user", "历史")])]

    assert chunks == ["摘要"]
    assert client.total_actual_tokens == 6


@pytest.mark.asyncio
async def test_online_suite_runs_requested_repetitions_and_keeps_failures(
    monkeypatch,
) -> None:
    """测试在线主任务会运行全部文件任务和代码任务并保留失败。"""

    calls: list[str] = []

    async def fake_run_file(task):
        calls.append(task.name)
        if task.name == "online_multi_file_edit":
            raise RuntimeError("模拟在线失败")
        return EvaluationResult(
            scenario=task.name,
            duration_ms=10,
            evaluation_type="real-task",
            assertions=(EvaluationAssertion("ok", True),),
        )

    async def fake_run_code(task):
        calls.append(task.name)
        return EvaluationResult(
            scenario=task.name,
            duration_ms=10,
            evaluation_type="code-correctness",
            assertions=(EvaluationAssertion("ok", True),),
        )

    monkeypatch.setattr("evaluation.online.run_online_file_task", fake_run_file)
    monkeypatch.setattr("evaluation.online.run_online_code_task", fake_run_code)

    results = await run_online_suite(repetitions=1)

    expected_names = [task.name for task in ONLINE_FILE_TASKS] + [
        task.name for task in ONLINE_CODE_TASKS
    ]
    assert calls == expected_names
    assert [result.repetition for result in results] == [1] * len(expected_names)
    assert [result.passed for result in results] == [True, False, True, True, True]


@pytest.mark.asyncio
async def test_online_suite_default_repetitions_reach_performance_sample_count(
    monkeypatch,
) -> None:
    """测试默认主套件会生成至少二十个性能样本。"""

    async def fake_run_file(task):
        return EvaluationResult(
            scenario=task.name,
            duration_ms=10,
            evaluation_type="real-task",
            assertions=(EvaluationAssertion("ok", True),),
        )

    async def fake_run_code(task):
        return EvaluationResult(
            scenario=task.name,
            duration_ms=10,
            evaluation_type="code-correctness",
            assertions=(EvaluationAssertion("ok", True),),
        )

    monkeypatch.setattr("evaluation.online.run_online_file_task", fake_run_file)
    monkeypatch.setattr("evaluation.online.run_online_code_task", fake_run_code)

    results = await run_online_suite()

    assert DEFAULT_ONLINE_REPETITIONS == 7
    total_tasks = len(ONLINE_FILE_TASKS) + len(ONLINE_CODE_TASKS)
    assert len(results) == DEFAULT_ONLINE_REPETITIONS * total_tasks
    assert len(results) >= 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "responses"),
    [
        (
            ONLINE_FILE_TASKS[0],
            [
                [ToolCallEvent(ToolCall("read", "read_file", {"path": "note.txt"}))],
                [
                    ToolCallEvent(
                        ToolCall(
                            "edit",
                            "edit_file",
                            {
                                "path": "note.txt",
                                "old_content": "before\n",
                                "new_content": "after\n",
                            },
                        )
                    )
                ],
                [TextDelta("note.txt 已完成修改")],
            ],
        ),
        (
            ONLINE_FILE_TASKS[1],
            [
                [ToolCallEvent(ToolCall("read-config", "read_file", {"path": "config.txt"}))],
                [
                    ToolCallEvent(
                        ToolCall(
                            "edit-config",
                            "edit_file",
                            {
                                "path": "config.txt",
                                "old_content": "old-config\n",
                                "new_content": "new-config\n",
                            },
                        )
                    )
                ],
                [ToolCallEvent(ToolCall("read-note", "read_file", {"path": "note.txt"}))],
                [
                    ToolCallEvent(
                        ToolCall(
                            "edit-note",
                            "edit_file",
                            {
                                "path": "note.txt",
                                "old_content": "old-note\n",
                                "new_content": "new-note\n",
                            },
                        )
                    ),
                ],
                [TextDelta("config.txt 和 note.txt 已完成修改")],
            ],
        ),
        (
            ONLINE_FILE_TASKS[2],
            [
                [ToolCallEvent(ToolCall("missing", "read_file", {"path": "missing.txt"}))],
                [ToolCallEvent(ToolCall("read", "read_file", {"path": "note.txt"}))],
                [
                    ToolCallEvent(
                        ToolCall(
                            "edit",
                            "edit_file",
                            {
                                "path": "note.txt",
                                "old_content": "before\n",
                                "new_content": "after\n",
                            },
                        )
                    )
                ],
                [TextDelta("note.txt 已完成修改")],
            ],
        ),
    ],
)
async def test_online_file_tasks_validate_expected_agent_behavior(
    task,
    responses,
    monkeypatch,
) -> None:
    """测试三类真实任务均验证文件、工具顺序和最终回复。"""

    client = FakeModelClient(responses)
    monkeypatch.setattr(
        "evaluation.online.load_settings",
        lambda: Settings("https://example.com", "test", "key"),
    )
    monkeypatch.setattr("evaluation.online.OpenAICompatibleClient", lambda settings: client)

    result = await run_online_file_task(task)

    assert result.passed


def test_online_file_task_requires_each_target_read_before_edit() -> None:
    """测试多文件任务允许交错读写，但拒绝先编辑目标文件。"""

    task = ONLINE_FILE_TASKS[1]
    interleaved_events = [
        {"type": "tool_call", "name": "read_file", "arguments": {"path": "config.txt"}},
        {"type": "tool_call", "name": "edit_file", "arguments": {"path": "config.txt"}},
        {"type": "tool_call", "name": "read_file", "arguments": {"path": "note.txt"}},
        {"type": "tool_call", "name": "edit_file", "arguments": {"path": "note.txt"}},
    ]
    edit_first_events = [
        {"type": "tool_call", "name": "edit_file", "arguments": {"path": "config.txt"}},
        {"type": "tool_call", "name": "read_file", "arguments": {"path": "config.txt"}},
        {"type": "tool_call", "name": "read_file", "arguments": {"path": "note.txt"}},
        {"type": "tool_call", "name": "edit_file", "arguments": {"path": "note.txt"}},
    ]

    assert _has_expected_file_read_before_edit(interleaved_events, task)
    assert not _has_expected_file_read_before_edit(edit_first_events, task)


@pytest.mark.asyncio
async def test_online_code_task_validates_pytest_result(monkeypatch) -> None:
    """测试代码正确性任务以独立 pytest 结果判断成功。"""

    task = ONLINE_CODE_TASKS[0]
    client = FakeModelClient(
        [
            [
                ToolCallEvent(ToolCall("read-1", "read_file", {"path": "math_utils.py"})),
                ToolCallEvent(ToolCall("read-2", "read_file", {"path": "test_math_utils.py"})),
            ],
            [
                ToolCallEvent(
                    ToolCall(
                        "write-1",
                        "write_file",
                        {
                            "path": "math_utils.py",
                            "content": "def add(left, right):\n    return left + right\n",
                        },
                    )
                )
            ],
            [TextDelta("已修复 math_utils.py 并完成")],
        ]
    )
    monkeypatch.setattr(
        "evaluation.online.load_settings",
        lambda: Settings("https://example.com", "test", "key"),
    )
    monkeypatch.setattr(
        "evaluation.online.OpenAICompatibleClient",
        lambda settings: client,
    )

    result = await run_online_code_task(task)

    assert result.passed
    assert result.evaluation_type == "code-correctness"


@pytest.mark.asyncio
async def test_online_code_task_fails_when_test_file_modified(monkeypatch) -> None:
    """测试模型篡改测试文件时任务失败。"""

    task = ONLINE_CODE_TASKS[0]
    client = FakeModelClient(
        [
            [ToolCallEvent(ToolCall("read", "read_file", {"path": "math_utils.py"}))],
            [
                ToolCallEvent(
                    ToolCall(
                        "write",
                        "write_file",
                        {
                            "path": "test_math_utils.py",
                            "content": "def test_fake():\n    assert True\n",
                        },
                    )
                )
            ],
            [TextDelta("完成")],
        ]
    )
    monkeypatch.setattr(
        "evaluation.online.load_settings",
        lambda: Settings("https://example.com", "test", "key"),
    )
    monkeypatch.setattr(
        "evaluation.online.OpenAICompatibleClient",
        lambda settings: client,
    )

    result = await run_online_code_task(task)

    assert not result.passed
    assert any(
        assertion.name == "test-file-unchanged" and not assertion.passed
        for assertion in result.assertions
    )


@pytest.mark.asyncio
async def test_online_code_task_fails_when_pytest_does_not_pass(monkeypatch) -> None:
    """测试模型修复后测试仍未通过时任务失败。"""

    task = ONLINE_CODE_TASKS[0]
    client = FakeModelClient(
        [
            [ToolCallEvent(ToolCall("read", "read_file", {"path": "math_utils.py"}))],
            [
                ToolCallEvent(
                    ToolCall(
                        "write",
                        "write_file",
                        {
                            "path": "math_utils.py",
                            "content": "def add(left, right):\n    return left * right\n",
                        },
                    )
                )
            ],
            [TextDelta("完成")],
        ]
    )
    monkeypatch.setattr(
        "evaluation.online.load_settings",
        lambda: Settings("https://example.com", "test", "key"),
    )
    monkeypatch.setattr(
        "evaluation.online.OpenAICompatibleClient",
        lambda settings: client,
    )

    result = await run_online_code_task(task)

    assert not result.passed
    assert any(
        assertion.name == "pytest-passed" and not assertion.passed
        for assertion in result.assertions
    )


@pytest.mark.asyncio
async def test_online_suite_dispatches_network_error_scenario(monkeypatch) -> None:
    """测试在线评测套件能调度网络异常专项场景"""

    async def fake_run():
        return EvaluationResult(
            scenario="online_network_error",
            duration_ms=10,
            assertions=(EvaluationAssertion("ok", True),),
        )

    monkeypatch.setattr("evaluation.online.run_online_network_error_smoke", fake_run)

    results = await run_online_suite(repetitions=1, scenario="network-error")

    assert results[0].scenario == "online_network_error"
    assert results[0].passed


@pytest.mark.asyncio
async def test_online_smoke_keeps_configuration_failure_diagnostics(monkeypatch) -> None:
    """测试在线场景会保留配置失败的类别和阶段。"""

    def fail_load_settings():
        raise ConfigError("缺少模型配置")

    monkeypatch.setattr("evaluation.online.load_settings", fail_load_settings)

    from evaluation.online import run_online_smoke

    result = await run_online_smoke()

    assert result.passed is False
    assert result.error_category == "configuration"
    assert result.error_stage == "load-settings"
    assert "ConfigError" in (result.error_message or "")
    assert result.events[-1]["type"] == "evaluation_error"
