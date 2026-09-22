"""测试长任务阶段验证：固定测试命令与官方 Harness 回归检查。"""

from pathlib import Path

import pytest

from core.tools.command_executor import CommandExecution
from evaluation.long_task_validation import (
    ORDERING_TESTS_COMMAND,
    run_harness_check,
    run_ordering_tests,
)
from evaluation.swebench import HarnessResult, SwebenchTask


class _FakeExecutor:
    """记录调用并返回预置退出码的假命令执行后端。"""

    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[str, Path, float]] = []

    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecution:
        self.calls.append((command, cwd, timeout_seconds))
        return CommandExecution(self.stdout, self.stderr, self.returncode)


def _task() -> SwebenchTask:
    """构造最小任务元数据。"""

    return SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")


@pytest.mark.asyncio
async def test_run_ordering_tests_passes_on_zero_exit(tmp_path: Path) -> None:
    """测试固定测试命令退出码为 0 时判为通过。"""

    executor = _FakeExecutor(0, stdout=b"OK")

    result = await run_ordering_tests(executor, tmp_path)

    assert result.kind == "ordering-tests"
    assert result.passed is True
    assert result.exit_code == 0
    assert executor.calls[0][0] == ORDERING_TESTS_COMMAND
    assert executor.calls[0][1] == tmp_path


@pytest.mark.asyncio
async def test_run_ordering_tests_fails_and_keeps_output_tail(tmp_path: Path) -> None:
    """测试非零退出码判为失败并保留输出尾部。"""

    executor = _FakeExecutor(1, stderr=b"Traceback: boom")

    result = await run_ordering_tests(executor, tmp_path)

    assert result.passed is False
    assert result.exit_code == 1
    assert "boom" in result.detail


@pytest.mark.asyncio
async def test_run_harness_check_reports_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 Harness 通过时阶段验证为 resolved。"""

    monkeypatch.setattr(
        "evaluation.long_task_validation.create_patch",
        lambda *args: (("f.py",), "patch"),
    )
    monkeypatch.setattr(
        "evaluation.long_task_validation.verify_patch",
        lambda *args: HarnessResult(True),
    )

    result = await run_harness_check(
        _task(), tmp_path / "baseline", tmp_path / "workspace", tmp_path / "root", "python"
    )

    assert result.kind == "harness"
    assert result.passed is True
    assert result.resolved is True


@pytest.mark.asyncio
async def test_run_harness_check_marks_environment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 Harness 环境失败时保留环境失败原因。"""

    monkeypatch.setattr(
        "evaluation.long_task_validation.create_patch",
        lambda *args: ((), ""),
    )
    monkeypatch.setattr(
        "evaluation.long_task_validation.verify_patch",
        lambda *args: HarnessResult(False, "官方 Harness 环境失败"),
    )

    result = await run_harness_check(
        _task(), tmp_path / "baseline", tmp_path / "workspace", tmp_path / "root", "python"
    )

    assert result.passed is False
    assert result.environment_error == "官方 Harness 环境失败"
