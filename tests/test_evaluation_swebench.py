"""验证 SWE-bench 评测器的纯本地补丁处理。"""

from pathlib import Path

import pytest

from core.model import ModelClientError
from core.session import Session
from core.tools import ToolManager
from evaluation.swebench import _changed_files, _normalise_patch_paths, create_patch
from evaluation.swebench import _context_builder
from evaluation.swebench import (
    DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS,
    EVALUATION_COMMAND_TIMEOUT_SECONDS,
    HarnessResult,
    SwebenchTask,
    _model_error_record,
    _result,
    _tool_manager,
    prepare_repository,
    verify_patch,
)


def test_create_patch_uses_repository_relative_paths(tmp_path: Path) -> None:
    """测试临时目录中的文件修改会变成可应用的相对补丁。"""

    baseline = tmp_path / "baseline"
    workspace = tmp_path / "workspace"
    baseline.mkdir()
    workspace.mkdir()
    (baseline / "module.py").write_text("value = 1\n", encoding="utf-8")
    (workspace / "module.py").write_text("value = 2\n", encoding="utf-8")

    changed_files, patch = create_patch(baseline, workspace)

    assert "a/module.py" in patch
    assert "b/module.py" in patch
    assert changed_files == ("module.py",)


def test_create_patch_ignores_python_runtime_cache(tmp_path: Path) -> None:
    """测试运行时产生的字节码不会成为待验证补丁。"""

    baseline = tmp_path / "baseline"
    workspace = tmp_path / "workspace"
    baseline.mkdir()
    (workspace / "__pycache__").mkdir(parents=True)
    (workspace / "__pycache__" / "module.pyc").write_bytes(b"cache")

    changed_files, patch = create_patch(baseline, workspace)

    assert changed_files == ()
    assert patch == ""


def test_normalise_patch_paths_preserves_standard_headers(tmp_path: Path) -> None:
    """测试路径归一化不破坏标准补丁头部。"""

    baseline = tmp_path / "baseline"
    workspace = tmp_path / "workspace"
    patch = (
        f"diff --git a/{baseline.resolve().as_posix().lstrip('/')}/a.py "
        f"b/{workspace.resolve().as_posix().lstrip('/')}/a.py\n"
        f"--- a/{baseline.resolve().as_posix().lstrip('/')}/a.py\n"
        f"+++ b/{workspace.resolve().as_posix().lstrip('/')}/a.py\n"
    )

    normalised = _normalise_patch_paths(patch, baseline, workspace)

    assert normalised == "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    assert _changed_files(normalised) == ("a.py",)


@pytest.mark.asyncio
async def test_evaluation_context_uses_production_system_prompt(tmp_path: Path) -> None:
    """测试真实评测请求与生产入口一样包含系统提示词。"""

    session = Session(tmp_path)
    session.add_user_message("修复一个问题")
    build_context = _context_builder(
        session,
        object(),  # type: ignore[arg-type]
        False,
        [],
        "test-model",
        ToolManager(),
    )

    result = await build_context(session.get_messages(), False)

    assert result.messages[0].role == "system"
    assert "test-model" in result.messages[0].content
    session.close()


def test_model_request_error_keeps_model_failure_category() -> None:
    """测试模型请求异常不会被误记为环境失败。"""

    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")
    result = _result(
        task,
        0.0,
        None,
        [],
        (),
        (),
        False,
        False,
        "agent-loop: ModelClientError: failed",
        "model",
    )

    assert result.error_category == "model"


def test_swebench_result_keeps_agent_stop_trace() -> None:
    """测试真实任务结果保留 Agent 的回合数和停止原因。"""

    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")

    result = _result(
        task,
        0.0,
        None,
        [],
        (),
        (),
        False,
        False,
        None,
        tool_rounds=4,
        stop_reason="tool_limit",
    )

    assert result.tool_rounds == 4
    assert result.stop_reason == "tool_limit"


def test_model_error_record_omits_request_content() -> None:
    """测试模型失败轨迹只保留类别和异常类型。"""

    event = _model_error_record(
        ModelClientError("safe message", category="network", retryable=True, cause=ValueError("secret"))
    )

    assert event == {
        "type": "model_error",
        "category": "network",
        "retryable": True,
        "cause_type": "ValueError",
    }


def test_prepare_repository_skips_fetch_when_commit_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试已有基线提交时不会重复访问远端仓库。"""

    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")
    repository = tmp_path / "example__repo"
    repository.mkdir()
    monkeypatch.setattr("evaluation.swebench._has_commit", lambda *_: True)
    calls: list[list[str]] = []
    monkeypatch.setattr("evaluation.swebench._run_git", calls.append)

    assert prepare_repository(task, tmp_path) == repository
    assert calls == []


def test_evaluation_command_uses_longer_timeout(tmp_path: Path) -> None:
    """测试评测命令可容纳仓库自身的较长测试。"""

    definition = next(
        item for item in _tool_manager(tmp_path).list_definitions() if item.name == "run_command"
    )

    assert definition.name == "run_command"
    assert EVALUATION_COMMAND_TIMEOUT_SECONDS == 300.0
    assert DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS == 40


def test_verify_patch_marks_harness_exit_without_report_as_environment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 Harness 未产出报告且非零退出时不会被误判为 Agent 失败。"""

    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")

    class Completed:
        returncode = 17
        stdout = "harness failed"

    monkeypatch.setattr("evaluation.swebench.subprocess.run", lambda *args, **kwargs: Completed())

    assert verify_patch(task, "", tmp_path, "python") == HarnessResult(
        False, "官方 Harness 执行失败，退出码 17"
    )
