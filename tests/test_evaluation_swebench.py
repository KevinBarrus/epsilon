"""验证 SWE-bench 评测器的纯本地补丁处理。"""

import sys
import types
from pathlib import Path

import pytest

from core.model import ModelClientError
from core.agent_loop import AgentRunResult
from core.model import ToolCall, ToolResult
from core.session import Session
from core.tools import CommandExecution, ToolManager
from evaluation.models import EvaluationAssertion
from evaluation.swebench import _agent_prompt, _changed_files, _normalise_patch_paths, _project_guide_paths, create_patch, load_task
from evaluation.swebench import _context_builder
from evaluation.swebench import (
    DEFAULT_SWEBENCH_MAX_TOOL_ROUNDS,
    EVALUATION_COMMAND_TIMEOUT_SECONDS,
    HarnessResult,
    SWEBENCH_ENVIRONMENT_CONTRACT_VERSION,
    SwebenchTask,
    _configuration_record,
    _agent_end_record,
    _local_verification_status,
    _model_error_record,
    _official_harness_status,
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


def test_agent_prompt_states_evaluation_workspace_contract(tmp_path: Path) -> None:
    """测试评测任务提示词提供可执行的环境事实。"""

    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")

    prompt = _agent_prompt(task, tmp_path)

    assert "workspace is already selected for every tool" in prompt
    assert "run_command already runs from the repository workspace root" in prompt
    assert "search_files.path must be an existing directory" in prompt
    assert "source snapshot without Git history" in prompt
    assert "diagnose the test entry point" in prompt


def test_agent_prompt_lists_existing_repository_guides_without_reading_content(
    tmp_path: Path,
) -> None:
    """测试评测提示词只列出固定候选资料的相对路径。"""

    (tmp_path / "README.md").write_text("do not preload this", encoding="utf-8")
    (tmp_path / "CONTRIBUTING.rst").write_text("contributing", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("project", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "README.txt").write_text("tests", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore", encoding="utf-8")
    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")

    prompt = _agent_prompt(task, tmp_path)

    assert _project_guide_paths(tmp_path) == (
        "README.md",
        "CONTRIBUTING.rst",
        "pyproject.toml",
        "tests/README.txt",
    )
    assert "- README.md" in prompt
    assert "- tests/README.txt" in prompt
    assert "notes.md" not in prompt
    assert "do not preload this" not in prompt


def test_configuration_record_keeps_environment_contract_version() -> None:
    """测试评测轨迹记录环境引导版本，便于复跑归因。"""

    assert _configuration_record(40) == {
        "type": "configuration",
        "max_tool_rounds": 40,
        "swebench_environment_contract": SWEBENCH_ENVIRONMENT_CONTRACT_VERSION,
    }


def test_agent_end_record_keeps_write_verification_trace() -> None:
    """测试评测轨迹保留写后验证提醒和命令结果。"""

    record = _agent_end_record(
        AgentRunResult(
            (),
            "完成",
            verification_reminder_injected=True,
            write_count=2,
            post_write_command_results=(
                ToolResult("check-1", "failed", is_error=True, error_category="tool"),
            ),
        )
    )

    assert record["verification_reminder_injected"] is True
    assert record["write_count"] == 2
    assert record["post_write_command_results"] == [
        {"call_id": "check-1", "is_error": True, "error_category": "tool"}
    ]


def test_verification_statuses_keep_local_and_official_results_separate() -> None:
    """测试宿主命令失败不会被误写为官方 Harness 环境失败。"""

    run = AgentRunResult(
        (),
        "完成",
        write_count=1,
        post_write_command_results=(ToolResult("check-1", "failed", is_error=True),),
    )

    assert _local_verification_status(run) == "failed"
    assert _official_harness_status(HarnessResult(False)) == "failed"
    assert _official_harness_status(HarnessResult(False, "Docker failed")) == "environment-error"


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
        [{"type": "tool_batch", "execution_mode": "parallel", "duration_ms": 12.5}],
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
    assert result.parallel_tool_batches == 1
    assert result.tool_batch_duration_ms == 12.5


def test_swebench_result_keeps_verification_classification() -> None:
    """测试结果分别保存本地命令和官方 Harness 的分类。"""

    task = SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")
    result = _result(
        task,
        0.0,
        None,
        [],
        (EvaluationAssertion("official-harness", False, "官方 Harness 未通过"),),
        (),
        False,
        False,
        None,
        error_stage="official-harness",
        agent_execution_environment="official-instance-container",
        agent_verification_status="failed",
        official_harness_status="failed",
    )

    assert result.error_stage == "official-harness"
    assert result.agent_execution_environment == "official-instance-container"
    assert result.agent_verification_status == "failed"
    assert result.official_harness_status == "failed"


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


def test_load_task_keeps_official_instance_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试任务加载保留官方 Harness 提供的实例镜像。"""

    dataset_module = types.ModuleType("datasets")
    dataset_module.load_dataset = lambda *args, **kwargs: [
        {
            "instance_id": "example__1",
            "repo": "example/repo",
            "base_commit": "base",
            "problem_statement": "issue",
            "image": "sweb.eval.x86_64.example__1",
        }
    ]
    monkeypatch.setitem(sys.modules, "datasets", dataset_module)

    task = load_task("example__1", "swebench-lite")

    assert task.instance_image == "sweb.eval.x86_64.example__1"


@pytest.mark.asyncio
async def test_tool_manager_passes_injected_executor_to_run_command(tmp_path: Path) -> None:
    """测试评测工具管理器会将命令交给评测执行后端。"""

    class FakeExecutor:
        def __init__(self) -> None:
            self.command: str | None = None

        async def execute(
            self,
            command: str,
            cwd: Path,
            timeout_seconds: float,
        ) -> CommandExecution:
            self.command = command
            return CommandExecution(b"container output", b"", 0)

    executor = FakeExecutor()
    manager = _tool_manager(tmp_path, executor)

    result = await manager.execute(ToolCall("call-1", "run_command", {"command": "pwd"}))

    assert result.content == "container output"
    assert executor.command == "pwd"
