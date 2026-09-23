"""测试长任务主流程：单 Session 多阶段、验证顺序、阶段差值与落盘汇总。"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent_loop import AgentRunResult
from core.model import ToolCall, ToolCallEvent
from evaluation.long_task import (
    LongTaskSpec,
    LongTaskStage,
    _delta,
    _snapshot_workspace,
    default_long_task_spec,
    run_long_task,
)
from evaluation.long_task_validation import StageValidation
from evaluation.swebench import SwebenchTask
from evaluation.swebench_workspace import EvaluationWorkspace


def _task() -> SwebenchTask:
    """构造最小任务元数据。"""

    return SwebenchTask("example__1", "example/repo", "base", "issue", "swebench-lite")


def _harness_validation(passed: bool = True) -> StageValidation:
    """构造官方 Harness 阶段验证结论。"""

    return StageValidation(
        kind="harness",
        passed=passed,
        detail="resolved" if passed else "官方 Harness 未通过",
        resolved=passed,
    )


class _FakeTimedClient:
    """按真实 TimedModelClient 语义维护用量，空请求时返回 None。"""

    def __init__(self, client) -> None:
        self.requests: list = []
        self.usages: list = []
        self.durations_ms: list[float] = []
        self.cache_hit_rate = None

    def _completed(self):
        if not self.usages or any(usage is None for usage in self.usages):
            return None
        return list(self.usages)

    @property
    def total_actual_tokens(self):
        usages = self._completed()
        if usages is None:
            return None
        return sum(usage.total_tokens for usage in usages)

    @property
    def total_cached_tokens(self):
        usages = self._completed()
        if usages is None:
            return None
        return sum(usage.cached_tokens for usage in usages)

    async def close(self) -> None:
        pass


class _FakeAgentLoop:
    """每次 run 追加一次用量、发出一次 artifact:// 取回事件。"""

    def __init__(self, client, manager, **kwargs) -> None:
        self.client = client

    async def run(self, messages, on_event=None, build_context=None) -> AgentRunResult:
        self.client.requests.append(list(messages))
        self.client.usages.append(
            SimpleNamespace(total_tokens=1_000, cached_tokens=800)
        )
        if on_event is not None:
            await on_event(
                ToolCallEvent(
                    ToolCall("call-1", "read_file", {"path": "artifact://3"})
                )
            )
        return AgentRunResult((), "完成", stop_reason="completed", tool_rounds=2)


class _FakeContainer:
    """无副作用的任务容器替身。"""

    def __init__(self, image: str, workspace: Path) -> None:
        pass

    @asynccontextmanager
    async def running(self):
        yield "container-1"


def _install_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patches: list):
    """安装替身，返回验证调用顺序、create_patch 的 reference 序列与基线路径。"""

    baseline = tmp_path / "baseline"
    workspace = tmp_path / "workspace"
    session_root = tmp_path / "session"
    for path in (baseline, workspace, session_root):
        path.mkdir()
    workspaces = iter(
        (
            EvaluationWorkspace(baseline, session_root),
            EvaluationWorkspace(workspace, session_root),
        )
    )
    calls: list[str] = []
    references: list[Path] = []
    patch_iter = iter(patches)

    async def fake_harness(*args, **kwargs) -> StageValidation:
        calls.append("harness")
        return _harness_validation()

    async def fake_tests(*args, **kwargs) -> StageValidation:
        calls.append("tests")
        return StageValidation(
            kind="ordering-tests",
            passed=True,
            detail="exit code 0",
            command="cmd",
            exit_code=0,
        )

    def fake_create_patch(reference: Path, workspace: Path):
        assert isinstance(reference, Path), type(reference)
        assert isinstance(workspace, Path), type(workspace)
        references.append(reference)
        return next(patch_iter), "patch"

    monkeypatch.setattr("evaluation.long_task.load_task", lambda *args: _task())
    monkeypatch.setattr("evaluation.long_task.prepare_repository", lambda *args: tmp_path)
    monkeypatch.setattr(
        "evaluation.long_task.prepare_evaluation_workspace", lambda *args: next(workspaces)
    )
    monkeypatch.setattr("evaluation.long_task.SwebenchTaskContainer", _FakeContainer)
    monkeypatch.setattr(
        "evaluation.long_task.OpenAICompatibleClient", lambda settings: object()
    )
    monkeypatch.setattr("evaluation.long_task.TimedModelClient", _FakeTimedClient)
    monkeypatch.setattr("evaluation.long_task.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr(
        "evaluation.long_task.load_settings", lambda: SimpleNamespace(model_name="test-model")
    )
    monkeypatch.setattr("evaluation.long_task.run_harness_check", fake_harness)
    monkeypatch.setattr("evaluation.long_task.run_ordering_tests", fake_tests)
    monkeypatch.setattr("evaluation.long_task.create_patch", fake_create_patch)
    return calls, references, baseline


def test_delta_treats_missing_before_as_zero() -> None:
    """测试首阶段没有前置快照时仍能归因用量。"""

    assert _delta(None, 1_000) == 1_000
    assert _delta(100, 250) == 150
    assert _delta(100, None) == 0
    assert _delta(200, 150) == 0


def test_snapshot_workspace_preserves_symlinks(tmp_path: Path) -> None:
    """测试阶段快照保留符号链接，避免阶段差分出现幻影改动。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "target.txt").write_text("body", encoding="utf-8")
    (workspace / "link.txt").symlink_to("target.txt")

    snapshot = _snapshot_workspace(workspace, tmp_path / "result", "T1")

    assert (snapshot / "link.txt").is_symlink()
    assert (snapshot / "link.txt").read_text(encoding="utf-8") == "body"


@pytest.mark.asyncio
async def test_run_long_task_runs_stages_and_writes_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试三阶段顺序执行、T1 用量归因、验证顺序与 JSONL 落盘。"""

    calls, references, baseline = _install_fakes(
        monkeypatch,
        tmp_path,
        [
            ("django/db/models/sql/compiler.py",),
            ("tests/ordering/tests.py",),
            ("django/db/models/sql/compiler.py", "tests/ordering/tests.py"),
        ],
    )
    result_root = tmp_path / "result"

    result = await run_long_task(
        default_long_task_spec(_task()),
        result_root,
        "python",
        eviction_enabled=True,
        eviction_threshold_tokens=20_000,
    )

    assert [stage.stage for stage in result.stages] == ["T1", "T2", "T3"]
    assert all(stage.passed for stage in result.stages)
    # T1 的硬验证必须先于 T2 的测试验证
    assert calls == ["harness", "tests", "harness", "tests", "harness"]
    # T1 的用量必须计入（首阶段 before 为 None）
    assert [stage.actual_tokens for stage in result.stages] == [1_000, 1_000, 1_000]
    assert result.total_actual_tokens == 3_000
    assert result.total_artifact_read_calls == 3
    assert result.stages[0].cache_hit_rate == pytest.approx(0.8)

    # T1 用未修改的原始基线作起点，T2/T3 用阶段开始时的快照
    assert references[0] == baseline
    assert all("workspace-snapshots" in str(path) for path in references[1:])

    records = [
        json.loads(line)
        for line in (result_root / "long_task.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["type"] for record in records] == [
        "long_task_stage",
        "long_task_stage",
        "long_task_stage",
        "long_task_arm",
    ]
    assert records[0]["stage"] == "T1"
    assert records[0]["validation_kind"] == "harness"
    assert records[1]["validation_kind"] == "ordering-tests"
    assert records[1]["harness_passed"] is True
    assert records[3]["total_actual_tokens"] == 3_000
    assert records[3]["total_eviction_gate_rejections"] == 0


@pytest.mark.asyncio
async def test_run_long_task_fails_stage_outside_allowed_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试 T2 改动生产代码时该阶段判为失败。"""

    _install_fakes(
        monkeypatch,
        tmp_path,
        [
            ("django/db/models/sql/compiler.py",),
            ("django/db/models/sql/compiler.py",),
            ("django/db/models/sql/compiler.py",),
        ],
    )

    result = await run_long_task(
        default_long_task_spec(_task()),
        tmp_path / "result",
        "python",
        eviction_enabled=False,
        eviction_threshold_tokens=None,
    )

    assert result.stages[0].passed is True
    assert result.stages[1].passed is False
    assert result.stages[1].allowed_changes_ok is False


def test_default_long_task_spec_declares_three_stages() -> None:
    """测试默认规格包含 T1/T2/T3 与各阶段验证方式。"""

    spec = default_long_task_spec(_task())

    assert [stage.name for stage in spec.stages] == ["T1", "T2", "T3"]
    assert spec.stages[0].validation == "harness"
    assert spec.stages[1].allowed_changes == ("tests/ordering/",)
    assert "django/db/models/sql/compiler.py" in spec.stages[2].allowed_changes
    assert "tests/runtests.py ordering" in spec.stages[1].prompt
