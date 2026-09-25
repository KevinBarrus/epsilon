"""验证带 Goal 的大任务评测只认独立证据。"""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evaluation.big_task_single_goal import (
    TOKEN_FUSE, TIME_FUSE_SECONDS, TASK, DELEGATED_TASK, coverage, completion_verdict,
    usage_breakdown, prepare_copy, typecheck, BudgetedClient, TokenBudgetReached,
    peak_worker_concurrency, run_with_wall_clock,
)
from core.agent_loop import AgentLoopCancelled
from core.model import UsageEvent, UsageLedger, UsageTrackingClient
from core.subagent import SubagentRunMetrics
from core.worktree import commit_worktree
from evaluation.online import TimedModelClient


def test_big_task_only_has_high_safety_fuses() -> None:
    """评测不再用 15M 小预算人为截断任务。"""
    assert TOKEN_FUSE == 120_000_000
    assert TIME_FUSE_SECONDS == 10_800


def test_coverage_excludes_dependencies_and_maps_modules(tmp_path: Path) -> None:
    """node_modules 不计入产物，__init__ 映射到 index.ts。"""
    for name in ("ts/core/agent.ts", "ts/src/tools/index.ts", "ts/core/missing.d.ts", "ts/node_modules/pkg/extra.ts"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("export {};", encoding="utf-8")

    state = coverage(tmp_path, ["agent.py", "tools/__init__.py", "missing.py"])

    assert state["ts_file_count"] == 2
    assert state["py_modules_converted"] == 2
    assert state["modules_without_matching_ts"] == ["missing.py"]


def test_completion_verdict_flags_false_claims() -> None:
    """模型 complete 但模块缺失或 tsc 失败，都必须记为虚报。"""
    assert completion_verdict("complete", 72, 73, True) == "false_completion_claim"
    assert completion_verdict("complete", 73, 73, False) == "false_completion_claim"
    assert completion_verdict("complete", 73, 73, True) == "structurally_complete"
    assert completion_verdict("budget_limited", 72, 73, False) == "budget_limited_incomplete"


def test_delegated_task_preserves_baseline_and_requires_worker_reviewer() -> None:
    """委派档只追加协作要求，迁移目标仍与单 Agent 一致。"""
    assert DELEGATED_TASK.startswith(TASK)
    assert "spawn_worker" in DELEGATED_TASK
    assert "spawn_reviewer" in DELEGATED_TASK


def test_usage_breakdown_counts_each_child_including_compaction() -> None:
    """分角色统计从客户端请求取数，不能漏掉子 Agent 的压缩请求。"""
    parent = TimedModelClient(None)
    worker = TimedModelClient(None)
    reviewer = TimedModelClient(None)
    parent.usages = [UsageEvent(10, 2, 12)]
    worker.usages = [UsageEvent(20, 3, 23), UsageEvent(5, 1, 6)]
    reviewer.usages = [UsageEvent(8, 2, 10)]
    split = usage_breakdown(parent, {"worker": [worker], "reviewer": [reviewer]})
    assert split["parent"] == 12
    assert split["worker"] == [29]
    assert split["reviewer"] == [10]


def test_isolated_copy_uses_fixed_source_and_git_baseline(tmp_path: Path, monkeypatch) -> None:
    """并行档仍用旧版任务源码，Git 基线包含被项目忽略的 AGENTS。"""
    from evaluation import big_task_single_goal as benchmark

    source = tmp_path / "source"
    (source / "src/core").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "src/core/agent_loop.py").write_text("new = True\n", encoding="utf-8")
    (source / "src/core/worktree.py").write_text("extra = True\n", encoding="utf-8")
    (source / "AGENTS.md").write_text("instructions", encoding="utf-8")
    (source / ".gitignore").write_text("AGENTS.md\n", encoding="utf-8")
    old_core = tmp_path / "old-core"
    old_core.mkdir()
    (old_core / "agent_loop.py").write_text("old = True\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "SOURCE", source)

    with pytest.raises(ValueError, match="archived source_core"):
        prepare_copy(isolation_enabled=True)

    workspace = prepare_copy(isolation_enabled=True, source_core=old_core)

    assert (workspace / "src/core/agent_loop.py").read_text(encoding="utf-8") == "old = True\n"
    assert not (workspace / "src/core/worktree.py").exists()
    assert (workspace / ".git").is_dir()
    assert subprocess.run(["git", "-C", str(workspace), "status", "--porcelain"], capture_output=True, check=True).stdout == b""
    assert subprocess.run(["git", "-C", str(workspace), "ls-files", "AGENTS.md"], capture_output=True, check=True).stdout.strip() == b"AGENTS.md"
    (workspace / "ts").mkdir()
    (workspace / "ts/AGENTS.md").write_text("new instructions", encoding="utf-8")
    (workspace / ".npm/_logs").mkdir(parents=True)
    (workspace / ".npm/_logs/install.log").write_text("generated cache", encoding="utf-8")
    (workspace / "node_modules/pkg").mkdir(parents=True)
    (workspace / "node_modules/pkg/index.js").write_text("dep", encoding="utf-8")
    commit_worktree(workspace)
    assert subprocess.run(["git", "-C", str(workspace), "show", "HEAD:ts/AGENTS.md"],
                          capture_output=True, check=True).stdout == b"new instructions"
    assert subprocess.run(["git", "-C", str(workspace), "ls-files", ".npm"],
                          capture_output=True, check=True).stdout == b""
    assert subprocess.run(["git", "-C", str(workspace), "ls-files", "node_modules"],
                          capture_output=True, check=True).stdout == b""
    assert json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))["source_modules"] == ["agent_loop.py"]


@pytest.mark.skipif(shutil.which("tsc") is None, reason="tsc not installed")
def test_typecheck_accepts_root_tsconfig(tmp_path: Path) -> None:
    """根目录 tsconfig 同样是有效的 TypeScript 工程布局。"""
    (tmp_path / "ts").mkdir()
    (tmp_path / "ts/x.ts").write_text("export const x: number = 1;\n", encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions":{"strict":true,"noEmit":true},"include":["ts/**/*.ts"]}', encoding="utf-8")
    assert typecheck(tmp_path)["tsc_pass"] is True


@pytest.mark.asyncio
async def test_budget_guard_rejects_new_child_request() -> None:
    """共享总账越线后，子 Agent 不能继续发新模型请求。"""
    class DummyClient:
        calls = 0

        async def stream_response(self, messages, tools=(), thinking_level=None):
            self.calls += 1
            yield UsageEvent(1, 1, 2)

    ledger = UsageLedger()
    first = DummyClient()
    first_client = BudgetedClient(UsageTrackingClient(first, ledger), ledger, 2)
    async for _ in first_client.stream_response([]):
        pass
    assert ledger.total_tokens == 2
    dummy = DummyClient()
    client = BudgetedClient(UsageTrackingClient(dummy, ledger), ledger, 2)
    with pytest.raises(TokenBudgetReached):
        async for _ in client.stream_response([]):
            pass
    assert dummy.calls == 0


@pytest.mark.asyncio
async def test_budget_guard_can_be_disabled_for_approved_two_hour_run() -> None:
    """用户允许超出 50M 时，模型请求继续计账但不再被 token 熔断。"""
    class DummyClient:
        async def stream_response(self, messages, tools=(), thinking_level=None):
            yield UsageEvent(1, 1, 2)

    ledger = UsageLedger()
    client = BudgetedClient(UsageTrackingClient(DummyClient(), ledger), ledger, None)
    for _ in range(2):
        async for _ in client.stream_response([]):
            pass
    assert ledger.total_tokens == 4


def test_peak_worker_concurrency_uses_actual_run_intervals() -> None:
    """真实 Worker 执行区间重叠才算并行，不只看父模型一次发了几条调用。"""
    metrics = [
        SubagentRunMetrics("a", "completed", 0, 2000, 0, 0, "worker", started_at=1, finished_at=3),
        SubagentRunMetrics("b", "completed", 0, 2000, 0, 0, "worker", started_at=2, finished_at=4),
        SubagentRunMetrics("c", "completed", 0, 1000, 0, 0, "worker", started_at=4, finished_at=5),
    ]
    assert peak_worker_concurrency(metrics) == 2


@pytest.mark.asyncio
async def test_wall_clock_turns_agent_loop_cancel_into_timeout() -> None:
    """批量工具在墙钟熔断时包装取消异常，也要进入预算收尾路径。"""
    async def wrapped_cancel():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError as exc:
            raise AgentLoopCancelled(()) from exc

    with pytest.raises(TimeoutError):
        await run_with_wall_clock(wrapped_cancel(), 0.01)
