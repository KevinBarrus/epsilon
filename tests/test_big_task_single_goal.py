"""验证带 Goal 的大任务评测只认独立证据。"""

from pathlib import Path

from evaluation.big_task_single_goal import (
    TOKEN_FUSE, TIME_FUSE_SECONDS, coverage, completion_verdict,
)


def test_big_task_only_has_high_safety_fuses() -> None:
    """评测不再用 15M 小预算人为截断任务。"""
    assert TOKEN_FUSE == 50_000_000
    assert TIME_FUSE_SECONDS == 7200


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
