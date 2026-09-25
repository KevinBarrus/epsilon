"""只读汇总实验主程序的测试（不调模型）。"""

import json
from pathlib import Path

from core.subagent import SubagentRunMetrics
from evaluation.read_summary_compare import (
    ARMS,
    DEFAULT_SCOUT_MODE,
    copy_repository,
    peak_concurrency,
    prepare,
    read_duplication,
    repository_hash,
    scout_summary,
)


def _metric(started: float, finished: float) -> SubagentRunMetrics:
    """构造一条只有运行区间的子 Agent 指标。"""

    return SubagentRunMetrics(
        call_id=f"c-{started}",
        outcome="completed",
        total_tokens=10,
        duration_ms=(finished - started) * 1000,
        summary_chars=0,
        context_chars=0,
        started_at=started,
        finished_at=finished,
    )


def test_copy_repository_skips_excluded_dirs_and_large_files(tmp_path: Path) -> None:
    """复制时跳过元数据目录与超过 1MB 的文件。"""

    source = tmp_path / "source"
    (source / "docs").mkdir(parents=True)
    (source / "docs/index.md").write_text("hello", encoding="utf-8")
    (source / "node_modules/pkg").mkdir(parents=True)
    (source / "node_modules/pkg/index.js").write_text("dep", encoding="utf-8")
    (source / ".venv/lib").mkdir(parents=True)
    (source / ".venv/lib/x.py").write_text("venv", encoding="utf-8")
    (source / "big.bin").write_bytes(b"0" * 1_000_001)

    target = tmp_path / "copy"
    stats = copy_repository(source, target)

    assert stats == {"files": 1, "bytes": 5, "skipped_large": 1}
    assert (target / "docs/index.md").read_text(encoding="utf-8") == "hello"
    assert not (target / "node_modules").exists()
    assert not (target / ".venv").exists()
    assert not (target / "big.bin").exists()


def test_repository_hash_ignores_excluded_dirs(tmp_path: Path) -> None:
    """哈希不受排除目录影响，内容变化则哈希变化。"""

    (tmp_path / "a.md").write_text("one", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/x.js").write_text("dep", encoding="utf-8")
    first = repository_hash(tmp_path)

    (tmp_path / "node_modules/x.js").write_text("changed dep", encoding="utf-8")
    assert repository_hash(tmp_path) == first

    (tmp_path / "a.md").write_text("two", encoding="utf-8")
    assert repository_hash(tmp_path) != first


def test_prepare_writes_baseline(tmp_path: Path) -> None:
    """prepare 建副本并写 baseline，含档位与两侧哈希。"""

    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")

    workspace = prepare("single", source)
    try:
        baseline = json.loads((workspace.parent / "baseline.json").read_text(encoding="utf-8"))
        assert baseline["arm"] == "single"
        assert baseline["source_hash"] == baseline["copy_hash"]
        assert baseline["files"] == 1
        assert workspace.name == "workspace"
        assert workspace.parent.name.startswith("epsilon-read-single-")
    finally:
        import shutil

        shutil.rmtree(workspace.parent, ignore_errors=True)


def test_task_includes_source_questions_for_every_arm() -> None:
    """源码级问题段对所有档完全相同，只有档位句不同。"""

    from evaluation.read_summary_compare import ARMS, SOURCE_QUESTIONS, TASK, _ARM_SENTENCE

    tasks = {arm: TASK + _ARM_SENTENCE.get(arm, "") for arm in ARMS}
    assert all(SOURCE_QUESTIONS in task for task in tasks.values())
    # 去掉档位句后三档任务完全一致
    stripped = {task.replace(_ARM_SENTENCE.get(arm, ""), "") for arm, task in tasks.items()}
    assert len(stripped) == 1


def test_default_scout_mode_matches_arm() -> None:
    """三档各自的默认子 Agent 模式：A 无委派、B fresh 扇出、C fork。"""

    assert ARMS == ("single", "fanout_fresh", "map_then_fork")
    assert DEFAULT_SCOUT_MODE["single"] is None
    assert DEFAULT_SCOUT_MODE["fanout_fresh"] == "fresh"
    assert DEFAULT_SCOUT_MODE["map_then_fork"] == "fork"


def test_scout_summary_aggregates_by_mode() -> None:
    """按模式汇总 token、缓存命中与读调用次数。"""

    metrics = [
        SubagentRunMetrics(
            "s1", "completed", 100, 1.0, 0, 0, mode="fresh",
            cache_hit_tokens=20, prompt_tokens=80,
        ),
        SubagentRunMetrics(
            "s2", "completed", 300, 1.0, 0, 0, mode="fork",
            cache_hit_tokens=200, prompt_tokens=250,
        ),
    ]
    child_events = [
        {"type": "batch", "agent_run_id": "s1", "calls": [{"tool": "read_file"}]},
        {
            "type": "batch",
            "agent_run_id": "s2",
            "calls": [{"tool": "read_file"}, {"tool": "run_command"}],
        },
    ]

    summary = scout_summary(metrics, child_events)

    assert summary["fresh"]["read_calls"] == 1
    assert summary["fresh"]["cache_hit_rate"] == 0.25
    assert summary["fork"]["read_calls"] == 1
    assert summary["fork"]["cache_hit_rate"] == 0.8
    assert summary["fork"]["tokens"] == 300


def test_peak_concurrency_uses_overlapping_intervals() -> None:
    """并发峰值按真实区间重叠计算。"""

    metrics = [_metric(1, 4), _metric(2, 5), _metric(3, 4), _metric(10, 11)]
    assert peak_concurrency(metrics) == 3


def test_read_duplication_counts_repeat_reads() -> None:
    """重复读率按“同参数读了多少次”计算。"""

    from core.loop_guard import action_digest

    same = action_digest("read_file", {"path": "a.md"})
    parent_events = [
        {"type": "tool_started", "name": "read_file", "arguments": {"path": "a.md"}},
        {"type": "tool_started", "name": "list_files", "arguments": {"path": "."}},
    ]
    child_events = [
        {
            "type": "batch",
            "agent_run_id": "s1",
            "calls": [
                {"call_id": "c1", "tool": "read_file", "args_digest": same},
            ],
        },
        {
            "type": "batch",
            "agent_run_id": "s2",
            "calls": [
                {"call_id": "c2", "tool": "read_file", "args_digest": same},
                {"call_id": "c3", "tool": "read_file", "args_digest": "d-b"},
            ],
        },
    ]

    result = read_duplication(parent_events, child_events)

    # 父读 a、s1 读 a、s2 读 a 与 b：共 4 次读，唯一 2 份
    assert result["total_reads"] == 4
    assert result["unique_reads"] == 2
    assert result["duplication_rate"] == 0.5
    assert result["agents_with_reads"] == 3
