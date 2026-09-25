"""大语料只读实验评分模块的测试。"""

from pathlib import Path

from evaluation.big_repo_read_score import (
    SUBSYSTEM_CHECKLIST,
    coverage_by_subsystem,
    score,
)


def test_checklist_shape() -> None:
    """清单按子系统组织，合计 30 条，每子系统 2~3 条。"""

    total = sum(len(items) for items in SUBSYSTEM_CHECKLIST.values())
    assert total == 30
    assert 8 <= len(SUBSYSTEM_CHECKLIST) <= 12
    for name, items in SUBSYSTEM_CHECKLIST.items():
        assert 2 <= len(items) <= 4, name
        assert all(item.key.startswith(name.split("-")[0]) or item.key.split("_")[0] for item in items)


def test_empty_report_scores_zero() -> None:
    """空报告源码级覆盖率为 0。"""

    result = coverage_by_subsystem("")

    assert result["matched_count"] == 0
    assert result["rate"] == 0.0
    assert result["hit_subsystems"] == []


def test_source_report_matches_identifiers() -> None:
    """提到源码标识符的报告会被命中，且按子系统归类。"""

    text = (
        "core 用 MAX_CONCURRENT_ANCESTOR_PROBES 限制并发探测；"
        "app-server 的超时退出码常量是 EXEC_TIMEOUT_EXIT_CODE；"
        "MCP 的 stdout 上限常量叫 MAX_MCP_STDOUT_LINE_BYTES。"
    )
    result = coverage_by_subsystem(text)

    assert result["matched_count"] == 3
    assert set(result["hit_subsystems"]) == {"core", "app-server", "mcp"}
    assert result["subsystems"]["core"]["matched"] == ["core_agents_probe"]
    assert "core_agents_separator" in result["subsystems"]["core"]["missing"]


def test_doc_level_wording_does_not_match() -> None:
    """只复述文档级说法（不带标识符）不会被命中。"""

    text = "codex 支持 MCP、沙箱、app-server 与 SQLite 持久化，使用 bwrap 做隔离。"
    result = coverage_by_subsystem(text)

    assert result["matched_count"] == 0


def test_score_returns_subsystem_coverage_and_precision(tmp_path: Path) -> None:
    """score 同时给出子系统覆盖率与路径精确率。"""

    (tmp_path / "codex-rs").mkdir()
    (tmp_path / "codex-rs/core.rs").write_text("x", encoding="utf-8")
    result = score("见 codex-rs/core.rs 与 codex-rs/ghost.rs", tmp_path)

    assert set(result) == {"coverage_source", "precision"}
    assert result["precision"]["rate"] == 0.5
    assert "subsystems" in result["coverage_source"]
