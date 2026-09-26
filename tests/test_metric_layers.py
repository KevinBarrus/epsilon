"""评测产物主/辅指标分层的测试（design/evaluation.md 第十四节）。

要点：**辅助指标必须带题型构成（`item_types`）**——否则读者无法判断这把尺子的可靠度。
"""

import inspect
from pathlib import Path

from evaluation import read_summary_compare
from evaluation.big_repo_read_score import ALL_ITEMS
from evaluation.metric_layers import (
    build_layers,
    build_primary,
    build_secondary,
    item_types,
    split_coverage,
)
from evaluation.read_summary_score import CHECKLIST_SOURCE


def test_item_types_reports_composition() -> None:
    """清单题型构成：40 条 = 30 常量题 + 10 机制题。"""

    assert item_types(ALL_ITEMS) == {"constant": 30, "mechanism": 10}


def test_secondary_metrics_always_carry_item_types_and_caveat() -> None:
    """`item_types` 与 `caveat` 是必填项，缺失即测试失败。"""

    secondary = build_secondary(coverage=None, items=CHECKLIST_SOURCE)

    coverage = secondary["fact_coverage"]
    assert coverage["item_types"]  # 必填
    assert sum(coverage["item_types"].values()) == len(CHECKLIST_SOURCE)
    assert coverage["caveat"]
    assert "下界" in coverage["caveat"]


def test_split_coverage_splits_by_item_kind() -> None:
    """覆盖率能按题型拆开：机制题命中多少、常量题命中多少。"""

    coverage = {"matched": [ALL_ITEMS[0].key], "matched_count": 1, "total": 40, "rate": 0.025}
    split = split_coverage(coverage, ALL_ITEMS)

    assert split["constant"]["total"] == 30
    assert split["mechanism"]["total"] == 10
    assert split["constant"]["matched"] + split["mechanism"]["matched"] == 1


class _FakeCheck:
    """最小检查结果替身（只用到 kind / status / detail）。"""

    def __init__(self, kind: str, status: str) -> None:
        """记录类型与三态状态。"""

        self.kind = kind
        self.status = status
        self.detail = f"{kind}-{status}"


def test_primary_metrics_use_objective_checks_only() -> None:
    """主指标只由客观项构成：交付物完整性 / 无编造 / 客观通过项 / 成本。"""

    primary = build_primary(
        tokens=1234,
        wall_clock_seconds=5.5,
        tool_errors=2,
        retries=1,
        check_results=[
            _FakeCheck("sections_cover", "passed"),
            _FakeCheck("identifiers_per_section", "unverified"),
            _FakeCheck("mentioned_paths_exist", "passed"),
            _FakeCheck("command_passed", "failed"),
        ],
    )

    completeness = primary["deliverable_completeness"]
    assert completeness["passed"] == 1 and completeness["total"] == 2
    assert completeness["items"] == [
        {"kind": "sections_cover", "status": "passed"},
        {"kind": "identifiers_per_section", "status": "unverified"},
    ]
    assert primary["fabrication_free"]["status"] == "passed"
    assert primary["objective_pass"]["status"] == "failed"
    assert primary["cost"] == {
        "tokens": 1234,
        "wall_clock_seconds": 5.5,
        "tool_errors": 2,
        "retries": 1,
    }


def test_layers_keep_both_halves(tmp_path: Path) -> None:
    """`build_layers` 同时给出主/辅两层，且辅层从 coverage_source 取值。"""

    from evaluation.big_repo_read_score import score

    quality = score("# 报告\n读到 `MAX_STDOUT_BYTES`。\n", tmp_path)
    layers = build_layers(
        quality=quality, items=ALL_ITEMS, tokens=10, wall_clock_seconds=1.0
    )

    assert set(layers) == {"primary_metrics", "secondary_metrics"}
    assert layers["secondary_metrics"]["fact_coverage"]["total"] == 40
    assert "primary_metrics" in layers


def test_run_merges_layers_into_result_json() -> None:
    """接线测试：result.json 的组装必须把两层并进顶层（否则分层等于没做）。"""

    source = inspect.getsource(read_summary_compare.run)

    assert "**layers" in source
    assert "build_layers(" in source
