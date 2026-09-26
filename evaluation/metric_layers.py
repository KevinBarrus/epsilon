"""评测产物的主/辅指标分层（见 `design/evaluation.md` 第十四节）。

**主指标**：客观、可复现、与"考哪几个"无关——交付物完整性、无编造、客观通过项、成本。

**辅助指标**：事实清单覆盖率。它是**下界**：事实无穷多，命中主要反映运气与报告篇幅，
因此**必须带题型构成**（常量题 / 机制题各多少条），**不得单独支撑结论**。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SPEC_REFERENCE = "design/evaluation.md 第十四节"
SECONDARY_CAVEAT = (
    "事实无穷多，命中主要反映运气与报告篇幅；该指标为下界，不得单独支撑结论"
)
# 交付物完整性类（结构化要求逐项 pass/fail）的检查器
COMPLETENESS_KINDS = frozenset(
    {
        "path_exists",
        "sections_cover",
        "paths_per_section",
        "identifiers_per_section",
        "question_types_per_section",
        "files_changed",
    }
)
FABRICATION_KIND = "mentioned_paths_exist"
OBJECTIVE_KIND = "command_passed"


def item_kind(item: object) -> str:
    """题型：需要多个标识符的算机制题，其余算常量题（具名常量/字面值）。"""

    required = int(getattr(item, "required_hits", 1) or 1)
    return "mechanism" if required >= 2 else "constant"


def item_types(items: Sequence[object]) -> dict[str, int]:
    """清单的题型构成——报告覆盖率时**必须**同时给出它。"""

    counts = {"constant": 0, "mechanism": 0}
    for item in items:
        counts[item_kind(item)] += 1
    return counts


def _check_status(results: Sequence[object], kind: str) -> dict[str, str] | None:
    """取某一类检查的状态（passed / failed / unverified）。"""

    for result in results:
        if str(getattr(result, "kind", "")) == kind:
            return {
                "status": str(getattr(result, "status", "")),
                "detail": str(getattr(result, "detail", "")),
            }
    return None


def build_primary(
    *,
    tokens: int,
    wall_clock_seconds: float,
    tool_errors: int = 0,
    retries: int = 0,
    check_results: Sequence[object] = (),
    precision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造主指标：全部是客观、可复现的量。"""

    completeness = [
        {"kind": str(getattr(r, "kind", "")), "status": str(getattr(r, "status", ""))}
        for r in check_results
        if str(getattr(r, "kind", "")) in COMPLETENESS_KINDS
    ]
    passed = sum(1 for item in completeness if item["status"] == "passed")
    fabrication = _check_status(check_results, FABRICATION_KIND)
    if fabrication is None and precision is not None:
        fabrication = {
            "status": "unknown",
            "detail": f"引用的路径精确率 {precision.get('rate')}",
        }
    return {
        "deliverable_completeness": {
            "passed": passed,
            "total": len(completeness),
            "items": completeness,
            "note": "结构化要求逐项 pass/fail，判据在跑前写死",
        },
        "fabrication_free": fabrication,
        "objective_pass": _check_status(check_results, OBJECTIVE_KIND),
        "cost": {
            "tokens": tokens,
            "wall_clock_seconds": wall_clock_seconds,
            "tool_errors": tool_errors,
            "retries": retries,
        },
    }


def split_coverage(
    coverage: Mapping[str, object] | None,
    items: Sequence[object],
) -> dict[str, dict[str, object]]:
    """按题型拆开覆盖率（命中集 / 全集各自按常量题、机制题分组）。"""

    matched = {str(key) for key in (coverage or {}).get("matched", []) or []}
    split: dict[str, dict[str, object]] = {}
    for kind in ("constant", "mechanism"):
        group = [item for item in items if item_kind(item) == kind]
        hit = [item for item in group if getattr(item, "key", "") in matched]
        split[kind] = {
            "matched": len(hit),
            "total": len(group),
            "rate": round(len(hit) / len(group), 4) if group else None,
        }
    return split


def build_secondary(
    *,
    coverage: Mapping[str, object] | None,
    items: Sequence[object] = (),
) -> dict[str, object]:
    """构造辅助指标：事实覆盖率——下界，必须带题型构成与 caveat。"""

    by_kind = split_coverage(coverage, items)
    return {
        "fact_coverage": {
            "rate": (coverage or {}).get("rate"),
            "matched": (coverage or {}).get("matched_count"),
            "total": (coverage or {}).get("total"),
            "item_types": item_types(items),
            "by_item_type": by_kind,
            "caveat": SECONDARY_CAVEAT,
        },
        "spec": SPEC_REFERENCE,
    }


def build_layers(
    *,
    quality: Mapping[str, object] | None,
    items: Sequence[object] = (),
    tokens: int,
    wall_clock_seconds: float,
    tool_errors: int = 0,
    retries: int = 0,
    check_results: Sequence[object] = (),
) -> dict[str, object]:
    """把现有 quality 结构翻译成主/辅两层（原字段保留，不删证据）。"""

    coverage = (quality or {}).get("coverage_source") or (quality or {}).get("coverage")
    precision = (quality or {}).get("precision")
    return {
        "primary_metrics": build_primary(
            tokens=tokens,
            wall_clock_seconds=wall_clock_seconds,
            tool_errors=tool_errors,
            retries=retries,
            check_results=check_results,
            precision=precision if isinstance(precision, Mapping) else None,
        ),
        "secondary_metrics": build_secondary(coverage=coverage, items=items),
    }
