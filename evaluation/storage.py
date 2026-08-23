"""保存和读取评测结果 JSONL"""

import json
from pathlib import Path

from .models import EvaluationAssertion, EvaluationResult


def append_result(path: Path, result: EvaluationResult) -> None:
    """向 JSONL 文件追加一条评测结果"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(_result_to_record(result), file, ensure_ascii=False)
        file.write("\n")


def load_results(path: Path) -> list[EvaluationResult]:
    """按文件顺序读取全部评测结果"""

    if not path.exists():
        return []
    results: list[EvaluationResult] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(_result_from_record(json.loads(line)))
    return results


def _result_to_record(result: EvaluationResult) -> dict[str, object]:
    """将评测结果转换为 JSON 对象"""

    return {
        "run_id": result.run_id,
        "scenario": result.scenario,
        "evaluation_type": result.evaluation_type,
        "repetition": result.repetition,
        "passed": result.passed,
        "duration_ms": result.duration_ms,
        "model_requests": result.model_requests,
        "tool_calls": result.tool_calls,
        "tool_failures": result.tool_failures,
        "retries": result.retries,
        "compactions": result.compactions,
        "estimated_tokens": result.estimated_tokens,
        "actual_tokens": result.actual_tokens,
        "persistence_degraded": result.persistence_degraded,
        "error_category": result.error_category,
        "error_stage": result.error_stage,
        "error_message": result.error_message,
        "model_request_durations_ms": list(result.model_request_durations_ms),
        "events": list(result.events),
        "assertions": [
            {
                "name": assertion.name,
                "passed": assertion.passed,
                "message": assertion.message,
            }
            for assertion in result.assertions
        ],
    }


def _result_from_record(record: object) -> EvaluationResult:
    """将 JSON 对象转换为评测结果"""

    if not isinstance(record, dict):
        raise ValueError("评测结果必须是 JSON 对象")
    assertions = record.get("assertions", [])
    if not isinstance(assertions, list):
        raise ValueError("评测断言必须是数组")
    return EvaluationResult(
        scenario=_required_string(record, "scenario"),
        duration_ms=_required_number(record, "duration_ms"),
        evaluation_type=_evaluation_type(record.get("evaluation_type", "core-regression")),
        run_id=str(record.get("run_id", "legacy")),
        repetition=_required_int_or_default(record, "repetition", 1),
        model_requests=_required_int(record, "model_requests"),
        tool_calls=_required_int(record, "tool_calls"),
        tool_failures=_required_int(record, "tool_failures"),
        retries=_required_int(record, "retries"),
        compactions=_required_int(record, "compactions"),
        estimated_tokens=_required_int(record, "estimated_tokens"),
        actual_tokens=record.get("actual_tokens"),
        persistence_degraded=bool(record.get("persistence_degraded", False)),
        error_category=_optional_string(record.get("error_category")),
        error_stage=_optional_string(record.get("error_stage")),
        error_message=_optional_string(record.get("error_message")),
        model_request_durations_ms=tuple(
            float(value)
            for value in record.get("model_request_durations_ms", [])
        ),
        events=tuple(_event_from_record(item) for item in record.get("events", [])),
        assertions=tuple(_assertion_from_record(item) for item in assertions),
    )


def _event_from_record(record: object) -> dict[str, object]:
    """校验并恢复一条评测事件"""

    if not isinstance(record, dict):
        raise ValueError("评测事件必须是 JSON 对象")
    return dict(record)


def _assertion_from_record(record: object) -> EvaluationAssertion:
    """将 JSON 断言对象转换为评测断言"""

    if not isinstance(record, dict):
        raise ValueError("评测断言必须是 JSON 对象")
    return EvaluationAssertion(
        name=_required_string(record, "name"),
        passed=bool(record.get("passed", False)),
        message=str(record.get("message", "")),
    )


def _required_string(record: dict[str, object], key: str) -> str:
    """读取必需的字符串字段"""

    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"评测结果字段无效：{key}")
    return value


def _evaluation_type(value: object) -> str:
    """读取兼容旧结果的评测类型。"""

    if value not in {
        "core-regression",
        "real-task",
        "online-special",
        "code-correctness",
    }:
        raise ValueError("评测结果字段无效：evaluation_type")
    return value


def _optional_string(value: object) -> str | None:
    """读取允许为空的字符串字段。"""

    if value is None or isinstance(value, str):
        return value
    raise ValueError("评测结果可选字符串字段无效")


def _required_int(record: dict[str, object], key: str) -> int:
    """读取必需的整数数字段"""

    value = record.get(key)
    if not isinstance(value, int):
        raise ValueError(f"评测结果字段无效：{key}")
    return value


def _required_int_or_default(
    record: dict[str, object],
    key: str,
    default: int,
) -> int:
    """读取兼容旧结果的整数数字段"""

    value = record.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"评测结果字段无效：{key}")
    return value


def _required_number(record: dict[str, object], key: str) -> float:
    """读取必需的数值字段"""

    value = record.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"评测结果字段无效：{key}")
    return float(value)
