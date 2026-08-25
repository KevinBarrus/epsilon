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
        "task_id": result.task_id,
        "source": result.source,
        "evaluation_group": result.evaluation_group,
        "base_commit": result.base_commit,
        "changed_files": list(result.changed_files),
        "passed": result.passed,
        "duration_ms": result.duration_ms,
        "model_requests": result.model_requests,
        "tool_rounds": result.tool_rounds,
        "parallel_tool_batches": result.parallel_tool_batches,
        "tool_batch_duration_ms": result.tool_batch_duration_ms,
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
        "stop_reason": result.stop_reason,
        "agent_execution_environment": result.agent_execution_environment,
        "agent_verification_status": result.agent_verification_status,
        "official_harness_status": result.official_harness_status,
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
        task_id=_optional_string(record.get("task_id")),
        source=_optional_string(record.get("source")),
        evaluation_group=_optional_string(record.get("evaluation_group")),
        base_commit=_optional_string(record.get("base_commit")),
        changed_files=_string_tuple(record.get("changed_files", [])),
        model_requests=_required_int(record, "model_requests"),
        tool_rounds=_required_int_or_default(record, "tool_rounds", 0),
        parallel_tool_batches=_required_int_or_default(record, "parallel_tool_batches", 0),
        tool_batch_duration_ms=_required_number_or_default(record, "tool_batch_duration_ms", 0.0),
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
        stop_reason=_optional_string(record.get("stop_reason")),
        agent_execution_environment=_optional_string(record.get("agent_execution_environment")),  # type: ignore[arg-type]
        agent_verification_status=_optional_string(
            record.get("agent_verification_status", record.get("local_verification_status"))
        ),  # type: ignore[arg-type]
        official_harness_status=_optional_string(record.get("official_harness_status")),  # type: ignore[arg-type]
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


def _string_tuple(value: object) -> tuple[str, ...]:
    """读取字符串数组字段。"""

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("评测结果字符串数组字段无效")
    return tuple(value)


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


def _required_number_or_default(
    record: dict[str, object],
    key: str,
    default: float,
) -> float:
    """读取兼容旧结果的数值字段。"""

    value = record.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"评测结果字段无效：{key}")
    return float(value)
