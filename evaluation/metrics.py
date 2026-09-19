"""根据评测结果计算汇总指标"""

from dataclasses import dataclass

from .models import EvaluationResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """保存一组评测结果的汇总指标"""

    scenario_count: int
    passed_scenarios: int
    task_completion_rate: float
    assertion_pass_rate: float
    tool_success_rate: float
    tool_recovery_rate: float
    persistence_success_rate: float
    degradation_rate: float
    average_duration_ms: float
    average_model_requests: float
    total_retries: int
    total_compactions: int
    p50_duration_ms: float
    p95_duration_ms: float
    average_model_request_duration_ms: float
    p50_model_request_duration_ms: float
    p95_model_request_duration_ms: float
    average_cache_hit_rate: float | None
    total_cached_tokens: int | None


def calculate_metrics(results: list[EvaluationResult]) -> EvaluationMetrics:
    """根据场景结果计算设计文档中的核心指标"""

    scenario_count = len(results)
    passed_scenarios = sum(result.passed for result in results)
    assertion_count = sum(len(result.assertions) for result in results)
    passed_assertions = sum(
        assertion.passed
        for result in results
        for assertion in result.assertions
    )
    tool_calls = sum(result.tool_calls for result in results)
    tool_failures = sum(result.tool_failures for result in results)
    failed_tool_scenarios = sum(result.tool_failures > 0 for result in results)
    recovered_tool_scenarios = sum(
        result.tool_failures > 0 and result.passed for result in results
    )
    degraded_scenarios = sum(result.persistence_degraded for result in results)
    return EvaluationMetrics(
        scenario_count=scenario_count,
        passed_scenarios=passed_scenarios,
        task_completion_rate=_rate(passed_scenarios, scenario_count),
        assertion_pass_rate=_rate(passed_assertions, assertion_count),
        tool_success_rate=_rate(tool_calls - tool_failures, tool_calls),
        tool_recovery_rate=_rate(
            recovered_tool_scenarios,
            failed_tool_scenarios,
        ),
        persistence_success_rate=_rate(
            scenario_count - degraded_scenarios,
            scenario_count,
        ),
        degradation_rate=_rate(degraded_scenarios, scenario_count),
        average_duration_ms=_average(
            [result.duration_ms for result in results]
        ),
        average_model_requests=_average(
            [result.model_requests for result in results]
        ),
        total_retries=sum(result.retries for result in results),
        total_compactions=sum(result.compactions for result in results),
        p50_duration_ms=_percentile(
            [result.duration_ms for result in results], 0.50
        ),
        p95_duration_ms=_percentile(
            [result.duration_ms for result in results], 0.95
        ),
        average_model_request_duration_ms=_average(
            [
                duration
                for result in results
                for duration in result.model_request_durations_ms
            ]
        ),
        p50_model_request_duration_ms=_percentile(
            [
                duration
                for result in results
                for duration in result.model_request_durations_ms
            ],
            0.50,
        ),
        p95_model_request_duration_ms=_percentile(
            [
                duration
                for result in results
                for duration in result.model_request_durations_ms
            ],
            0.95,
        ),
        average_cache_hit_rate=(
            _average([result.cache_hit_rate for result in results if result.cache_hit_rate is not None])
            if any(result.cache_hit_rate is not None for result in results)
            else None
        ),
        total_cached_tokens=(
            sum(result.cached_tokens for result in results if result.cached_tokens is not None)
            if any(result.cached_tokens is not None for result in results)
            else None
        ),
    )


def _rate(numerator: int, denominator: int) -> float:
    """计算比例，避免空评测集产生除零异常"""

    return numerator / denominator if denominator else 0.0


def _average(values: list[float | int]) -> float:
    """计算平均值，避免空评测集产生除零异常"""

    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float | int], percentile: float) -> float:
    """使用线性插值计算百分位数"""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
