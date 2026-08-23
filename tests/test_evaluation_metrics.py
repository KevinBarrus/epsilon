from pathlib import Path

import pytest

from evaluation.metrics import calculate_metrics
from evaluation.models import EvaluationAssertion, EvaluationResult
from evaluation.storage import append_result, load_results


def _result(
    scenario: str,
    passed: bool,
    *,
    tool_calls: int = 0,
    tool_failures: int = 0,
    persistence_degraded: bool = False,
) -> EvaluationResult:
    """构造测试用评测结果"""

    return EvaluationResult(
        scenario=scenario,
        duration_ms=10,
        model_requests=2,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        retries=1,
        compactions=1,
        persistence_degraded=persistence_degraded,
        assertions=(EvaluationAssertion("result", passed),),
    )


def test_calculate_metrics_aggregates_task_tool_and_recovery_rates() -> None:
    """测试指标计算覆盖任务、工具和异常恢复数据"""

    metrics = calculate_metrics(
        [
            _result("success", True, tool_calls=2),
            _result("recovery", True, tool_calls=2, tool_failures=1),
            _result("failed", False, persistence_degraded=True),
        ]
    )

    assert metrics.scenario_count == 3
    assert metrics.passed_scenarios == 2
    assert metrics.task_completion_rate == 2 / 3
    assert metrics.tool_success_rate == 3 / 4
    assert metrics.tool_recovery_rate == 1
    assert metrics.persistence_success_rate == 2 / 3
    assert metrics.total_retries == 3


def test_calculate_metrics_computes_percentiles_and_request_latency() -> None:
    """测试指标计算包含总耗时和模型请求耗时分布"""

    results = [
        EvaluationResult(
            scenario=f"scenario-{index}",
            duration_ms=duration,
            model_request_durations_ms=(duration / 2,),
        )
        for index, duration in enumerate((100, 200, 300, 400, 500))
    ]

    metrics = calculate_metrics(results)

    assert metrics.p50_duration_ms == 300
    assert metrics.p95_duration_ms == 480
    assert metrics.average_model_request_duration_ms == 150
    assert metrics.p50_model_request_duration_ms == 150
    assert metrics.p95_model_request_duration_ms == 240


@pytest.mark.asyncio
async def test_memory_scenario_uses_cjk_aware_token_estimation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试中文会话评测复用核心 Token 估算函数。"""

    from evaluation import scenarios
    calls = 0

    def estimate(messages) -> int:
        """模拟核心 Token 估算函数并记录调用。"""

        nonlocal calls
        calls += 1
        return 7

    monkeypatch.setattr(scenarios, "estimate_context_tokens", estimate)
    result = await scenarios.run_memory_scenario(tmp_path)

    assert calls == 2
    assert result.estimated_tokens == 14


def test_results_can_round_trip_through_jsonl(tmp_path: Path) -> None:
    """测试评测结果可以写入并从 JSONL 恢复"""

    path = tmp_path / "results.jsonl"
    expected = _result("round-trip", True, tool_calls=1)
    expected = EvaluationResult(
        scenario=expected.scenario,
        duration_ms=expected.duration_ms,
        evaluation_type="code-correctness",
        run_id="run-1",
        repetition=2,
        model_requests=expected.model_requests,
        tool_calls=expected.tool_calls,
        tool_failures=expected.tool_failures,
        retries=expected.retries,
        compactions=expected.compactions,
        persistence_degraded=expected.persistence_degraded,
        events=({"type": "tool_call", "name": "read_file"},),
        assertions=expected.assertions,
    )

    append_result(path, expected)

    assert load_results(path) == [expected]
