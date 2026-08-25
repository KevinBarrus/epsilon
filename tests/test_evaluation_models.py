from evaluation.models import (
    EvaluationAssertion,
    EvaluationResult,
    EvaluationScenario,
)
from evaluation.storage import append_result, load_results


def test_evaluation_result_passes_when_all_assertions_pass() -> None:
    """测试全部断言通过时场景结果通过"""

    result = EvaluationResult(
        scenario="memory",
        duration_ms=12.5,
        assertions=(
            EvaluationAssertion("history", True),
            EvaluationAssertion("tool", True),
        ),
    )

    assert result.passed


def test_evaluation_result_fails_when_any_assertion_fails() -> None:
    """测试任意断言失败时场景结果失败"""

    result = EvaluationResult(
        scenario="tool-recovery",
        duration_ms=18,
        assertions=(
            EvaluationAssertion("tool-error", False, "工具未返回错误"),
        ),
    )

    assert not result.passed


def test_evaluation_models_keep_scenario_and_actual_token_metadata() -> None:
    """测试评测模型保留场景描述和实际 Token 信息"""

    scenario = EvaluationScenario("session-restore", "测试会话恢复")
    result = EvaluationResult(
        scenario=scenario.name,
        duration_ms=20,
        estimated_tokens=100,
        actual_tokens=80,
    )

    assert scenario.description == "测试会话恢复"
    assert result.estimated_tokens == 100
    assert result.actual_tokens == 80
    assert result.evaluation_type == "core-regression"
    assert result.error_stage is None


def test_evaluation_models_keep_real_task_identity() -> None:
    """测试真实任务结果保留来源、基线和变更文件。"""

    result = EvaluationResult(
        scenario="django__django-10914",
        duration_ms=20,
        task_id="django__django-10914",
        source="swebench-lite",
        evaluation_group="normal",
        base_commit="base",
        changed_files=("django/conf/global_settings.py",),
    )

    assert result.task_id == "django__django-10914"
    assert result.source == "swebench-lite"
    assert result.changed_files == ("django/conf/global_settings.py",)


def test_result_storage_preserves_verification_classification(tmp_path) -> None:
    """测试 JSONL 往返不会丢失两类验证状态。"""

    path = tmp_path / "results.jsonl"
    append_result(
        path,
        EvaluationResult(
            scenario="task",
            duration_ms=1,
            local_verification_status="failed",
            official_harness_status="environment-error",
        ),
    )

    restored = load_results(path)[0]

    assert restored.local_verification_status == "failed"
    assert restored.official_harness_status == "environment-error"
