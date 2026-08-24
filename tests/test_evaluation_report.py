from pathlib import Path

from evaluation.models import EvaluationAssertion, EvaluationResult
from evaluation.baseline import RegressionReport
from evaluation.report import generate_report, render_report


def test_render_report_contains_metrics_and_scenario_status() -> None:
    """测试报告包含核心指标和场景状态"""

    html = render_report(
        [
            EvaluationResult(
                scenario="demo",
                duration_ms=12,
                error_category="configuration",
                error_stage="agent-loop",
                error_message="ConfigError: <MODEL_NAME> is missing",
                assertions=(EvaluationAssertion("done", True),),
            )
        ]
    )

    assert "任务完成率" in html
    assert "核心链路回归" in html
    assert "固定脚本验证模块协作，不衡量模型能力" in html
    assert "P50 耗时" in html
    assert "P95 耗时" in html
    assert "demo" in html
    assert "configuration" in html
    assert "agent-loop" in html
    assert "ConfigError: &lt;MODEL_NAME&gt; is missing" in html
    assert "通过" in html


def test_render_report_separates_regression_and_real_task_metrics() -> None:
    """测试固定脚本结果不会混入真实任务指标。"""

    html = render_report(
        [
            EvaluationResult(
                scenario="offline-script",
                duration_ms=10,
                assertions=(EvaluationAssertion("done", True),),
            ),
            EvaluationResult(
                scenario="online-task",
                duration_ms=20,
                evaluation_type="real-task",
                assertions=(EvaluationAssertion("done", False),),
            ),
        ]
    )

    regression_section, real_task_section, _ = html.split("<h2>")[1:4]
    assert "固定脚本验证模块协作，不衡量模型能力" in regression_section
    assert "样本数：1，通过数：1" in regression_section
    assert "offline-script" in regression_section
    assert "online-task" not in regression_section
    assert "使用真实模型任务衡量 Agent 的任务完成能力" in real_task_section
    assert "样本数：1，通过数：0" in real_task_section
    assert "online-task" in real_task_section
    assert "offline-script" not in real_task_section


def test_render_report_separates_swebench_sources_and_groups() -> None:
    """测试 Lite、完整集和压缩专项不会共享同一个指标分母。"""

    html = render_report(
        [
            EvaluationResult(
                scenario="lite",
                task_id="lite",
                source="swebench-lite",
                evaluation_group="normal",
                duration_ms=1,
            ),
            EvaluationResult(
                scenario="full",
                task_id="full",
                source="swebench-full",
                evaluation_group="compaction",
                duration_ms=1,
            ),
        ]
    )

    assert "SWE-bench：swebench-lite · normal" in html
    assert "SWE-bench：swebench-full · compaction" in html


def test_render_report_marks_small_sample_percentiles_as_observations() -> None:
    """测试小样本报告不会把百分位数表达为稳定性能结论。"""

    html = render_report(
        [
            EvaluationResult(
                scenario=f"task-{index}",
                duration_ms=10 + index,
                model_request_durations_ms=(5 + index,),
                evaluation_type="real-task",
                assertions=(EvaluationAssertion("done", True),),
            )
            for index in range(6)
        ]
    )

    real_task_section = html.split("<h2>真实任务评测</h2>", 1)[1].split(
        "<h2>在线专项</h2>",
        1,
    )[0]
    assert "样本数：6，通过数：6" in real_task_section
    assert "样本少于 20，P50/P95 仅为观察值，不代表稳定性能结论" in real_task_section
    assert "P50 耗时（观察值）" in real_task_section
    assert "P95 耗时（观察值）" in real_task_section
    assert "请求 P50（观察值）" in real_task_section
    assert "请求 P95（观察值）" in real_task_section


def test_render_report_lists_all_failed_assertions() -> None:
    """测试同一场景的多个失败原因都会显示。"""

    html = render_report(
        [
            EvaluationResult(
                scenario="failed-task",
                duration_ms=10,
                assertions=(
                    EvaluationAssertion("file-content", False, "文件内容不正确"),
                    EvaluationAssertion("tool-order", False, "工具顺序不正确"),
                ),
            )
        ]
    )

    assert "file-content" in html
    assert "文件内容不正确" in html
    assert "tool-order" in html
    assert "工具顺序不正确" in html


def test_render_report_distinguishes_estimated_and_actual_tokens() -> None:
    """测试报告不会将本地估算值误写为服务端用量。"""

    html = render_report(
        [
            EvaluationResult(
                scenario="estimated-only",
                duration_ms=10,
                estimated_tokens=12,
                assertions=(EvaluationAssertion("done", True),),
            ),
            EvaluationResult(
                scenario="with-usage",
                duration_ms=10,
                estimated_tokens=20,
                actual_tokens=18,
                assertions=(EvaluationAssertion("done", True),),
            ),
        ]
    )

    assert "估算上下文 Token 基于本地字符估算，不等同于服务端 usage" in html
    assert "估算上下文 Token" in html
    assert "服务端实际 Token" in html
    assert "<td>12</td><td></td>" in html
    assert "<td>20</td><td>18</td>" in html


def test_render_report_contains_baseline_regression() -> None:
    """测试报告包含 baseline 回归结果"""

    html = render_report(
        [],
        RegressionReport(
            new_failures=("task#1",),
            known_failures=(),
            missing_runs=(),
            duplicate_runs=(),
        ),
    )

    assert "新增失败" in html
    assert "指标回归" in html
    assert "指标观察" in html
    assert "配置不匹配" in html
    assert "task#1" in html
    assert "回归门禁：失败" in html


def test_generate_report_writes_static_html(tmp_path: Path) -> None:
    """测试报告可以写入静态 HTML 文件"""

    path = tmp_path / "evaluation-report.html"
    generate_report(path, [])

    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
