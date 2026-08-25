"""将评测结果渲染为可直接打开的静态 HTML 报告"""

from html import escape
from pathlib import Path

from .baseline import RegressionReport
from .metrics import calculate_metrics
from .models import EvaluationAssertion, EvaluationResult


MIN_STABLE_SAMPLE_COUNT = 20


def generate_report(
    path: Path,
    results: list[EvaluationResult],
    regression: RegressionReport | None = None,
) -> None:
    """根据评测结果生成静态 HTML 文件"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results, regression), encoding="utf-8")


def render_report(
    results: list[EvaluationResult],
    regression: RegressionReport | None = None,
) -> str:
    """将评测结果转换为 HTML 文本"""

    regression_html = _regression_section(regression)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>epsilon Evaluation Report</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1100px; margin: 2rem auto; color: #222; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }}
    .metric {{ padding: 1rem; background: #f1f3f5; border-radius: .4rem; }}
    .value {{ display: block; font-size: 1.5rem; font-weight: bold; margin-top: .35rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: .6rem; text-align: left; }}
    .pass {{ color: #16803c; }}
    .fail {{ color: #b42318; }}
  </style>
</head>
<body>
  <h1>epsilon Evaluation Report</h1>
  {_evaluation_section(
      "核心链路回归",
      "固定脚本验证模块协作，不衡量模型能力",
      _results_of_type(results, "core-regression"),
      "场景通过率",
  )}
  {_evaluation_section(
      "真实任务评测",
      "使用真实模型任务衡量 Agent 的任务完成能力",
      [
          result
          for result in _results_of_type(results, "real-task")
          if result.source is None
      ],
      "任务完成率",
  )}
  {_swebench_sections(results)}
  {_evaluation_section(
      "代码正确性任务",
      "使用独立 pytest 结果验证模型修改的代码，不依赖回复关键词",
      _results_of_type(results, "code-correctness"),
      "任务完成率",
  )}
  {_evaluation_section(
      "在线专项",
      "验证压缩、网络等指定运行时能力，不计入真实任务指标",
      _results_of_type(results, "online-special"),
      "场景通过率",
  )}
  {regression_html}
</body>
</html>
"""


def _results_of_type(
    results: list[EvaluationResult],
    evaluation_type: str,
) -> list[EvaluationResult]:
    """筛选同一来源的评测结果，避免不同性质指标混合。"""

    return [result for result in results if result.evaluation_type == evaluation_type]


def _swebench_sections(results: list[EvaluationResult]) -> str:
    """按来源和运行分组分别汇总真实 SWE-bench 结果。"""

    groups: dict[tuple[str, str], list[EvaluationResult]] = {}
    for result in results:
        if result.source is None:
            continue
        key = (result.source, result.evaluation_group or "normal")
        groups.setdefault(key, []).append(result)
    return "".join(
        _evaluation_section(
            f"SWE-bench：{source} · {group}",
            "独立工作区中的真实仓库修复，使用官方 Harness 验证补丁",
            group_results,
            "任务完成率",
        )
        for (source, group), group_results in sorted(groups.items())
    )


def _evaluation_section(
    title: str,
    description: str,
    results: list[EvaluationResult],
    completion_label: str,
) -> str:
    """渲染一种评测来源的独立汇总、结果和失败断言。"""

    metrics = calculate_metrics(results)
    rows = "\n".join(_result_row(result) for result in results)
    rows = rows or "<tr><td colspan=20>暂无结果</td></tr>"
    failures = "\n".join(
        _failure_row(result, assertion)
        for result in results
        for assertion in result.assertions
        if not assertion.passed
    )
    failures = failures or "<tr><td colspan=3>无失败断言</td></tr>"
    return f"""<section>
  <h2>{escape(title)}</h2>
  <p>{escape(description)}</p>
  <p>样本数：{metrics.scenario_count}，通过数：{metrics.passed_scenarios}</p>
  <p>估算上下文 Token 基于本地字符估算，不等同于服务端 usage；服务端实际 Token 无数据时留空</p>
  {_sample_size_note(metrics.scenario_count)}
  <section class="metrics">
    {_metric(completion_label, _percent(metrics.task_completion_rate))}
    {_metric("断言通过率", _percent(metrics.assertion_pass_rate))}
    {_metric("工具成功率", _percent(metrics.tool_success_rate))}
    {_metric("工具恢复率", _percent(metrics.tool_recovery_rate))}
    {_metric("持久化成功率", _percent(metrics.persistence_success_rate))}
    {_metric("降级率", _percent(metrics.degradation_rate))}
    {_metric("平均耗时", f"{metrics.average_duration_ms:.2f} ms")}
    {_metric(_percentile_label("P50 耗时", metrics.scenario_count), f"{metrics.p50_duration_ms:.2f} ms")}
    {_metric(_percentile_label("P95 耗时", metrics.scenario_count), f"{metrics.p95_duration_ms:.2f} ms")}
    {_metric("平均模型请求", f"{metrics.average_model_requests:.2f}")}
    {_metric("平均请求耗时", f"{metrics.average_model_request_duration_ms:.2f} ms")}
    {_metric(_percentile_label("请求 P50", metrics.scenario_count), f"{metrics.p50_model_request_duration_ms:.2f} ms")}
    {_metric(_percentile_label("请求 P95", metrics.scenario_count), f"{metrics.p95_model_request_duration_ms:.2f} ms")}
  </section>
  <h3>场景结果</h3>
  <table>
    <thead><tr><th>场景</th><th>任务 ID</th><th>来源</th><th>分组</th><th>基线提交</th><th>变更文件</th><th>类型</th><th>状态</th><th>错误类别</th><th>失败阶段</th><th>错误详情</th><th>停止原因</th><th>耗时</th><th>模型请求</th><th>工具回合</th><th>工具调用</th><th>重试</th><th>压缩</th><th>估算上下文 Token</th><th>服务端实际 Token</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h3>失败断言</h3>
  <table>
    <thead><tr><th>场景</th><th>断言</th><th>原因</th></tr></thead>
    <tbody>{failures}</tbody>
  </table>
</section>"""


def _result_row(result: EvaluationResult) -> str:
    """生成单个场景的 HTML 行"""

    status = "通过" if result.passed else "失败"
    status_class = "pass" if result.passed else "fail"
    return (
        f"<tr><td>{escape(result.scenario)}</td>"
        f"<td>{escape(result.task_id or '-')}</td>"
        f"<td>{escape(result.source or '-')}</td>"
        f"<td>{escape(result.evaluation_group or '-')}</td>"
        f"<td>{escape(result.base_commit or '-')}</td>"
        f"<td>{escape(', '.join(result.changed_files) or '-')}</td>"
        f"<td>{_evaluation_type_label(result.evaluation_type)}</td>"
        f"<td class=\"{status_class}\">{status}</td>"
        f"<td>{escape(result.error_category or '-')}</td>"
        f"<td>{escape(result.error_stage or '-')}</td>"
        f"<td>{escape(result.error_message or '-')}</td>"
        f"<td>{escape(result.stop_reason or '-')}</td>"
        f"<td>{result.duration_ms:.2f} ms</td>"
        f"<td>{result.model_requests}</td><td>{result.tool_rounds}</td>"
        f"<td>{result.tool_calls}</td>"
        f"<td>{result.retries}</td><td>{result.compactions}</td>"
        f"<td>{result.estimated_tokens}</td>"
        f"<td>{'' if result.actual_tokens is None else result.actual_tokens}</td></tr>"
    )


def _evaluation_type_label(evaluation_type: str) -> str:
    """将评测类型转换为报告中的中文标签。"""

    labels = {
        "core-regression": "核心链路回归",
        "real-task": "真实任务",
        "online-special": "在线专项",
        "code-correctness": "代码正确性",
    }
    return labels[evaluation_type]


def _failure_row(result: EvaluationResult, assertion: EvaluationAssertion) -> str:
    """生成一条失败断言的 HTML 行。"""

    return (
        f"<tr><td>{escape(result.scenario)}</td>"
        f"<td>{escape(assertion.name)}</td>"
        f"<td>{escape(assertion.message)}</td></tr>"
    )


def _metric(name: str, value: str) -> str:
    """生成指标卡片"""

    return f'<div class="metric">{escape(name)}<span class="value">{escape(value)}</span></div>'


def _sample_size_note(sample_count: int) -> str:
    """在小样本时明确百分位数仅供观察。"""

    if sample_count < MIN_STABLE_SAMPLE_COUNT:
        return (
            "<p>样本少于 20，P50/P95 仅为观察值，"
            "不代表稳定性能结论</p>"
        )
    return ""


def _percentile_label(name: str, sample_count: int) -> str:
    """为小样本百分位数增加观察性标签。"""

    return f"{name}（观察值）" if sample_count < MIN_STABLE_SAMPLE_COUNT else name


def _percent(value: float) -> str:
    """将比例格式化为百分比"""

    return f"{value:.1%}"


def _regression_section(regression: RegressionReport | None) -> str:
    """生成 baseline 回归比较区域"""

    if regression is None:
        return ""
    status = "通过" if regression.passed else "失败"
    status_class = "pass" if regression.passed else "fail"
    return f"""<h2>Baseline 回归</h2>
<p class="{status_class}">回归门禁：{status}</p>
<ul>
  <li>新增失败：{_list_or_none(regression.new_failures)}</li>
  <li>历史已知失败：{_list_or_none(regression.known_failures)}</li>
  <li>缺失运行：{_list_or_none(regression.missing_runs)}</li>
  <li>重复运行：{_list_or_none(regression.duplicate_runs)}</li>
  <li>指标回归：{_list_or_none(regression.metric_regressions)}</li>
  <li>指标观察：{_list_or_none(regression.metric_observations)}</li>
  <li>配置不匹配：{_list_or_none(regression.metadata_mismatches)}</li>
</ul>"""


def _list_or_none(values: tuple[str, ...]) -> str:
    """格式化回归项列表"""

    return escape(", ".join(values) if values else "无")
