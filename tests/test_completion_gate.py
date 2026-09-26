"""Completion Gate（完成门）的测试。

覆盖指令第三节的 10 条，外加一条"防绕过"测试：配置了验证器时，
`observe_tool_results` 的同步完成路径绝不能生效——否则模型只要不调用
`goal(op=complete)` 就能把自己标成完成，门就是纸做的。
"""

import asyncio
from pathlib import Path

import pytest

from core.goal import (
    CheckVerdict,
    CompletionOutcome,
    Goal,
    GoalPolicy,
    create_goal_tool,
    parse_verdict,
)
from core.model import Message, ToolCall, ToolResult
from core.tools import ToolManager

CRITERIA = "报告必须覆盖 A、B、C 三个子系统"


def _policy(**kwargs) -> GoalPolicy:
    """构造一个带验收标准的目标策略。"""

    goal = Goal("把项目讲清楚", acceptance_criteria=kwargs.pop("criteria", CRITERIA))
    return GoalPolicy(goal, **kwargs)


def _ok_verifier(text: str = "VERDICT: met\nUNMET: 无"):
    """返回固定结论的假验证器（v2：输入是有界证据简报，不是 Goal）。"""

    async def verify(brief: str) -> str:
        return text

    return verify


async def _complete(policy: GoalPolicy) -> CompletionOutcome:
    return await policy.request_completion()


# --- 1. 验收标准为空时退回 objective ---------------------------------------


@pytest.mark.asyncio
async def test_empty_criteria_falls_back_to_objective() -> None:
    """没有验收标准时，用 objective 作为判定依据。"""

    seen: list[str] = []

    async def verify(brief: str) -> str:
        seen.append(brief)
        return "VERDICT: met"

    policy = GoalPolicy(Goal("读完整仓库"), verifier=verify)
    outcome = await _complete(policy)

    assert outcome.accepted
    assert seen and "读完整仓库" in seen[0]


def test_verifier_prompt_uses_criteria_or_objective() -> None:
    """判定依据优先用验收标准，为空才退回 objective。"""

    from core.goal import verifier_brief

    with_criteria = Goal("目标", acceptance_criteria=CRITERIA)
    without = Goal("目标")

    assert CRITERIA in verifier_brief(with_criteria)
    assert "目标" in verifier_brief(without)


# --- 2. met ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_met_accepts_and_marks_verified() -> None:
    """met → 接受完成并标记 verified。"""

    policy = _policy(verifier=_ok_verifier())
    outcome = await _complete(policy)

    assert outcome.accepted and outcome.verified
    assert policy.goal.status == "complete"
    assert policy.goal.verified is True


# --- 3. not_met -----------------------------------------------------------


@pytest.mark.asyncio
async def test_not_met_rejects_and_keeps_active() -> None:
    """not_met → 拒绝、回注未满足项、目标保持 active、拒绝计数 +1。"""

    async def verify(brief: str) -> str:
        return "VERDICT: not_met\nUNMET: B 未覆盖\nUNMET: C 未覆盖"

    policy = _policy(verifier=verify)
    outcome = await _complete(policy)

    assert not outcome.accepted
    assert outcome.unmet == ("B 未覆盖", "C 未覆盖")
    assert outcome.reason == "not_met"
    assert policy.goal.status == "active"
    assert policy.goal.completion_rejections == 1
    assert "B 未覆盖" in outcome.message


# --- 4. inconclusive ------------------------------------------------------


@pytest.mark.asyncio
async def test_inconclusive_is_also_rejected() -> None:
    """inconclusive 与 not_met 一样拒绝（无法验证 ≠ 已完成）。"""

    policy = _policy(verifier=_ok_verifier("VERDICT: inconclusive\nUNMET: 无"))
    outcome = await _complete(policy)

    assert not outcome.accepted
    assert outcome.reason == "inconclusive"
    assert policy.goal.completion_rejections == 1
    assert policy.goal.status == "active"


# --- 5. 验证器故障与超时 ----------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_error_is_rejected_fail_closed() -> None:
    """验证器抛错 → 按拒绝处理（fail-closed），并单独归类为 verifier_error。"""

    async def boom(brief: str) -> str:
        raise RuntimeError("verifier exploded")

    policy = _policy(verifier=boom, max_rejections=5)
    outcome = await _complete(policy)

    assert not outcome.accepted
    assert outcome.reason == "verifier_error"
    assert policy.goal.completion_rejections == 1


@pytest.mark.asyncio
async def test_verifier_timeout_is_rejected() -> None:
    """验证器超时 → 拒绝，但 reason 必须是 verifier_error（验证器故障，不是证据不足）。"""

    async def slow(brief: str) -> str:
        await asyncio.sleep(5)
        return "VERDICT: met"

    policy = _policy(verifier=slow, verifier_timeout_seconds=0.05, max_rejections=5)
    outcome = await _complete(policy)

    assert not outcome.accepted
    assert outcome.reason == "verifier_error"


@pytest.mark.asyncio
async def test_verifier_budget_error_is_recorded_as_verifier_error() -> None:
    """验证器抛预算类错误 → verifier_error。

    v1 把验证器故障全记成 inconclusive，等于把实现缺陷归咎于模型没做完——
    这条回归测试就是防止再次误归因。
    """

    class BudgetError(RuntimeError):
        """模拟 BudgetedClient 的 TokenBudgetReached。"""

    async def broke(brief: str) -> str:
        raise BudgetError("token budget exhausted")

    policy = _policy(verifier=broke, max_rejections=5)
    outcome = await _complete(policy)

    assert not outcome.accepted
    assert outcome.reason == "verifier_error"
    assert "BudgetError" in outcome.message


# --- 6. 拒绝上限 -----------------------------------------------------------


@pytest.mark.asyncio
async def test_releases_after_max_rejections_but_marks_unverified() -> None:
    """连拒到上限后进入 unverified 终态（不再是 complete），并标注未通过验证。"""

    policy = _policy(
        verifier=_ok_verifier("VERDICT: not_met\nUNMET: 还差 B"),
        max_rejections=2,
    )

    first = await _complete(policy)
    second = await _complete(policy)
    third = await _complete(policy)

    assert [first.accepted, second.accepted, third.accepted] == [False, False, False]
    assert third.verified is False
    assert policy.goal.status == "unverified", "超限后不许再写 complete"
    assert policy.goal.verified is False
    assert "未通过验证" in third.message
    assert policy.goal.completion_rejections == 2


# --- 7. 防退化 nudge ------------------------------------------------------


def test_no_tool_nudge_after_consecutive_rounds() -> None:
    """连续 N 轮没有工具调用 → 注入"复述计划不是进展"。"""

    policy = _policy(verifier=_ok_verifier(), no_tool_nudge_rounds=3)
    messages = [policy.follow_up_message("我在想") for _ in range(3)]

    assert "复述计划" not in messages[1].content
    assert "复述计划" in messages[2].content


def test_no_tool_counter_resets_after_tool_use() -> None:
    """一旦有工具调用，无工具轮数归零。"""

    policy = _policy(verifier=_ok_verifier(), no_tool_nudge_rounds=2)
    policy.follow_up_message("一")
    policy.observe_tool_results((ToolCall("c", "read_file", {"path": "a"}),), (ToolResult("c", "ok"),))
    message = policy.follow_up_message("二")

    assert "复述计划" not in message.content


def test_repeated_summary_nudge() -> None:
    """连续两次归一化后完全相同的总结 → 注入"换一个下一步"。"""

    policy = _policy(verifier=_ok_verifier(), no_tool_nudge_rounds=99)
    policy.follow_up_message("已经完成了 A！")
    message = policy.follow_up_message("已经完成了 A ！！")

    assert "实质不同的下一步" in message.content


def test_different_summary_does_not_nudge() -> None:
    """总结不同就不打扰（宁可漏，不要误伤）。"""

    policy = _policy(verifier=_ok_verifier(), no_tool_nudge_rounds=99)
    policy.follow_up_message("完成了 A")
    message = policy.follow_up_message("现在做 B")

    assert "实质不同的下一步" not in message.content


# --- 8. 关闭后行为不变 ----------------------------------------------------


def test_disabled_gate_keeps_legacy_behaviour() -> None:
    """未配置验证器时，仍然走原来的同步完成路径。"""

    policy = _policy()
    policy.observe_tool_results(
        (ToolCall("c", "goal", {"op": "complete"}),),
        (ToolResult("c", "目标已显式完成"),),
    )

    assert policy.goal.status == "complete"


# --- 10. 防绕过：同步兜底路径必须失效 --------------------------------------


@pytest.mark.asyncio
async def test_sync_fallback_cannot_bypass_the_gate() -> None:
    """配置了验证器时，observe_tool_results 不得走同步完成路径。"""

    policy = _policy(verifier=_ok_verifier("VERDICT: not_met\nUNMET: 还差 C"), max_rejections=2)

    policy.observe_tool_results(
        (ToolCall("c", "goal", {"op": "complete"}),),
        (ToolResult("c", "目标已显式完成"),),
    )

    assert policy.goal.status == "active", "同步兜底路径绕过了完成门"
    assert policy.goal.completion_rejections == 0


@pytest.mark.asyncio
async def test_goal_tool_returns_unmet_items_to_the_model() -> None:
    """拒绝时，未满足项要通过工具结果回注给模型。"""

    async def verify(brief: str) -> str:
        return "VERDICT: not_met\nUNMET: 缺 B\nUNMET: 缺 C"

    policy = _policy(verifier=verify, max_rejections=2)
    _, handler = create_goal_tool(lambda: policy)
    result = await handler(ToolCall("c", "goal", {"op": "complete"}))

    assert result.is_error is True
    assert "缺 B" in result.content and "缺 C" in result.content
    assert policy.goal.status == "active"


@pytest.mark.asyncio
async def test_goal_tool_reports_verified_completion() -> None:
    """通过验证时，工具结果要说明已通过独立验证。"""

    policy = _policy(verifier=_ok_verifier())
    _, handler = create_goal_tool(lambda: policy)
    result = await handler(ToolCall("c", "goal", {"op": "complete"}))

    assert result.is_error is False
    assert policy.goal.status == "complete"
    assert "验证" in result.content


# --- 9. 可观测 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_gate_event_is_emitted_and_serialisable() -> None:
    """每次判定都要发事件，且能被评测轨迹序列化。"""

    from evaluation.events import event_to_record

    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    policy = _policy(
        verifier=_ok_verifier("VERDICT: not_met\nUNMET: 缺 A"),
        on_event=collect,
        max_rejections=2,
    )
    await _complete(policy)

    from core.goal import CompletionGateEvent

    gate_events = [event for event in events if isinstance(event, CompletionGateEvent)]
    assert gate_events and gate_events[0].verdict == "not_met"
    assert gate_events[0].unmet == ("缺 A",)
    assert gate_events[0].attempt == 1

    record = event_to_record(gate_events[0])
    assert record["type"] == "completion_gate"
    assert record["verdict"] == "not_met"


def test_parse_verdict_reads_met_and_unmet() -> None:
    """解析验证器输出：结论 + 未满足项清单。"""

    verdict = parse_verdict("VERDICT: not_met\nUNMET: 缺 A\nUNMET: 缺 B")
    assert verdict.outcome == "not_met"
    assert verdict.unmet == ("缺 A", "缺 B")

    # 验证器常把 UNMET 写在 VERDICT 同一行，也要能解析
    inline = parse_verdict("VERDICT: not_met UNMET: - 缺 A - 缺 B")
    assert inline.outcome == "not_met" and inline.unmet == ("缺 A", "缺 B")

    assert parse_verdict("VERDICT: met").outcome == "met"
    assert parse_verdict("VERDICT: met").unmet == ()
    # 解析不出来时按 inconclusive 处理，绝不能默认放行
    assert parse_verdict("我看挺好的").outcome == "inconclusive"


def test_check_verdict_shape() -> None:
    """CheckVerdict 的三值就是 met / not_met / inconclusive。"""

    assert CheckVerdict.__args__ == ("met", "not_met", "inconclusive")


# --- 11. UNMET 解析：把验证器真实用过的三种格式都锁住 ----------------------


def test_parse_verdict_multiline_bullets_after_marker() -> None:
    """marker 在行尾、条目在后续行——真实验证器就是这么写的（曾经解析成全空）。"""

    verdict = parse_verdict(
        "VERDICT: not_met UNMET:\n"
        "- 缺 A 小节\n"
        "- 缺 B 小节\n"
        "- 路径编造\n"
    )

    assert verdict.outcome == "not_met"
    assert verdict.unmet == ("缺 A 小节", "缺 B 小节", "路径编造")


def test_parse_verdict_standalone_unmet_section() -> None:
    """marker 单独成行、条目在后续行。"""

    verdict = parse_verdict("VERDICT: not_met\nUNMET:\n- 缺 A\n- 缺 B\n")

    assert verdict.outcome == "not_met"
    assert verdict.unmet == ("缺 A", "缺 B")


def test_parse_verdict_ignores_trailing_prose_after_items() -> None:
    """条目后面的普通说明不当作未满足项。"""

    verdict = parse_verdict("VERDICT: not_met\nUNMET:\n- 缺 A\n请继续推进。\n")

    assert verdict.unmet == ("缺 A",)
