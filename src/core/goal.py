"""维护跨模型回复持续存在的目标、预算与显式完成入口。

完成门（Completion Gate）：把"完成"的决定权从模型的自我声明，改成"独立验证器的判定"。
各层职责见 todo/completion-gate.md。
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Literal

from .end_policy import EndPolicySummary
from .model import Message, ToolCall, ToolResult, UsageEvent, UsageLedger
from .tools.args import string_argument
from .tools.types import ToolDefinition, ToolHandler


GoalStatus = Literal["active", "complete", "budget_limited"]
CheckVerdict = Literal["met", "not_met", "inconclusive"]
RejectionReason = Literal["not_met", "inconclusive", "verifier_error"]

# 独立验证器的审计要点：照抄 minimax-code 的精神（把完成当作未证实、必须证明完成）
VERIFIER_AUDIT_BRIEF = (
    "把完成当作未证实，对照当前实际状态逐条验证；"
    "对每一条验收项找出权威证据（文件内容、命令输出、测试结果、运行时行为）；"
    "不要用意图、部分进展、对早前工作的记忆、或一个看起来合理的答案当作完成的证据；"
    "不确定或间接的证据一律算未达成；"
    "审计必须“证明完成”，而不是“没找到明显的剩余工作”。"
    "只输出两段：`VERDICT: met|not_met|inconclusive`，以及 `UNMET:` 逐条列出未满足项（没有则写“无”）。"
)

# 防退化提醒（措辞照抄指令）
NO_TOOL_NUDGE = "复述计划、状态或意图不是进展。用工具检查当前证据，执行下一个具体动作再汇报。"
REPEATED_SUMMARY_NUDGE = "不要重复同样的总结或停在同一个点。选一个实质不同的下一步并执行。"
_NO_UNMET = {"", "无", "none", "n/a", "无。", "-"}


@dataclass(frozen=True)
class Verdict:
    """一次独立验证的结论。"""

    outcome: CheckVerdict
    unmet: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompletionOutcome:
    """完成门的判定结果。"""

    accepted: bool
    verified: bool
    outcome: CheckVerdict
    reason: RejectionReason | None
    unmet: tuple[str, ...]
    attempt: int
    message: str


@dataclass(frozen=True)
class CompletionGateEvent:
    """一次完成门判定的可观测记录。"""

    verdict: CheckVerdict
    unmet: tuple[str, ...]
    attempt: int
    accepted: bool
    verified: bool
    reason: RejectionReason | None = None


def verifier_brief(goal: "Goal") -> str:
    """返回判定依据：优先验收标准，为空才退回 objective。

    验收标准必须由用户/调用方提供，不得由模型生成——否则等于自证。
    评测里传入的验收标准必须是**泛化**的，绝不点名任何隐藏清单条目。
    """
    criteria = goal.acceptance_criteria.strip()
    return criteria or goal.objective


def verifier_task(goal: "Goal") -> str:
    """拼出交给独立验证器的任务文本。"""

    return (
        "独立审计下面这个目标的完成声明。\n"
        f"目标：{goal.objective}\n"
        f"验收标准：\n{verifier_brief(goal)}\n\n"
        "审计方式（务必按此执行，不要漫无目的地遍历仓库）：\n"
        "1. 先看交付物本身（报告、改动、产物），以它为主要证据；\n"
        "2. 对每条验收标准只做**有界抽查**：最多读少量能直接证实或证伪的文件、\n"
        "   必要时跑一条命令；**不要审计整个仓库，不要把时间花在通读代码上**；\n"
        "3. 判定只依据你实际看到的证据。\n\n"
        f"{VERIFIER_AUDIT_BRIEF}"
    )


def parse_verdict(text: str) -> Verdict:
    """解析验证器输出；解析不出来一律按 inconclusive 处理（绝不放行）。"""

    outcome: CheckVerdict = "inconclusive"
    unmet: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        upper = stripped.upper()
        if upper.startswith("VERDICT:"):
            value = stripped.split(":", 1)[1].strip().lower().replace("-", "_")
            if value in {"met", "not_met", "inconclusive"}:
                outcome = value
        elif upper.startswith("UNMET:"):
            item = stripped.split(":", 1)[1].strip()
            if item.lower() not in _NO_UNMET:
                unmet.append(item)
    if outcome == "met":
        return Verdict("met", ())
    return Verdict(outcome, tuple(unmet))


def _normalize_summary(text: str) -> str:
    """归一化总结：去空白与标点、统一大小写；只用于精确比较，不做模糊匹配。"""

    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


@dataclass
class Goal:
    """保存目标、预算与可持久化的执行进度。"""

    objective: str
    acceptance_criteria: str = ""
    max_rounds: int | None = None
    token_budget: int | None = None
    time_budget_seconds: int | None = None
    status: GoalStatus = "active"
    verified: bool = False
    completion_rejections: int = 0
    tokens_used: int = 0
    rounds_started: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        """拒绝空目标和非法预算，避免恢复损坏的记录。"""
        self.objective = self.objective.strip()
        if not self.objective:
            raise ValueError("goal objective must not be empty")
        for name in ("max_rounds", "token_budget", "time_budget_seconds"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"goal {name} must be a nonnegative integer")
        if self.status not in {"active", "complete", "budget_limited"}:
            raise ValueError("invalid goal status")
        if not isinstance(self.acceptance_criteria, str):
            raise ValueError("invalid goal acceptance_criteria")
        if type(self.verified) is not bool:
            raise ValueError("invalid goal verified")
        if type(self.completion_rejections) is not int or self.completion_rejections < 0:
            raise ValueError("invalid goal completion_rejections")
        if type(self.tokens_used) is not int or self.tokens_used < 0:
            raise ValueError("invalid goal tokens_used")
        if type(self.rounds_started) is not int or self.rounds_started < 0:
            raise ValueError("invalid goal rounds_started")
        if not isinstance(self.elapsed_seconds, int | float) or self.elapsed_seconds < 0:
            raise ValueError("invalid goal elapsed_seconds")


class GoalPolicy:
    """在模型停下时判断续跑、显式完成或预算收尾。"""

    def __init__(
        self,
        goal: Goal,
        *,
        completion_check: Callable[[], bool] | None = None,
        on_change: Callable[[Goal], None] | None = None,
        clock: Callable[[], float] = monotonic,
        usage_ledger: UsageLedger | None = None,
        verifier: Callable[[Goal], Awaitable[str]] | None = None,
        max_rejections: int = 2,
        verifier_timeout_seconds: float = 300.0,
        no_tool_nudge_rounds: int = 3,
        on_event: Callable[[object], Awaitable[None]] | None = None,
    ) -> None:
        """共享可变 Goal，并以回调持久化每次状态变化。"""
        self.goal = goal
        self._completion_check = completion_check
        self._on_change = on_change
        self._clock = clock
        self._last_tick = clock()
        self._closing_sent = False
        self._usage_ledger = usage_ledger
        # 完成门：配置了验证器才启用（否则保持旧的自我声明语义）
        self._verifier = verifier
        self._max_rejections = max(0, max_rejections)
        self._verifier_timeout_seconds = verifier_timeout_seconds
        self._no_tool_nudge_rounds = no_tool_nudge_rounds
        self._on_event = on_event
        self._rounds_without_tools = 0
        self._last_summary: str | None = None
        self._nudges: list[str] = []
        self._initial_tokens = goal.tokens_used
        self._ledger_baseline = usage_ledger.total_tokens if usage_ledger is not None else 0
        if usage_ledger is not None:
            usage_ledger.set_observer(self._observe_total_usage)

    @property
    def summary(self) -> EndPolicySummary:
        """兼容 AgentLoop 的通用收尾统计接口。"""
        return EndPolicySummary(
            False, 0, (), (), self.goal.completion_rejections
        )

    @property
    def final_response_only(self) -> bool:
        """预算收尾阶段不再向模型暴露工具。"""
        return self.goal.status == "budget_limited" and self._closing_sent

    def goal_summary(self) -> dict[str, object]:
        """返回用户和模型都可读取的目标状态快照。"""
        self._tick()
        return {
            "objective": self.goal.objective,
            "status": self.goal.status,
            "rounds_started": self.goal.rounds_started,
            "max_rounds": self.goal.max_rounds,
            "tokens_used": self.goal.tokens_used,
            "token_budget": self.goal.token_budget,
            "elapsed_seconds": round(self.goal.elapsed_seconds, 3),
            "time_budget_seconds": self.goal.time_budget_seconds,
        }

    def mark_complete(self) -> bool:
        """仅在 active 且可选硬验证通过后接受模型的完成声明。"""
        if self.goal.status != "active":
            return False
        if self._completion_check is not None and not self._completion_check():
            return False
        self._tick()
        self.goal.status = "complete"
        self._persist()
        return True

    def observe_usage(self, usage: UsageEvent) -> None:
        """累计服务端实际用量，供预算与 resume 使用。"""
        if self._usage_ledger is not None:
            return
        self._tick()
        self.goal.tokens_used += usage.total_tokens
        self._persist()

    def _observe_total_usage(self, total_tokens: int) -> None:
        """按共享总账的增量更新 Goal，避免主循环事件重复记账。"""
        self._tick()
        self.goal.tokens_used = self._initial_tokens + total_tokens - self._ledger_baseline
        self._persist()

    def observe_tool_results(
        self, tool_calls: Sequence[ToolCall], results: Sequence[ToolResult]
    ) -> None:
        """接收一批工具结果；启用完成门后，完成只能由 goal 工具（异步）判定。

        防绕过：配置了验证器时**不得**走这里的同步 mark_complete() 路径，
        否则模型只要不调用 goal(op=complete) 就能把自己标成完成，门就是纸做的。
        """
        self._rounds_without_tools = 0
        if self._verifier is not None:
            return
        for call, result in zip(tool_calls, results):
            if call.name == "goal" and call.arguments.get("op") == "complete" and not result.is_error:
                self.mark_complete()

    def follow_up_message(self, assistant_content: str = "") -> Message | None:
        """模型自然停下时续跑，预算耗尽时只收尾一次。"""
        self._tick()
        if self.goal.status != "active":
            return None
        if self._budget_exhausted():
            self.goal.status = "budget_limited"
            self._closing_sent = True
            self._persist()
            return Message(role="system", content="目标预算已用尽。只总结实际完成和未完成的部分，不再调用工具；预算耗尽不等于目标完成。")
        self._update_degeneration(assistant_content)
        self.goal.rounds_started += 1
        self._persist()
        return Message(role="system", content=self._continuation_text())

    def instruction_message(self) -> Message:
        """在目标开始或恢复时告知模型完整目标、验收标准与完成入口。

        长契约只在 kickoff 注入一次，后续 `follow_up_message` 只给短提示，
        以保持继承前缀稳定（对齐 prompt cache 的 cache-prefix stable 要求）。
        """
        criteria = self.goal.acceptance_criteria.strip()
        criteria_block = (
            f"\n验收标准（完成前必须逐条满足，并给出对应证据）：\n{criteria}\n"
            if criteria
            else ""
        )
        gate_block = (
            "完成声明会交给一个独立、只读的验证器对照验收标准核查："
            "未通过会被拒绝，并返回未满足项。\n"
            "不要把不确定或间接的证据当作完成的证据；审计必须“证明完成”，"
            "而不是“没找到明显的剩余工作”。\n"
            if self._verifier is not None
            else ""
        )
        return Message(
            role="system",
            content=(
                f"当前持续目标：{self.goal.objective}\n"
                f"{criteria_block}"
                "目标不会因为一次回复结束而结束。完成全部目标并审计证据后，"
                "必须调用 goal({\"op\":\"complete\"})；"
                "只回复‘完成’或只停止工具调用都不算完成。\n"
                f"{gate_block}"
            ),
        )

    async def request_completion(self) -> CompletionOutcome:
        """向完成门申请完成：跑独立验证器并据此接受或拒绝。"""

        attempt = self.goal.completion_rejections + 1
        if self.goal.status != "active":
            return CompletionOutcome(
                False, False, "inconclusive", None, (), attempt, "目标当前不是进行中状态"
            )
        if self._verifier is None:
            # 未配置验证器：退回原来的同步语义，行为与完成门之前一致
            accepted = self.mark_complete()
            message = "目标已显式完成" if accepted else "目标尚未通过完成检查，继续完成原目标"
            return CompletionOutcome(
                accepted, accepted, "met" if accepted else "inconclusive", None, (), attempt, message
            )
        verdict, reason, raw = await self._run_verifier()
        return await self._apply_verdict(verdict, reason, attempt, raw)

    async def _run_verifier(self) -> tuple[Verdict, RejectionReason | None, str]:
        """跑一次独立验证；超时按 inconclusive、抛错按 verifier_error。

        返回 (判定, 拒绝原因, 验证器原文节选)——原文要跟着拒绝消息回注给模型，
        否则"未满足项"只有一句话，模型拿不到可执行的信息。
        """

        assert self._verifier is not None
        try:
            raw = await asyncio.wait_for(
                self._verifier(self.goal), self._verifier_timeout_seconds
            )
        except (TimeoutError, asyncio.TimeoutError):
            # 超时是"证据不足"而不是"验证器坏了"
            return Verdict("inconclusive"), "inconclusive", "验证器超时，未能取得证据"
        except Exception as exc:  # noqa: BLE001 - 验证器不可用一律 fail-closed
            return Verdict("inconclusive"), "verifier_error", f"验证器不可用：{type(exc).__name__}"
        verdict = parse_verdict(raw)
        excerpt = re.sub(r"\s+", " ", raw).strip()[:600]
        return verdict, (None if verdict.outcome == "met" else verdict.outcome), excerpt

    async def _apply_verdict(
        self,
        verdict: Verdict,
        reason: RejectionReason | None,
        attempt: int,
        raw_excerpt: str = "",
    ) -> CompletionOutcome:
        """按判定结果接受、拒绝或（超限后）放行并标注未验证。"""

        if verdict.outcome == "met":
            self._tick()
            self.goal.status = "complete"
            self.goal.verified = True
            self._persist()
            outcome = CompletionOutcome(
                True, True, "met", None, (), attempt, "目标已完成，并通过独立验证。"
            )
        elif self.goal.completion_rejections < self._max_rejections:
            unmet_text = (
                "\n".join(f"- {item}" for item in verdict.unmet)
                or "- 验证器未给出具体未满足项，请自行复核对验收标准"
            )
            self.goal.completion_rejections += 1
            self._persist()
            outcome = CompletionOutcome(
                False,
                False,
                verdict.outcome,
                reason,
                verdict.unmet,
                attempt,
                "完成被拒绝：独立验证未通过（"
                + str(reason)
                + "）。\n未满足项：\n"
                + unmet_text
                + "\n请继续推进原目标，不要重复声明完成。",
            )
        else:
            self._tick()
            self.goal.status = "complete"
            self.goal.verified = False
            self._persist()
            outcome = CompletionOutcome(
                True,
                False,
                verdict.outcome,
                reason,
                verdict.unmet,
                attempt,
                "已完成，但未通过验证（已达拒绝上限 "
                + str(self._max_rejections)
                + " 次，放行）。请在总结中标注“未通过验证”。",
            )
        await self._emit(outcome)
        return outcome

    async def _emit(self, outcome: CompletionOutcome) -> None:
        """把判定结果发给观察者（评测据此统计拒绝次数与分类）。"""

        if self._on_event is None:
            return
        await self._on_event(
            CompletionGateEvent(
                verdict=outcome.outcome,
                unmet=outcome.unmet,
                attempt=outcome.attempt,
                accepted=outcome.accepted,
                verified=outcome.verified,
                reason=outcome.reason,
            )
        )

    def _update_degeneration(self, assistant_content: str) -> None:
        """更新防退化信号：连续无工具轮数 + 重复总结（只做归一化精确比较）。"""

        self._nudges = []
        self._rounds_without_tools += 1
        if self._no_tool_nudge_rounds > 0 and self._rounds_without_tools >= self._no_tool_nudge_rounds:
            self._nudges.append(NO_TOOL_NUDGE)
        normalized = _normalize_summary(assistant_content)
        if normalized and self._last_summary is not None and normalized == self._last_summary:
            self._nudges.append(REPEATED_SUMMARY_NUDGE)
        if normalized:
            self._last_summary = normalized

    def _continuation_text(self) -> str:
        """提醒模型维持原目标，不把阶段性成果冒充完成。"""
        return (
            "继续推进当前目标。目标在跨轮次持续存在，结束本轮不代表要把目标缩成现在能做完的较小版本。\n\n"
            f"目标：{self.goal.objective}\n"
            f"预算：已用 token {self.goal.tokens_used} / {self.goal.token_budget if self.goal.token_budget is not None else '不限'}，"
            f"已用轮次 {self.goal.rounds_started} / {self.goal.max_rounds if self.goal.max_rounds is not None else '不限'}。\n\n"
            "- 绝不允许把成功重新定义成一个更小、更容易、或已经完成的子集。\n"
            "- 预算耗尽 ≠ 完成，不要因为快没预算就声明完成。\n"
            "- 声明完成前必须审计当前仓库实际状态，用直接证据证明目标达成；"
            "覆盖范围不足、只间接证据、未检查的‘看起来对’，一律视为未完成，继续工作。\n"
            "- 只做推进目标的事，不要叙述‘我接下来要继续’。"
            + ("\n\n" + "\n".join(self._nudges) if self._nudges else "")
        )

    def _budget_exhausted(self) -> bool:
        """任一已设置预算达到上限即停止续跑。"""
        goal = self.goal
        return (
            goal.max_rounds is not None and goal.rounds_started >= goal.max_rounds
            or goal.token_budget is not None and goal.tokens_used >= goal.token_budget
            or goal.time_budget_seconds is not None and goal.elapsed_seconds >= goal.time_budget_seconds
        )

    def _tick(self) -> None:
        """把本进程已经经过的时间累积到可持久化字段。"""
        now = self._clock()
        self.goal.elapsed_seconds += max(0.0, now - self._last_tick)
        self._last_tick = now

    def _persist(self) -> None:
        """向 Session 回调追加当前目标快照。"""
        if self._on_change is not None:
            self._on_change(self.goal)


def create_goal_tool(policy_provider: Callable[[], GoalPolicy | None]) -> tuple[ToolDefinition, ToolHandler]:
    """创建无需用户审批的目标状态与显式完成工具。"""
    async def handle(call: ToolCall) -> ToolResult:
        policy = policy_provider()
        if policy is None:
            return ToolResult(call.call_id, "当前没有目标", is_error=True)
        operation = string_argument(call, "op")
        if operation == "status":
            return ToolResult(call.call_id, json.dumps(policy.goal_summary(), ensure_ascii=False))
        if operation != "complete":
            return ToolResult(call.call_id, f"不支持的 goal 操作：{operation}", is_error=True)
        outcome = await policy.request_completion()
        return ToolResult(call.call_id, outcome.message, is_error=not outcome.accepted)

    return (
        ToolDefinition(
            name="goal",
            description="查看持续目标状态，或在审计整个目标确实完成后用 op=complete 显式声明完成。阶段完成不算。",
            parameters={
                "type": "object",
                "properties": {"op": {"type": "string", "enum": ["status", "complete"]}},
                "required": ["op"],
            },
            source="local",
            permission="read",
            idempotent=False,
        ),
        handle,
    )
