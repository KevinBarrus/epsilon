"""维护跨模型回复持续存在的目标、预算与显式完成入口。"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from .end_policy import EndPolicySummary
from .model import Message, ToolCall, ToolResult, UsageEvent
from .tools.args import string_argument
from .tools.types import ToolDefinition, ToolHandler


GoalStatus = Literal["active", "complete", "budget_limited"]


@dataclass
class Goal:
    """保存目标、预算与可持久化的执行进度。"""

    objective: str
    max_rounds: int | None = None
    token_budget: int | None = None
    time_budget_seconds: int | None = None
    status: GoalStatus = "active"
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
    ) -> None:
        """共享可变 Goal，并以回调持久化每次状态变化。"""
        self.goal = goal
        self._completion_check = completion_check
        self._on_change = on_change
        self._clock = clock
        self._last_tick = clock()
        self._closing_sent = False

    @property
    def summary(self) -> EndPolicySummary:
        """兼容 AgentLoop 的通用收尾统计接口。"""
        return EndPolicySummary(False, 0, (), ())

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
        self._tick()
        self.goal.tokens_used += usage.total_tokens
        self._persist()

    def observe_tool_results(
        self, tool_calls: Sequence[ToolCall], results: Sequence[ToolResult]
    ) -> None:
        """只接受成功执行的 goal(complete)，拒绝工具错误伪装完成。"""
        for call, result in zip(tool_calls, results):
            if call.name == "goal" and call.arguments.get("op") == "complete" and not result.is_error:
                self.mark_complete()

    def follow_up_message(self) -> Message | None:
        """模型自然停下时续跑，预算耗尽时只收尾一次。"""
        self._tick()
        if self.goal.status != "active":
            return None
        if self._budget_exhausted():
            self.goal.status = "budget_limited"
            self._closing_sent = True
            self._persist()
            return Message(role="system", content="目标预算已用尽。只总结实际完成和未完成的部分，不再调用工具；预算耗尽不等于目标完成。")
        self.goal.rounds_started += 1
        self._persist()
        return Message(role="system", content=self._continuation_text())

    def instruction_message(self) -> Message:
        """在目标开始或恢复时告知模型完整目标与完成入口。"""
        return Message(
            role="system",
            content=(
                f"当前持续目标：{self.goal.objective}\n"
                "目标不会因为一次回复结束而结束。完成全部目标并审计证据后，必须调用 goal({\"op\":\"complete\"})；"
                "只回复‘完成’或只停止工具调用都不算完成。"
            ),
        )

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
        if not policy.mark_complete():
            return ToolResult(call.call_id, "目标尚未通过完成检查，继续完成原目标", is_error=True)
        return ToolResult(call.call_id, "目标已显式完成")

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
