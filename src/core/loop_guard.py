"""Agent 空转检测与纠偏（Loop Guard）。

把"空转"当作信号而不是时长问题：检测到局部循环后，注入一条系统消息把模型
拽出来，而不设置任何轮次或时间上限（生产长任务不能被掐死）。

两种检测形态：
- 检测 A：同一个工具调用（名称 + 规范化参数）连续多轮重复出现；
- 检测 B：连续多轮没有任何"有进展"的工具调用（成功且 capability 不是
  ``file.read`` 的调用才算进展，委派与命令同样计为进展）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .model import Message, ToolCall, ToolResult

if TYPE_CHECKING:  # 仅用于类型标注，避免 config → loop_guard → model 的循环导入
    from .config import Settings

# 达到最后一个阈值后，每隔多少轮继续施压一次（检测 A/B 共用节流口径）
REPEAT_REMINDER_INTERVAL = 4
# 只读角色天然没有写入，跳过"无进展"检测，避免误报
READ_ONLY_ROLES = frozenset({"scout"})
# 只看调用签名，不看结果；这里统一截断参数与结果摘要长度
EXCERPT_LIMIT = 200

REPEATED_CALL_REMINDER = (
    "你在用完全相同的参数重复调用同一个工具。先仔细分析上一次的结果："
    "如果任务还没完成，请换一种方法或换参数，不要重复同样的调用。"
)

DetectionKind = Literal["repeated_call", "no_progress"]
ReminderLevel = Literal["mild", "detailed"]


@dataclass(frozen=True)
class LoopGuardConfig:
    """Loop Guard 的可配置项。"""

    enabled: bool = True
    thresholds: tuple[int, ...] = (3, 5, 8)
    exempt_tools: tuple[str, ...] = ()
    no_progress_rounds: int = 8

    def __post_init__(self) -> None:
        """校验阈值为非空递增正整数，无进展轮数非负。"""

        if not self.thresholds:
            raise ValueError("thresholds must not be empty")
        previous = 0
        for threshold in self.thresholds:
            if threshold <= previous:
                raise ValueError("thresholds must be strictly increasing positive ints")
            previous = threshold
        if self.no_progress_rounds < 0:
            raise ValueError("no_progress_rounds must be >= 0")


def config_from_settings(settings: "Settings") -> LoopGuardConfig:
    """从运行配置构造空转检测配置。"""

    return LoopGuardConfig(
        enabled=settings.loop_guard_enabled,
        thresholds=settings.loop_guard_thresholds,
        exempt_tools=settings.loop_guard_exempt_tools,
        no_progress_rounds=settings.loop_guard_no_progress_rounds,
    )


@dataclass(frozen=True)
class LoopGuardEvent:
    """一次纠偏注入的可观测记录。"""

    kind: DetectionKind
    level: ReminderLevel
    tool_name: str | None
    count: int


@dataclass(frozen=True)
class LoopGuardInjection:
    """一次纠偏注入：注入的消息与对应事件。"""

    message: Message
    event: LoopGuardEvent


def _canonical_arguments(arguments: Mapping[str, object]) -> str:
    """把工具参数规范化成稳定字符串，用作调用签名的一部分。"""

    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    """截断长文本，避免纠偏消息本身撑大上下文。"""

    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def _detailed_repeat_message(
    tool_name: str,
    count: int,
    arguments: str,
    last_result: str,
) -> str:
    """渲染重复调用的详细提醒。"""

    return (
        "检测到重复工具调用：\n"
        f"- 工具：{tool_name}\n"
        f"- 连续次数：{count}\n"
        f"- 参数：{_excerpt(arguments)}\n"
        f"- 上次结果：{_excerpt(last_result)}\n"
        "不要继续重复；换策略，或明确说明当前阻塞。"
    )


def _no_progress_message(count: int) -> str:
    """渲染长期无进展的提醒。"""

    return (
        f"你已经连续 {count} 轮没有产生任何文件改动或命令执行结果。"
        "重新评估：是卡住了、缺信息，还是该换个方案？如果确实无法推进，请说明阻塞原因。"
    )


class ToolCallLoopGuard:
    """检测单个 Agent 的空转并给出纠偏消息。

    每个 AgentLoop（含每个子 Agent）各持一个实例，计数互不影响。
    """

    def __init__(self, config: LoopGuardConfig | None = None, role: str = "parent") -> None:
        """创建 guard，记录角色以决定是否启用无进展检测。"""

        self._config = config or LoopGuardConfig()
        self._role = role
        self._counts: dict[str, int] = {}
        self._signatures: dict[str, tuple[str, str]] = {}
        self._last_results: dict[str, str] = {}
        self._no_progress = 0
        self._injections = 0

    @property
    def injections(self) -> int:
        """返回累计注入次数。"""

        return self._injections

    def observe(
        self,
        tool_calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
        capabilities: Mapping[str, str | None],
    ) -> LoopGuardInjection | None:
        """观察一轮工具结果，命中空转时返回一条纠偏注入。"""

        if not self._config.enabled:
            return None
        calls = tuple(tool_calls)
        completed = tuple(results)
        if self._has_progress(calls, completed, capabilities):
            # 有进展说明循环在推进，所有计数归零
            self._reset()
            return None

        self._no_progress += 1
        candidates = self._update_signature_counts(calls, completed)
        repeated = self._repeated_call_injection(candidates)
        if repeated is not None:
            return self._emit(repeated)
        no_progress = self._no_progress_injection()
        if no_progress is not None:
            return self._emit(no_progress)
        return None

    def _has_progress(
        self,
        calls: tuple[ToolCall, ...],
        results: tuple[ToolResult, ...],
        capabilities: Mapping[str, str | None],
    ) -> bool:
        """判断本轮是否存在"有进展"的调用：成功且不是纯读文件。"""

        for call, result in zip(calls, results):
            if result.is_error:
                continue
            if capabilities.get(call.name) != "file.read":
                return True
        return False

    def _update_signature_counts(
        self,
        calls: tuple[ToolCall, ...],
        results: tuple[ToolResult, ...],
    ) -> dict[str, int]:
        """按签名更新连续计数，返回本轮出现的签名计数。"""

        present: dict[str, tuple[str, str]] = {}
        for call, result in zip(calls, results):
            if call.name in self._config.exempt_tools:
                continue
            key = self._signature(call)
            present[key] = (call.name, _canonical_arguments(call.arguments))
            self._last_results[key] = result.content

        for key in list(self._counts):
            if key not in present:
                # 该签名本轮没有出现，计数归零
                self._counts.pop(key, None)
                self._signatures.pop(key, None)
                self._last_results.pop(key, None)

        for key, (name, arguments) in present.items():
            self._counts[key] = self._counts.get(key, 0) + 1
            self._signatures[key] = (name, arguments)
        return {key: self._counts[key] for key in present}

    def _signature(self, call: ToolCall) -> str:
        """把调用规范化为 `名称 + 排序参数` 的签名。"""

        return f"{call.name}:{_canonical_arguments(call.arguments)}"

    def _repeated_call_injection(
        self,
        candidates: Mapping[str, int],
    ) -> LoopGuardInjection | None:
        """返回计数最高且达到阈值的重复调用提醒。"""

        best_key: str | None = None
        best_level: ReminderLevel | None = None
        for key, count in candidates.items():
            level = self._repeat_level(count)
            if level is None:
                continue
            if best_key is None or count > candidates[best_key]:
                best_key = key
                best_level = level
        if best_key is None or best_level is None:
            return None

        tool_name, arguments = self._signatures[best_key]
        count = candidates[best_key]
        if best_level == "mild":
            content = REPEATED_CALL_REMINDER
        else:
            content = _detailed_repeat_message(
                tool_name,
                count,
                arguments,
                self._last_results.get(best_key, ""),
            )
        return LoopGuardInjection(
            Message(role="system", content=content),
            LoopGuardEvent("repeated_call", best_level, tool_name, count),
        )

    def _repeat_level(self, count: int) -> ReminderLevel | None:
        """按阈值决定提醒级别：命中阈值为温和/详细，之后每 4 轮继续施压。"""

        thresholds = self._config.thresholds
        if count in thresholds:
            return "mild" if count == thresholds[0] else "detailed"
        last = thresholds[-1]
        if count > last and (count - last) % REPEAT_REMINDER_INTERVAL == 0:
            return "detailed"
        return None

    def _no_progress_injection(self) -> LoopGuardInjection | None:
        """连续无进展达到阈值时返回提醒，之后按节流继续施压。"""

        limit = self._config.no_progress_rounds
        if limit == 0 or self._role in READ_ONLY_ROLES:
            return None
        count = self._no_progress
        if count < limit:
            return None
        if count != limit and (count - limit) % REPEAT_REMINDER_INTERVAL != 0:
            return None
        return LoopGuardInjection(
            Message(role="system", content=_no_progress_message(count)),
            LoopGuardEvent("no_progress", "detailed", None, count),
        )

    def _emit(self, injection: LoopGuardInjection) -> LoopGuardInjection:
        """记录注入次数并返回该注入。"""

        self._injections += 1
        return injection

    def _reset(self) -> None:
        """清空所有计数。"""

        self._counts.clear()
        self._signatures.clear()
        self._last_results.clear()
        self._no_progress = 0
